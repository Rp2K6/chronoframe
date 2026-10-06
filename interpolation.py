"""Contribution 2: joint multi-frame latent interpolation.

JointInterpolator predicts all 3 in-between latents for a keyframe pair in ONE forward
pass (T keyframes -> 4T-3 frames). SingleFrameInterpolator is the baseline that predicts
one frame per pass (like Masked Frame Interpolation), so you can benchmark the speed-up.
NOTE: this is a lightweight regression version for a mini project, not the paper's
diffusion U-Net.
"""
import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class JointInterpolator(nn.Module):
    def __init__(self, latent_ch=4, hidden=64, n_interp=3, n_blocks=3):
        super().__init__()
        self.n, self.c = n_interp, latent_ch
        self.inp = nn.Conv2d(2 * latent_ch, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*[_ResBlock(hidden) for _ in range(n_blocks)])
        self.out = nn.Conv2d(hidden, n_interp * latent_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _linear(self, a, b):
        ts = [(i + 1) / (self.n + 1) for i in range(self.n)]
        return torch.stack([(1 - t) * a + t * b for t in ts], dim=1)  # (N, n, C, H, W)

    def predict_pairs(self, a, b):
        res = self.out(self.blocks(self.inp(torch.cat([a, b], dim=1))))
        res = res.reshape(a.shape[0], self.n, self.c, *a.shape[2:])
        return self._linear(a, b) + res

    def forward(self, keyframes):
        """keyframes: (B, T, C, H, W) -> (B, (T-1)*(n+1)+1, C, H, W)  (= 4T-3 for n=3)."""
        B, T, C, H, W = keyframes.shape
        a = keyframes[:, :-1].reshape(-1, C, H, W)
        b = keyframes[:, 1:].reshape(-1, C, H, W)
        mid = self.predict_pairs(a, b).reshape(B, T - 1, self.n, C, H, W)
        seq = torch.cat([keyframes[:, :-1].unsqueeze(2), mid], dim=2)
        seq = seq.reshape(B, (T - 1) * (self.n + 1), C, H, W)
        return torch.cat([seq, keyframes[:, -1:]], dim=1)


class SingleFrameInterpolator(JointInterpolator):
    """Baseline: one in-between frame per forward pass, conditioned on time t."""

    def __init__(self, latent_ch=4, hidden=64, n_interp=3, n_blocks=3):
        super().__init__(latent_ch, hidden, n_interp, n_blocks)
        self.inp = nn.Conv2d(2 * latent_ch + 1, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, latent_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def predict_pairs(self, a, b):
        frames = []
        for i in range(self.n):
            t = (i + 1) / (self.n + 1)
            tmap = torch.full_like(a[:, :1], t)
            res = self.out(self.blocks(self.inp(torch.cat([a, b, tmap], dim=1))))
            frames.append((1 - t) * a + t * b + res)
        return torch.stack(frames, dim=1)
