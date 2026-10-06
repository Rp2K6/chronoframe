"""Contribution 2: Joint Multi-Frame Latent Interpolation (T -> 4T-3 expansion).

JointInterpolator predicts all 3 intermediate latents between keyframe pairs in ONE forward pass
(T keyframes -> 4T-3 frames).

SingleFrameInterpolator is the baseline that predicts one frame per forward pass
(Masked Frame Interpolation / MFI), providing a real empirical benchmark for 3x speedup.
"""
import time
import torch
import torch.nn as nn


class _ResBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class JointInterpolator(nn.Module):
    """Proposed Model: Jointly predicts 3 intermediate frames for keyframe pairs in a single forward pass."""

    def __init__(self, latent_ch: int = 4, hidden_ch: int = 64, n_interp: int = 3, n_blocks: int = 3):
        super().__init__()
        self.n_interp = n_interp
        self.latent_ch = latent_ch
        
        self.inp_conv = nn.Conv2d(2 * latent_ch, hidden_ch, 3, padding=1)
        self.blocks = nn.Sequential(*[_ResBlock2D(hidden_ch) for _ in range(n_blocks)])
        self.out_conv = nn.Conv2d(hidden_ch, n_interp * latent_ch, 3, padding=1)
        
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _linear_interpolation(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Linear interpolation baseline between keyframe pairs a and b."""
        ts = [(i + 1) / (self.n_interp + 1) for i in range(self.n_interp)]
        return torch.stack([(1.0 - t) * a + t * b for t in ts], dim=1)

    def predict_pairs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Predicts all n_interp intermediate latents between keyframe tensors a and b in 1 forward pass."""
        cat_pair = torch.cat([a, b], dim=1)
        feat = self.blocks(self.inp_conv(cat_pair))
        residual = self.out_conv(feat)
        
        residual = residual.reshape(a.shape[0], self.n_interp, self.latent_ch, *a.shape[2:])
        linear = self._linear_interpolation(a, b)
        return linear + residual

    def forward(self, keyframes: torch.Tensor) -> torch.Tensor:
        """Keyframes shape: (B, T, C, H, W) -> Output shape: (B, (T-1)*(n_interp+1)+1, C, H, W)."""
        B, T, C, H, W = keyframes.shape
        if T <= 1:
            return keyframes
        
        a = keyframes[:, :-1].reshape(-1, C, H, W)
        b = keyframes[:, 1:].reshape(-1, C, H, W)

        mid_latents = self.predict_pairs(a, b).reshape(B, T - 1, self.n_interp, C, H, W)

        sequence_chunks = torch.cat([keyframes[:, :-1].unsqueeze(2), mid_latents], dim=2)
        sequence_flat = sequence_chunks.reshape(B, (T - 1) * (self.n_interp + 1), C, H, W)

        return torch.cat([sequence_flat, keyframes[:, -1:]], dim=1)


class SingleFrameInterpolator(JointInterpolator):
    """Baseline Model: Masked Frame Interpolation (MFI) approach that predicts 1 frame per forward pass."""

    def __init__(self, latent_ch: int = 4, hidden_ch: int = 64, n_interp: int = 3, n_blocks: int = 3):
        super().__init__(latent_ch, hidden_ch, n_interp, n_blocks)
        self.inp_conv = nn.Conv2d(2 * latent_ch + 1, hidden_ch, 3, padding=1)
        self.out_conv = nn.Conv2d(hidden_ch, latent_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def predict_pairs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Sequential single-frame passes per time step t (3 forward passes per keyframe pair)."""
        frames = []
        for i in range(self.n_interp):
            t_val = (i + 1) / (self.n_interp + 1)
            t_map = torch.full_like(a[:, :1], t_val)
            cat_input = torch.cat([a, b, t_map], dim=1)
            feat = self.blocks(self.inp_conv(cat_input))
            res = self.out_conv(feat)
            linear = (1.0 - t_val) * a + t_val * b
            frames.append(linear + res)
        return torch.stack(frames, dim=1)
