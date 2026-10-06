"""Contribution 1: separate Temporal Blocks (Conv1D + temporal attention, learnable merge).

Idea from ImproveYourVideos / FusionFrames: instead of interleaving temporal layers
inside the spatial blocks, keep the (frozen) spatial block as is and add a *separate*
temporal block after it. The block's output is merged with its input through a
learnable scalar that starts at 0, so training starts from the pretrained image model.
"""
import torch
import torch.nn as nn


def _groups(channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return g


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 8, kernel_size: int = 3):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.norm_conv = nn.GroupNorm(_groups(channels), channels)
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.alpha = nn.Parameter(torch.zeros(1))  # learnable merge, identity at init

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """x: (B*T, C, H, W) -> (B*T, C, H, W)."""
        bt, c, h, w = x.shape
        b = bt // num_frames
        # every spatial location becomes one sequence of length T
        seq = x.reshape(b, num_frames, c, h, w).permute(0, 3, 4, 2, 1).reshape(b * h * w, c, num_frames)
        delta = self.conv(self.norm_conv(seq))                       # temporal Conv1D
        tokens = self.norm_attn(seq.transpose(1, 2))                 # (N, T, C)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        delta = delta + attn_out.transpose(1, 2)                     # + temporal attention
        out = seq + self.alpha * delta                               # learnable merging
        return out.reshape(b, h, w, c, num_frames).permute(0, 4, 3, 1, 2).reshape(bt, c, h, w)


class FrozenSpatialWithTemporal(nn.Module):
    """Wrap an existing (pretrained) spatial block; freeze it; add a trainable TemporalBlock."""

    def __init__(self, spatial: nn.Module, channels: int, **kw):
        super().__init__()
        self.spatial = spatial
        for p in self.spatial.parameters():
            p.requires_grad = False
        self.temporal = TemporalBlock(channels, **kw)

    def forward(self, x, num_frames, *args, **kwargs):
        x = self.spatial(x, *args, **kwargs)
        return self.temporal(x, num_frames)
