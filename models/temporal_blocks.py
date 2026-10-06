"""Contribution 1: Separate Temporal Blocks (Conv1D + Temporal Attention with learnable merging).

As specified in ImproveYourVideos / KandinskyVideo architecture:
Instead of interleaving temporal layers inside pretrained spatial blocks,
keep the frozen spatial block as is and add a *separate* temporal block after it.
The temporal block output is merged with its input through a learnable scalar parameter
'alpha' initialized to 0.0, ensuring training starts from the exact pretrained state.
"""
import torch
import torch.nn as nn


def _calculate_groups(channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class TemporalBlock(nn.Module):
    """Separate Temporal Block combining Conv1D and Multihead Temporal Attention with zero-init alpha merging."""

    def __init__(self, channels: int, heads: int = 8, kernel_size: int = 3):
        super().__init__()
        assert channels % heads == 0, f"channels ({channels}) must be divisible by heads ({heads})"
        self.channels = channels
        self.heads = heads
        
        self.norm_conv = nn.GroupNorm(_calculate_groups(channels), channels)
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        
        # Learnable temporal merge parameter, initialized to 0.0 for identity mapping at start
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Input x shape: (B * T, C, H, W).
        Returns x shape: (B * T, C, H, W).
        """
        bt, c, h, w = x.shape
        b = bt // num_frames

        # Reshape to treat every spatial location (h, w) across batch b as a sequence of length T
        # (B*T, C, H, W) -> (B, T, C, H, W) -> (B, H, W, C, T) -> (B*H*W, C, T)
        seq = x.reshape(b, num_frames, c, h, w).permute(0, 3, 4, 2, 1).reshape(b * h * w, c, num_frames)

        # 1. 1D Temporal Convolution
        delta_conv = self.conv(self.norm_conv(seq))  # (B*H*W, C, T)

        # 2. Multi-head Temporal Attention
        tokens = self.norm_attn(seq.transpose(1, 2))  # (B*H*W, T, C)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        delta_attn = attn_out.transpose(1, 2)  # (B*H*W, C, T)

        # Combine temporal convolution and temporal attention
        delta = delta_conv + delta_attn

        # 3. Learnable Temporal Merging (identity at init when alpha = 0.0)
        out = seq + self.alpha * delta  # (B*H*W, C, T)

        # Reshape back to (B * T, C, H, W)
        out = out.reshape(b, h, w, c, num_frames).permute(0, 4, 3, 1, 2).reshape(bt, c, h, w)
        return out


class FrozenSpatialWithTemporal(nn.Module):
    """Wraps an existing spatial block; freezes its spatial parameters; appends a trainable TemporalBlock."""

    def __init__(self, spatial_module: nn.Module, channels: int, **kwargs):
        super().__init__()
        self.spatial = spatial_module
        for param in self.spatial.parameters():
            param.requires_grad = False
        self.temporal = TemporalBlock(channels, **kwargs)

    def forward(self, x: torch.Tensor, num_frames: int, *args, **kwargs) -> torch.Tensor:
        x = self.spatial(x, *args, **kwargs)
        return self.temporal(x, num_frames)
