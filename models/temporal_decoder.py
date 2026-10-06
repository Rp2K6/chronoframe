"""Contribution 3: Lightweight Temporal MoVQ-GAN-inspired Decoder.

Extends standard 2D image decoder with 3x3x3 Temporal Convolutions and Temporal Attention layers,
zero-initialized so it starts with identical reconstruction capabilities as the image decoder,
while eliminating frame flickering artifacts across video sequences.
"""
import torch
import torch.nn as nn
from models.temporal_blocks import TemporalBlock


class TemporalConv3D(nn.Module):
    """3x3x3 Temporal Convolution zero-initialized for flicker-free temporal consistency."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Input x shape: (B * T, C, H, W). Output shape: (B * T, C, H, W)."""
        bt, c, h, w = x.shape
        b = bt // num_frames
        
        # (B*T, C, H, W) -> (B, C, T, H, W)
        v = x.reshape(b, num_frames, c, h, w).permute(0, 2, 1, 3, 4)
        v = v + self.conv(v)
        
        # (B, C, T, H, W) -> (B*T, C, H, W)
        return v.permute(0, 2, 1, 3, 4).reshape(bt, c, h, w)


class TemporalDecoderLayer(nn.Module):
    """Combined 3D Temporal Convolution and Temporal Attention layer for MoVQ-GAN video decoding."""

    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        self.conv3d = TemporalConv3D(channels)
        self.attn = TemporalBlock(channels, heads=heads)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        return self.attn(self.conv3d(x, num_frames), num_frames)


class TinyEncoder(nn.Module):
    """Encoder mapping RGB video frames into 4-channel latent space."""

    def __init__(self, latent_ch: int = 4, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base_ch, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, latent_ch, 4, stride=2, padding=1),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v: (B, T, 3, H, W) -> z: (B, T, C, H/4, W/4)."""
        B, T, C, H, W = v.shape
        z = self.net(v.reshape(B * T, C, H, W))
        return z.reshape(B, T, *z.shape[1:])


class TinyDecoder(nn.Module):
    """Decoder mapping 4-channel latents back to RGB video frames.
    
    Setting temporal=True activates the Lightweight Temporal MoVQ-GAN-inspired Decoder.
    Setting temporal=False operates as a standard 2D Image Decoder baseline.
    """

    def __init__(self, latent_ch: int = 4, base_ch: int = 64, temporal: bool = True):
        super().__init__()
        self.temporal = temporal
        self.inp_conv = nn.Conv2d(latent_ch, base_ch, 3, padding=1)
        
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(base_ch, base_ch, 3, padding=1),
                nn.GroupNorm(8, base_ch),
                nn.SiLU()
            )
            for _ in range(2)
        ])
        
        self.tlayers = nn.ModuleList([TemporalDecoderLayer(base_ch) for _ in range(2)]) if temporal else None
        self.out_conv = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, C, H, W) -> RGB: (B, T, 3, 4*H, 4*W)."""
        B, T, C, H, W = z.shape
        x = self.inp_conv(z.reshape(B * T, C, H, W))
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.temporal and self.tlayers is not None:
                x = self.tlayers[i](x, T)
                
        x = self.out_conv(x)
        return x.reshape(B, T, 3, H * 4, W * 4)
