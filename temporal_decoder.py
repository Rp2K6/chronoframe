"""Contribution 3: temporal video decoder (3x3x3 Conv + temporal attention).

Real MoVQ-GAN weights live in the KandinskyVideo repo. Here we implement the *idea*
(extend an image decoder with temporal 3D conv + temporal attention, zero-initialised
so it starts as the image decoder) on a tiny decoder you can train on a laptop.
"""
import torch
import torch.nn as nn
from temporal_blocks import TemporalBlock


class TemporalConv3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, num_frames):
        bt, c, h, w = x.shape
        b = bt // num_frames
        v = x.reshape(b, num_frames, c, h, w).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        v = v + self.conv(v)
        return v.permute(0, 2, 1, 3, 4).reshape(bt, c, h, w)


class TemporalDecoderLayer(nn.Module):
    def __init__(self, ch, heads=8):
        super().__init__()
        self.conv3d = TemporalConv3D(ch)
        self.attn = TemporalBlock(ch, heads=heads)

    def forward(self, x, num_frames):
        return self.attn(self.conv3d(x, num_frames), num_frames)


class TinyEncoder(nn.Module):
    def __init__(self, latent_ch=4, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.SiLU(), nn.Conv2d(ch, latent_ch, 4, 2, 1)
        )

    def forward(self, v):  # (B, T, 3, S, S) -> (B, T, C, S/4, S/4)
        B, T, C, H, W = v.shape
        z = self.net(v.reshape(B * T, C, H, W))
        return z.reshape(B, T, *z.shape[1:])


class TinyDecoder(nn.Module):
    """Image decoder (temporal=False) or temporal decoder (temporal=True)."""

    def __init__(self, latent_ch=4, ch=64, temporal=True):
        super().__init__()
        self.temporal = temporal
        self.inp = nn.Conv2d(latent_ch, ch, 3, padding=1)
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                          nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.SiLU())
            for _ in range(2)
        ])
        self.tlayers = nn.ModuleList([TemporalDecoderLayer(ch) for _ in range(2)]) if temporal else None
        self.out = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, z):  # (B, T, C, h, w) -> (B, T, 3, 4h, 4w)
        B, T, C, H, W = z.shape
        x = self.inp(z.reshape(B * T, C, H, W))
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.temporal:
                x = self.tlayers[i](x, T)
        x = self.out(x)
        return x.reshape(B, T, 3, H * 4, W * 4)
