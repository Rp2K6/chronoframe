"""Unit test for TinyDecoder and TemporalConv3D."""
import pytest
import torch
from models.temporal_decoder import TinyDecoder, TemporalConv3D


def test_temporal_conv3d_shape_and_zero_init():
    B, T, C, H, W = 2, 4, 32, 8, 8
    x = torch.randn(B * T, C, H, W)
    tconv = TemporalConv3D(channels=C)
    
    out = tconv(x, num_frames=T)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-6), "TemporalConv3D zero-initialization must start as identity mapping"


def test_tiny_decoder_2d_vs_temporal():
    B, T, C, H, W = 1, 4, 4, 8, 8
    z = torch.randn(B, T, C, H, W)
    
    dec_2d = TinyDecoder(temporal=False).eval()
    dec_temp = TinyDecoder(temporal=True).eval()
    
    with torch.no_grad():
        out_2d = dec_2d(z)
        out_temp = dec_temp(z)
        
    expected_shape = (B, T, 3, H * 4, W * 4)  # (1, 4, 3, 32, 32)
    assert out_2d.shape == expected_shape
    assert out_temp.shape == expected_shape
