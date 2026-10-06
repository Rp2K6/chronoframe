"""Unit test for TemporalBlock and FrozenSpatialWithTemporal."""
import pytest
import torch
from models.temporal_blocks import TemporalBlock, FrozenSpatialWithTemporal


def test_temporal_block_zero_init_identity():
    B, T, C, H, W = 2, 4, 32, 8, 8
    x = torch.randn(B * T, C, H, W)
    block = TemporalBlock(channels=C, heads=4)
    
    # At init alpha=0.0, output must exactly match input
    out = block(x, num_frames=T)
    assert torch.allclose(out, x, atol=1e-6), "TemporalBlock must preserve identity at initialization when alpha=0.0"
    assert out.shape == x.shape


def test_temporal_block_alpha_activation():
    B, T, C, H, W = 2, 4, 32, 8, 8
    x = torch.randn(B * T, C, H, W)
    block = TemporalBlock(channels=C, heads=4)
    block.alpha.data.fill_(1.0)
    
    out = block(x, num_frames=T)
    assert out.shape == (B * T, C, H, W)
    assert not torch.allclose(out, x), "When alpha > 0, TemporalBlock must apply temporal modifications"


def test_frozen_spatial_wrapper():
    class DummySpatial(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(16, 16, 3, padding=1)
        def forward(self, x):
            return self.conv(x)

    dummy = DummySpatial()
    wrapper = FrozenSpatialWithTemporal(dummy, channels=16)
    
    # Verify spatial parameters are frozen
    for param in wrapper.spatial.parameters():
        assert not param.requires_grad, "Spatial parameters must be frozen"

    x = torch.randn(4, 16, 8, 8)
    out = wrapper(x, num_frames=2)
    assert out.shape == x.shape
