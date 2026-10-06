"""Unit test for JointInterpolator and SingleFrameInterpolator (T -> 4T-3)."""
import pytest
import torch
from models.interpolation import JointInterpolator, SingleFrameInterpolator


def test_joint_interpolator_shape_expansion():
    B, T, C, H, W = 2, 4, 4, 8, 8
    keyframes = torch.randn(B, T, C, H, W)
    model = JointInterpolator(latent_ch=C, n_interp=3).eval()
    
    with torch.no_grad():
        out = model(keyframes)
        
    expected_frames = 4 * T - 3  # For T=4, 4(4) - 3 = 13
    assert out.shape == (B, expected_frames, C, H, W), f"Expected shape ({B}, {expected_frames}, {C}, {H}, {W}), got {out.shape}"


def test_single_frame_interpolator_shape_expansion():
    B, T, C, H, W = 2, 4, 4, 8, 8
    keyframes = torch.randn(B, T, C, H, W)
    model = SingleFrameInterpolator(latent_ch=C, n_interp=3).eval()
    
    with torch.no_grad():
        out = model(keyframes)
        
    expected_frames = 4 * T - 3  # 13
    assert out.shape == (B, expected_frames, C, H, W)


def test_six_keyframe_expansion():
    B, T, C, H, W = 1, 6, 4, 16, 16
    keyframes = torch.randn(B, T, C, H, W)
    model = JointInterpolator(latent_ch=C, n_interp=3).eval()
    
    with torch.no_grad():
        out = model(keyframes)
        
    expected_frames = 4 * 6 - 3  # 21
    assert out.shape == (B, 21, C, H, W)
