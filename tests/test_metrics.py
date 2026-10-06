"""Unit test for evaluation metrics in utils/metrics.py."""
import pytest
import torch
from utils.metrics import psnr, ssim, temporal_flicker, clipsim


def test_psnr_identical():
    a = torch.rand(1, 4, 3, 16, 16)
    assert psnr(a, a) == float("inf")


def test_ssim_identical():
    a = torch.rand(1, 4, 3, 16, 16)
    score = ssim(a, a)
    assert abs(score - 1.0) < 1e-4, f"SSIM of identical tensors must be ~1.0, got {score}"


def test_temporal_flicker():
    # Constant video across time should have 0.0 temporal flicker
    frame = torch.rand(1, 1, 3, 16, 16)
    smooth_video = frame.repeat(1, 5, 1, 1, 1)
    flicker_score = temporal_flicker(smooth_video)
    assert abs(flicker_score) < 1e-6, f"Smooth video flicker should be 0, got {flicker_score}"

    # Random video across time should have non-zero flicker
    random_video = torch.rand(1, 5, 3, 16, 16)
    assert temporal_flicker(random_video) > 0.0


def test_clipsim_metric():
    frames = [torch.zeros((16, 16, 3), dtype=torch.uint8).numpy() for _ in range(3)]
    sim = clipsim(frames, "A test prompt")
    assert 0.0 <= sim <= 1.0
