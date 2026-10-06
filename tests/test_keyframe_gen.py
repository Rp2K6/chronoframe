"""Unit test for Text-Conditioned KeyframeGenerator."""
import pytest
import torch
from models.keyframe_generator import KeyframeGenerator


def test_keyframe_generation_prompt_responsiveness():
    gen = KeyframeGenerator().eval()
    
    with torch.no_grad():
        kf1 = gen(prompt="A panda making latte art", num_keyframes=4, latent_size=8, device="cpu")
        kf2 = gen(prompt="A futuristic cyberpunk city at night", num_keyframes=4, latent_size=8, device="cpu")
        
    assert kf1.shape == (1, 4, 4, 8, 8)
    assert kf2.shape == (1, 4, 4, 8, 8)
    
    # Verify that different text prompts generate distinct keyframe latents
    diff = (kf1 - kf2).abs().mean().item()
    assert diff > 0.05, f"Different text prompts must produce distinct keyframe latents, got diff={diff}"


def test_keyframe_temporal_flag():
    gen = KeyframeGenerator().eval()
    
    with torch.no_grad():
        kf_spatial = gen(prompt="Test prompt", num_keyframes=4, latent_size=8, device="cpu", apply_temporal=False)
        kf_temp = gen(prompt="Test prompt", num_keyframes=4, latent_size=8, device="cpu", apply_temporal=True)
        
    assert kf_spatial.shape == (1, 4, 4, 8, 8)
    assert kf_temp.shape == (1, 4, 4, 8, 8)
