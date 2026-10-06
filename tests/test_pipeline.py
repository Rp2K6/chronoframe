"""Integration test for EnhancedTextToVideoPipeline."""
import os
import pytest
from models.pipeline import EnhancedTextToVideoPipeline


def test_pipeline_proposed_mode(tmp_path):
    pipeline = EnhancedTextToVideoPipeline(device="cpu")
    out_file = str(tmp_path / "test_proposed.mp4")
    
    res = pipeline.generate(
        prompt="A panda making latte art",
        mode="proposed",
        num_keyframes=4,
        resolution=(128, 128),
        fps=8,
        output_path=out_file
    )
    
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 0
    assert res["frame_count"] == 13  # 4 * 4 - 3 = 13
    assert len(res["keyframes"]) == 4
    assert len(res["generated_frames"]) == 13
    assert "metrics" in res
    assert "psnr" in res["metrics"]
    assert "ssim" in res["metrics"]
    assert "flicker" in res["metrics"]


def test_pipeline_baseline_mode(tmp_path):
    pipeline = EnhancedTextToVideoPipeline(device="cpu")
    out_file = str(tmp_path / "test_baseline.mp4")
    
    res = pipeline.generate(
        prompt="A panda making latte art",
        mode="baseline",
        num_keyframes=4,
        resolution=(128, 128),
        fps=8,
        output_path=out_file
    )
    
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 0
    assert res["frame_count"] == 13
    assert res["mode"] == "baseline"
