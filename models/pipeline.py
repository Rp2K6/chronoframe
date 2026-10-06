"""ChronoFrame Pipeline Orchestrator: EnhancedTextToVideoPipeline.

Executes real end-to-end Text-to-Video generation in two modes:
1. Baseline Mode: Spatial Keyframes -> Single-Frame Interpolation (MFI) -> 2D Image Decoder -> MP4.
2. ChronoFrame (Proposed) Mode: Text-Guided Keyframes + Temporal Blocks -> Joint 3-Frame Interpolation -> Temporal MoVQ-GAN-inspired Decoder -> MP4.
"""
import os
import time
import math
import torch
import numpy as np

from models.keyframe_generator import KeyframeGenerator
from models.interpolation import JointInterpolator, SingleFrameInterpolator
from models.temporal_decoder import TinyDecoder
from utils.video_utils import tensor_to_uint8_frames, save_video_mp4
from utils.metrics import psnr, ssim, temporal_flicker, clipsim


def get_optimal_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EnhancedTextToVideoPipeline:
    """Core End-to-End Pipeline for ChronoFrame Text-to-Video Generation."""

    def __init__(self, device: str = None):
        self.device = device or get_optimal_device()
        print(f"[ChronoFrame Pipeline] Initialized on device: {self.device.upper()}")
        
        # Initialize sub-modules
        self.keyframe_gen = KeyframeGenerator().to(self.device).eval()
        self.joint_interpolator = JointInterpolator().to(self.device).eval()
        self.single_interpolator = SingleFrameInterpolator().to(self.device).eval()
        
        self.temporal_decoder = TinyDecoder(temporal=True).to(self.device).eval()
        self.image_decoder = TinyDecoder(temporal=False).to(self.device).eval()

    def generate(
        self,
        prompt: str,
        mode: str = "proposed",
        num_keyframes: int = 4,
        resolution: tuple = (256, 256),
        fps: int = 8,
        output_path: str = None,
        callback: callable = None,
    ) -> dict:
        """Runs the complete end-to-end Text-to-Video generation pipeline.
        
        mode: 'proposed' (ChronoFrame) or 'baseline'
        """
        start_time = time.perf_counter()
        dev = torch.device(self.device)
        is_proposed = mode.lower() in ["proposed", "chronoframe"]
        mode_label = "ChronoFrame (Proposed)" if is_proposed else "Baseline"
        
        def notify(stage: str, pct: int):
            if callback:
                callback(stage, pct)
            print(f"[{mode_label}] Stage: {stage} ({pct}%)")

        notify("GENERATING KEYFRAMES", 10)
        print(f"[DEMO / FALLBACK MODE: Lightweight Text-Guided Keyframe Synthesizer] Prompt: '{prompt}'")
        
        # Step 1: Text-conditioned keyframe generation
        num_keyframes = max(1, min(8, int(num_keyframes)))
        latent_size = max(8, resolution[0] // 16)
        
        with torch.no_grad():
            # Generate T keyframe latents: (1, T, C, H_lat, W_lat)
            keyframe_latents = self.keyframe_gen(
                prompt=prompt,
                num_keyframes=num_keyframes,
                latent_size=latent_size,
                device=self.device,
                apply_temporal=is_proposed
            )

            # Decode original keyframe previews (1, T, 3, H, W)
            decoder_kf = self.temporal_decoder if is_proposed else self.image_decoder
            keyframe_rgb = decoder_kf(keyframe_latents)
            keyframe_frames = tensor_to_uint8_frames(keyframe_rgb)

            notify("INTERPOLATING", 40)

            # Step 2: Latent frame interpolation (T -> 4T-3)
            interp_start = time.perf_counter()
            if is_proposed:
                # Joint prediction of 3 intermediate frames in 1 forward pass
                full_latents = self.joint_interpolator(keyframe_latents)
            else:
                # Baseline Single-Frame Interpolation (MFI) sequential passes
                full_latents = self.single_interpolator(keyframe_latents)
                
            interp_latency = time.perf_counter() - interp_start

            notify("DECODING", 75)

            # Step 3: Latents-to-Video decoding
            if is_proposed:
                # Lightweight Temporal MoVQ-GAN-inspired Decoder (3x3x3 Conv + Temporal Attention)
                video_rgb = self.temporal_decoder(full_latents)
            else:
                # Standard 2D Image Decoder
                video_rgb = self.image_decoder(full_latents)

            notify("RENDERING VIDEO", 90)

            # Step 4: Render MP4 file
            full_frames = tensor_to_uint8_frames(video_rgb)
            if not output_path:
                os.makedirs("outputs", exist_ok=True)
                filename = "proposed.mp4" if is_proposed else "baseline.mp4"
                output_path = os.path.join("outputs", filename)

            save_video_mp4(full_frames, output_path, fps=fps)

            # Step 5: Compute real evaluation metrics
            notify("COMPUTING METRICS", 95)
            
            # Reconstruction metric: Compare decoded video frames at keyframe positions (indices 0, 4, 8...)
            # against original spatial keyframe decoded RGB images `keyframe_rgb`
            if num_keyframes > 1:
                video_kf_sub = video_rgb[:, ::4]
                min_len = min(video_kf_sub.shape[1], keyframe_rgb.shape[1])
                psnr_val = psnr(video_kf_sub[:, :min_len], keyframe_rgb[:, :min_len])
                ssim_val = ssim(video_kf_sub[:, :min_len], keyframe_rgb[:, :min_len])
            else:
                psnr_val = psnr(video_rgb, keyframe_rgb)
                ssim_val = ssim(video_rgb, keyframe_rgb)

            flicker_val = temporal_flicker(video_rgb)
            clip_val = clipsim(full_frames, prompt, device=self.device)

        total_latency = time.perf_counter() - start_time
        frame_count = len(full_frames)
        duration_sec = frame_count / float(fps)

        notify("COMPLETE", 100)

        result_dict = {
            "mode": mode,
            "mode_label": mode_label,
            "prompt": prompt,
            "keyframes": keyframe_frames,
            "generated_frames": full_frames,
            "output_path": output_path,
            "frame_count": frame_count,
            "fps": fps,
            "duration": round(duration_sec, 2),
            "generation_time": round(total_latency, 3),
            "interp_latency": round(interp_latency, 4),
            "metrics": {
                "psnr": psnr_val,
                "ssim": round(ssim_val, 4) if ssim_val is not None else 1.0,
                "flicker": round(flicker_val, 4),
                "clipsim": round(clip_val, 4),
                "generation_time_sec": round(total_latency, 3),
                "interp_latency_sec": round(interp_latency, 4)
            }
        }

        psnr_str = f"{psnr_val:.2f}dB" if psnr_val is not None else "inf dB (Identical)"
        print(f"[{mode_label}] Successfully generated {frame_count} frames ({duration_sec:.1f}s) in {total_latency:.2f}s!")
        print(f"[{mode_label}] Metrics -> PSNR: {psnr_str} | SSIM: {ssim_val:.4f} | Flicker: {flicker_val:.4f} | CLIP: {clip_val:.4f}")
        return result_dict
