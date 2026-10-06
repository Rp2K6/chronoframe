"""ChronoFrame Interactive CLI Demo Runner.

Generates video outputs and evaluates baseline vs proposed (ChronoFrame) pipeline.
Usage:
    python run_demo.py --prompt "A panda making latte art" --mode both
"""
import os
import argparse
import time
from models.pipeline import EnhancedTextToVideoPipeline


def format_psnr(val):
    if val is None:
        return "∞ dB (Identical)"
    return f"{val:.2f} dB"


def print_comparison_table(res_baseline: dict, res_proposed: dict):
    m_b = res_baseline["metrics"]
    m_p = res_proposed["metrics"]
    
    speedup = m_b["interp_latency_sec"] / max(1e-4, m_p["interp_latency_sec"])
    flicker_diff = ((m_b["flicker"] - m_p["flicker"]) / max(1e-4, m_b["flicker"])) * 100.0

    psnr_b_str = format_psnr(m_b["psnr"])
    psnr_p_str = format_psnr(m_p["psnr"])
    
    flicker_str = f"{flicker_diff:.1f}% Smoother"
    speedup_str = f"{speedup:.2f}x Faster"
    
    gen_b_str = f"{m_b['generation_time_sec']:.3f}s"
    gen_p_str = f"{m_p['generation_time_sec']:.3f}s"

    print("\n" + "=" * 76)
    print("        CHRONOFRAME vs BASELINE EMPIRICAL COMPARISON BENCHMARK")
    print("=" * 76)
    print(f"{'Metric / Property':<30} | {'Baseline (MFI + 2D)':<20} | {'ChronoFrame (Proposed)':<20}")
    print("-" * 76)
    print(f"{'Keyframe Generation':<30} | {'Spatial 2D':<20} | {'Conv1D + Attn (Alpha Init)':<20}")
    print(f"{'Interpolation Mode':<30} | {'Sequential (MFI)':<20} | {'Joint 3-Frame (1-Pass)':<20}")
    print(f"{'Interpolation Latency':<30} | {m_b['interp_latency_sec']:<20.4f}s | {m_p['interp_latency_sec']:<20.4f}s")
    print(f"{'Interpolation Speedup':<30} | {'1.0x (Baseline)':<20} | {speedup_str:<20}")
    print(f"{'Decoder Architecture':<30} | {'Standard 2D Image':<20} | {'3D Conv + Temp Attn':<20}")
    print(f"{'Temporal Flicker Score':<30} | {m_b['flicker']:<20.4f} | {m_p['flicker']:<20.4f}")
    print(f"{'Flicker Reduction':<30} | {'0.0%':<20} | {flicker_str:<20}")
    print(f"{'PSNR (Reconstruction)':<30} | {psnr_b_str:<20} | {psnr_p_str:<20}")
    print(f"{'SSIM (Structure)':<30} | {m_b['ssim']:<20.4f} | {m_p['ssim']:<20.4f}")
    print(f"{'CLIP Text Alignment':<30} | {m_b['clipsim']:<20.4f} | {m_p['clipsim']:<20.4f}")
    print(f"{'Total Generation Latency':<30} | {gen_b_str:<20} | {gen_p_str:<20}")
    print("=" * 76)
    print(f"Saved Baseline Video: {res_baseline['output_path']}")
    print(f"Saved Proposed Video: {res_proposed['output_path']}")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ChronoFrame Text-to-Video Generation CLI Demo")
    parser.add_argument("--prompt", type=str, default="A panda making latte art", help="Input text prompt")
    parser.add_argument("--mode", type=str, default="both", choices=["baseline", "proposed", "both"], help="Execution mode")
    parser.add_argument("--keyframes", type=int, default=4, help="Number of keyframes to generate (default: 4)")
    parser.add_argument("--resolution", type=int, default=256, help="Square resolution of generated video (default: 256)")
    parser.add_argument("--fps", type=int, default=8, help="Video framerate (default: 8)")
    parser.add_argument("--out", type=str, default=None, help="Custom output video filepath")

    args = parser.parse_args()

    pipeline = EnhancedTextToVideoPipeline()
    res_tuple = (args.resolution, args.resolution)

    os.makedirs("outputs", exist_ok=True)

    if args.mode == "both":
        print("\n>>> Running Baseline Pipeline Mode...")
        out_b = args.out if (args.out and "baseline" in args.out) else os.path.join("outputs", "baseline.mp4")
        res_baseline = pipeline.generate(
            prompt=args.prompt,
            mode="baseline",
            num_keyframes=args.keyframes,
            resolution=res_tuple,
            fps=args.fps,
            output_path=out_b
        )

        print("\n>>> Running ChronoFrame Proposed Pipeline Mode...")
        out_p = args.out if (args.out and "proposed" in args.out) else os.path.join("outputs", "proposed.mp4")
        res_proposed = pipeline.generate(
            prompt=args.prompt,
            mode="proposed",
            num_keyframes=args.keyframes,
            resolution=res_tuple,
            fps=args.fps,
            output_path=out_p
        )

        print_comparison_table(res_baseline, res_proposed)

    elif args.mode == "proposed":
        out_p = args.out or os.path.join("outputs", "proposed.mp4")
        res = pipeline.generate(
            prompt=args.prompt,
            mode="proposed",
            num_keyframes=args.keyframes,
            resolution=res_tuple,
            fps=args.fps,
            output_path=out_p
        )
        print(f"\n[Success] Generated {res['frame_count']} frames saved to {res['output_path']}")

    else:
        out_b = args.out or os.path.join("outputs", "baseline.mp4")
        res = pipeline.generate(
            prompt=args.prompt,
            mode="baseline",
            num_keyframes=args.keyframes,
            resolution=res_tuple,
            fps=args.fps,
            output_path=out_b
        )
        print(f"\n[Success] Generated {res['frame_count']} frames saved to {res['output_path']}")


if __name__ == "__main__":
    main()
