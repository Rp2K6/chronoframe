"""Generate a video with an open text-to-video model (needs a CUDA GPU) and score it.

Uses ModelScope text-to-video via diffusers as a runnable baseline. For the real
ImproveYourVideos / Kandinsky Video model, use the official repo (see README).
"""
import argparse
import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from metrics import clipsim, temporal_flicker


def to_uint8(frames):
    out = []
    for f in frames:
        a = np.asarray(f)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
        out.append(a)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A panda making latte art")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--out", default="out.mp4")
    args = ap.parse_args()

    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    frames = to_uint8(pipe(args.prompt, num_inference_steps=args.steps, num_frames=args.frames).frames[0])
    export_to_video(frames, args.out, fps=8)

    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255).unsqueeze(0)
    print("saved:", args.out)
    print("CLIPSIM :", round(clipsim(frames, args.prompt, device="cuda"), 4))
    print("flicker :", round(temporal_flicker(video), 4))
