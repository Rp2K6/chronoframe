"""Video rendering and format conversion utilities for ChronoFrame."""
import os
import cv2
import base64
import numpy as np
import torch
from PIL import Image


def tensor_to_uint8_frames(video_tensor: torch.Tensor) -> list:
    """Converts a video tensor of shape (B, T, C, H, W) or (T, C, H, W) in [0, 1] to a list of HxWxC uint8 NumPy arrays (RGB)."""
    if video_tensor.ndim == 5:
        video_tensor = video_tensor[0]  # Take first batch item: (T, C, H, W)

    video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
    # (T, C, H, W) -> (T, H, W, C)
    frames_np = (video_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

    return [frames_np[i] for i in range(frames_np.shape[0])]


def save_video_mp4(frames: list, output_path: str, fps: int = 8) -> str:
    """Saves a list of RGB uint8 frames or PIL Images to an MP4 video file using OpenCV."""
    if not frames:
        raise ValueError("Frames list cannot be empty.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Normalize frames to RGB numpy uint8
    converted_frames = []
    for f in frames:
        if isinstance(f, Image.Image):
            arr = np.array(f.convert("RGB"))
        else:
            arr = np.asarray(f)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        converted_frames.append(arr)

    height, width, _ = converted_frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    for frame in converted_frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

    writer.release()
    return output_path


def frame_to_base64(frame: np.ndarray) -> str:
    """Converts an RGB uint8 image array into a PNG base64 data URI string for frontend display."""
    pil_img = Image.fromarray(frame)
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
