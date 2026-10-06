# ChronoFrame — Technical Implementation Guide

This document provides a detailed technical breakdown of the architecture, tensor flow, module responsibilities, API interfaces, and frontend-backend communications for **ChronoFrame**.

---

## 1. System Architecture & Tensor Flow

ChronoFrame operates as a modular, latent-space video generation pipeline. Below is the step-by-step tensor dimension transformations across the 4 key stages for a standard resolution of $256 \times 256$ and $T=4$ keyframes:

```
[ Input Text Prompt ] ──> String: "A panda making latte art"
          │
          ▼  Stage 1: Text-Conditioned Keyframe Generation
[ KeyframeGenerator ]
  • Encodes prompt into 128-dim semantic feature vector
  • Projects vector into spatial latent seed: (1, C=4, H_lat=16, W_lat=16)
  • Applies time-evolving spatial roll trajectory across T=4 keyframes
  • Applies TemporalBlock refinement (Conv1D + Temporal Attention + Alpha Init)
  • Output Keyframe Latents shape: (1, T=4, C=4, H_lat=16, W_lat=16)
          │
          ▼  Stage 2: Joint Multi-Frame Latent Interpolation (T → 4T-3)
[ JointInterpolator ]
  • Extracts adjacent keyframe pairs (a, b) of shape (N_pairs=3, C=4, H_lat=16, W_lat=16)
  • Concatenates pairs along channel dimension -> (3, 8, 16, 16)
  • Single forward pass through Residual ResNet -> Outputs 3 intermediate latents per pair (3, 12, 16, 16)
  • Interleaves keyframes & intermediate triads: [K1, I1, I2, I3, K2, I4, I5, I6, K3, I7, I8, I9, K4]
  • Output Interpolated Latents shape: (1, T_total=13, C=4, H_lat=16, W_lat=16)
          │
          ▼  Stage 3: Latents-to-Video Decoding
[ TinyDecoder (temporal=True) ]
  • Reshapes input to (B*T_total, C=4, H_lat=16, W_lat=16) = (13, 4, 16, 16)
  • Passes through 2D upsampling stages interleaved with TemporalDecoderLayer (3x3x3 Conv + Temporal Attn)
  • Upscales spatial dimensions 4x (16x16 -> 64x64 or 256x256)
  • Output Video Tensor shape: (1, T_total=13, C_rgb=3, H=256, W=256)
          │
          ▼  Stage 4: MP4 Video Compilation & Metrics Evaluation
[ save_video_mp4 & metrics ]
  • Converts normalized float tensor [0, 1] to uint8 RGB NumPy arrays
  • OpenCV VideoWriter encodes frames to H.264 / MP4 container at 8 FPS
  • Evaluates PSNR, SSIM, Temporal Flicker Score, and CLIP alignment score
  • Saved MP4 File Path: "outputs/proposed.mp4"
```

---

## 2. Module Responsibilities

### `models/temporal_blocks.py`
- **`TemporalBlock`**: Core temporal module appending 1D temporal convolution (`nn.Conv1d`) and multi-head temporal attention (`nn.MultiheadAttention`) along sequence dimension $T$. Features a learnable scalar parameter `alpha` initialized to `0.0` to preserve pretrained spatial identity at initialization.
- **`FrozenSpatialWithTemporal`**: Wrapper that freezes spatial module parameters (`requires_grad = False`) and attaches a trainable `TemporalBlock`.

### `models/interpolation.py`
- **`JointInterpolator`**: Proposed interpolation model that predicts all 3 intermediate latent frames between keyframe pairs in a single forward pass, expanding $T$ keyframes to $4T-3$ frames ($4 \to 13$, $6 \to 21$).
- **`SingleFrameInterpolator`**: Baseline interpolation model simulating Masked Frame Interpolation (MFI) by executing 3 sequential forward passes per keyframe pair. Used for empirical speedup benchmarking.

### `models/temporal_decoder.py`
- **`TemporalConv3D`**: Zero-initialized $3 \times 3 \times 3$ 3D Convolution layer for inter-frame smoothing.
- **`TemporalDecoderLayer`**: Integrates `TemporalConv3D` and `TemporalBlock`.
- **`TinyEncoder` & `TinyDecoder`**: Encoder converts RGB to latent space; Decoder converts latents to RGB frames. `temporal=True` enables the 3D temporal decoder; `temporal=False` enables the standard 2D image decoder baseline.

### `models/keyframe_generator.py`
- **`KeyframeGenerator`**: Generates text-conditioned keyframe latents. Maps text prompt strings to deterministic semantic feature vectors that influence latent spatial patterns, color schemes, and motion trajectories.

### `models/pipeline.py`
- **`EnhancedTextToVideoPipeline`**: Primary end-to-end interface. Coordinates keyframe generation, interpolation, decoding, rendering, and metric calculations for both Proposed and Baseline modes.

### `utils/metrics.py` & `utils/video_utils.py`
- Implements Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), Temporal Flicker Score ($|x_{t+1} - 2x_t + x_{t-1}|$), CLIP alignment score, OpenCV MP4 video encoding, and base64 image conversion.

---

## 3. Frontend-Backend Communication API

The web application uses a RESTful JSON API implemented in `web/app.py`:

| Endpoint | Method | Payload / Arguments | Response | Purpose |
|---|---|---|---|---|
| `/` | `GET` | None | HTML Page | Serves `web/static/index.html` UI |
| `/api/generate` | `POST` | `{ prompt, mode, keyframes, resolution, fps }` | `{ job_id, status, progress }` | Triggers background pipeline job |
| `/api/status/<job_id>` | `GET` | Job ID URL param | `{ status, progress, result, error }` | Polled by UI every 350ms for live progress |
| `/api/compare` | `POST` | `{ prompt, keyframes, resolution, fps }` | `{ baseline: {...}, proposed: {...} }` | Runs side-by-side benchmark comparison |
| `/outputs/<filename>` | `GET` | `download=1` query param | Binary MP4 Video File | Video playback & direct MP4 download |

---

## 4. Limitations & Future Improvements

### Current Prototype Limitations
1. **Model Scale:** Scaled down to micro ResNets/ConvNets (~3M parameters) for CPU/laptop execution instead of multi-billion parameter diffusion U-Nets.
2. **Latent Space Resolution:** Latent grid is $16 \times 16$ upscaled to $256 \times 256$ RGB.
3. **Training Data:** Trained on synthetic moving blob trajectories and prompt feature maps rather than 1.7M high-resolution text-video pairs.

### Future Improvements
1. **ONNX / TensorRT Export:** Export pipeline modules to ONNX runtime for ultra-fast GPU inference.
2. **Audio Alignment:** Integrate audio/speech generation to match video frame timestamps.
3. **Higher Frame Expansion:** Extend interpolator to support $T \to 8T-7$ or dynamic frame interpolation rates.
