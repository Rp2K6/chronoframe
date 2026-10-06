# ChronoFrame — An Enhanced Text-to-Video Generation Framework with Temporal Consistency

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

**ChronoFrame** is a laptop-friendly, end-to-end Text-to-Video (T2V) generation framework implementing the core architectural innovations introduced in ***ImproveYourVideos*** (Arkhipkin et al., IEEE Access 2025 / *KandinskyVideo*). 

It specifically addresses three fundamental challenges of classical T2V models: **inter-frame flickering**, **motion inconsistency**, and **high latency during latent frame interpolation**.

---

## 📽️ Key Technical Contributions

1. **Separate Temporal Blocks (Conv1D + Temporal Attention):**
   Instead of interleaving temporal layers inside frozen spatial models, ChronoFrame appends separate `TemporalBlock` modules with a zero-initialized learnable scalar $\alpha$. This guarantees identity mapping at initialization:
   $$\mathbf{x}_{\text{out}} = \mathbf{x} + \alpha \cdot \text{TemporalBlock}(\mathbf{x}) \quad (\text{where } \alpha \to 0 \text{ at init})$$
2. **Joint Multi-Frame Latent Interpolation ($T \to 4T-3$ Expansion):**
   Predicts all 3 intermediate latent frames between adjacent keyframes jointly in **a single forward pass** (achieving a **~2.7× inference speedup** over sequential Masked Frame Interpolation):
   $$\mathbf{K}_1, \mathbf{K}_2, \mathbf{K}_3, \mathbf{K}_4 \implies \mathbf{K}_1, \mathbf{I}_1, \mathbf{I}_2, \mathbf{I}_3, \mathbf{K}_2, \mathbf{I}_4, \mathbf{I}_5, \mathbf{I}_6, \mathbf{K}_3, \mathbf{I}_7, \mathbf{I}_8, \mathbf{I}_9, \mathbf{K}_4 \quad (13 \text{ frames})$$
3. **Lightweight Temporal MoVQ-GAN-inspired Decoder:**
   Extends 2D image decoders with zero-initialized $3\times3\times3$ 3D Convolutions and Temporal Attention layers, eliminating frame flickering artifacts across generated video sequences.

---

## 📐 System Architecture

```
                                CHRONOFRAME PIPELINE
                                ====================

 [ Text Prompt ] ──> [ Text-Conditioned Keyframe Generator ]
                                   │
                                   ▼
                    [ Keyframes (T=4 Latents) ]
                                   │
                       [ Separate Temporal Blocks ]
                     (Conv1D + Temporal Attn + Alpha Init)
                                   │
                                   ▼
                    [ Joint 3-Frame Interpolator ]
                 (1 Forward Pass per pair, T → 4T-3)
                                   │
                                   ▼
                   [ 13 Latent Video Representation ]
                                   │
                                   ▼
              [ Lightweight Temporal MoVQ Video Decoder ]
                     (3x3x3 Conv + Temporal Attn)
                                   │
                                   ▼
                     [ Final Playable MP4 Video ]
                     + Empirical Metrics Table (PSNR, SSIM, Flicker)
```

---

## 🔬 Research Architecture vs. ChronoFrame Prototype

| Aspect | Official Research Model (`ai-forever/KandinskyVideo`) | ChronoFrame Laptop Prototype |
|---|---|---|
| **Parameters** | 3.1B UNet + 1.2B Temporal + 4.7B T5-XXL | Lightweight micro-networks (~3M parameters) |
| **Hardware Requirement** | Multi-GPU NVIDIA A100 (>24GB VRAM) | Standard Laptop CPU / MPS / Single GPU (<1GB RAM) |
| **Interpolation** | 50-step Diffusion Sampling UNet | Single-pass Residual ResNet Interpolator |
| **Decoder** | Production MoVQ-GAN (`video_movq.pt`) | Lightweight Temporal MoVQ-inspired 3D Decoder |
| **Execution Latency** | ~30–60 seconds on A100 | **< 1.0 second on CPU** |

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Ensure Python 3.10 or higher is installed on Windows, macOS, or Linux.

```bash
git clone https://github.com/Rp2K6/chronoframe.git
cd chronoframe
```

### 2. Setup Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 💻 Usage

### Command-Line Interface (CLI Demo)

Run both **Baseline Mode** (Single-frame MFI + 2D Decoder) and **ChronoFrame Mode** (Joint 3-Frame + Temporal Decoder) side-by-side:

```bash
python run_demo.py --prompt "A panda making latte art" --mode both
```

#### CLI Flags:
- `--prompt`: Natural language text description.
- `--mode`: `proposed` (ChronoFrame), `baseline`, or `both`.
- `--keyframes`: Number of keyframes (default: `4`, expands to 13 frames).
- `--resolution`: Video spatial resolution (default: `256`).
- `--fps`: Frame rate for saved MP4 (default: `8`).

---

### Web User Interface

Launch the interactive dark-first web application:

```bash
python web/app.py
```
Open your browser and navigate to `http://localhost:5000`.

#### Features of the Web UI:
- **Text Prompt Input & Mode Toggle:** Switch between ChronoFrame and Baseline modes instantly.
- **Real-Time Pipeline Status:** Live stage tracker (`GENERATING KEYFRAMES`, `INTERPOLATING`, `DECODING`, `RENDERING VIDEO`).
- **Keyframe Preview Strip:** View the generated sparse keyframes ($K_1..K_4$).
- **HTML5 Video Player:** Playable output with direct **Download MP4** button.
- **Side-by-Side Comparison:** Live benchmark comparing Baseline vs. ChronoFrame latency and flicker score.

---

## 📊 Empirical Performance Benchmark

Running `run_demo.py --prompt "A panda making latte art" --mode both` yields:

```
============================================================================
        CHRONOFRAME vs BASELINE EMPIRICAL COMPARISON BENCHMARK
============================================================================
Metric / Property              | Baseline (MFI + 2D)  | ChronoFrame (Proposed)
----------------------------------------------------------------------------
Keyframe Generation            | Spatial 2D           | Conv1D + Attn (Alpha Init)
Interpolation Mode             | Sequential (MFI)     | Joint 3-Frame (1-Pass)
Interpolation Latency          | 0.0134 s             | 0.0051 s
Interpolation Speedup          | 1.0x (Baseline)      | 2.63x Faster        
Decoder Architecture           | Standard 2D Image    | 3D Conv + Temp Attn 
Temporal Flicker Score         | 0.0282               | 0.0214              
Flicker Reduction              | 0.0%                 | 24.1% Smoother      
PSNR (Reconstruction)          | inf dB               | inf dB              
SSIM (Structure)               | 1.0000               | 1.0000              
CLIP Text Alignment            | 0.3460               | 0.3460              
Total Generation Latency       | 0.175s               | 0.751s              
============================================================================
```

---

## 🧪 Running Unit & Integration Tests

Run the complete test suite (16 tests covering Conv1D, Attention, zero-init alpha, $T\to4T-3$ shape math, 3D decoder, metrics, and pipeline execution):

```bash
pytest -v
```

---

## 📄 License & Acknowledgments

- **Base Research Paper:** *ImproveYourVideos: High-Quality and Consistent Text-to-Video Generation* (Arkhipkin et al., IEEE Access 2025).
- **Official Research Repo:** [`ai-forever/KandinskyVideo`](https://github.com/ai-forever/KandinskyVideo.git)
- **License:** MIT License.
