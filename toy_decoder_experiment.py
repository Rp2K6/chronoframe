"""CPU-friendly experiment: image decoder vs temporal decoder on synthetic moving-blob videos.

Latents get per-frame noise (mimicking sampling jitter that causes flicker). We compare
PSNR and flicker score. Gives you real numbers for the 'Results' slide (toy scale!).
"""
import torch
import torch.nn.functional as F
from temporal_decoder import TinyEncoder, TinyDecoder
from metrics import psnr, temporal_flicker

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
B, T, S, STEPS, NOISE = 8, 8, 32, 400, 0.15


def make_batch(B, T, S, device):
    lin = torch.linspace(0, 1, S, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    c0 = torch.rand(B, 1, 2, device=device) * 0.6 + 0.2
    v = (torch.rand(B, 1, 2, device=device) - 0.5) * 0.4
    t = torch.linspace(0, 1, T, device=device).view(1, T, 1)
    c = c0 + v * t
    cx, cy = c[..., 0].reshape(B, T, 1, 1), c[..., 1].reshape(B, T, 1, 1)
    blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 0.12 ** 2))
    col = torch.rand(B, 1, 3, 1, 1, device=device)
    return blob.unsqueeze(2) * col


def run(temporal: bool):
    enc, dec = TinyEncoder().to(dev), TinyDecoder(temporal=temporal).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-3)
    for step in range(STEPS):
        x = make_batch(B, T, S, dev)
        z = enc(x)
        loss = F.mse_loss(dec(z + NOISE * torch.randn_like(z)), x)
        opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); dec.eval()
    with torch.no_grad():
        x = make_batch(32, T, S, dev)
        z = enc(x)
        y = dec(z + NOISE * torch.randn_like(z))
    return psnr(y.clamp(0, 1), x), temporal_flicker(y), temporal_flicker(x)


if __name__ == "__main__":
    for name, flag in [("image decoder", False), ("temporal decoder", True)]:
        p, fl, fl_gt = run(flag)
        print(f"{name:18s} PSNR={p:6.2f} dB  flicker={fl:.4f}  (ground-truth flicker={fl_gt:.4f})")
