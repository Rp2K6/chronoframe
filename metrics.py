import torch
import torch.nn.functional as F


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(a, b).item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()


def temporal_flicker(video: torch.Tensor) -> float:
    """Proxy flicker score: mean |x[t+1] - 2x[t] + x[t-1]|. video: (B, T, C, H, W). Lower = smoother."""
    d2 = video[:, 2:] - 2 * video[:, 1:-1] + video[:, :-2]
    return d2.abs().mean().item()


def clipsim(frames, prompt: str, device: str = "cpu", model_name: str = "openai/clip-vit-base-patch32") -> float:
    """Mean cosine similarity between the prompt and every frame (PIL images or uint8 arrays)."""
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    inputs = proc(text=[prompt], images=list(frames), return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
    img = F.normalize(out.image_embeds, dim=-1)
    txt = F.normalize(out.text_embeds, dim=-1)
    return (img @ txt.T).mean().item()
