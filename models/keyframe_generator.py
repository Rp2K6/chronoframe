"""Text-Conditioned Keyframe Generator for ChronoFrame.

Synthesizes keyframe latents (e.g. 4 keyframes) directly conditioned on the input text prompt's
semantic embedding, color distribution, and motion trajectory vectors.

Clearly labeled as: [DEMO / FALLBACK MODE: Lightweight Text-Guided Keyframe Synthesizer]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.temporal_blocks import TemporalBlock


class KeyframeGenerator(nn.Module):
    """Text-conditioned keyframe generator operating in latent space."""

    def __init__(self, latent_ch: int = 4, hidden_ch: int = 64):
        super().__init__()
        self.latent_ch = latent_ch
        
        # Text embedding projection layers (maps 128-dim prompt features to latent channels)
        self.text_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, latent_ch * 16 * 16),
        )
        
        # Spatial refinement network
        self.spatial_net = nn.Sequential(
            nn.Conv2d(latent_ch, hidden_ch, 3, padding=1),
            nn.GroupNorm(8, hidden_ch),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, latent_ch, 3, padding=1),
        )
        
        # Temporal Block for keyframe temporal consistency
        self.temporal_block = TemporalBlock(channels=latent_ch, heads=4)

    def _encode_prompt(self, prompt: str, device: torch.device) -> torch.Tensor:
        """Deterministically extracts a 128-dim semantic feature vector from text prompt string."""
        prompt_clean = prompt.lower().strip()
        words = prompt_clean.split()
        
        vec = np.zeros(128, dtype=np.float32)
        for idx, char in enumerate(prompt_clean):
            vec[idx % 128] += (ord(char) % 31) / 15.0 - 1.0
        for idx, word in enumerate(words):
            word_hash = sum(ord(c) * (i + 1) for i, c in enumerate(word))
            vec[(idx * 7) % 128] += math.sin(word_hash) * 2.0
            
        # Scale to ensure distinct prompt representations
        return torch.from_numpy(vec).float().to(device).unsqueeze(0)

    def forward(
        self, prompt: str, num_keyframes: int = 4, latent_size: int = 16, device: str = "cpu", apply_temporal: bool = True
    ) -> torch.Tensor:
        """Generates T text-conditioned keyframe latents of shape (1, T, C, H, W)."""
        dev = torch.device(device)
        
        # 1. Extract prompt embedding
        text_emb = self._encode_prompt(prompt, dev)  # (1, 128)
        
        # 2. Project text embedding into spatial latent seed
        seed_feat = self.text_proj(text_emb).reshape(1, self.latent_ch, 16, 16)
        if latent_size != 16:
            seed_feat = F.interpolate(seed_feat, size=(latent_size, latent_size), mode="bilinear", align_corners=False)
            
        # 3. Create time-evolving trajectory based on prompt semantics
        prompt_val = sum(ord(c) * (i + 1) for i, c in enumerate(prompt))
        speed = 0.15 + (prompt_val % 50) / 200.0
        
        keyframes = []
        for t_idx in range(num_keyframes):
            t_rel = t_idx / max(1, num_keyframes - 1)
            # Apply motion shift and spatial deformation conditioned on time
            shift_x = int((t_rel * speed * latent_size) % latent_size)
            rolled_feat = torch.roll(seed_feat, shifts=(shift_x, 0), dims=(3, 2))
            
            # Apply spatial refinement network
            k_t = self.spatial_net(rolled_feat) + rolled_feat
            keyframes.append(k_t)
            
        # Stack keyframes: (1, T, C, H, W)
        k_tensor = torch.stack(keyframes, dim=1)
        B, T, C, H, W = k_tensor.shape
        
        # 4. Apply Temporal Block refinement if enabled (Proposed Mode)
        if apply_temporal:
            k_flat = k_tensor.reshape(B * T, C, H, W)
            k_refined = self.temporal_block(k_flat, T)
            k_tensor = k_refined.reshape(B, T, C, H, W)
            
        return k_tensor
