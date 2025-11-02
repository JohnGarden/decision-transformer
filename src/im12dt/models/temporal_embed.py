from __future__ import annotations
import torch
import torch.nn as nn

class TimeEncodingFourier(nn.Module):
    """Embedding temporal contínuo a partir de Δt usando features de Fourier.

    Opções: normalizar por p95 (clipping) e/ou aplicar log1p antes da codificação.
    """
    def __init__(self, d_model: int, n_freq: int = 16, use_log1p: bool = True, clip_p95: float | None = None):
        super().__init__()
        self.n_freq = n_freq
        self.use_log1p = use_log1p
        self.clip_p95 = clip_p95
        self.proj = nn.Linear(2 * n_freq, d_model)
        # Frequências baseadas em potência de 2
        freqs = torch.logspace(start=0, end=n_freq - 1, steps=n_freq, base=2.0)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # dt: (B, K) em segundos (ou unidade relativa)
        x = dt
        if self.clip_p95 is not None:
            x = torch.clamp(x, max=self.clip_p95)
        if self.use_log1p:
            x = torch.log1p(x)
        # (B,K,2F)
        angles = x.unsqueeze(-1) * self.freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(emb)


class TimeEncodingMLP(nn.Module):
    """Alternativa simples via MLP sobre Δt escalar."""
    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, d_model)
        )
    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return self.net(dt.unsqueeze(-1))