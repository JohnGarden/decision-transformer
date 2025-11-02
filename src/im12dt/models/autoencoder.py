from __future__ import annotations
import torch
import torch.nn as nn

class TabularAutoencoder(nn.Module):
    """Autoencoder simples para compressão de contínuos tabulares.

    Use quando `config.autoencoder.use=true`. Treinaremos futuramente.
    """
    def __init__(self, in_dim: int, bottleneck: int = 32, hidden: list[int] | None = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or [256, 128]
        enc_layers = []
        d = in_dim
        for h in hidden:
            enc_layers += [nn.Linear(d, h), nn.GELU()]
            if dropout > 0: enc_layers += [nn.Dropout(dropout)]
            d = h
        enc_layers += [nn.Linear(d, bottleneck)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        d = bottleneck
        for h in reversed(hidden):
            dec_layers += [nn.Linear(d, h), nn.GELU()]
            if dropout > 0: dec_layers += [nn.Dropout(dropout)]
            d = h
        dec_layers += [nn.Linear(d, in_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z