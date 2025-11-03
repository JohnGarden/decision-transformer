from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional
import math
import torch
import torch.nn as nn


@dataclass
class CatSpec:
    name: str
    n_tokens: int
    embed_dim: int


def _rule_embed_dim(n_cat: int, rule: str = "sqrt", fixed: int = 16) -> int:
    if rule == "fixed":
        return fixed
    if rule == "log":
        d = int(max(4, min(32, round(math.log2(max(2, n_cat)) * 4))))
        return d
    # sqrt default
    d = int(max(4, min(32, round(math.sqrt(max(1, n_cat)) * 2))))
    return d


class NumericProjector(nn.Module):
    """Projeção linear (com LayerNorm) dos contínuos para um espaço latente."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, D_num)
        return self.norm(self.proj(x))


class CategoricalTokenizer(nn.Module):
    """Embeddings independentes por coluna categórica.

    Espera receber tensores inteiros (B, K) em `cats[name]`.
    """
    def __init__(self, specs: Iterable[CatSpec]):
        super().__init__()
        self.emb = nn.ModuleDict({s.name: nn.Embedding(s.n_tokens, s.embed_dim) for s in specs})
        self.specs = {s.name: s for s in specs}
    def forward(self, cats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for name, tens in cats.items():
            if tens.device != next(self.emb[name].parameters()).device:
                raise RuntimeError(f"[{name}] indices on {tens.device}, embedding on {next(self.emb[name].parameters()).device}")
            out[name] = self.emb[name](tens)
        return out


class StateTokenizer(nn.Module):
    """Combina contínuos + categóricos em um *token de estado* de dimensão `state_embed_dim`.

    Opções de combinação: concatenação seguida de projeção + LayerNorm.
    """
    def __init__(
        self,
        numeric_dim: int,
        cat_specs: Iterable[CatSpec],
        state_embed_dim: int,
    ):
        super().__init__()
        self.num_tok = NumericProjector(numeric_dim, state_embed_dim)
        self.cat_tok = CategoricalTokenizer(cat_specs)
        cat_total = sum(s.embed_dim for s in cat_specs)
        self.proj = nn.Linear(state_embed_dim + cat_total, state_embed_dim)
        self.norm = nn.LayerNorm(state_embed_dim)

    def forward(self, num: torch.Tensor, cats: Dict[str, torch.Tensor]) -> torch.Tensor:
        # num: (B, K, D_num)
        # cats[name]: (B, K)
        num_e = self.num_tok(num)  # (B, K, E)
        cat_e = self.cat_tok(cats) # dict: (B, K, d_i)
        if len(cat_e) > 0:
            cat_stack = torch.cat([cat_e[k] for k in sorted(cat_e.keys())], dim=-1)
            x = torch.cat([num_e, cat_stack], dim=-1)
        else:
            x = num_e
        return self.norm(self.proj(x))  # (B, K, E)


class ActionTokenizer(nn.Module):
    """Embedding de ações discretas (inclui START-id no vocabulário de ação)."""
    def __init__(self, n_actions_plus_start: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_actions_plus_start, embed_dim)
    def forward(self, a_in: torch.Tensor) -> torch.Tensor:
        # a_in: (B, K) inteiros
        return self.emb(a_in)


class RTGTokenizer(nn.Module):
    """Projeção de escalar RTG por passo para o espaço de embeddings."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    def forward(self, rtg: torch.Tensor) -> torch.Tensor:
        # rtg: (B, K)
        return self.mlp(rtg.unsqueeze(-1))