from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]):
        # x: (B, S, D)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(attn_out)
        x = self.ln1(x)
        ff_out = self.ff(x)
        x = x + self.drop(ff_out)
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    """Decision Transformer minimalista com 3 tokens por passo: [RTG_t, s_t, a_{t-1}].

    - Recebe *embeddings* já projetados: state_emb, action_emb, rtg_emb, time_emb.
    - Projeta cada tipo para `d_model` se necessário e soma `time_emb` a todos tokens do passo.
    - Aplica máscara causal sobre a sequência intercalada (R,S,A,R,S,A,...).
    - Prediz a ação em `t` a partir das **posições de estado** `s_t` (cabeça linear).
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float, n_actions: int):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_actions)
        # Projeções para forçar a mesma dimensão
        self.proj_state: nn.Module = nn.Identity()
        self.proj_action: nn.Module = nn.Identity()
        self.proj_rtg:   nn.Module = nn.Identity()
        self.proj_time:  nn.Module = nn.Identity()



    def ensure_projections(self, E_state: int, E_action: int, E_rtg: int, E_time: int, device: torch.device):
        """Prepare as projeções para d_model no device correto, antes do treino."""
        self.proj_state = nn.Identity() if E_state == self.d_model else nn.Linear(E_state, self.d_model, device=device)
        self.proj_action = nn.Identity() if E_action == self.d_model else nn.Linear(E_action, self.d_model, device=device)
        self.proj_rtg   = nn.Identity() if E_rtg   == self.d_model else nn.Linear(E_rtg,   self.d_model, device=device)
        self.proj_time  = nn.Identity() if E_time  == self.d_model else nn.Linear(E_time,  self.d_model, device=device)


    @staticmethod
    def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # True = mascarar
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(
        self,
        state_emb: torch.Tensor,   # (B, K, E_state)
        action_emb: torch.Tensor,  # (B, K, E_action)
        rtg_emb: torch.Tensor,     # (B, K, E_rtg)
        time_emb: torch.Tensor,    # (B, K, E_time)
        attn_mask_steps: torch.Tensor,  # (B, K) 1=válido, 0=pad
    ) -> torch.Tensor:
        B, K, E_s = state_emb.shape
        _, _, E_a = action_emb.shape
        _, _, E_r = rtg_emb.shape
        _, _, E_t = time_emb.shape
        device = state_emb.device

        # Projeções para d_model (já preparadas no device certo)
        s = self.proj_state(state_emb)
        a = self.proj_action(action_emb)
        r = self.proj_rtg(rtg_emb)
        t = self.proj_time(time_emb)

        # Soma o embedding temporal a todos os tokens do passo
        s = s + t
        a = a + t
        r = r + t

        # Intercalar tokens: (B, K, 3, D) → (B, 3K, D)
        x = torch.stack([r, s, a], dim=2).reshape(B, 3 * K, self.d_model)

        # Máscaras: causal (S,S) e padding (B,S)
        S = 3 * K
        causal = self.build_causal_mask(S, device)
        key_pad = (~attn_mask_steps.bool()).repeat_interleave(3, dim=1)  # True nos pads

        # Transformer
        for blk in self.layers:
            x = blk(x, attn_mask=causal, key_padding_mask=key_pad)

        # Pegamos as posições de estado (índices 1,4,7,...) para prever a ação de t
        idx = torch.arange(1, S, 3, device=device)  # tamanho K
        x_states = x[:, idx, :]  # (B, K, D)

        logits = self.head(x_states)  # (B, K, n_actions)
        # Blindagem final contra NaN/Inf
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        
        return logits