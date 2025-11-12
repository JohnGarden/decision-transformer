from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pathlib import Path
import warnings
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import average_precision_score, f1_score

from im12dt.data.dataset_seq import UNSWSequenceDataset
from im12dt.models.tokens import (
    StateTokenizer, ActionTokenizer, RTGTokenizer, CatSpec, _rule_embed_dim
)
from im12dt.models.temporal_embed import TimeEncodingFourier
from im12dt.models.model_dt import DecisionTransformer

warnings.filterwarnings("ignore", message="No positive class found in y_true")


# ---------------------------- Config ----------------------------

@dataclass
class TrainerConfig:
    batch_size: int
    max_epochs: int
    steps_per_epoch: Optional[int]
    grad_clip: float
    lr: float
    betas: Tuple[float, float]
    weight_decay: float
    wait_threshold: float
    class_weights: Tuple[float, float]
    label_smoothing: float
    reward_weights: Dict[str, float]        # cTP,cTN,cFP,cFN,cWAIT
    # opcionais (retro-compat)
    eval_no_rtg: bool = False
    # Importance Sampling (pode vir vazio; se None, tentamos ler do YAML externo)
    is_dt: Optional[Dict] = None


# ---------------------------- Utils ----------------------------

def build_cat_specs(ds: UNSWSequenceDataset, cat_cols: List[str], rule: str, fixed_dim: int) -> List[CatSpec]:
    specs = []
    for c in cat_cols:
        if hasattr(ds, "_cat_maps") and c in ds._cat_maps:
            n_tokens = len(ds._cat_maps[c])
            d = _rule_embed_dim(n_tokens, rule=rule, fixed=fixed_dim)
            specs.append(CatSpec(c, n_tokens, d))
    return specs


def make_weighted_sampler(ds: UNSWSequenceDataset, pos_weight: float = 4.0) -> WeightedRandomSampler:
    """Pesa mais janelas (exemplos) que contenham ao menos um token positivo."""
    weights: List[float] = []
    for ex in ds.examples:
        pos = int(ex.actions_out.sum() > 0)
        weights.append(pos_weight if pos else 1.0)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def cross_entropy_masked(
    logits: torch.Tensor,         # (B,K,C)
    targets: torch.Tensor,        # (B,K) em {0,1}
    mask: torch.Tensor,           # (B,K) em {0,1}
    weight: Optional[torch.Tensor],
    label_smoothing: float = 0.0,
    token_weights: Optional[torch.Tensor] = None,   # (B,K) ou None
) -> torch.Tensor:
    B, K, C = logits.shape
    logits  = logits.reshape(B * K, C)
    targets = targets.reshape(B * K)
    mask    = mask.reshape(B * K).float()
    w_tok   = token_weights.reshape(B * K).float() if token_weights is not None else None

    # Blindagem numérica
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

    per_tok = F.cross_entropy(
        logits, targets,
        weight=weight,
        reduction='none',
        label_smoothing=(label_smoothing if label_smoothing > 0.0 else 0.0)
    )  # (B*K,)

    w = mask if w_tok is None else (mask * w_tok)
    denom = w.sum().clamp(min=1.0)
    loss  = (per_tok * w).sum() / denom
    loss  = torch.nan_to_num(loss, nan=0.0, posinf=1e3, neginf=1e3)

    if loss.ndim != 0:
        loss = loss.mean()

    return loss


def decode_with_wait(logits: torch.Tensor, wait_threshold: float) -> torch.Tensor:
    """Decodifica ações {0,1,2} com política 'wait' quando conf < limiar."""
    probs = torch.softmax(logits, dim=-1)  # (B,K,C)
    conf, argmax = probs.max(dim=-1)
    pred = argmax.clone()
    pred[conf < wait_threshold] = 2  # 2 = wait
    return pred


def compute_reward(pred: torch.Tensor, y: torch.Tensor, c: Dict[str, float]) -> torch.Tensor:
    """Recompensa por token dadas as ações {0,1,2} e rótulos {0,1}."""
    r = torch.zeros_like(pred, dtype=torch.float32)
    r[(pred == 1) & (y == 1)] = c['cTP']
    r[(pred == 0) & (y == 0)] = c['cTN']
    r[(pred == 1) & (y == 0)] = c['cFP']
    r[(pred == 0) & (y == 1)] = c['cFN']
    r[(pred == 2)] = c['cWAIT']
    return r


# ---------------------------- Trainer ----------------------------

class DTTrainer:
    def __init__(
        self,
        ds_train: UNSWSequenceDataset,
        ds_val: UNSWSequenceDataset,
        cfg: TrainerConfig,
        model_cfg: dict,
        cat_cfg: dict,
        is_cfg: Optional[Dict] = None,   # opcional (pode vir por fora)
    ):
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Tokenizers
        Dnum = ds_train.examples[0].states.shape[1]
        cat_cols = cat_cfg.get('cols', [])
        specs = build_cat_specs(
            ds_train, cat_cols, rule=cat_cfg.get('embed_rule', 'sqrt'), fixed_dim=cat_cfg.get('fixed_dim', 16)
        )

        self.state_tok = StateTokenizer(numeric_dim=Dnum, cat_specs=specs,
                                        state_embed_dim=model_cfg['embeddings']['state_embed_dim']).to(self.device)
        self.action_tok = ActionTokenizer(n_actions_plus_start=4,
                                          embed_dim=model_cfg['embeddings']['state_embed_dim']).to(self.device)
        self.rtg_tok = RTGTokenizer(embed_dim=model_cfg['embeddings']['rtg_dim']).to(self.device)
        self.time_tok = TimeEncodingFourier(d_model=model_cfg['embeddings']['time_dim'],
                                            n_freq=16, use_log1p=True).to(self.device)

        # Modelo
        self.model = DecisionTransformer(
            d_model=model_cfg['d_model'],
            n_layers=model_cfg['n_layers'],
            n_heads=model_cfg['n_heads'],
            d_ff=model_cfg['d_ff'],
            dropout=model_cfg.get('Dropout', 0.1),
            n_actions=model_cfg['vocab']['n_actions'],
        ).to(self.device)

        # Ajustar projeções para d_model (antes do otimizador)
        E_state = model_cfg['embeddings']['state_embed_dim']
        E_action = model_cfg['embeddings']['state_embed_dim']
        E_rtg = model_cfg['embeddings']['rtg_dim']
        E_time = model_cfg['embeddings']['time_dim']
        self.model.ensure_projections(E_state, E_action, E_rtg, E_time, device=self.device)

        # Otimizador
        self.opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, betas=self.cfg.betas, weight_decay=self.cfg.weight_decay)

        # Pesos da CE
        self.ce_weight = torch.tensor(self.cfg.class_weights, dtype=torch.float32, device=self.device)

        # ---- Importance Sampling config ----
        # 1) prioridade: is_cfg recebido no construtor
        # 2) depois: cfg.is_dt (TrainerConfig)
        # 3) fallback: ler do YAML externo (configs/trainer.yaml) se existir
        if is_cfg is not None:
            self.is_cfg = is_cfg
        elif getattr(self.cfg, "is_dt", None) is not None:
            self.is_cfg = self.cfg.is_dt
        else:
            try:
                _cfg_yaml = yaml.safe_load(Path("configs/trainer.yaml").read_text())
                self.is_cfg = _cfg_yaml.get("is_dt", {"enable": False})
            except Exception:
                self.is_cfg = {"enable": False}

        # Estimativa de fração positiva global do TREINO (por token válido) + EMA
        with torch.no_grad():
            tot_pos, tot_valid = 0.0, 0.0
            for ex in self.ds_train.examples:
                y = torch.from_numpy(ex.actions_out)
                m = torch.from_numpy(ex.attn_mask.astype('float32'))
                tot_pos  += float((y * m).sum().item())
                tot_valid+= float(m.sum().item())
            self.global_pos_frac = (tot_pos / max(tot_valid, 1.0)) if tot_valid > 0 else 0.01
            self.pos_ema = float(self.global_pos_frac)

    # ---- helpers ----

    def parameters(self):
        return (list(self.state_tok.parameters()) +
                list(self.action_tok.parameters()) +
                list(self.rtg_tok.parameters()) +
                list(self.time_tok.parameters()) +
                list(self.model.parameters()))

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if torch.is_tensor(v) else v
        return out

    def _cats_from_batch(self, batch: Dict[str, torch.Tensor], cols: List[str]) -> Dict[str, torch.Tensor]:
        return {c: batch[f'cat_{c}'] for c in cols if f'cat_{c}' in batch}

    # ---- IS weights ----

    def _build_token_weights_by_label(self, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Importance Sampling simples por rótulo (positivos recebem peso maior)."""
        is_dt = self.is_cfg or {}
        if not is_dt.get("enable", False):
            return torch.ones_like(m, dtype=torch.float32, device=m.device)

        mode = is_dt.get("mode", "by_label")
        if mode != "by_label":
            # ganchos para outros modos
            return torch.ones_like(m, dtype=torch.float32, device=m.device)

        power = float(is_dt.get("power", 0.5))
        w_min = float(is_dt.get("w_min", 1.0))
        w_max = float(is_dt.get("w_max", 10.0))
        normalize_batch = bool(is_dt.get("normalize_batch", True))

        # EMA da fração positiva do batch
        with torch.no_grad():
            pos = (y * m).sum()
            val = m.sum().clamp(min=1.0)
            batch_pos_frac = (pos / val).item()
            self.pos_ema = 0.9 * self.pos_ema + 0.1 * float(batch_pos_frac)

        f_pos = max(self.pos_ema, 1e-6)
        w_pos = (1.0 / f_pos) ** power
        w_pos = max(min(w_pos, w_max), w_min)

        w_tok = torch.ones_like(m, dtype=torch.float32, device=m.device)
        w_tok = torch.where(y > 0, torch.as_tensor(w_pos, device=m.device, dtype=torch.float32), w_tok)

        if normalize_batch:
            denom = (w_tok * m).sum().clamp(min=1.0)
            scale = (m.sum().clamp(min=1.0) / denom)
            w_tok = w_tok * scale

        return w_tok

    def _build_token_weights_by_attack_cat(self, attack_idx: Optional[torch.Tensor], m: torch.Tensor) -> torch.Tensor:
        """Gancho para IS por categoria (requer exportar attack_cat_idx no dataset)."""
        is_dt = self.is_cfg or {}
        if not is_dt.get("enable", False):
            return torch.ones_like(m, dtype=torch.float32, device=m.device)

        mode = is_dt.get("mode", "by_label")
        if mode != "by_attack_cat" or attack_idx is None:
            return torch.ones_like(m, dtype=torch.float32, device=m.device)

        # TODO: implementar se/quando o dataset expor attack_cat_idx e freq_cat.
        # Por ora, retorna vetor de 1s para não alterar o treinamento.
        return torch.ones_like(m, dtype=torch.float32, device=m.device)

    # ---- loop base (forward sem IS; útil para validação e compatibilidade) ----

    def step(self, batch: Dict[str, torch.Tensor], cat_cols: List[str]) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch = self._to_device(batch)
        cats = self._cats_from_batch(batch, cat_cols)

        s_emb = self.state_tok(batch['states'].float(), cats)
        a_emb = self.action_tok(batch['actions_in'])
        r_emb = self.rtg_tok(batch['rtg'].float())
        t_emb = self.time_tok(batch['delta_t'].float())

        logits = self.model(s_emb, a_emb, r_emb, t_emb, batch['attn_mask'])

        loss = cross_entropy_masked(
            logits, batch['actions_out'], batch['attn_mask'],
            weight=self.ce_weight, label_smoothing=self.cfg.label_smoothing
        )

        return loss, {}

    # ---------------------------- Treino + Validação ----------------------------

    def fit(self, dl_train: DataLoader, dl_val: DataLoader, cat_cols: List[str]):
        for epoch in range(self.cfg.max_epochs):
            # ---------- TREINO ----------
            self.model.train()
            running_loss = 0.0
            n_steps = 0

            for it, batch in enumerate(dl_train):
                self.opt.zero_grad(set_to_none=True)

                # forward (explícito aqui para injetar IS por token)
                batch_dev = self._to_device(batch)
                cats = self._cats_from_batch(batch_dev, cat_cols)

                s_emb = self.state_tok(batch_dev['states'].float(), cats)
                a_emb = self.action_tok(batch_dev['actions_in'])
                r_emb = self.rtg_tok(batch_dev['rtg'].float())
                t_emb = self.time_tok(batch_dev['delta_t'].float())

                logits = self.model(s_emb, a_emb, r_emb, t_emb, batch_dev['attn_mask'])

                # --- Importance Sampling por token (by_label) ---
                w_tok = self._build_token_weights_by_label(
                    y=batch_dev['actions_out'],
                    m=batch_dev['attn_mask'].float()
                )
                # (ganchos) por categoria, quando disponível:
                # w_tok_cat = self._build_token_weights_by_attack_cat(
                #     attack_idx=batch_dev.get('cat_attack', None),
                #     m=batch_dev['attn_mask'].float()
                # )
                # w_tok = w_tok * w_tok_cat

                loss = cross_entropy_masked(
                    logits, batch_dev['actions_out'], batch_dev['attn_mask'],
                    weight=self.ce_weight,
                    label_smoothing=self.cfg.label_smoothing,
                    token_weights=w_tok,
                )

                if loss.ndim != 0:
                    loss = loss.mean()

                loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip)
                self.opt.step()

                running_loss += float(loss.detach().item())
                n_steps += 1

                # logging leve do IS
                if (it + 1) % 50 == 0:
                    with torch.no_grad():
                        m = batch_dev['attn_mask'].float()
                        pos_frac = float((batch_dev['actions_out'] * m).sum() / m.sum().clamp(min=1))
                        mean_w = float((w_tok * m).sum() / m.sum().clamp(min=1))
                        print(f"[IS] pos_frac_batch={pos_frac:.4f} ema={self.pos_ema:.4f} mean_w={mean_w:.3f}")

                if self.cfg.steps_per_epoch and n_steps >= self.cfg.steps_per_epoch:
                    break

            print(f"[Epoch {epoch+1}] train loss={running_loss / max(n_steps, 1):.4f}")

            # ---------- VALIDAÇÃO ----------
            self.model.eval()
            val_loss_sum = 0.0
            n_batches = 0
            all_y = []; all_p = []; all_m = []
            all_pred_act = []

            with torch.no_grad():
                for batch in dl_val:
                    batch = self._to_device(batch)
                    cats = self._cats_from_batch(batch, cat_cols)

                    s_emb = self.state_tok(batch['states'].float(), cats)
                    a_emb = self.action_tok(batch['actions_in'])
                    r_emb = self.rtg_tok(batch['rtg'].float())
                    t_emb = self.time_tok(batch['delta_t'].float())

                    # ablação "No-RTG" em avaliação:
                    if getattr(self.cfg, "eval_no_rtg", False):
                        r_emb.zero_()

                    logits = self.model(s_emb, a_emb, r_emb, t_emb, batch['attn_mask'])
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

                    vloss = cross_entropy_masked(
                        logits, batch['actions_out'], batch['attn_mask'],
                        weight=self.ce_weight, label_smoothing=self.cfg.label_smoothing
                    )
                    val_loss_sum += float(vloss.item()); n_batches += 1

                    probs1 = torch.softmax(logits, dim=-1)[..., 1]
                    all_y.append(batch['actions_out'].reshape(-1).cpu().numpy())
                    all_p.append(probs1.reshape(-1).cpu().numpy())
                    all_m.append(batch['attn_mask'].reshape(-1).cpu().numpy())

                    pred_act = decode_with_wait(logits, self.cfg.wait_threshold)  # {0,1,2}
                    all_pred_act.append(pred_act.reshape(-1).cpu().numpy())

            # agrega e calcula métricas com blindagem
            y_np = np.concatenate(all_y) if all_y else np.array([])
            p_np = np.concatenate(all_p) if all_p else np.array([])
            m_np = (np.concatenate(all_m) > 0.5) if all_m else np.array([], dtype=bool)
            pa_np = np.concatenate(all_pred_act) if all_pred_act else np.array([])

            y_np = y_np[m_np]
            p_np = p_np[m_np]
            pa_np = pa_np[m_np] if pa_np.size else pa_np

            p_np = np.nan_to_num(p_np, nan=0.0, posinf=1.0, neginf=0.0)
            p_np = np.clip(p_np, 0.0, 1.0)

            def safe_ap(y, p):
                if y.size == 0 or y.max() == 0:
                    return 0.0
                return float(average_precision_score(y, p))

            pr_auc = safe_ap(y_np, p_np)
            f1 = float(f1_score(y_np, (p_np >= 0.5).astype(int), zero_division=0))

            if pa_np.size:
                c = self.cfg.reward_weights
                r = ((pa_np==1)&(y_np==1))*c['cTP'] + ((pa_np==0)&(y_np==0))*c['cTN'] \
                    + ((pa_np==1)&(y_np==0))*c['cFP'] + ((pa_np==0)&(y_np==1))*c['cFN'] \
                    + (pa_np==2)*c['cWAIT']
                reward = float(r.mean())
            else:
                reward = 0.0

            print(f"[Epoch {epoch+1}]  val loss={val_loss_sum/max(n_batches,1):.4f} f1={f1:.4f} pr_auc={pr_auc:.4f} reward={reward:.4f}")
