# src/im12dt/training/trainer_dt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

import warnings
warnings.filterwarnings("ignore", message="No positive class found in y_true")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class TrainerConfig:
    # otimização
    batch_size: int
    max_epochs: int
    steps_per_epoch: Optional[int]
    grad_clip: float
    lr: float
    betas: Tuple[float, float]
    weight_decay: float

    # perda / política
    wait_threshold: float
    class_weights: Tuple[float, ...] | None
    label_smoothing: float
    reward_weights: Dict[str, float]  # {cTP,cTN,cFP,cFN,cWAIT}

    # avaliação
    eval_no_rtg: bool = False

    # early stopping (novos campos com default para manter compatibilidade)
    early_stopping: bool = True
    early_metric: str = "pr_auc"   # "pr_auc" ou "f1"
    patience: int = 3
    min_delta: float = 1e-6


# ---------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------
def build_cat_specs(ds: UNSWSequenceDataset, cat_cols: List[str], rule: str, fixed_dim: int) -> List[CatSpec]:
    specs: List[CatSpec] = []
    for c in cat_cols:
        if hasattr(ds, "_cat_maps") and c in ds._cat_maps:
            n_tokens = len(ds._cat_maps[c])
            d = _rule_embed_dim(n_tokens, rule=rule, fixed=fixed_dim)
            specs.append(CatSpec(c, n_tokens, d))
    return specs


def make_weighted_sampler(ds: UNSWSequenceDataset, pos_weight: float = 4.0) -> WeightedRandomSampler:
    """Amostra janelas com pelo menos um positivo com mais peso."""
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
) -> torch.Tensor:
    B, K, C = logits.shape
    logits  = logits.reshape(B * K, C)
    targets = targets.reshape(B * K)
    mask    = mask.reshape(B * K).float()

    # blindagem numérica
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

    per_tok = F.cross_entropy(
        logits, targets,
        weight=weight,
        reduction="none",
        label_smoothing=(label_smoothing if label_smoothing > 0.0 else 0.0),
    )

    denom = mask.sum().clamp(min=1.0)
    loss  = (per_tok * mask).sum() / denom
    loss  = torch.nan_to_num(loss, nan=0.0, posinf=1e3, neginf=1e3)
    if loss.ndim != 0:
        loss = loss.mean()
    return loss


def decode_with_wait(logits: torch.Tensor, wait_threshold: float) -> torch.Tensor:
    """Retorna {0,1,2} (0=normal, 1=ataque, 2=wait) via confiança."""
    probs = torch.softmax(logits, dim=-1)         # (B,K,C)
    conf, argmax = probs.max(dim=-1)              # (B,K)
    pred = argmax.clone()
    pred[conf < wait_threshold] = 2               # 2 = wait
    return pred


def compute_reward(pred: torch.Tensor, y: torch.Tensor, c: Dict[str, float]) -> torch.Tensor:
    """pred: (B,K) em {0,1,2}; y: (B,K) em {0,1}"""
    r = torch.zeros_like(pred, dtype=torch.float32)
    r[(pred == 1) & (y == 1)] = c["cTP"]
    r[(pred == 0) & (y == 0)] = c["cTN"]
    r[(pred == 1) & (y == 0)] = c["cFP"]
    r[(pred == 0) & (y == 1)] = c["cFN"]
    r[(pred == 2)]            = c["cWAIT"]
    return r


# ---------------------------------------------------------------------
# Treinador
# ---------------------------------------------------------------------
class DTTrainer:
    def __init__(self, ds_train: UNSWSequenceDataset, ds_val: UNSWSequenceDataset,
                 cfg: TrainerConfig, model_cfg: dict, cat_cfg: dict):
        self.ds_train = ds_train
        self.ds_val   = ds_val
        self.cfg      = cfg
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Tokenizers ---
        Dnum     = ds_train.examples[0].states.shape[1]
        cat_cols = cat_cfg.get("cols", [])
        specs    = build_cat_specs(ds_train, cat_cols,
                                   rule=cat_cfg.get("embed_rule", "sqrt"),
                                   fixed_dim=cat_cfg.get("fixed_dim", 16))

        self.state_tok = StateTokenizer(
            numeric_dim=Dnum,
            cat_specs=specs,
            state_embed_dim=model_cfg["embeddings"]["state_embed_dim"],
        ).to(self.device)

        self.action_tok = ActionTokenizer(
            n_actions_plus_start=4,
            embed_dim=model_cfg["embeddings"]["state_embed_dim"],
        ).to(self.device)

        self.rtg_tok = RTGTokenizer(embed_dim=model_cfg["embeddings"]["rtg_dim"]).to(self.device)
        self.time_tok = TimeEncodingFourier(
            d_model=model_cfg["embeddings"]["time_dim"], n_freq=16, use_log1p=True
        ).to(self.device)

        # --- Modelo ---
        self.model = DecisionTransformer(
            d_model   = model_cfg["d_model"],
            n_layers  = model_cfg["n_layers"],
            n_heads   = model_cfg["n_heads"],
            d_ff      = model_cfg["d_ff"],
            dropout   = model_cfg.get("Dropout", 0.1),
            n_actions = model_cfg["vocab"]["n_actions"],   # 2 (normal/attack)
        ).to(self.device)

        # Projeções (garantir dims corretas)
        self.model.ensure_projections(
            model_cfg["embeddings"]["state_embed_dim"],   # E_state
            model_cfg["embeddings"]["state_embed_dim"],   # E_action (mesmo dim)
            model_cfg["embeddings"]["rtg_dim"],           # E_rtg
            model_cfg["embeddings"]["time_dim"],          # E_time
            device=self.device,
        )

        # --- Otimizador: TODOS os parâmetros (model + tokenizers) ---
        self.opt = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr,
            betas=self.cfg.betas, weight_decay=self.cfg.weight_decay
        )

        # --- Pesos da CE (robusto a comprimentos inesperados) ---
        n_classes = model_cfg["vocab"]["n_actions"]
        ce_w = None
        if self.cfg.class_weights and len(self.cfg.class_weights) >= n_classes:
            ce_w = torch.tensor(self.cfg.class_weights[:n_classes], dtype=torch.float32, device=self.device)
        self.ce_weight = ce_w  # pode ser None

        # --- Early stopping state ---
        self._best = {
            "metric": -float("inf"),
            "epoch":  -1,
            "state":  None,
            "epochs_no_improve": 0,
        }

    # -----------------------------------------------------------------
    # Infra
    # -----------------------------------------------------------------
    def parameters(self):
        """Parâmetros treináveis (modelo + tokenizers)."""
        return list(self.state_tok.parameters()) \
             + list(self.action_tok.parameters()) \
             + list(self.rtg_tok.parameters()) \
             + list(self.time_tok.parameters()) \
             + list(self.model.parameters())

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _cats_from_batch(self, batch: Dict[str, torch.Tensor], cols: List[str]) -> Dict[str, torch.Tensor]:
        return {c: batch[f"cat_{c}"].to(self.device) for c in cols if f"cat_{c}" in batch}

    # -----------------------------------------------------------------
    # Passo de treino
    # -----------------------------------------------------------------
    def step(self, batch: Dict[str, torch.Tensor], cat_cols: List[str]) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch = self._to_device(batch)
        cats  = self._cats_from_batch(batch, cat_cols)

        # forward
        s_emb = self.state_tok(batch["states"].float(), cats)
        a_emb = self.action_tok(batch["actions_in"])
        r_emb = self.rtg_tok(batch["rtg"].float())
        t_emb = self.time_tok(batch["delta_t"].float())

        logits = self.model(s_emb, a_emb, r_emb, t_emb, batch["attn_mask"])

        loss = cross_entropy_masked(
            logits, batch["actions_out"], batch["attn_mask"],
            weight=self.ce_weight, label_smoothing=self.cfg.label_smoothing
        )
        return loss, {}

    # -----------------------------------------------------------------
    # Loop de treino + validação + early stopping
    # -----------------------------------------------------------------
    def fit(self, dl_train: DataLoader, dl_val: DataLoader, cat_cols: List[str]):
        for epoch in range(self.cfg.max_epochs):
            # ------- TREINO -------
            self.model.train()
            running_loss = 0.0
            n_steps = 0

            for it, batch in enumerate(dl_train):
                self.opt.zero_grad(set_to_none=True)

                loss, _ = self.step(batch, cat_cols)
                if loss.ndim != 0:
                    loss = loss.mean()

                loss.backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip)
                self.opt.step()

                running_loss += float(loss.detach().item())
                n_steps += 1
                if self.cfg.steps_per_epoch and n_steps >= self.cfg.steps_per_epoch:
                    break

            train_loss = running_loss / max(n_steps, 1)
            print(f"[Epoch {epoch+1}] train loss={train_loss:.4f}")

            # ------- VALIDAÇÃO -------
            self.model.eval()
            val_loss_sum = 0.0
            n_batches    = 0
            all_y, all_p, all_m, all_pred_act = [], [], [], []

            with torch.no_grad():
                for batch in dl_val:
                    batch = self._to_device(batch)
                    cats  = self._cats_from_batch(batch, cat_cols)

                    s_emb = self.state_tok(batch["states"].float(), cats)
                    a_emb = self.action_tok(batch["actions_in"])
                    r_emb = self.rtg_tok(batch["rtg"].float())
                    t_emb = self.time_tok(batch["delta_t"].float())

                    if self.cfg.eval_no_rtg:
                        r_emb.zero_()

                    logits = self.model(s_emb, a_emb, r_emb, t_emb, batch["attn_mask"])
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

                    vloss = cross_entropy_masked(
                        logits, batch["actions_out"], batch["attn_mask"],
                        weight=self.ce_weight, label_smoothing=self.cfg.label_smoothing
                    )
                    val_loss_sum += float(vloss.item()); n_batches += 1

                    probs1 = torch.softmax(logits, dim=-1)[..., 1]
                    all_y.append(batch["actions_out"].reshape(-1).cpu().numpy())
                    all_p.append(probs1.reshape(-1).cpu().numpy())
                    all_m.append(batch["attn_mask"].reshape(-1).cpu().numpy())

                    pred_act = decode_with_wait(logits, self.cfg.wait_threshold)
                    all_pred_act.append(pred_act.reshape(-1).cpu().numpy())

            # agrega métricas
            y_np = np.concatenate(all_y) if all_y else np.array([])
            p_np = np.concatenate(all_p) if all_p else np.array([])
            m_np = (np.concatenate(all_m) > 0.5) if all_m else np.array([], dtype=bool)
            pa_np = np.concatenate(all_pred_act) if all_pred_act else np.array([])

            y_np = y_np[m_np]
            p_np = np.clip(np.nan_to_num(p_np[m_np], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            if pa_np.size:
                pa_np = pa_np[m_np]

            def safe_ap(y, p):
                if y.size == 0 or y.max() == 0:
                    return 0.0
                return float(average_precision_score(y, p))

            pr_auc = safe_ap(y_np, p_np)
            f1     = float(f1_score(y_np, (p_np >= 0.5).astype(int), zero_division=0))
            reward = 0.0
            if pa_np.size:
                c = self.cfg.reward_weights
                r = ((pa_np==1)&(y_np==1))*c['cTP'] + ((pa_np==0)&(y_np==0))*c['cTN'] \
                    + ((pa_np==1)&(y_np==0))*c['cFP'] + ((pa_np==0)&(y_np==1))*c['cFN'] \
                    + (pa_np==2)*c['cWAIT']
                reward = float(r.mean())

            val_loss = val_loss_sum / max(n_batches, 1)
            print(f"[Epoch {epoch+1}]  val loss={val_loss:.4f} f1={f1:.4f} pr_auc={pr_auc:.4f} reward={reward:.4f}")

            # ------- Early stopping -------
            metric = pr_auc if self.cfg.early_metric.lower() == "pr_auc" else f1
            if not np.isfinite(metric):
                metric = -float("inf")

            improved = (metric > self._best["metric"] + self.cfg.min_delta)
            if improved:
                self._best.update({
                    "metric": metric,
                    "epoch":  epoch + 1,
                    "state":  self._snapshot_state(),
                    "epochs_no_improve": 0,
                })
            else:
                self._best["epochs_no_improve"] += 1

            if self.cfg.early_stopping and self._best["epochs_no_improve"] >= self.cfg.patience:
                print(f"[EARLY] stopped on epoch {epoch+1}, best {self.cfg.early_metric}={self._best['metric']:.4f} @epoch {self._best['epoch']}")
                self._restore_best()
                return  # encerra treinamento

        # fim do loop — se treinou tudo, restaura o melhor também (boa prática)
        if self._best["state"] is not None:
            self._restore_best()

    # -----------------------------------------------------------------
    # Snapshot / Restore (modelo + tokenizers)
    # -----------------------------------------------------------------
    def _snapshot_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "model":      self.model.state_dict(),
            "state_tok":  self.state_tok.state_dict(),
            "action_tok": self.action_tok.state_dict(),
            "rtg_tok":    self.rtg_tok.state_dict(),
            "time_tok":   self.time_tok.state_dict(),
        }

    def _restore_best(self):
        if self._best["state"] is None:
            return
        self.model.load_state_dict(self._best["state"]["model"])
        self.state_tok.load_state_dict(self._best["state"]["state_tok"])
        self.action_tok.load_state_dict(self._best["state"]["action_tok"])
        self.rtg_tok.load_state_dict(self._best["state"]["rtg_tok"])
        self.time_tok.load_state_dict(self._best["state"]["time_tok"])
