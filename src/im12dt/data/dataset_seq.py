# src/im12dt/data/dataset_seq.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .sequence_builder import build_trajectory_windows  # exige patch do Dia 5 (abs_time/flow_id/cats)


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
@dataclass
class SeqDatasetConfig:
    csv_path: str
    flow_keys: List[str]
    time_col: Optional[str]
    context_length: int
    start_action: int
    pad_token: int
    normalize: bool
    label_col: str
    attack_cat_col: str
    # novas: lista de colunas categóricas a mapear (ex.: ["proto","service","state"])
    categorical_cols: Optional[List[str]] = None


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _ensure_2d_row(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x


def _select_numeric_columns(df: pd.DataFrame, drop: List[str] | set[str]) -> List[str]:
    drop_set = set(drop)
    cols = []
    for c in df.columns:
        if c in drop_set:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _valid_flow_keys(df: pd.DataFrame, keys: List[str] | None) -> List[str]:
    if not keys:
        return []
    return [k for k in keys if k in df.columns]


def _to_np32(x: Any) -> np.ndarray:
    """Converte list/np/tensor -> np.float32."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
class UNSWSequenceDataset(Dataset):
    """
    Constrói janelas causais (K) a partir do CSV UNSW_NB15, com:
      - normalização opcional (overrideável com stats do treino),
      - mapeamento categórico por coluna (overrideável com stoi do treino),
      - exporta 'flow_id' e 'abs_time' para avaliação de TTR.
    Requer que sequence_builder tenha sido patchado para propagar abs_time/flow_id/cats.
    """

    def __init__(
        self,
        cfg: SeqDatasetConfig,
        max_rows: Optional[int] = None,
        *,
        stats_override: Optional[Dict[str, np.ndarray]] = None,
        cat_maps_override: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        super().__init__()
        self.cfg = cfg

        # ------------------ leitura ------------------
        df = pd.read_csv(cfg.csv_path)
        if max_rows is not None:
            df = df.head(int(max_rows))

        # ------------------ categóricas -> stoi/ids ------------------
        cat_cols = [c for c in (cfg.categorical_cols or []) if c in df.columns]
        self.cat_cols: List[str] = cat_cols
        self._cat_maps: Dict[str, Dict[str, int]] = {}

        if cat_maps_override:
            # usa mapas do treino; garante <UNK>=0
            for c in cat_cols:
                base = dict(cat_maps_override.get(c, {}))
                if "<UNK>" not in base:
                    base = {"<UNK>": 0, **{k: v for k, v in base.items() if k != "<UNK>"}}
                else:
                    # assegura que <UNK> é 0; se não for, reindexa
                    if base.get("<UNK>", None) != 0:
                        inv = sorted(base.items(), key=lambda kv: kv[1])
                        # realoca mantendo as demais posições
                        new = {"<UNK>": 0}
                        nxt = 1
                        for k, v in inv:
                            if k == "<UNK>":
                                continue
                            new[k] = nxt
                            nxt += 1
                        base = new
                self._cat_maps[c] = base
        else:
            # cria stoi por ordem de aparecimento no CSV
            for c in cat_cols:
                vals = df[c].astype(str).fillna("<UNK>")
                uniq = list(dict.fromkeys(vals.tolist()))
                stoi = {"<UNK>": 0}
                for u in uniq:
                    if u != "<UNK>" and u not in stoi:
                        stoi[u] = len(stoi)
                self._cat_maps[c] = stoi

        # materializa colunas __cat_* por linha
        for c in cat_cols:
            stoi = self._cat_maps[c]
            df[f"__cat_{c}"] = df[c].astype(str).fillna("<UNK>").map(lambda s: stoi.get(s, 0)).astype("int64")

        # ------------------ seleção numérica ------------------
        # sempre remove originais categóricas, ids, rótulos e tempo
        drop_cols = set(
            [cfg.label_col, cfg.attack_cat_col, "id"]   # ← NÃO removemos cfg.time_col aqui
            + [c for c in cat_cols]
            + [f"__cat_{c}" for c in cat_cols]
        ) - {None}
        num_cols = _select_numeric_columns(df, drop=drop_cols)
        self.num_cols: List[str] = num_cols

        X = df[num_cols].to_numpy(dtype=np.float32)  # (N, D_num)
        y = df[cfg.label_col].astype(int).to_numpy()  # (N,)
        attack_cat = df[cfg.attack_cat_col].astype(str).fillna("Normal").to_numpy()

        # ------------------ tempo absoluto e Δt ------------------
        if cfg.time_col and cfg.time_col in df.columns:
            t_abs = df[cfg.time_col].astype(float).to_numpy()
        else:
            # surrogate: índice crescente
            t_abs = np.arange(len(df), dtype=np.float64)
        # Δt por linha (em relação ao anterior globalmente); ok porque agrupamos por índice depois
        dt_global = np.diff(t_abs, prepend=t_abs[0])

        # ------------------ normalização ------------------
        self._stats: Optional[Dict[str, np.ndarray]] = None
        if cfg.normalize:
            if stats_override and ("mean" in stats_override) and ("std" in stats_override):
                m = _to_np32(stats_override["mean"])
                s = _to_np32(stats_override["std"])
                # shapes esperadas: (1, D) ou (D,)
                m = _ensure_2d_row(m)
                s = _ensure_2d_row(s)
            else:
                m = np.nanmean(X, axis=0, keepdims=True)
                s = np.nanstd(X, axis=0, keepdims=True)
            s = s + 1e-6
            X = (np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) - m) / s
            self._stats = {"mean": m.astype(np.float32), "std": s.astype(np.float32)}

        # ------------------ flow grouping ------------------
        flow_keys = _valid_flow_keys(df, cfg.flow_keys)
        if flow_keys:
            # hash determinístico por conjunto de colunas
            df["_flow_key"] = pd.util.hash_pandas_object(df[flow_keys], index=False).astype(np.int64)
            groups = [g for _, g in df.groupby("_flow_key", sort=False)]
        else:
            df["_flow_key"] = 0
            groups = [df]

        # mapeamento flow_id -> categoria dominante (para diagnóstico)
        self._flow_attack_cat: Dict[int, str] = {}

        # ------------------ construir janelas ------------------
        K = int(cfg.context_length)
        start_id = int(cfg.start_action)
        examples = []

        for g in groups:
            rows = g.index.values
            S = X[rows]                                     # (T, D)
            A = y[rows].astype(np.int64)                    # (T,)
            R = (A == 1).astype(np.float32)                 # reward simples para RTG
            # Tempo local do fluxo:
            if cfg.time_col and (cfg.time_col in df.columns):
                # Ex.: cfg.time_col == "dur" (segundos por registro)
                dt_flow = df.loc[rows, cfg.time_col].astype(float).to_numpy().astype(np.float32)
            elif "dur" in df.columns:
                dt_flow = df.loc[rows, "dur"].astype(float).to_numpy().astype(np.float32)
            else:
                dt_flow = np.ones(len(rows), dtype=np.float32)
            at_flow = np.cumsum(dt_flow, dtype=np.float64)
            DT = dt_flow
            AT = at_flow
            FID = int(g["_flow_key"].iloc[0])

            # categóricas alinhadas às linhas do grupo
            CATS: Optional[Dict[str, np.ndarray]] = None
            if len(cat_cols) > 0:
                CATS = {c: df.loc[rows, f"__cat_{c}"].to_numpy(dtype=np.int64) for c in cat_cols}

            # attack_cat dominante no fluxo (dos positivos; se não houver, majoritário global)
            g_labels = y[rows]
            g_attack = attack_cat[rows]
            if (g_labels == 1).any():
                vals, counts = np.unique(g_attack[g_labels == 1], return_counts=True)
            else:
                vals, counts = np.unique(g_attack, return_counts=True)
            self._flow_attack_cat[FID] = str(vals[np.argmax(counts)])

            traj = {
                "states": S,
                "actions": A,
                "rewards": R,
                "delta_t": DT,
                "abs_time": AT,
                "flow_id": np.full_like(A, FID, dtype=np.int64),
                "cats": CATS,
            }
            windows = build_trajectory_windows(traj, K, start_id)
            examples.extend(windows)

        self.examples = examples

    # ------------------ dataset protocol ------------------
    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        out: Dict[str, torch.Tensor] = {
            "states": torch.from_numpy(ex.states),           # (K, D)
            "actions_in": torch.from_numpy(ex.actions_in),   # (K,)
            "actions_out": torch.from_numpy(ex.actions_out), # (K,)
            "rtg": torch.from_numpy(ex.rtg),                 # (K,)
            "delta_t": torch.from_numpy(ex.delta_t),         # (K,)
            "attn_mask": torch.from_numpy(ex.attn_mask),     # (K,) uint8
            "length": torch.tensor(int(ex.length), dtype=torch.int64),
        }
        # extras do Dia 5
        if hasattr(ex, "abs_time") and ex.abs_time is not None:
            out["abs_time"] = torch.from_numpy(ex.abs_time.astype(np.float64))  # (K,)
        if hasattr(ex, "flow_id") and ex.flow_id is not None:
            out["flow_id"] = torch.tensor(int(ex.flow_id), dtype=torch.int64)   # escalar

        # categóricas janeladas
        if hasattr(ex, "cats") and isinstance(ex.cats, dict):
            for c, arr in ex.cats.items():
                out[f"cat_{c}"] = torch.from_numpy(arr.astype(np.int64))        # (K,)

        return out

    # ------------------ utilidades ------------------
    def get_flow_attack_cat(self, fid: int) -> str:
        """Retorna a attack_cat dominante (no fluxo) para diagnóstico."""
        return self._flow_attack_cat.get(int(fid), "Unknown")


# -------------------------------------------------------------------
# Collate
# -------------------------------------------------------------------
def seq_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Empilha a lista de amostras (cada uma com forma (K, *)) em tensores (B, K, *).
    Inclui também campos opcionais: cat_*, abs_time e flow_id.
    """
    # obrigatórios
    states = torch.stack([b["states"] for b in batch], dim=0).to(torch.float32)            # (B, K, D)
    actions_in = torch.stack([b["actions_in"] for b in batch], dim=0).to(torch.long)       # (B, K)
    actions_out = torch.stack([b["actions_out"] for b in batch], dim=0).to(torch.long)     # (B, K)
    rtg = torch.stack([b["rtg"] for b in batch], dim=0).to(torch.float32)                  # (B, K)
    delta_t = torch.stack([b["delta_t"] for b in batch], dim=0).to(torch.float32)          # (B, K)
    attn_mask = torch.stack([b["attn_mask"] for b in batch], dim=0).to(torch.uint8)        # (B, K)
    length = torch.stack([b["length"] for b in batch], dim=0).to(torch.long)               # (B,)

    out: Dict[str, torch.Tensor] = {
        "states": states,
        "actions_in": actions_in,
        "actions_out": actions_out,
        "rtg": rtg,
        "delta_t": delta_t,
        "attn_mask": attn_mask,
        "length": length,
    }

    # opcionais: abs_time (B,K), flow_id (B,)
    if "abs_time" in batch[0]:
        abs_time = torch.stack([b["abs_time"] for b in batch], dim=0).to(torch.float64)     # (B, K)
        out["abs_time"] = abs_time
    if "flow_id" in batch[0]:
        flow_id = torch.stack([b["flow_id"] for b in batch], dim=0).to(torch.long)          # (B,)
        out["flow_id"] = flow_id

    # categóricas: todos os campos cat_* presentes na primeira amostra
    cat_keys = [k for k in batch[0].keys() if k.startswith("cat_")]
    for k in cat_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0).to(torch.long)                   # (B, K)

    return out
