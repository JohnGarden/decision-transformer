from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .sequence_builder import SequenceExample, build_trajectory_windows

_NUMERIC_KINDS = set("iufcb")  # int, unsigned, float, complex, bool

# -------------------------------
# Utilidades de features
# -------------------------------

def _select_flow_keys(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def _derive_label(df: pd.DataFrame, label_col: str, attack_cat_col: Optional[str]) -> pd.Series:
    if label_col in df.columns:
        return df[label_col].astype(int)
    if attack_cat_col and attack_cat_col in df.columns:
        ac = df[attack_cat_col].astype(str).str.lower()
        return (ac != "normal").astype(int)
    raise KeyError("Não foi possível derivar a coluna de label. Informe 'label_col' ou 'attack_cat_col'.")


def _time_from_row(df: pd.DataFrame, preferred: Optional[str]) -> np.ndarray:
    # Prioridade: coluna informada -> 'timestamp'/'time'/'stime' -> 'dur' acumulado -> passo=1
    cols = []
    if preferred and preferred in df.columns:
        cols = [preferred]
    else:
        for c in ["timestamp", "time", "stime"]:
            if c in df.columns:
                cols = [c]; break
    if cols:
        t = pd.to_datetime(df[cols[0]], errors="coerce")
        if t.notna().any():
            # segundos relativos ao primeiro observado
            ts = t.view("int64") / 1e9  # ns → s
            ts = ts - np.nanmin(ts)
            ts[np.isnan(ts)] = 0.0
            return ts.astype(np.float32)
    if "dur" in df.columns:
        # trata dur como delta por linha e faz cumulativa
        d = pd.to_numeric(df["dur"], errors="coerce").fillna(0.0).astype(float).values
        return np.cumsum(d).astype(np.float32)
    # fallback: passos uniformes
    return np.arange(len(df), dtype=np.float32)


def _delta_from_time(t: np.ndarray) -> np.ndarray:
    if t.size == 0:
        return t
    dt = np.diff(t, prepend=t[:1])
    dt[0] = 0.0
    return dt.astype(np.float32)


def _split_by_flows(df: pd.DataFrame, flow_keys: List[str]) -> List[pd.DataFrame]:
    if not flow_keys:
        # Sem chaves: retorna uma trajetória única com a ordem original.
        return [df]
    groups = []
    for _, g in df.groupby(flow_keys, sort=False, dropna=False):
        groups.append(g)
    return groups


def _select_numeric_columns(df: pd.DataFrame, drop: List[str]) -> List[str]:
    cols = []
    for c, s in df.items():
        if c in drop:  # ignora colunas de controle/label
            continue
        if s.dtype.kind in _NUMERIC_KINDS:
            cols.append(c)
    return cols


def _standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    return mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std

# -------------------------------
# Dataset
# -------------------------------

@dataclass
class SeqDatasetConfig:
    # ---- todos sem default primeiro ----
    csv_path: str
    flow_keys: List[str]
    time_col: Optional[str]
    context_length: int
    start_action: int
    pad_token: int
    normalize: bool
    label_col: str
    attack_cat_col: Optional[str]
    # ---- só depois vêm os com default ----
    categorical_cols: list[str] | None = None


class UNSWSequenceDataset(Dataset):
    """Constrói janelas K (SequenceExample) a partir de um CSV UNSW-NB15.

    - Gera ações alvo a partir da coluna `label` (0=benign, 1=malicious).
    - Recompensa de **baseline**: r_t = 1.0 para todo passo (imitando especialista),
      o que torna RTG_t := T - t. Pesos sofisticados entram em etapas futuras.
    - Δt derivado da coluna temporal, de `dur` ou de passos discretos.
    - Apenas **colunas numéricas** são usadas em `states` inicialmente (as categóricas serão
      incorporadas via embeddings futuramente).
    """

    def __init__(self, cfg: SeqDatasetConfig, max_rows: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.max_rows = max_rows

        df = pd.read_csv(cfg.csv_path)
        if max_rows is not None:
            df = df.head(max_rows)

        # Ordenação determinística (se houver uma coluna de índice/tempo)
        if cfg.time_col and cfg.time_col in df.columns:
            df = df.sort_values(cfg.time_col, kind="mergesort").reset_index(drop=True)
        elif "id" in df.columns:
            df = df.sort_values("id", kind="mergesort").reset_index(drop=True)

        # Label alvo
        y = _derive_label(df, cfg.label_col, cfg.attack_cat_col)
        df = df.assign(_label=y.values)

        # Construção de tempo e Δt
        t = _time_from_row(df, cfg.time_col)
        dt = _delta_from_time(t)
        df = df.assign(_time=t, _dt=dt)

        # Seleção de features numéricas
        #drop_cols = set([cfg.label_col, cfg.attack_cat_col, cfg.time_col, "_label", "_time", "_dt"]) - {None}
        '''
        # retirar id explicitamente do conjunto de features e tornar a seleção numérica robusta
        # __init__ já ordena por time_col quando existe; com time_col=null, ele ordena por id (que está presente). O Δt permanece derivado de dur cumulativo
        drop_cols = set([cfg.label_col, cfg.attack_cat_col, cfg.time_col, "_label", "_time", "_dt", "id"]) - {None}
        num_cols = _select_numeric_columns(df, drop=list(drop_cols))
        X = df[num_cols].to_numpy(dtype=np.float32)
        if cfg.normalize and X.size > 0:
            mean, std = _standardize_fit(X)
            X = _standardize_apply(X, mean, std)
            self._stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "num_cols": num_cols}
        else:
            self._stats = {"mean": None, "std": None, "num_cols": num_cols}
        '''


        # Seleção de features numéricas
        drop_cols = set([cfg.label_col, cfg.attack_cat_col, cfg.time_col, "_label", "_time", "_dt", "id"]) - {None}
        num_cols = _select_numeric_columns(df, drop=list(drop_cols))
        # 1) coagir valores não numéricos -> NaN; 2) imputar faltas; 3) normalizar com nan-safe
        if num_cols:
            df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
            df[num_cols] = df[num_cols].fillna(0.0)
            X = df[num_cols].to_numpy(dtype=np.float32)
        else:
            X = np.zeros((len(df), 0), dtype=np.float32)
        if cfg.normalize and X.size > 0:
            # médias/desvios nan-safe
            
            mean = np.nanmean(X, axis=0, keepdims=True)
            std  = np.nanstd (X, axis=0, keepdims=True) + 1e-8
            X = (X - mean) / std
            # CLIPPING de z-score para evitar estouro no attention
            np.clip(X, -8.0, 8.0, out=X)
            # blindagem final
            X[~np.isfinite(X)] = 0.0            
            
            self._stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "num_cols": num_cols}
        else:
            self._stats = {"mean": None, "std": None, "num_cols": num_cols}



##############


        # Ações (0=benign, 1=malicious) — START será aplicado no builder
        A = df["_label"].to_numpy(dtype=np.int64)
        R = np.ones_like(A, dtype=np.float32)  # baseline: +1 por passo
        DT = df["_dt"].to_numpy(dtype=np.float32)


       # Categóricas → IDs
        cat_cols = [c for c in (cfg.categorical_cols or []) if c in df.columns]
        cat_maps: dict[str, dict[str, int]] = {}
        cat_ids_df = {}
        for c in cat_cols:
            vals = df[c].astype(str).fillna("<UNK>")
            uniq = list(dict.fromkeys(vals.tolist()))  # ordem de aparecimento
            stoi = {u: i for i, u in enumerate(["<UNK>"] + [u for u in uniq if u != "<UNK>"])}
            cat_maps[c] = stoi
            cat_ids_df[c] = vals.map(lambda s: stoi.get(s, 0)).astype(int)
        self._cat_maps = cat_maps


        # Quebra por flows
        flow_keys = _select_flow_keys(df, cfg.flow_keys)
        if flow_keys:
            df["_flow_key"] = pd.util.hash_pandas_object(df[flow_keys], index=False).astype(np.int64)
            groups = [g for _, g in df.groupby("_flow_key", sort=False)]
        else:
            groups = [df]

        # Constrói janelas por flow
        K = cfg.context_length
        start_action_id = cfg.start_action
        examples: List[SequenceExample] = []

        for g in groups:
            rows = g.index.values
            S = X[rows]
            AA = A[rows]
            RR = R[rows]
            # Δt sempre finito e não-negativo
            DDT = DT[rows]
            DDT = np.nan_to_num(DDT, nan=0.0, posinf=1e6, neginf=0.0)
            DDT = np.maximum(DDT, 0.0)

            traj = {"states": S, "actions": AA, "rewards": RR, "delta_t": DDT}
            # anexar categóricas como dict de arrays alinhados
            if cat_cols:
                traj["cats"] = {c: cat_ids_df[c].to_numpy(dtype=int)[rows] for c in cat_cols}

            windows = build_trajectory_windows(traj, K, start_action_id)
            examples.extend(windows)

        self.examples = examples

    # --------------- PyTorch API ---------------
    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        out = {
            "states": torch.from_numpy(ex.states),        # (K,d)
            "actions_in": torch.from_numpy(ex.actions_in),
            "actions_out": torch.from_numpy(ex.actions_out),
            "rtg": torch.from_numpy(ex.rtg),
            "delta_t": torch.from_numpy(ex.delta_t),
            "attn_mask": torch.from_numpy(ex.attn_mask),
            "length": torch.tensor(ex.length, dtype=torch.int64),
        }
        # exporta categóricas
        if hasattr(ex, "cats") and isinstance(ex.cats, dict):
            for c, arr in ex.cats.items():
                out[f"cat_{c}"] = torch.from_numpy(arr.astype("int64"))
        return out

def seq_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in batch[0].keys():
        if k == "length":
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out