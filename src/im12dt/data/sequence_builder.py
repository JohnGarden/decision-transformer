# src/im12dt/data/sequence_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


# ------------------------------------------------------------
# Estrutura da janela / exemplo
# ------------------------------------------------------------
@dataclass
class SequenceExample:
    # tensores janelados, todos com comprimento K após padding
    states: np.ndarray        # (K, D) float32
    actions_in: np.ndarray    # (K,)   int64   (inclui <START> no t=0)
    actions_out: np.ndarray   # (K,)   int64   (rótulo/token alvo por passo)
    rtg: np.ndarray           # (K,)   float32 (return-to-go por passo)
    delta_t: np.ndarray       # (K,)   float32 (Δt relativo por passo)
    attn_mask: np.ndarray     # (K,)   uint8   (1 = válido, 0 = padding)
    length: int               # T real da janela (<= K)

    # campos extras (Dia 5)
    abs_time: Optional[np.ndarray] = None       # (K,) float64 - carimbo temporal absoluto
    flow_id: Optional[int] = None               # id do fluxo (constante na janela)
    cats: Optional[Dict[str, np.ndarray]] = None  # {col: (K,)} ids categóricos


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def right_pad_to_K(x: np.ndarray, K: int, pad_value=0, axis: int = 0) -> Tuple[np.ndarray, int]:
    """
    Faz padding à direita ao longo do eixo `axis` até comprimento K.
    Retorna (x_pad, L), onde L = tamanho original (limitado a K).
    """
    x = np.asarray(x)
    T = x.shape[axis]
    L = int(min(T, K))

    if x.ndim == 1:
        out = np.full((K,), pad_value, dtype=x.dtype)
        out[:L] = x[:L]
        return out, L

    # padding para 2D/ND ao longo do eixo 0 (mais comum: tempo)
    if axis != 0:
        raise NotImplementedError("right_pad_to_K atualmente implementa padding ao longo de axis=0.")

    pad_shape = list(x.shape)
    pad_shape[0] = K
    out = np.full(pad_shape, pad_value, dtype=x.dtype)
    out[:L] = x[:L]
    return out, L


def sliding_windows(T: int, K: int, stride: Optional[int] = None):
    """
    Gera janelas [s:e) sobre um trajeto de comprimento T.
    Por padrão, usa janelas não sobrepostas (stride=K).
    """
    if T <= 0:
        return
    if stride is None or stride <= 0:
        stride = K
    s = 0
    while s < T:
        e = min(s + K, T)
        yield s, e
        s += stride


def _rtg_from_rewards(r: np.ndarray) -> np.ndarray:
    """
    Return-to-go (soma de recompensas futuras) por posição, calculado reversamente.
    Entrada: (L,)
    Saída:   (L,)
    """
    r = np.asarray(r, dtype=np.float32)
    out = np.zeros_like(r, dtype=np.float32)
    acc = 0.0
    for i in range(r.shape[0] - 1, -1, -1):
        acc = r[i] + acc
        out[i] = acc
    return out


# ------------------------------------------------------------
# Construção de janelas
# ------------------------------------------------------------
def build_window(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    delta_t: np.ndarray,
    K: int,
    start_action_id: int,
    *,
    abs_time: Optional[np.ndarray] = None,
    flow_id: Optional[int] = None,
    cats: Optional[Dict[str, np.ndarray]] = None,
) -> SequenceExample:
    """
    Constrói uma janela causal com padding à direita até K, gerando:
      - actions_in: [<START>, a_0, a_1, ...] (shift à direita)
      - actions_out: [a_0, a_1, ...]
      - rtg: soma reversa das recompensas por passo
      - attn_mask: 1 nos passos válidos (até L), 0 no padding
    Também propaga abs_time, flow_id e ids categóricos (cats) por janela.
    """
    # Garantir formas
    states = np.asarray(states, dtype=np.float32)          # (T, D)
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)    # (T,)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)  # (T,)
    delta_t = np.asarray(delta_t, dtype=np.float32).reshape(-1)  # (T,)

    T = states.shape[0]
    assert actions.shape[0] == T and rewards.shape[0] == T and delta_t.shape[0] == T, \
        "Comprimentos inconsistentes em (states, actions, rewards, delta_t)."

    # Return-to-go (antes do padding)
    rtg = _rtg_from_rewards(rewards)  # (T,)

    # Padding à direita
    states_pad, L = right_pad_to_K(states, K, pad_value=0.0, axis=0)  # (K, D)
    actions_out_pad, _ = right_pad_to_K(actions, K, pad_value=0, axis=0)  # (K,)
    rtg_pad, _ = right_pad_to_K(rtg, K, pad_value=0.0, axis=0)            # (K,)
    dt_pad, _ = right_pad_to_K(delta_t, K, pad_value=0.0, axis=0)         # (K,)

    # actions_in: <START> no t=0; em seguida, ações deslocadas
    actions_in_pad = np.zeros((K,), dtype=np.int64)
    actions_in_pad[0] = int(start_action_id)
    if L > 0:
        # copia ações da janela para posições 1..L-1
        actions_in_pad[1:L] = actions_out_pad[:L - 1]

    # máscara causal de atenção (1 = válido, 0 = padding)
    mask = np.zeros((K,), dtype=np.uint8)
    mask[:L] = 1

    # abs_time (opcional)
    abs_pad = None
    if abs_time is not None:
        abs_time = np.asarray(abs_time, dtype=np.float64).reshape(-1)
        abs_pad, _ = right_pad_to_K(abs_time, K, pad_value=0.0, axis=0)  # (K,)

    # cats (opcional): cada coluna categórica vira vetor (K,)
    cats_pad = None
    if cats:
        cats_pad = {}
        for name, arr in cats.items():
            arr = np.asarray(arr, dtype=np.int64).reshape(-1)
            catp, _ = right_pad_to_K(arr, K, pad_value=0, axis=0)
            cats_pad[name] = catp

    return SequenceExample(
        states=states_pad.astype(np.float32),
        actions_in=actions_in_pad.astype(np.int64),
        actions_out=actions_out_pad.astype(np.int64),
        rtg=rtg_pad.astype(np.float32),
        delta_t=dt_pad.astype(np.float32),
        attn_mask=mask.astype(np.uint8),
        length=int(L),
        abs_time=abs_pad,
        flow_id=(int(flow_id) if flow_id is not None else None),
        cats=cats_pad,
    )


def build_trajectory_windows(
    traj: Dict[str, np.ndarray],
    K: int,
    start_action_id: int,
    *,
    stride: Optional[int] = None,
) -> List[SequenceExample]:
    """
    Recebe um trajeto (dicionário de arrays 1D/2D) e o decompõe em janelas de tamanho K.
    Por padrão, usa janelas NÃO sobrepostas (stride=K).
    Campos esperados em `traj`:
      - states:  (T, D)
      - actions: (T,)
      - rewards: (T,)
      - delta_t: (T,)
    Campos opcionais:
      - abs_time: (T,)
      - flow_id:  (T,) ou escalar
      - cats:     dict[col] -> (T,)
    """
    S = np.asarray(traj["states"], dtype=np.float32)
    A = np.asarray(traj["actions"], dtype=np.int64).reshape(-1)
    R = np.asarray(traj["rewards"], dtype=np.float32).reshape(-1)
    DT = np.asarray(traj["delta_t"], dtype=np.float32).reshape(-1)

    T = S.shape[0]
    assert A.shape[0] == T and R.shape[0] == T and DT.shape[0] == T, \
        "Comprimentos inconsistentes no trajeto."

    AT = traj.get("abs_time", None)
    if AT is not None:
        AT = np.asarray(AT, dtype=np.float64).reshape(-1)
        assert AT.shape[0] == T, "abs_time e states com comprimentos diferentes."

    FID = traj.get("flow_id", None)  # pode ser escalar ou array(T,)
    if FID is not None and np.ndim(FID) > 0:
        # se veio vetor por linha, fixamos como o primeiro valor (id constante do fluxo)
        flow_id_scalar = int(np.asarray(FID).reshape(-1)[0])
    else:
        flow_id_scalar = int(FID) if FID is not None else None

    CATS = traj.get("cats", None)
    if CATS is not None:
        CATS = {k: np.asarray(v, dtype=np.int64).reshape(-1) for k, v in CATS.items()}
        for k, v in CATS.items():
            assert v.shape[0] == T, f"cats['{k}'] e states com comprimentos diferentes."

    out: List[SequenceExample] = []
    for s, e in sliding_windows(T, K, stride=stride):
        cats_slice = {k: v[s:e] for k, v in CATS.items()} if CATS is not None else None
        abs_slice = AT[s:e] if AT is not None else None

        ex = build_window(
            states=S[s:e],
            actions=A[s:e],
            rewards=R[s:e],
            delta_t=DT[s:e],
            K=K,
            start_action_id=start_action_id,
            abs_time=abs_slice,
            flow_id=flow_id_scalar,
            cats=cats_slice,
        )
        out.append(ex)

    return out
