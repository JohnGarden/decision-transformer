from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple
import numpy as np


@dataclass
class SequenceExample:
    """Estrutura padronizada de um exemplo janela-K para o DT.

    Todas as matrizes têm comprimento fixo K (com padding à esquerda) e dtype estável.
    """
    states: np.ndarray      # (K, d_state)  float32
    actions_in: np.ndarray  # (K,) int64 — ação t-1 (com START na primeira posição válida)
    actions_out: np.ndarray # (K,) int64 — rótulo/ação alvo em t
    rtg: np.ndarray         # (K,) float32 — return-to-go alinhado por passo
    delta_t: np.ndarray     # (K,) float32 — Δt entre eventos
    attn_mask: np.ndarray   # (K,) uint8  — 1 para tokens válidos, 0 para padding
    length: int             # T real da janela (<= K)


def right_pad_to_K(x: np.ndarray, K: int, pad_value: float = 0.0, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Alinha a sequência à direita (como GPT) e devolve (seq_padded, mask).
    - `mask[i]=0` indica padding; `mask[i]=1` indica token válido.
    """
    x = np.asarray(x)
    T = x.shape[axis]
    pad = [(0, 0)] * x.ndim
    pad_len = max(0, K - T)
    pad[axis] = (pad_len, 0)
    out = np.pad(x, pad, constant_values=pad_value)
    mask = np.zeros((K,), dtype=np.uint8)
    mask[pad_len:] = 1
    return out, mask


def compute_rtg(rewards: np.ndarray) -> np.ndarray:
    """Return-To-Go por posição: RTG_t = sum_{τ=t}^T r_τ (cumsum reverso)."""
    r = rewards.astype(np.float32)
    return np.cumsum(r[::-1])[::-1]




def sliding_windows(T: int, K: int) -> Iterable[Tuple[int, int]]:
    """Gera janelas [start, end) de tamanho ≤ K, alinhadas no fim.
    Ex.: T=7, K=4 → (0,3), (1,4), (2,5), (3,6), (4,7).
    """
    if T <= 0:
        return []
    for end in range(1, T + 1):
        start = max(0, end - K)
        yield (start, end)


def build_window(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    delta_t: np.ndarray,
    K: int,
    start_action_id: int,
) -> SequenceExample:
    """Monta uma janela alinhada à direita com START token para actions_in.

    - `actions_in[t] = START` para o primeiro passo válido da janela.
    - `actions_out[t] = ação-alvo` daquele passo.
    - `rtg` calculado sobre a parte válida e depois alinhado/pad.
    - `delta_t` é alinhado; padding recebe 0.
    """
    T = states.shape[0]
    assert T == actions.shape[0] == rewards.shape[0] == delta_t.shape[0]

    rtg_valid = compute_rtg(rewards)

    # actions_in é deslocada de 1 passo com START no primeiro
    actions_in_valid = np.roll(actions, shift=1)
    actions_in_valid[0] = start_action_id

    # padding/alinhamento
    states_pad, mask = right_pad_to_K(states, K, pad_value=0.0, axis=0)
    actions_in_pad, _ = right_pad_to_K(actions_in_valid, K, pad_value=start_action_id, axis=0)
    actions_out_pad, _ = right_pad_to_K(actions, K, pad_value=0, axis=0)
    rtg_pad, _ = right_pad_to_K(rtg_valid, K, pad_value=0.0, axis=0)
    dt_pad, _ = right_pad_to_K(delta_t, K, pad_value=0.0, axis=0)

    return SequenceExample(
        states=states_pad.astype(np.float32),
        actions_in=actions_in_pad.astype(np.int64),
        actions_out=actions_out_pad.astype(np.int64),
        rtg=rtg_pad.astype(np.float32),
        delta_t=dt_pad.astype(np.float32),
        attn_mask=mask.astype(np.uint8),
        length=int(mask.sum()),
    )


def build_trajectory_windows(
    traj: Dict[str, np.ndarray],
    K: int,
    start_action_id: int,
) -> List[SequenceExample]:
    """Recebe uma trajetória completa (por flow) e explode em janelas K.

    Espera chaves: 'states' (T,d), 'actions' (T,), 'rewards' (T,), 'delta_t' (T,).
    """
    S = traj["states"]
    A = traj["actions"]
    R = traj["rewards"]
    DT = traj["delta_t"]
    T = S.shape[0]

    out: List[SequenceExample] = []
    for s, e in sliding_windows(T, K):
        ex = build_window(S[s:e], A[s:e], R[s:e], DT[s:e], K, start_action_id)
        out.append(ex)
    return out
