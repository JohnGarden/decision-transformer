#!/usr/bin/env python
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from im12dt.data.dataset_seq import UNSWSequenceDataset, SeqDatasetConfig, seq_collate
from im12dt.models.tokens import StateTokenizer, ActionTokenizer, RTGTokenizer, CatSpec, _rule_embed_dim
from im12dt.models.temporal_embed import TimeEncodingFourier
from im12dt.models.model_dt import DecisionTransformer

from sklearn.metrics import average_precision_score, f1_score


# --------------------------- utils ---------------------------

def _interpolate_templates(obj, ctx):
    if isinstance(obj, dict):
        return {k: _interpolate_templates(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_templates(v, ctx) for v in obj]
    if isinstance(obj, str):
        def repl(m):
            path = m.group(1).split(".")
            cur = ctx
            for key in path: cur = cur[key]
            return str(cur)
        return re.sub(r"\$\{([^}]+)\}", repl, obj)
    return obj


def build_cat_specs(ds, cat_cols, rule, fixed_dim):
    specs = []
    for c in cat_cols:
        if hasattr(ds, "_cat_maps") and c in ds._cat_maps:
            n_tokens = len(ds._cat_maps[c])
            d = _rule_embed_dim(n_tokens, rule=rule, fixed=fixed_dim)
            specs.append(CatSpec(c, n_tokens, d))
    return specs


def _latest_artifact(art_dir: Path) -> Optional[Path]:
    if not art_dir.exists():
        return None
    pts = sorted(art_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0] if pts else None


def _quantize_time(x: float, scale: float = 1e6) -> int:
    """Quantiza (em microssegundos) para deduplicar steps em any-step."""
    return int(round(x * scale))


# --------------------------- core eval ---------------------------

@torch.no_grad()
def run_eval(
    model_cfg: dict,
    trainer_cfg: dict,
    data_cfg: dict,
    ckpt_path: Optional[str | Path] = None,
    max_rows_val: Optional[int] = None,
    grid_wait: Iterable[float] = (0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80),
    debug_missed: bool = False,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- checkpoint ----------
    if ckpt_path is None:
        ckpt_path = trainer_cfg.get('inference', {}).get('checkpoint', 'artifacts/dt_day4.pt')

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        latest = _latest_artifact(Path("artifacts"))
        if latest is None:
            raise FileNotFoundError(f"Checkpoint '{ckpt_path}' não encontrado e não há .pt em 'artifacts/'.")
        print(f"[INFO] Checkpoint '{ckpt_path}' não encontrado. Usando mais recente: {latest}")
        ckpt_path = latest

    print(f"Carregando checkpoint: {ckpt_path}")

    # PyTorch 2.6+: habilitar safe globals para antigos pickles
    try:
        from numpy.core.multiarray import _reconstruct as _np_reconstruct
        import numpy as _np
        torch.serialization.add_safe_globals([_np_reconstruct, _np.dtype, _np.ndarray])
    except Exception:
        pass

    ckpt = None
    try:
        # Precisamos de objetos não-tensor (stats, maps) → weights_only=False
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        # fallback final (não recomendado, mas útil na prática)
        ckpt = torch.load(ckpt_path, map_location=device)

    norm_stats = ckpt.get('norm_stats', None)
    cat_maps   = ckpt.get('cat_maps', None)
    if norm_stats is not None:
        m = np.asarray(norm_stats.get("mean", []))
        s = np.asarray(norm_stats.get("std", []))
        print(f"[INFO] norm_stats carregados: mean/std shape = {m.shape} / {s.shape}")
    if cat_maps is not None:
        print(f"[INFO] cat_maps carregados: cols = {list(cat_maps.keys())}")

    # ---------- dataset ----------
    conf_val = SeqDatasetConfig(
        csv_path=str(Path(data_cfg['paths']['test_csv'])),
        flow_keys=data_cfg['processing']['flow_keys'],
        time_col=data_cfg['processing']['time_col'],
        context_length=data_cfg['sequence']['context_length'],
        start_action=data_cfg['sequence']['start_action'],
        pad_token=data_cfg['sequence']['pad_token'],
        normalize=data_cfg['processing']['normalize'],
        label_col=data_cfg['labels']['label_col'],
        attack_cat_col=data_cfg['labels']['attack_cat_col'],
        categorical_cols=model_cfg['categorical']['cols'],
    )

    # Tente passar overrides se o dataset aceitar
    try:
        ds_val = UNSWSequenceDataset(
            conf_val,
            max_rows=max_rows_val,
            stats_override=norm_stats,
            cat_maps_override=cat_maps,
        )
    except TypeError:
        print("[WARN] Seu UNSWSequenceDataset não aceita stats_override/cat_maps_override. Usando instância padrão.")
        ds_val = UNSWSequenceDataset(conf_val, max_rows=max_rows_val)

    dl_val = DataLoader(
        ds_val,
        batch_size=trainer_cfg['training']['batch_size'],
        shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=seq_collate
    )

    # ---------- tokenizers & model ----------
    Dnum  = ds_val.examples[0].states.shape[1]
    specs = build_cat_specs(ds_val, model_cfg['categorical']['cols'],
                            model_cfg['categorical']['embed_rule'],
                            model_cfg['categorical'].get('fixed_dim', 16))

    state_tok = StateTokenizer(Dnum, specs, model_cfg['embeddings']['state_embed_dim']).to(device)
    action_tok = ActionTokenizer(4, model_cfg['embeddings']['state_embed_dim']).to(device)
    rtg_tok = RTGTokenizer(model_cfg['embeddings']['rtg_dim']).to(device)
    time_tok = TimeEncodingFourier(model_cfg['embeddings']['time_dim'], n_freq=16, use_log1p=True).to(device)

    model = DecisionTransformer(
        d_model=model_cfg['d_model'],
        n_layers=model_cfg['n_layers'],
        n_heads=model_cfg['n_heads'],
        d_ff=model_cfg['d_ff'],
        dropout=model_cfg.get('Dropout', 0.1),
        n_actions=model_cfg['vocab']['n_actions'],
    ).to(device)
    model.ensure_projections(
        model_cfg['embeddings']['state_embed_dim'],
        model_cfg['embeddings']['state_embed_dim'],
        model_cfg['embeddings']['rtg_dim'],
        model_cfg['embeddings']['time_dim'],
        device,
    )
    # carregar pesos (se existirem)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    if 'state_tok' in ckpt:
        state_tok.load_state_dict(ckpt['state_tok'], strict=False)
    if 'action_tok' in ckpt:
        action_tok.load_state_dict(ckpt['action_tok'], strict=False)
    if 'rtg_tok' in ckpt:
        rtg_tok.load_state_dict(ckpt['rtg_tok'], strict=False)
    if 'time_tok' in ckpt:
        time_tok.load_state_dict(ckpt['time_tok'], strict=False)
    model.eval()

    # ---------- agregadores ----------
    all_y: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    all_m: List[np.ndarray] = []

    flows_attack_t: Dict[int, float] = {}           # fid -> min t(y==1)
    flows_last_stats: Dict[int, List[Tuple[float,int,float]]] = {}  # fid -> [(t_last, a_last, c_last), ...]
    flows_any_stats: Dict[int, List[Tuple[float,int,float]]]  = {}  # fid -> [(t, a, c), ...] (deduplicado por tempo)
    flows_any_seen: Dict[int, set] = {}             # fid -> {quantized_time,...}

    for batch in dl_val:
        # device
        for k, v in list(batch.items()):
            batch[k] = v.to(device) if torch.is_tensor(v) else v

        cats = {c: batch.get(f'cat_{c}') for c in model_cfg['categorical']['cols'] if f'cat_{c}' in batch}

        # forward
        s_emb = state_tok(batch['states'].float(), cats)
        a_emb = action_tok(batch['actions_in'])
        r_emb = rtg_tok(batch['rtg'].float())
        t_emb = time_tok(batch['delta_t'].float())
        logits = model(s_emb, a_emb, r_emb, t_emb, batch['attn_mask'])
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        probs1 = torch.softmax(logits, dim=-1)[..., 1]

        # token-level
        all_y.append(batch['actions_out'].reshape(-1).detach().cpu().numpy())
        all_p.append(probs1.reshape(-1).detach().cpu().numpy())
        all_m.append(batch['attn_mask'].reshape(-1).detach().cpu().numpy())

        # TTR (por fluxo)
        B, K, C = logits.shape
        lengths = batch['length']              # (B,)
        abs_t   = batch['abs_time']            # (B,K)
        y       = batch['actions_out']         # (B,K)
        fid     = batch['flow_id']             # (B,)

        max_conf, argmax = torch.softmax(logits, dim=-1).max(dim=-1)  # (B,K)

        for i in range(B):
            f = int(fid[i].item())
            L = int(lengths[i].item())
            # mapa de ataque (min tempo onde y==1)
            y_i = y[i, :L]
            t_i = abs_t[i, :L]
            pos_mask = (y_i == 1)
            if pos_mask.any():
                t_pos_min = float(t_i[pos_mask].min().item())
                if f not in flows_attack_t:
                    flows_attack_t[f] = t_pos_min
                else:
                    flows_attack_t[f] = min(flows_attack_t[f], t_pos_min)

            # last-step
            j_last = L - 1
            t_last = float(abs_t[i, j_last].item())
            a_last = int(argmax[i, j_last].item())
            c_last = float(max_conf[i, j_last].item())
            flows_last_stats.setdefault(f, []).append((t_last, a_last, c_last))

            # any-step (deduplicado por tempo)
            seen = flows_any_seen.setdefault(f, set())
            lst  = flows_any_stats.setdefault(f, [])
            for j in range(L):
                t_step = float(abs_t[i, j].item())
                a_step = int(argmax[i, j].item())
                c_step = float(max_conf[i, j].item())
                q = _quantize_time(t_step)
                if q not in seen:
                    seen.add(q)
                    lst.append((t_step, a_step, c_step))

    # ---------- métricas token-level ----------
    y_np = np.concatenate(all_y) if all_y else np.array([])
    p_np = np.concatenate(all_p) if all_p else np.array([])
    m_np = (np.concatenate(all_m) > 0.5) if all_m else np.array([], dtype=bool)
    y_np = y_np[m_np]; p_np = np.clip(np.nan_to_num(p_np[m_np], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    pr_auc = 0.0 if (y_np.size==0 or y_np.max()==0) else float(average_precision_score(y_np, p_np))
    f1 = float(f1_score(y_np, (p_np>=0.5).astype(int), zero_division=0))
    print(f"Token-level  PR-AUC={pr_auc:.4f}  F1={f1:.4f}")

    # ---------- TTR helpers ----------
    def summarize_ttr(stats: Dict[int, List[Tuple[float,int,float]]], thr: float) -> Tuple[int,int,float,float,float,float,float]:
        """stats: fid -> [(t, a_pred, conf), ...] (ordenado internamente aqui)"""
        malicious_flows = len(flows_attack_t)
        detected = 0
        ttrs: List[float] = []
        for f, t_attack in flows_attack_t.items():
            seq = sorted(stats.get(f, []))  # ordena por t
            t_detect = None
            for (t, a, c) in seq:
                if (a == 1) and (c >= thr):
                    t_detect = t
                    break
            if t_detect is not None:
                detected += 1
                ttr = max(0.0, float(t_detect - t_attack))
                ttrs.append(ttr)
        rate = detected / max(malicious_flows, 1)
        if len(ttrs) > 0:
            p50 = float(np.percentile(ttrs, 50))
            p90 = float(np.percentile(ttrs, 90))
            avg = float(np.mean(ttrs))
            worst = float(np.max(ttrs))
        else:
            p50 = p90 = avg = worst = float('nan')
        return malicious_flows, detected, rate, p50, p90, avg, worst

    # ---------- prints ----------
    grid_wait = list(grid_wait) if grid_wait else [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]

    print("\nTTR summary (last-step):")
    print("thr\tflows+\tdetected\trate\tTTR_P50\tTTR_P90\tTTR_avg\tTTR_max")
    for thr in grid_wait:
        mf, d, rate, p50, p90, avg, worst = summarize_ttr(flows_last_stats, thr)
        print(f"{thr:.2f}\t{mf}\t{d}\t{rate:.3f}\t{p50:.3f}\t{p90:.3f}\t{avg:.3f}\t{worst:.3f}")

    print("\nTTR summary (any-step):")
    print("thr\tflows+\tdetected\trate\tTTR_P50\tTTR_P90\tTTR_avg\tTTR_max")
    for thr in grid_wait:
        mf, d, rate, p50, p90, avg, worst = summarize_ttr(flows_any_stats, thr)
        print(f"{thr:.2f}\t{mf}\t{d}\t{rate:.3f}\t{p50:.3f}\t{p90:.3f}\t{avg:.3f}\t{worst:.3f}")

    # ---------- debug: fluxos não detectados ----------
    if debug_missed:
        print("\n[DEBUG] Missed flows (amostra):")
        missed = []
        for f, t_attack in flows_attack_t.items():
            seq_last = sorted(flows_last_stats.get(f, []))
            seq_any  = sorted(flows_any_stats.get(f, []))
            # max confs
            max_last = max([c for (_, a, c) in seq_last], default=0.0)
            max_any  = max([c for (_, a, c) in seq_any],  default=0.0)
            # detectado a 0.55?
            det_last = any((a==1 and c>=0.55) for (_, a, c) in seq_last)
            det_any  = any((a==1 and c>=0.55) for (_, a, c) in seq_any)
            if not det_last:
                missed.append((f, max_last, max_any))
        missed.sort(key=lambda x: x[2], reverse=True)
        for f, ml, ma in missed[:10]:
            print(f"flow {f}: max_conf_last={ml:.3f}  max_conf_any={ma:.3f}")


# --------------------------- main ---------------------------

if __name__ == "__main__":
    # carregar configs
    cfg_data = _interpolate_templates(
        __import__("yaml").safe_load(Path('configs/data.yaml').read_text()),
        __import__("yaml").safe_load(Path('configs/data.yaml').read_text())
    )
    cfg_model = __import__("yaml").safe_load(Path('configs/model_dt.yaml').read_text())
    cfg_trn   = __import__("yaml").safe_load(Path('configs/trainer.yaml').read_text())

    # thresholds: pega da config se houver
    grid = cfg_trn.get('inference', {}).get('grid_wait', [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80])

    run_eval(
        model_cfg=cfg_model,
        trainer_cfg=cfg_trn,
        data_cfg=cfg_data,
        ckpt_path=cfg_trn.get('inference', {}).get('checkpoint', None),
        max_rows_val=None,
        grid_wait=grid,
        debug_missed=True,  # mude para True se quiser amostras dos perdidos
    )
