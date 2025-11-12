#!/usr/bin/env python
from __future__ import annotations

import re
import os
from datetime import datetime
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from im12dt.data.dataset_seq import (
    UNSWSequenceDataset,
    SeqDatasetConfig,
    seq_collate,
)
from im12dt.training.trainer_dt import (
    DTTrainer,
    TrainerConfig,
    make_weighted_sampler,
)

# ----------------- helpers -----------------

def _interpolate_templates(obj, ctx):
    if isinstance(obj, dict):
        return {k: _interpolate_templates(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_templates(v, ctx) for v in obj]
    if isinstance(obj, str):
        def repl(m):
            path = m.group(1).split(".")
            cur = ctx
            for key in path:
                cur = cur[key]
            return str(cur)
        return re.sub(r"\$\{([^}]+)\}", repl, obj)
    return obj


def main():
    # --------- load configs ---------
    cfg_data = yaml.safe_load(Path("configs/data.yaml").read_text())
    cfg_data = _interpolate_templates(cfg_data, cfg_data)
    cfg_model = yaml.safe_load(Path("configs/model_dt.yaml").read_text())
    cfg_trn   = yaml.safe_load(Path("configs/trainer.yaml").read_text())

    # Acessos convenientes com defaults
    trn = cfg_trn.get("training", {})
    inf = cfg_trn.get("inference", {})
    opt_old = cfg_trn.get("optimizer", {})       # legado
    reward_old = cfg_trn.get("reward", {})       # legado

    # --------- datasets ---------
    conf_train = SeqDatasetConfig(
        csv_path=str(Path(cfg_data["paths"]["train_csv"])),
        flow_keys=cfg_data["processing"]["flow_keys"],
        time_col=cfg_data["processing"]["time_col"],
        context_length=cfg_data["sequence"]["context_length"],
        start_action=cfg_data["sequence"]["start_action"],
        pad_token=cfg_data["sequence"]["pad_token"],
        normalize=cfg_data["processing"]["normalize"],
        label_col=cfg_data["labels"]["label_col"],
        attack_cat_col=cfg_data["labels"]["attack_cat_col"],
        categorical_cols=cfg_model["categorical"]["cols"],
    )
    conf_val = SeqDatasetConfig(
        **{**conf_train.__dict__, "csv_path": str(Path(cfg_data["paths"]["test_csv"]))}
    )

    train_ds = UNSWSequenceDataset(conf_train, max_rows=None)
    # importante: usar estatísticas do *train* no *val* (sem vazamento)
    val_ds   = UNSWSequenceDataset(
        conf_val,
        max_rows=None,
        stats_override=getattr(train_ds, "_stats", None),
        cat_maps_override=getattr(train_ds, "_cat_maps", None),
    )

    # --- Sampler (retro-compatível) ---
    sampler_cfg = cfg_trn.get("sampler", {})
    use_weighted = bool(sampler_cfg.get("use_weighted_windows", True))

    sampler = None
    if use_weighted:
        # novo nome → fallback para antigo
        pos_w = sampler_cfg.get("window_pos_weight", sampler_cfg.get("pos_weight", 4.0))
        sampler = make_weighted_sampler(train_ds, pos_weight=float(pos_w))

    train_dl = DataLoader(
        train_ds,
        batch_size=int(trn.get("batch_size", 128)),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=True,
        collate_fn=seq_collate,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=int(trn.get("batch_size", 128)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=seq_collate,
    )

    # --------- hiperparâmetros (retro-compat: training.* ou optimizer.* / reward.*) ---------
    lr = float(trn.get("lr", opt_old.get("lr", 3.0e-4)))
    betas = tuple(trn.get("betas", opt_old.get("betas", [0.9, 0.999])))
    weight_decay = float(trn.get("weight_decay", opt_old.get("weight_decay", 0.01)))
    label_smoothing = float(trn.get("label_smoothing", cfg_model.get("loss", {}).get("label_smoothing", 0.0)))

    # class_weights pode vir de training.* (preferido) ou model.loss.* (legado)
    class_weights = trn.get("class_weights", cfg_model.get("loss", {}).get("class_weights", [1.0, 1.0]))
    class_weights = tuple(float(x) for x in class_weights)

    wait_threshold = float(trn.get("wait_threshold", inf.get("wait_threshold", 0.55)))

    reward_weights = trn.get("reward_weights", None)
    if reward_weights is None:
        # fallback para bloco reward legado
        reward_weights = {
            "cTP": float(reward_old.get("cTP", 1.0)),
            "cTN": float(reward_old.get("cTN", 0.10)),
            "cFP": float(reward_old.get("cFP", -0.10)),
            "cFN": float(reward_old.get("cFN", -1.0)),
            "cWAIT": float(reward_old.get("cWAIT", 0.0)),
        }

    # --------- trainer ---------
    trainer = DTTrainer(
        ds_train=train_ds,
        ds_val=val_ds,
        cfg=TrainerConfig(
            batch_size=int(trn.get("batch_size", 128)),
            max_epochs=int(trn.get("max_epochs", 10)),
            steps_per_epoch=trn.get("steps_per_epoch", None),
            grad_clip=float(trn.get("grad_clip", 1.0)),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            wait_threshold=wait_threshold,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            reward_weights={k: float(v) for k, v in reward_weights.items()},
        ),
        model_cfg=cfg_model,
        cat_cfg=cfg_model["categorical"],
    )

    # --------- train ---------
    trainer.fit(train_dl, val_dl, cat_cols=cfg_model["categorical"]["cols"])

    # --------- save checkpoint (modelo + tokenizers + stats + cat_maps + cfg) ---------
    os.makedirs("artifacts", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_path = Path("artifacts") / f"dt_baseline_{ts}.pt"

    ckpt = {
        "model": trainer.model.state_dict(),
        "state_tok": trainer.state_tok.state_dict(),
        "action_tok": trainer.action_tok.state_dict(),
        "rtg_tok": trainer.rtg_tok.state_dict(),
        "time_tok": trainer.time_tok.state_dict(),
        "norm_stats": getattr(train_ds, "_stats", None),       # mean/std usados na normalização
        "cat_maps": getattr(train_ds, "_cat_maps", None),      # stoi das colunas categóricas (treino)
        "cfg": {"model": cfg_model, "trainer": cfg_trn, "data": cfg_data},
    }
    torch.save(ckpt, ckpt_path)
    print(f"[CKPT] saved to: {ckpt_path.resolve()}")


if __name__ == "__main__":
    # Info de device (útil para logs reprodutíveis)
    print(f"[DEVICE] torch={torch.__version__} | cuda_available={torch.cuda.is_available()} | cuda_build={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"[DEVICE] gpu={torch.cuda.get_device_name(0)} | capability={torch.cuda.get_device_capability(0)}")
    main()
