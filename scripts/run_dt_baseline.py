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
    val_ds   = UNSWSequenceDataset(conf_val,   max_rows=None, stats_override=train_ds._stats)

    # --------- sampler / dataloaders ---------
    sampler = make_weighted_sampler(train_ds, pos_weight=cfg_trn["sampler"]["pos_weight"])

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg_trn["training"]["batch_size"],
        sampler=sampler,
        num_workers=0,                 # Windows-friendly
        pin_memory=True,
        collate_fn=seq_collate,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg_trn["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=seq_collate,
    )

    # --------- trainer ---------
    trainer = DTTrainer(
        ds_train=train_ds,
        ds_val=val_ds,
        cfg=TrainerConfig(
            batch_size=cfg_trn["training"]["batch_size"],
            max_epochs=cfg_trn["training"]["max_epochs"],
            steps_per_epoch=cfg_trn["training"]["steps_per_epoch"],
            grad_clip=cfg_trn["training"]["grad_clip"],
            lr=cfg_trn["optimizer"]["lr"],
            betas=tuple(cfg_trn["optimizer"]["betas"]),
            weight_decay=cfg_trn["optimizer"]["weight_decay"],
            wait_threshold=cfg_trn["inference"]["wait_threshold"],
            class_weights=tuple(cfg_model["loss"]["class_weights"]),
            label_smoothing=cfg_model["loss"]["label_smoothing"],
            reward_weights={k: float(v) for k, v in cfg_trn["reward"].items()},
            # opcional: ablação sem RTG na validação
            # eval_no_rtg=cfg_trn.get("inference", {}).get("eval_no_rtg", False),
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
        "norm_stats": train_ds._stats,                       # mean/std usados na normalização
        "cat_maps": getattr(train_ds, "_cat_maps", None),    # stoi das colunas categóricas (treino)
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
