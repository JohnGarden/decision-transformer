#!/usr/bin/env python
from __future__ import annotations
import re
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from im12dt.data.dataset_seq import UNSWSequenceDataset, SeqDatasetConfig, seq_collate
from im12dt.training.trainer_dt import DTTrainer, TrainerConfig, make_weighted_sampler

# ------------- helpers -------------

def _interpolate_templates(obj, ctx):
    import re
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

cfg_data = yaml.safe_load(Path('configs/data.yaml').read_text()); cfg_data = _interpolate_templates(cfg_data, cfg_data)
cfg_model = yaml.safe_load(Path('configs/model_dt.yaml').read_text())
cfg_trn   = yaml.safe_load(Path('configs/trainer.yaml').read_text())

# Datasets
conf_train = SeqDatasetConfig(
    csv_path=str(Path(cfg_data['paths']['train_csv'])),
    flow_keys=cfg_data['processing']['flow_keys'],
    time_col=cfg_data['processing']['time_col'],
    context_length=cfg_data['sequence']['context_length'],
    start_action=cfg_data['sequence']['start_action'],
    pad_token=cfg_data['sequence']['pad_token'],
    normalize=cfg_data['processing']['normalize'],
    label_col=cfg_data['labels']['label_col'],
    attack_cat_col=cfg_data['labels']['attack_cat_col'],
    categorical_cols=cfg_model['categorical']['cols'],
)
conf_val = conf_train.__class__(**{**conf_train.__dict__, 'csv_path': str(Path(cfg_data['paths']['test_csv']))})

train_ds = UNSWSequenceDataset(conf_train, max_rows=None)
val_ds   = UNSWSequenceDataset(conf_val,   max_rows=50_000)

sampler = make_weighted_sampler(train_ds, pos_weight=cfg_trn['sampler']['pos_weight'])

train_dl = DataLoader(train_ds, batch_size=cfg_trn['training']['batch_size'], sampler=sampler, num_workers=0, collate_fn=seq_collate)
val_dl   = DataLoader(val_ds,   batch_size=cfg_trn['training']['batch_size'], shuffle=False, num_workers=0, collate_fn=seq_collate)

# Trainer
trainer = DTTrainer(
    ds_train=train_ds,
    ds_val=val_ds,
    cfg=TrainerConfig(
        batch_size=cfg_trn['training']['batch_size'],
        max_epochs=cfg_trn['training']['max_epochs'],
        steps_per_epoch=cfg_trn['training']['steps_per_epoch'],
        grad_clip=cfg_trn['training']['grad_clip'],
        lr=cfg_trn['optimizer']['lr'],
        betas=tuple(cfg_trn['optimizer']['betas']),
        weight_decay=cfg_trn['optimizer']['weight_decay'],
        wait_threshold=cfg_trn['inference']['wait_threshold'],
        class_weights=tuple(cfg_model['loss']['class_weights']),
        label_smoothing=cfg_model['loss']['label_smoothing'],
        reward_weights={k: float(v) for k, v in cfg_trn['reward'].items()},
    ),
    model_cfg=cfg_model,
    cat_cfg=cfg_model['categorical'],
)

trainer.fit(train_dl, val_dl, cat_cols=cfg_model['categorical']['cols'])