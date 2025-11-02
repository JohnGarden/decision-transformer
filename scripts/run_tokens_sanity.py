#!/usr/bin/env python
from __future__ import annotations
import re
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from im12dt.data.dataset_seq import UNSWSequenceDataset, SeqDatasetConfig, seq_collate
from im12dt.models.tokens import StateTokenizer, ActionTokenizer, RTGTokenizer, CatSpec, _rule_embed_dim
from im12dt.models.temporal_embed import TimeEncodingFourier

# -------- helpers --------
import yaml, re

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

cfg_data = yaml.safe_load(Path("configs/data.yaml").read_text())
cfg_data = _interpolate_templates(cfg_data, cfg_data)
cfg_model = yaml.safe_load(Path("configs/model_dt.yaml").read_text())

conf = SeqDatasetConfig(
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

ds = UNSWSequenceDataset(conf, max_rows=50_000)
dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, collate_fn=seq_collate)

batch = next(iter(dl))
B, K, Dnum = batch["states"].shape

# specs categóricas a partir dos mapas do dataset
cat_specs = []
for col in cfg_model["categorical"]["cols"]:
    if f"cat_{col}" in batch:
        n_tokens = max(int(ds._cat_maps[col].__len__()), 1)
        d = _rule_embed_dim(n_tokens, rule=cfg_model["categorical"]["embed_rule"], fixed=cfg_model["categorical"].get("fixed_dim", 16))
        cat_specs.append(CatSpec(col, n_tokens, d))

state_tok = StateTokenizer(
    numeric_dim=Dnum,
    cat_specs=cat_specs,
    state_embed_dim=cfg_model["embeddings"]["state_embed_dim"],
)
action_tok = ActionTokenizer(n_actions_plus_start=4, embed_dim=cfg_model["embeddings"]["state_embed_dim"])
rtg_tok = RTGTokenizer(embed_dim=cfg_model["embeddings"]["rtg_dim"])
time_tok = TimeEncodingFourier(d_model=cfg_model["embeddings"]["time_dim"], n_freq=16, use_log1p=True)

# preparar dict de cats
cats = {c: batch[f"cat_{c}"] for c in cfg_model["categorical"]["cols"] if f"cat_{c}" in batch}

with torch.no_grad():
    s_emb = state_tok(batch["states"].float(), cats)    # (B,K,Estate)
    a_emb = action_tok(batch["actions_in"])            # (B,K,Estate)
    r_emb = rtg_tok(batch["rtg"].float())              # (B,K,Ertg)
    t_emb = time_tok(batch["delta_t"].float())         # (B,K,Etime)

print("states_emb:", tuple(s_emb.shape))
print("actions_emb:", tuple(a_emb.shape))
print("rtg_emb:", tuple(r_emb.shape))
print("time_emb:", tuple(t_emb.shape))