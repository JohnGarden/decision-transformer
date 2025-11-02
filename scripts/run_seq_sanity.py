#!/usr/bin/env python
from __future__ import annotations
import re
from pathlib import Path
import yaml
from torch.utils.data import DataLoader
from im12dt.data.dataset_seq import UNSWSequenceDataset, SeqDatasetConfig, seq_collate

def _interpolate_templates(obj, ctx):
    """Substitui padrões ${a.b.c} por ctx['a']['b']['c'] recursivamente."""
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

# 1) Carregar YAML e interpolar ${...}
cfg_data = yaml.safe_load(Path("configs/data.yaml").read_text())
cfg_data = _interpolate_templates(cfg_data, cfg_data)

# 2) Construir caminhos e validar existência
data_root = Path(cfg_data["paths"]["data_root"])
train_csv = Path(cfg_data["paths"]["train_csv"])
test_csv  = Path(cfg_data["paths"]["test_csv"])

missing = [p for p in [data_root, train_csv, test_csv] if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Arquivos/caminhos não encontrados:\n  - "
        + "\n  - ".join(str(p) for p in missing)
        + "\nVerifique se você executa a partir da RAIZ do repo e se os CSVs estão em "
          f"{data_root} com os nomes: 'UNSW_NB15_training-set.csv' e 'UNSW_NB15_testing-set.csv'."
    )

# 3) Montar config do dataset já com caminhos resolvidos
conf = SeqDatasetConfig(
    csv_path=str(train_csv),
    flow_keys=cfg_data["processing"]["flow_keys"],
    time_col=cfg_data["processing"]["time_col"],
    context_length=cfg_data["sequence"]["context_length"],
    start_action=cfg_data["sequence"]["start_action"],
    pad_token=cfg_data["sequence"]["pad_token"],
    normalize=cfg_data["processing"]["normalize"],
    label_col=cfg_data["labels"]["label_col"],
    attack_cat_col=cfg_data["labels"]["attack_cat_col"],
)

# 4) Rodar sanity
ds = UNSWSequenceDataset(conf, max_rows=25_000)
dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=seq_collate)

print(f"Dataset: {len(ds)} janelas K={conf.context_length} | d_state inferido pela contagem de colunas numéricas")
batch = next(iter(dl))
for k, v in batch.items():
    if hasattr(v, "shape"):
        print(k, tuple(v.shape), v.dtype)

ai = batch["actions_in"][0].tolist()
ao = batch["actions_out"][0].tolist()
msk = batch["attn_mask"][0].tolist()
print("actions_in[0]:", ai)
print("actions_out[0]:", ao)
print("mask[0]:", msk)

has_padding = (batch["attn_mask"].sum(dim=1) < batch["attn_mask"].shape[1]).any().item()
has_positive_window = (batch["actions_out"].sum(dim=1) > 0).any().item()
print("has_padding_in_batch:", bool(has_padding))
print("has_positive_window_in_batch:", bool(has_positive_window))