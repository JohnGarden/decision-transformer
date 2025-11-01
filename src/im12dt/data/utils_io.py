from __future__ import annotations
from pathlib import Path
import pandas as pd


def read_unsw_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV não encontrado: {p}")
    df = pd.read_csv(p)
    return df