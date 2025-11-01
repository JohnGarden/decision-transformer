#!/usr/bin/env bash
set -euo pipefail

# 1) Descobrir o interpretador da venv (Linux/macOS vs Windows)
if [[ -x ".venv/bin/python" ]]; then
  PYBIN=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PYBIN=".venv/Scripts/python.exe"
else
  echo "Não encontrei a venv. Crie/instale com: bash scripts/setup_env.sh"
  exit 1
fi

echo "Usando Python da venv: $("$PYBIN" -c 'import sys; print(sys.executable)')"

# 2) Garantir que o pacote do projeto esteja importável
if ! "$PYBIN" -c "import im12dt" 2>/dev/null; then
  echo "ℹPacote 'im12dt' não importável. Instalando em modo editável…"
  if [[ -f "pyproject.toml" ]]; then
    "$PYBIN" -m pip install --upgrade pip
    "$PYBIN" -m pip install -e .
  else
    # Fallback para layout sem pyproject: usa PYTHONPATH
    export PYTHONPATH="src:${PYTHONPATH:-}"
    echo "Sem pyproject.toml. Ajustei PYTHONPATH=src (execução local)."
  fi

  # Rechecar import
  "$PYBIN" -c "import im12dt" >/dev/null
  echo "Pacote 'im12dt' OK."
fi

# 3) Executar o sanity inline com o Python da venv
"$PYBIN" - << 'PY'
from pathlib import Path
from im12dt.utils.seed import set_seed
from im12dt.utils.logging import get_logger
import yaml

set_seed(42)
log = get_logger(__name__)

cfg_data  = yaml.safe_load(Path('configs/data.yaml').read_text())
cfg_model = yaml.safe_load(Path('configs/model_dt.yaml').read_text())
cfg_trn   = yaml.safe_load(Path('configs/trainer.yaml').read_text())

log.info("Sanity check OK.")
log.info(f"data_root={cfg_data['paths']['data_root']}")
log.info(f"d_model={cfg_model['d_model']}, layers={cfg_model['n_layers']}")
log.info(f"batch_size={cfg_trn['training']['batch_size']}")
PY
