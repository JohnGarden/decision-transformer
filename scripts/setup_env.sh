#!/usr/bin/env bash
set -euo pipefail

python -m venv --prompt "im12-dt" .venv
source .venv/Scripts/activate

if [ -f pyproject.toml ]; then
    pip install -U pip
    pip install -e .
else
    pip install -U pip -r requirements.txt
fi

pip install pre-commit
pre-commit install

echo "Ambiente im12-dt criado. Ative com: source .venv/bin/activate"