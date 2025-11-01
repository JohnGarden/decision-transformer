#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data/raw/unsw-nb15/csv"
TRAIN="UNSW_NB15_training-set.csv"
TEST="UNSW_NB15_testing-set.csv"

# Garante que a pasta existe
mkdir -p "$DATA_DIR"

missing=()

# Verifica a existência dos arquivos esperados
[[ -f "$DATA_DIR/$TRAIN" ]] || missing+=("$TRAIN")
[[ -f "$DATA_DIR/$TEST"  ]] || missing+=("$TEST")

if (( ${#missing[@]} == 0 )); then
  echo "Datasets encontrados em '$DATA_DIR':"
  ls -lh "$DATA_DIR"/UNSW_NB15_* 2>/dev/null || true
  echo "Tudo configurado corretamente com os datasets."
  exit 0
fi

echo "Arquivos ausentes em '$DATA_DIR':"
for f in "${missing[@]}"; do
  echo "   - $f"
done
echo
echo "Por favor, copie os arquivos acima para '$DATA_DIR'."
echo "Dica: verifique nomes exatos (sensível a maiúsculas/minúsculas) e permissões."
exit 1
