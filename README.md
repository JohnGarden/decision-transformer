# Decision Transformer

Reprodução do artigo _Real-time Network Intrusion Detection via Importance Sampled Decision Transformers_

- DOI: 10.1109/MASS62177.2024.00022

## 1. Reprodutibilidade e Extensões

Reprodução do método para detecção de intrusões em tempo real com Decision Transformers, incluindo ação **wait**, **embedding temporal contínuo** e **DPE**.


### Setup rápido

```bash
#Lembre-se de instalar o cuda, caso você tenha GPU NVIDIA.
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# 1 criar e ativar ambiente
bash scripts/setup_env.sh
source .venv/Scripts/activate

# 2 preparar diretório de dados e colocar CSVs do UNSW-NB15
bash scripts/prepare_unsw.sh
# (copie os arquivos para data/raw/unsw-nb15/csv)

# 3 sanity check de configs e logging
bash scripts/run_sanity.sh




