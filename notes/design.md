## Design — IM12-DT+


- Sequencialização por flow, janela K, tokens [RTG_t, s_t, a_{t-1}] + e_time(Δt_t), máscara causal.
- Ação space: {benign, malicious, wait}. Recompensa com pesos cTP..cWAIT.
- Métricas: PR-AUC, F1, TTR (mediana/P90).
- Semana 2: DPE para reduzir variância de off-policy; behavior policy MLE.


```
flowchart LR
A[UNSW-NB15 CSVs] --> B[Pré-processamento\n(normalização, mapeamento de rótulos)]
B --> C[Reconstrução de flows\n(keys, ordenação temporal)]
C --> D[Trajetórias por flow\n(janelas de contexto K)]
D --> E[Tokens: \nRTG_t, s_t, a_{t-1}]
E --> F[Embedding temporal contínuo\nφ(Δt_t)]
F --> G[Máscara causal]
G --> H[Decision Transformer]
H --> I[Política alvo \n(π_θ(a_t|·))]
I --> J[Métricas: PR-AUC, F1, TTR, Recompensa]
```