# Parkinson com EEG de repouso e Transformers

Notebook que classifica sinais de EEG de repouso de pessoas com doença de Parkinson e de controles usando um modelo Transformer em PyTorch.

## Dados

Dataset público ds002778 do OpenNeuro, baixado direto no notebook com `openneuro` e lido com `mne-bids`.

## Etapas

1. **Pré-processamento:** filtro notch para o ruído da rede elétrica, filtro passa-banda e remoção de artefatos.
2. **Normalização e segmentação:** os sinais limpos são normalizados e cortados em janelas.
3. **Modelo:** separação em treino e teste e classificador baseado em Transformer em PyTorch.
4. **Validação:** métricas de classificação nos conjuntos de treino e teste.
5. **Interpretação:** atribuições com Captum e mapas topográficos para ver quais canais pesam na decisão do modelo.

## Arquivo

- `trabalho_2_predict.ipynb`: pipeline completo, do download dos dados à interpretação

## Bibliotecas

MNE, mne-bids, PyTorch, scikit-learn, Captum, NumPy, pandas, matplotlib e seaborn.
