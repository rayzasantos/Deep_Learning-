# EEG de repouso no Parkinson

Pipeline de análise de EEG de repouso na doença de Parkinson, com dados públicos de quatro centros.
O código cobre três perguntas:

- **P1**: o EEG de repouso separa pacientes com Parkinson de controles? A avaliação é feita por
  sujeito, deixando um centro inteiro de fora e num quarto centro usado só como teste externo.
- **P2**: o EEG separa as sessões com e sem levodopa (ON e OFF) do mesmo paciente?
- **Beta pareado**: a potência beta em C3/C4 muda entre ON e OFF, e quanto sinal é preciso para
  estimar essa diferença com confiabilidade?

## Dados

Todos os datasets são do [OpenNeuro](https://openneuro.org), no formato BIDS, e são baixados pela
própria pipeline.

| Dataset | Centro | Versão | Uso |
| --- | --- | --- | --- |
| [ds002778](https://openneuro.org/datasets/ds002778) | San Diego | 1.0.5 | treino |
| [ds003490](https://openneuro.org/datasets/ds003490) | Novo México | 1.1.0 | treino |
| [ds004584](https://openneuro.org/datasets/ds004584) | Iowa | 1.0.0 | treino |
| [ds007526](https://openneuro.org/datasets/ds007526) | Tel Aviv | 1.0.2 | teste externo |

O centro de teste externo fica numa pasta separada (`dados_externo/`) e não participa de nenhuma
decisão da pipeline: o modelo é treinado uma vez com os três centros e aplicado uma vez a ele.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

As versões em `requirements.txt` são as usadas nos resultados do artigo.

## Execução

A forma mais simples é o notebook `pipeline.ipynb`, que executa todas as etapas em ordem. Pela linha
de comando, primeiro baixe os dados e depois rode a pipeline completa:

```bash
python -m eeg_pd.cli baixar
python -m eeg_pd.cli verificar
python -m eeg_pd.cli baixar --datasets ds007526 --raiz dados_externo
python rodar_tudo.py
```

Cada etapa também pode ser executada isoladamente com `python -m eeg_pd.cli <etapa>`:

| Etapa | O que faz |
| --- | --- |
| `baixar`, `verificar`, `inspecionar` | download, conferência de integridade e resumo dos datasets |
| `preprocessar` | pré-processamento de cada gravação, com log JSON |
| `features` | potência relativa, expoente aperiódico, wPLI, beta em C3/C4 e covariâncias |
| `fluxo` | fluxograma de participantes e motivo de cada exclusão |
| `controles` | demografia, modelo só com idade, classificador de centro e permutação |
| `p1`, `profundos`, `comparar` | P1 com modelos clássicos, GCN e redes; comparação pelo DeLong |
| `auditoria` | confere que nenhum sujeito aparece no treino e no teste do mesmo fold |
| `p2` | P2 com p por permutação e correção de Holm |
| `pareado`, `blocos` | contraste ON − OFF do beta, efeito da duração e deriva ao longo da gravação |
| `sensibilidade` | confusão por QC do pré-processamento, sexo, banda gama e quantidade de sinal |
| `variacoes` | variações do modelo definidas antes da análise (regiões, pico alfa, stacking) |
| `externo` | teste no centro externo |
| `figuras`, `artigo` | figuras e tabelas |

O pré-processamento é a etapa mais longa (várias horas para as 270 gravações dos três centros); as
demais etapas levam cerca de 1 hora e meia num computador de 12 núcleos. ShallowNet, EEGNet e EEGConformer precisam de GPU e podem ser executadas
no Google Colab com `pipeline_colab.ipynb`.

## Pré-processamento

1. Conferência de integridade: duração lida contra a do JSON do BIDS e, em BDF/EDF, cabeçalho contra
   o tamanho do arquivo.
2. Leitura do arquivo, com os eventos tirados do `events.tsv`.
3. Só canais com nome do sistema 10-05 viram EEG; os canais de EOG são renomeados e o restante sai
   antes de qualquer referência.
4. Trechos de repouso de olhos abertos, com corte das bordas e das emendas.
5. Duas cópias, reamostradas para 250 Hz: 1-45 Hz (sinal analisado) e notch + 1-100 Hz (ICA).
6. Canais ruins por z robusto do desvio padrão e pela maior correlação com outro canal.
7. Referência média sem os canais ruins.
8. ICA infomax estendido na gravação inteira e classificação dos componentes pelo ICLabel (limiar
   0,8). Componentes de piscada abaixo do limiar são removidos pela correlação com o EOG.
9. Remoção dos componentes, interpolação dos canais ruins e nova referência média.
10. Janelas de 4 s dentro do repouso, sem emenda e com pico a pico abaixo de 200 µV.

Todos os parâmetros ficam em `eeg_pd/config.py`. Cada log guarda um hash da configuração, e o
`preprocessar` refaz só as gravações cuja configuração mudou.

## Controle de vazamento

- Os folds dividem por sujeito; nenhum sujeito aparece no treino e no teste do mesmo fold.
- Hiperparâmetros, padronização, harmonização e residualização são ajustados só no treino de cada fold.
- A etapa `auditoria` confere as predições salvas contra os arquivos de folds e interrompe a execução
  se encontrar sobreposição.
- Os dados de treino recusam qualquer gravação do centro externo.

## Estrutura

```
eeg_pd/
  config.py             parâmetros, datasets e caminhos
  dados.py              download, integridade e listagem das gravações
  preprocessamento.py   pré-processamento de uma gravação
  features.py           features por gravação
  modelos.py            modelos, harmonização, folds e previsão fora do fold
  estatistica.py        métricas, bootstrap, DeLong, Holm e testes pareados
  experimentos.py       P1, controles, P2, comparação, auditoria e fluxograma
  profundos.py          GCN e redes neurais
  pareado.py            análise pareada do beta
  sensibilidade.py      análises de sensibilidade
  variacoes.py          variações pré-especificadas do modelo
  externo.py            teste no centro externo
  figuras.py            figuras das análises
  figuras_artigo.py     figuras e tabelas finais
  cli.py                linha de comando
testes/                 testes unitários e dados simulados
rodar_tudo.py           pipeline completa
pipeline.ipynb          pipeline completa em notebook
pipeline_colab.ipynb    redes neurais em GPU
```

Os resultados ficam em `dados/resultados/` e, para o centro externo, em `dados_externo/resultados/`.

## Testes

```bash
python -m pytest testes/teste_unidades.py
python testes/rodar_teste.py        # pipeline inteira em dados simulados
```

## Licença

[MIT](LICENSE).
