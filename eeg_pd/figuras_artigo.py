"""Figuras e tabelas finais do artigo. São descritivas: não mudam nenhum resultado.

  fig_roc_b_c_externo.png   ROC do modelo principal: 10-fold por sujeito (B), deixando
                            um centro de fora (C) e no centro externo
  fig_calibracao.png        calibração por quantis da probabilidade prevista
  coeficientes_roi.csv      coeficientes padronizados da logística por região (ajuste
                            final com todos os sujeitos da P1 ON) e IC por bootstrap de
                            sujeitos; fig_coeficientes_roi.png
  coeficientes_canal.csv    coeficientes do modelo principal (por canal)

Saídas em resultados/figuras/ e resultados/tabelas/.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from . import estatistica as S
from . import modelos as M
from .config import N_FOLDS_INTERNOS, SEMENTE, Caminhos
from .experimentos import dados_p1
from .features import carregar
from .figuras import curva_roc
from .variacoes import PREFIXOS_ESPECTRAIS, REGIOES, media_por_regiao

N_BOOTSTRAP_COEFICIENTES = 500
CORES = ["#2a6f97", "#c4622d", "#5a8f3a"]
ROTULOS_LINHAS = ["delta", "teta", "alfa", "beta", "gama", "expoente"]
ESTABILIDADE_MINIMA = 0.95


def _predicoes(caminhos, caminhos_externos=None):
    pasta = caminhos.resultados / "p1_ON"
    conjuntos = {}
    for protocolo, rotulo in (("B_sujeito", "B: 10-fold por sujeito"), ("C_centro", "C: um centro de fora")):
        pred = pd.read_csv(pasta / f"pred__{protocolo}__espectral__logistica__semharm.csv")
        conjuntos[rotulo] = (pred["y"].to_numpy(), pred["prob"].to_numpy())
    if caminhos_externos is not None:
        arquivo = caminhos_externos.resultados / "externo" / "predicoes.csv"
        if arquivo.exists():
            externo = pd.read_csv(arquivo)
            conjuntos["Externo: Tel Aviv"] = (externo["y"].to_numpy(), externo["espectral (principal)"].to_numpy())
    return conjuntos


def _figura_roc(conjuntos, arquivo):
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for (rotulo, (y, prob)), cor in zip(conjuntos.items(), CORES):
        fpr, tpr = curva_roc(y, prob)
        ax.plot(fpr, tpr, color=cor, lw=1.6, label=f"{rotulo} (AUC {S.metricas(y, prob)['auc']:.2f}; n={len(y)})")
    ax.plot([0, 1], [0, 1], ":", color="#999999", lw=1)
    ax.set(xlabel="1 − especificidade", ylabel="sensibilidade", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=300)
    plt.close(fig)


def _pontos_de_calibracao(y, prob, n_quantis):
    limites = np.quantile(prob, np.linspace(0, 1, n_quantis + 1))
    quantil = np.clip(np.searchsorted(limites, prob, side="right") - 1, 0, n_quantis - 1)
    return np.array([[prob[quantil == k].mean(), y[quantil == k].mean()]
                     for k in range(n_quantis) if (quantil == k).any()])


def _figura_calibracao(conjuntos, arquivo):
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    linhas = []
    for (rotulo, (y, prob)), cor in zip(conjuntos.items(), CORES):
        pontos = _pontos_de_calibracao(y, prob, 5 if len(y) < 150 else 10)
        ax.plot(pontos[:, 0], pontos[:, 1], "o-", color=cor, ms=3.5, lw=1.2, label=rotulo)
        linhas.append({"conjunto": rotulo, "n": len(y), "prevalencia_pd": y.mean(), "prob_media": prob.mean(),
                       "brier": np.mean((prob - y) ** 2)})
    ax.plot([0, 1], [0, 1], ":", color="#999999", lw=1)
    ax.set(xlabel="probabilidade prevista de PD", ylabel="fração observada de PD", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=300)
    plt.close(fig)
    return pd.DataFrame(linhas)


def _ajuste_final(X, y, grupos):
    estimador, grade = M.modelo("logistica")
    cv = StratifiedGroupKFold(n_splits=N_FOLDS_INTERNOS, shuffle=True, random_state=SEMENTE)
    busca = GridSearchCV(estimador, grade, cv=cv, scoring="roc_auc").fit(X, y, groups=grupos)
    return busca.best_estimator_, busca.best_params_["logisticregression__C"]


def _bootstrap_de_coeficientes(X, y, C):
    """Coeficientes refeitos em reamostras de sujeitos, estratificadas por grupo, com C fixo."""
    rng = np.random.default_rng(SEMENTE)
    pacientes, controles = np.where(y == 1)[0], np.where(y == 0)[0]
    amostras = []
    for _ in range(N_BOOTSTRAP_COEFICIENTES):
        i = np.r_[rng.choice(pacientes, len(pacientes)), rng.choice(controles, len(controles))]
        estimador, _ = M.modelo("logistica")
        estimador.set_params(logisticregression__C=C).fit(X[i], y[i])
        amostras.append(estimador[-1].coef_[0])
    return np.array(amostras)


def _coeficientes_por_regiao(dados, comuns):
    regioes = media_por_regiao(dados, PREFIXOS_ESPECTRAIS, comuns)
    X, y, grupos = regioes.to_numpy(float), dados["y"].to_numpy(), dados["sujeito"].to_numpy()
    estimador, C = _ajuste_final(X, y, grupos)
    coeficientes = estimador[-1].coef_[0]
    amostras = _bootstrap_de_coeficientes(X, y, C)
    return pd.DataFrame({"feature": regioes.columns, "coef_padronizado": coeficientes,
                         "ic_inf": np.percentile(amostras, 2.5, 0), "ic_sup": np.percentile(amostras, 97.5, 0),
                         "frac_mesmo_sinal": (np.sign(amostras) == np.sign(coeficientes)).mean(0), "C": C})


def _figura_coeficientes(tabela, arquivo):
    por_feature = tabela.set_index("feature")
    nomes = [[f"{prefixo}{regiao}" for regiao in REGIOES] for prefixo in PREFIXOS_ESPECTRAIS]
    valores = np.array([[por_feature.loc[n, "coef_padronizado"] for n in linha] for linha in nomes])
    limite = np.nanmax(np.abs(valores))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    imagem = ax.imshow(valores, cmap="RdBu_r", vmin=-limite, vmax=limite)
    ax.set(xticks=range(len(REGIOES)), xticklabels=list(REGIOES),
           yticks=range(len(ROTULOS_LINHAS)), yticklabels=ROTULOS_LINHAS)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for (i, j), valor in np.ndenumerate(valores):
        estavel = por_feature.loc[nomes[i][j], "frac_mesmo_sinal"] >= ESTABILIDADE_MINIMA
        ax.text(j, i, f"{valor:.2f}" + ("*" if estavel else ""), ha="center", va="center", fontsize=6.5)
    fig.colorbar(imagem, ax=ax, shrink=0.8, label="coeficiente (positivo = PD)")
    ax.set_title("* sinal estável em ≥95% dos bootstraps", fontsize=7)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=300)
    plt.close(fig)


def gerar(caminhos: Caminhos, caminhos_externos: Caminhos | None = None):
    pasta_figuras, pasta_tabelas = caminhos.resultados / "figuras", caminhos.resultados / "tabelas"
    pasta_figuras.mkdir(parents=True, exist_ok=True)
    pasta_tabelas.mkdir(parents=True, exist_ok=True)

    conjuntos = _predicoes(caminhos, caminhos_externos)
    _figura_roc(conjuntos, pasta_figuras / "fig_roc_b_c_externo.png")
    calibracao = _figura_calibracao(conjuntos, pasta_figuras / "fig_calibracao.png")
    calibracao.to_csv(pasta_tabelas / "calibracao.csv", index=False)

    dados = dados_p1(carregar(caminhos), "ON", False)
    comuns = json.loads((caminhos.features / "canais_comuns.json").read_text())
    espectral = M.colunas(dados, "espectral")
    estimador, C = _ajuste_final(dados[espectral].to_numpy(float), dados["y"].to_numpy(), dados["sujeito"].to_numpy())
    (pd.DataFrame({"feature": espectral, "coef_padronizado": estimador[-1].coef_[0], "C": C})
     .sort_values("coef_padronizado", key=abs, ascending=False)
     .to_csv(pasta_tabelas / "coeficientes_canal.csv", index=False))

    coeficientes = _coeficientes_por_regiao(dados, comuns)
    coeficientes.to_csv(pasta_tabelas / "coeficientes_roi.csv", index=False)
    _figura_coeficientes(coeficientes, pasta_figuras / "fig_coeficientes_roi.png")

    print(calibracao.round(3).to_string(index=False))
    print(coeficientes.sort_values("coef_padronizado", key=abs, ascending=False).round(3).to_string(index=False))
