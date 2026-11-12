"""Figuras do artigo, a partir dos CSVs de resultados. Saída em resultados/figuras/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DATASETS, MARGEM_ERRO_DELTA, REFERENCIA_PRINCIPAL, Caminhos

CORES = {"ds002778": "#2a6f97", "ds003490": "#c4622d", "ds004584": "#5a8f3a", "todos": "#333333"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def _ic(s):
    a, b = s.strip("[]").split(",")
    return float(a), float(b)


def curva_roc(y, p):
    o = np.argsort(-np.asarray(p))
    y = np.asarray(y)[o]
    tpr = np.r_[0, np.cumsum(y) / y.sum()]
    fpr = np.r_[0, np.cumsum(1 - y) / (1 - y).sum()]
    return fpr, tpr


def gerar(caminhos: Caminhos, med="ON"):
    out = caminhos.resultados / "figuras"
    out.mkdir(parents=True, exist_ok=True)
    p1 = caminhos.resultados / f"p1_{med}"

    # 1. ROC do modelo de referência nos protocolos B e C
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for prot, est in (("B_sujeito", "-"), ("C_centro", "--")):
        f = p1 / f"pred__{prot}__espectral__logistica__semharm.csv"
        if f.exists():
            pr = pd.read_csv(f)
            fpr, tpr = curva_roc(pr["y"], pr["prob"])
            ax.plot(fpr, tpr, est, color="#222", label={"B_sujeito": "por sujeito (B)", "C_centro": "centro de fora (C)"}[prot])
    ax.plot([0, 1], [0, 1], ":", color="#999")
    ax.set(xlabel="1 - especificidade", ylabel="sensibilidade")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_roc_referencia.png")
    plt.close(fig)

    # 2. AUC por centro (protocolo C) para os modelos principais
    f = p1 / "resumo_por_centro.csv"
    if f.exists():
        c = pd.read_csv(f)
        c = c[(c["protocolo"] == "C_centro") & (~c["harmonizacao"])]
        modelos = c.groupby(["features", "modelo"])["auc"].mean().sort_values().index
        fig, ax = plt.subplots(figsize=(5, 0.35 * len(modelos) + 1))
        for k, (feat, mod) in enumerate(modelos):
            for j, (ds, g) in enumerate(c[(c["features"] == feat) & (c["modelo"] == mod)].groupby("centro")):
                lo, hi = _ic(g["auc_ic"].iat[0])
                y = k + (j - 1) * 0.22
                ax.plot([lo, hi], [y, y], color=CORES.get(ds, "k"), lw=1.2)
                ax.plot(g["auc"].iat[0], y, "o", color=CORES.get(ds, "k"), ms=3.5,
                        label=DATASETS[ds].centro if k == 0 else None)
        ax.axvline(0.5, ls=":", color="#999")
        ax.set_yticks(range(len(modelos)), [f"{m} ({f})" for f, m in modelos])
        ax.set_xlabel("AUC no centro deixado de fora (IC 95%)")
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        fig.savefig(out / "fig_auc_por_centro.png")
        plt.close(fig)

    # 3. duração: RMSE do delta ON - OFF por duração, com a margem
    f = caminhos.resultados / "pareado" / "duracao_sorteios.csv"
    if f.exists():
        s = pd.read_csv(f)
        s = s[(s["canal"] == "C3C4") & (s["ref"] == REFERENCIA_PRINCIPAL)]
        fig, ax = plt.subplots(figsize=(3.6, 3))
        for ds, g in s.groupby("dataset"):
            ax.errorbar(g["duracao_s"], g["rmse"], yerr=[g["rmse"] - g["rmse_ic_inf"], g["rmse_ic_sup"] - g["rmse"]],
                        fmt="o-", color=CORES[ds], ms=3.5, capsize=2, label=DATASETS[ds].centro)
        ax.axhline(MARGEM_ERRO_DELTA, ls="--", color="#999", label="margem")
        ax.set(xlabel="segundos de sinal limpo por sessão", ylabel="RMSE do Δ ON − OFF (log10)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_duracao_rmse.png")
        plt.close(fig)

    # 4. beta ao longo da gravação, ON e OFF
    f = caminhos.resultados / "pareado" / "blocos_resumo.csv"
    if f.exists():
        b = pd.read_csv(f)
        b = b[b["ref"] == REFERENCIA_PRINCIPAL]
        dss = sorted(b["dataset"].unique())
        fig, axs = plt.subplots(1, len(dss), figsize=(3.2 * len(dss), 2.8), squeeze=False)
        for ax, ds in zip(axs[0], dss):
            g = b[b["dataset"] == ds]
            ax.plot(g["inicio_s"], g["ON"], "o-", color="#2a6f97", ms=3, label="ON")
            ax.plot(g["inicio_s"], g["OFF"], "s-", color="#c4622d", ms=3, label="OFF")
            ax.set(title=DATASETS[ds].centro, xlabel="início do bloco (s)", ylabel="beta C3/C4 (log10)")
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_beta_blocos.png")
        plt.close(fig)
    print("figuras em", out)
