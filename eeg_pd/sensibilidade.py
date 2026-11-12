"""Análises de sensibilidade da P1 (ON).

Todas usam o modelo de referência (logística, sem harmonização) e os mesmos folds da P1:

  so_qc                  só as métricas do pré-processamento (componentes removidos por
                         tipo, canais ruins, sinal limpo). Se elas separam os grupos, o
                         pré-processamento difere entre eles e pode ser parte do que o
                         modelo aprende
  so_sexo, idade_sexo    confusão demográfica
  espectral_sem_gama     sem a banda de 30-45 Hz, a mais exposta a músculo
  espectral_resid_qc     features espectrais sem a parte explicada pelas métricas de QC
                         (regressão ajustada só no treino de cada fold)
  *_12_janelas           features recalculadas com as 12 primeiras janelas válidas (48 s)
                         de cada gravação: a mesma quantidade de sinal em todos os centros

Saídas em resultados/sensibilidade/: resumo.csv, qc_por_grupo.csv, gravacoes_12_janelas.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu

from . import estatistica as S
from . import modelos as M
from .config import N_BOOTSTRAP, Caminhos
from .experimentos import dados_p1, folds_p1
from .features import canais_comuns, carregar, features_gravacao, logs_ok

N_JANELAS_FIXO = 12  # 48 s: o mínimo do ds003490 (45 s) garante 12 janelas
COLUNAS_QC = ["qc_removidos", "qc_musculo", "qc_piscada", "qc_outros", "qc_canais_ruins", "qc_sinal_limpo_s"]
REFERENCIA = "espectral (referência)"


def metricas_qc(logs: pd.DataFrame) -> pd.DataFrame:
    """Métricas de qualidade do pré-processamento, uma linha por gravação."""
    por_tipo = logs["removidos_por_tipo"].apply(lambda d: d if isinstance(d, dict) else {})
    qc = pd.DataFrame({
        "rec_id": logs["rec_id"],
        "qc_removidos": logs["componentes_removidos"].astype(float),
        "qc_musculo": por_tipo.apply(lambda d: d.get("muscle artifact", 0)).astype(float),
        "qc_piscada": por_tipo.apply(lambda d: sum(v for k, v in d.items() if k.startswith("eye"))).astype(float),
        "qc_canais_ruins": logs["canais_ruins"].apply(len).astype(float),
        "qc_sinal_limpo_s": logs["duracao_limpa_s"].astype(float),
    })
    qc["qc_outros"] = qc["qc_removidos"] - qc["qc_musculo"] - qc["qc_piscada"]
    return qc


def comparar_qc_entre_grupos(dados: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney PD vs. HC de cada métrica de QC, dentro de cada centro, com Holm."""
    linhas = []
    for centro, grupo in dados.groupby("dataset"):
        for metrica in COLUNAS_QC:
            pd_, hc = grupo.loc[grupo["y"] == 1, metrica], grupo.loc[grupo["y"] == 0, metrica]
            varia = pd_.nunique() + hc.nunique() > 2
            linhas.append({"dataset": centro, "metrica": metrica, "media_pd": pd_.mean(), "media_hc": hc.mean(),
                           "p_mann_whitney": mannwhitneyu(pd_, hc).pvalue if varia else np.nan})
    tabela = pd.DataFrame(linhas)
    tabela["p_holm"] = S.holm(tabela["p_mann_whitney"].fillna(1))
    return tabela


def _features_com_12_janelas(caminhos, logs, comuns, n_jobs):
    arquivo = caminhos.resultados / "sensibilidade" / "gravacoes_12_janelas.parquet"
    if arquivo.exists():
        return pd.read_parquet(arquivo)
    resultados = Parallel(n_jobs=n_jobs)(delayed(features_gravacao)(r, caminhos, comuns, N_JANELAS_FIXO, True)
                                         for r in logs.to_dict("records"))
    tabela = pd.concat([logs[["rec_id"]].reset_index(drop=True), pd.DataFrame([r["feats"] for r in resultados])],
                       axis=1)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_parquet(arquivo, index=False)
    return tabela


def _resumir(protocolo, nome, predicoes, referencia=None):
    linha = {"protocolo": protocolo, "analise": nome, "n": len(predicoes),
             **S.resumo(predicoes["y"], predicoes["prob"], N_BOOTSTRAP)}
    if referencia is not None:
        ref = referencia.set_index("sujeito")
        alinhado = predicoes.set_index("sujeito").loc[ref.index]
        auc_ref, auc, p = S.delong(ref["y"], ref["prob"], alinhado["prob"])
        linha.update(dif_auc_vs_ref=auc - auc_ref, p_delong=p)
    return linha


def rodar(caminhos: Caminhos, n_jobs: int = 1):
    pasta = caminhos.resultados / "sensibilidade"
    pasta.mkdir(parents=True, exist_ok=True)
    logs = logs_ok(caminhos)
    gravacoes = carregar(caminhos).merge(metricas_qc(logs), on="rec_id")
    dados = dados_p1(gravacoes, "ON", False)
    folds = folds_p1(caminhos, dados, "ON", False)
    dados["sexo_f"] = dados["sexo"].astype(str).str.strip().str.upper().str.startswith("F").astype(float)
    dados.loc[dados["sexo"].isna(), "sexo_f"] = np.nan

    qc = comparar_qc_entre_grupos(dados)
    qc.to_csv(pasta / "qc_por_grupo.csv", index=False)
    print(qc.round(3).to_string(index=False))

    doze = dados[["rec_id"]].merge(_features_com_12_janelas(caminhos, logs, canais_comuns(logs), n_jobs),
                                   on="rec_id", how="left")
    assert doze["rec_id"].tolist() == dados["rec_id"].tolist()

    espectral = M.colunas(dados, "espectral")
    sem_gama = [c for c in espectral if not c.startswith("rel_gama_")]
    conectividade = M.colunas(dados, "conectividade")
    Q = dados[COLUNAS_QC].to_numpy(float)
    matriz = lambda tabela, colunas: tabela[colunas].to_numpy(float)
    # nome: (features, covariáveis removidas, análise de referência para o DeLong)
    analises = {
        REFERENCIA: (matriz(dados, espectral), None, None),
        "so_qc": (Q, None, None),
        "espectral_sem_gama": (matriz(dados, sem_gama), None, REFERENCIA),
        "espectral_resid_qc": (matriz(dados, espectral), Q, REFERENCIA),
        "espectral_12_janelas": (matriz(doze, espectral), None, REFERENCIA),
        "conectividade": (matriz(dados, conectividade), None, None),
        "conectividade_12_janelas": (matriz(doze, conectividade), None, "conectividade"),
    }
    linhas = []
    for protocolo, folds_protocolo in folds.items():
        predicoes = {}
        for nome, (X, covariaveis, referencia) in analises.items():
            predicoes[nome] = M.prever_fora_do_fold(dados, folds_protocolo, [X], covariaveis)
            predicoes[nome].to_csv(pasta / f"pred__{protocolo}__{nome.split(' ')[0]}.csv", index=False)
            linha = _resumir(protocolo, nome, predicoes[nome], predicoes.get(referencia))
            for centro, grupo in predicoes[nome].groupby("dataset"):
                linha[f"auc_{centro}"] = S.metricas(grupo["y"], grupo["prob"])["auc"]
            linhas.append(linha)
        for nome, colunas in (("so_sexo", ["sexo_f"]), ("idade_sexo", ["idade", "sexo_f"])):
            com_dado = dados.dropna(subset=colunas).reset_index(drop=True)
            pred = M.prever_fora_do_fold(com_dado, folds_protocolo, [matriz(com_dado, colunas)])
            linhas.append(_resumir(protocolo, nome, pred))

    resumo = pd.DataFrame(linhas)
    resumo.to_csv(pasta / "resumo.csv", index=False)
    colunas = ["protocolo", "analise", "n", "auc", "auc_ic", "acuracia_balanceada", "dif_auc_vs_ref", "p_delong",
               "auc_ds002778", "auc_ds003490", "auc_ds004584"]
    print(resumo[[c for c in colunas if c in resumo]].round(3).to_string(index=False))
    return resumo
