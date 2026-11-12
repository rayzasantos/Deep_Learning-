"""Variações pré-especificadas do modelo da P1 (ON).

Todas usam os mesmos folds e a mesma logística (C escolhido por validação interna por
sujeito) e são comparadas com a referência (espectral por canal) pelo DeLong, com Holm
entre as variações:

  espectral_roi            potência relativa e expoente aperiódico em média por região
  espectral+periodico      referência + parâmetros periódicos do FOOOF, por canal
  espectral_roi+periodico  as duas ideias juntas, por região
  so_periodico_roi         só os parâmetros periódicos (informativo)
  conectividade_roi        wPLI em média por par de regiões
  stacking                 logística sobre as predições fora do fold interno de
                           espectral_roi+periodico e conectividade_roi

Parâmetros periódicos (espectro mediano das janelas, FOOOF 2-40 Hz): frequência e
potência do pico alfa (7-14 Hz), centro de gravidade 4-14 Hz e log(teta/alfa). O offset
aperiódico fica de fora porque sua escala absoluta depende do amplificador de cada centro.
Gravações sem pico alfa recebem a mediana do treino.

Cada variação também é avaliada com as features residualizadas nas métricas de QC.

Saídas em resultados/variacoes/: resumo.csv, periodico.parquet, pred__*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import welch

from . import estatistica as S
from . import modelos as M
from .config import APERIODICO_FAIXA, N_BOOTSTRAP, PREPROC, Caminhos
from .experimentos import dados_p1, folds_p1
from .features import canais_comuns, carregar, logs_ok, recortar_janelas
from .sensibilidade import COLUNAS_QC, metricas_qc

REGIOES = {
    "frontal": ["Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8"],
    "central": ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4"],
    "temporal": ["T7", "T8"],
    "parietal": ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "P4", "P8"],
    "occipital": ["O1", "Oz", "O2"],
}
PARAMETROS_PERIODICOS = ("apf", "alfa_pw", "cog", "teta_alfa")
PREFIXOS_ESPECTRAIS = [f"rel_{b}_" for b in ("delta", "teta", "alfa", "beta", "gama")] + ["exp_"]
FAIXA_ALFA_PICO = (7, 14)
FAIXA_CENTRO_GRAVIDADE = (4, 14)


# --------------------------------------------------------------------------- features
def _pico_alfa(freqs, espectro):
    """(frequência, potência) do maior pico do FOOOF na faixa alfa, ou (nan, nan)."""
    from fooof import FOOOF

    modelo = FOOOF(aperiodic_mode="fixed", peak_width_limits=(1, 12), max_n_peaks=6, verbose=False)
    try:
        modelo.fit(freqs, espectro, list(APERIODICO_FAIXA))
        picos = np.atleast_2d(modelo.get_params("peak_params"))
    except Exception:
        return np.nan, np.nan
    if not picos.size or np.isnan(picos).all():
        return np.nan, np.nan
    alfa = picos[(picos[:, 0] >= FAIXA_ALFA_PICO[0]) & (picos[:, 0] <= FAIXA_ALFA_PICO[1])]
    if not len(alfa):
        return np.nan, np.nan
    frequencia, potencia = alfa[np.argmax(alfa[:, 1]), :2]
    return frequencia, potencia


def _parametros_periodicos(linha, caminhos, comuns):
    import mne

    raw = mne.io.read_raw_fif(caminhos.preproc / linha["arquivo_saida"], preload=True, verbose="ERROR")
    sfreq = raw.info["sfreq"]
    janelas = recortar_janelas(raw.pick(comuns), linha["janelas_inicio_s"], PREPROC.janela_s)
    nperseg = int(2 * sfreq)
    freqs, psd = welch(janelas, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    psd = np.median(psd, axis=0)
    faixa_cog = (freqs >= FAIXA_CENTRO_GRAVIDADE[0]) & (freqs <= FAIXA_CENTRO_GRAVIDADE[1])
    teta = psd[:, (freqs >= 4) & (freqs < 8)].sum(1)
    alfa = psd[:, (freqs >= 8) & (freqs < 13)].sum(1)
    saida = {"rec_id": linha["rec_id"]}
    for i, canal in enumerate(comuns):
        frequencia, potencia = _pico_alfa(freqs, psd[i])
        saida.update({
            f"apf_{canal}": frequencia,
            f"alfa_pw_{canal}": potencia,
            f"cog_{canal}": (freqs[faixa_cog] * psd[i, faixa_cog]).sum() / psd[i, faixa_cog].sum(),
            f"teta_alfa_{canal}": np.log10(teta[i] / alfa[i]),
        })
    return saida


def parametros_periodicos(caminhos, logs, comuns, n_jobs):
    arquivo = caminhos.resultados / "variacoes" / "periodico.parquet"
    if arquivo.exists():
        return pd.read_parquet(arquivo)
    linhas = Parallel(n_jobs=n_jobs)(delayed(_parametros_periodicos)(r, caminhos, comuns)
                                     for r in logs.to_dict("records"))
    tabela = pd.DataFrame(linhas)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_parquet(arquivo, index=False)
    return tabela


def media_por_regiao(tabela: pd.DataFrame, prefixos, comuns) -> pd.DataFrame:
    """Média, ignorando ausentes, de cada feature por canal dentro de cada região."""
    saida = {}
    for prefixo in prefixos:
        for regiao, canais in REGIOES.items():
            colunas = [f"{prefixo}{c}" for c in canais if c in comuns and f"{prefixo}{c}" in tabela]
            if colunas:
                saida[f"{prefixo}{regiao}"] = tabela[colunas].mean(axis=1, skipna=True)
    return pd.DataFrame(saida, index=tabela.index)


def wpli_por_regiao(tabela: pd.DataFrame) -> pd.DataFrame:
    """wPLI médio de cada par de regiões (inclusive dentro da mesma região)."""
    regiao_do_canal = {c: r for r, canais in REGIOES.items() for c in canais}
    saida = {}
    for banda in ("teta", "alfa", "beta"):
        prefixo = f"wpli_{banda}_"
        pares = {}
        for coluna in (c for c in tabela.columns if c.startswith(prefixo)):
            canal_a, canal_b = coluna[len(prefixo):].split("_")
            par = "-".join(sorted((regiao_do_canal[canal_a], regiao_do_canal[canal_b])))
            pares.setdefault(par, []).append(coluna)
        saida.update({f"{prefixo}{par}": tabela[colunas].mean(axis=1) for par, colunas in pares.items()})
    return pd.DataFrame(saida, index=tabela.index)


# --------------------------------------------------------------------------- avaliação
def _conjuntos_de_features(dados, comuns):
    matriz = lambda t: np.asarray(t, float)
    espectral = M.colunas(dados, "espectral")
    periodico = [f"{p}_{c}" for p in PARAMETROS_PERIODICOS for c in comuns]
    espectral_roi = media_por_regiao(dados, PREFIXOS_ESPECTRAIS, comuns)
    periodico_roi = media_por_regiao(dados, [f"{p}_" for p in PARAMETROS_PERIODICOS], comuns)
    conectividade_roi = wpli_por_regiao(dados)
    espectral_e_periodico_roi = pd.concat([espectral_roi, periodico_roi], axis=1)
    return {
        "referencia": [matriz(dados[espectral])],
        "espectral_roi": [matriz(espectral_roi)],
        "espectral+periodico": [matriz(dados[espectral + periodico])],
        "espectral_roi+periodico": [matriz(espectral_e_periodico_roi)],
        "so_periodico_roi": [matriz(periodico_roi)],
        "conectividade_roi": [matriz(conectividade_roi)],
        "stacking": [matriz(espectral_e_periodico_roi), matriz(conectividade_roi)],
    }


def _resumir(predicoes):
    linhas = []
    for (protocolo, nome, residualizado), pred in predicoes.items():
        linha = {"protocolo": protocolo, "variacao": nome, "resid_qc": residualizado, "n": len(pred),
                 **S.resumo(pred["y"], pred["prob"], N_BOOTSTRAP)}
        if nome != "referencia":
            ref = predicoes[(protocolo, "referencia", residualizado)].set_index("sujeito")
            alinhado = pred.set_index("sujeito").loc[ref.index]
            auc_ref, auc, p = S.delong(ref["y"], ref["prob"], alinhado["prob"])
            linha.update(dif_auc_vs_ref=auc - auc_ref, p_delong=p)
        for centro, grupo in pred.groupby("dataset"):
            linha[f"auc_{centro}"] = S.metricas(grupo["y"], grupo["prob"])["auc"]
        linhas.append(linha)
    resumo = pd.DataFrame(linhas)
    resumo["p_holm"] = np.nan
    for _, grupo in resumo[resumo["p_delong"].notna()].groupby(["protocolo", "resid_qc"]):
        resumo.loc[grupo.index, "p_holm"] = S.holm(grupo["p_delong"])
    return resumo.sort_values(["resid_qc", "protocolo", "auc"], ascending=[True, True, False])


def rodar(caminhos: Caminhos, n_jobs: int = 1):
    pasta = caminhos.resultados / "variacoes"
    pasta.mkdir(parents=True, exist_ok=True)
    logs = logs_ok(caminhos)
    comuns = canais_comuns(logs)
    gravacoes = (carregar(caminhos).merge(metricas_qc(logs), on="rec_id")
                 .merge(parametros_periodicos(caminhos, logs, comuns, n_jobs), on="rec_id"))
    dados = dados_p1(gravacoes, "ON", False)
    folds = folds_p1(caminhos, dados, "ON", False)
    conjuntos = _conjuntos_de_features(dados, comuns)
    print({nome: [b.shape[1] for b in blocos] for nome, blocos in conjuntos.items()})

    Q = dados[COLUNAS_QC].to_numpy(float)
    tarefas = [(protocolo, nome, residualizado) for protocolo in folds for nome in conjuntos
               for residualizado in (False, True)]
    resultados = Parallel(n_jobs=n_jobs)(
        delayed(M.prever_fora_do_fold)(dados, folds[protocolo], conjuntos[nome], Q if residualizado else None)
        for protocolo, nome, residualizado in tarefas)
    predicoes = dict(zip(tarefas, resultados))
    for (protocolo, nome, residualizado), pred in predicoes.items():
        pred.to_csv(pasta / f"pred__{protocolo}__{nome}{'__resid_qc' if residualizado else ''}.csv", index=False)

    resumo = _resumir(predicoes)
    resumo.to_csv(pasta / "resumo.csv", index=False)
    colunas = ["protocolo", "variacao", "resid_qc", "auc", "auc_ic", "acuracia_balanceada", "dif_auc_vs_ref",
               "p_delong", "p_holm", "auc_ds002778", "auc_ds003490", "auc_ds004584"]
    print(resumo[colunas].round(3).to_string(index=False))
    return resumo
