"""Teste externo num 4º centro (Tel Aviv, ds007526), que não participa de nenhuma
decisão da pipeline.

Plano fixado antes de ver qualquer resultado do centro externo:
  * pré-processamento idêntico (mesmo código e configuração; mudam só os campos do
    próprio dataset: rede de 50 Hz, nomes do EOG e o canal de referência VREF = Cz);
  * features nos mesmos canais comuns dos 3 centros, na mesma ordem;
  * modelo principal: logística sobre as features espectrais por canal, sem
    harmonização, com C escolhido por validação interna por sujeito, treinado uma
    vez com todos os sujeitos da P1 ON e aplicado uma vez ao centro externo;
  * secundário: o mesmo modelo com as features espectrais por região;
  * controles: idade, métricas de QC por grupo e correlação da probabilidade
    prevista com as escalas clínicas dos pacientes.

O centro externo fica numa raiz própria (--raiz-externa), separada dos dados de treino.
Saídas em <raiz-externa>/resultados/externo/.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2, mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression

from . import estatistica as S
from . import modelos as M
from .config import EXTERNOS, N_BOOTSTRAP, SEMENTE, Caminhos
from .experimentos import dados_p1
from .features import META, carregar, features_gravacao, logs_ok
from .sensibilidade import COLUNAS_QC, metricas_qc
from .variacoes import PREFIXOS_ESPECTRAIS, media_por_regiao

N_PERMUTACOES = 10000
MODELO_PRINCIPAL = "espectral (principal)"
VARIAVEIS_CLINICAS = ("updrs_part_iii", "disease_duration", "ledd", "moca")


def _features_externas(caminhos_externos, comuns, n_jobs):
    arquivo = caminhos_externos.features / "gravacoes.parquet"
    logs = logs_ok(caminhos_externos)
    logs = logs[logs["dataset"].isin(EXTERNOS)].reset_index(drop=True)
    if arquivo.exists():
        tabela = pd.read_parquet(arquivo)
        if sorted(tabela["rec_id"]) == sorted(logs["rec_id"]):
            return tabela, logs
    faltam = [c for c in comuns if any(c not in finais for finais in logs["canais_finais"])]
    assert not faltam, f"canais comuns ausentes no centro externo: {faltam}"
    resultados = Parallel(n_jobs=n_jobs)(delayed(features_gravacao)(r, caminhos_externos, comuns, None, True)
                                         for r in logs.to_dict("records"))
    tabela = pd.concat([logs[[c for c in META if c in logs]].reset_index(drop=True),
                        pd.DataFrame([r["feats"] for r in resultados])], axis=1)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_parquet(arquivo, index=False)
    return tabela, logs


def _amostra_externa(tabela):
    """Controles e pacientes em ON, fora as gravações excluídas da análise principal."""
    amostra = tabela[tabela["principal"]]
    amostra = amostra[(amostra["grupo"] == "HC") | (amostra["med"] == "ON")].reset_index(drop=True)
    amostra["y"] = (amostra["grupo"] == "PD").astype(int)
    return amostra


def _log_verossimilhanca(X, y):
    modelo = LogisticRegression(penalty=None, max_iter=5000).fit(X, y)
    p = np.clip(modelo.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def p_alem_da_idade(prob, idade, y):
    """Teste da razão de verossimilhança: o logit da probabilidade prevista acrescenta
    informação a um modelo que já tem a idade?"""
    ok = ~np.isnan(idade)
    logit = np.log(np.clip(prob, 1e-6, 1 - 1e-6) / np.clip(1 - prob, 1e-6, 1))
    so_idade = idade[ok].reshape(-1, 1)
    com_prob = np.column_stack([so_idade, logit[ok]])
    estatistica = 2 * (_log_verossimilhanca(com_prob, y[ok]) - _log_verossimilhanca(so_idade, y[ok]))
    return chi2.sf(estatistica, 1)


def p_permutacao_auc(y, prob, n=N_PERMUTACOES):
    rng = np.random.default_rng(SEMENTE)
    observada = S.metricas(y, prob)["auc"]
    nula = np.array([S.metricas(rng.permutation(y), prob)["auc"] for _ in range(n)])
    return (1 + np.sum(nula >= observada)) / (n + 1)


def _qc_por_grupo(amostra, logs):
    qc = amostra[["rec_id", "y"]].merge(metricas_qc(logs), on="rec_id")
    linhas = []
    for metrica in COLUNAS_QC:
        pd_, hc = qc.loc[qc["y"] == 1, metrica], qc.loc[qc["y"] == 0, metrica]
        linhas.append({"metrica": metrica, "media_pd": pd_.mean(), "media_hc": hc.mean(),
                       "p_mann_whitney": mannwhitneyu(pd_, hc).pvalue if qc[metrica].nunique() > 1 else np.nan})
    return pd.DataFrame(linhas)


def _correlacao_clinica(predicoes, amostra, caminhos_externos):
    participantes = pd.read_csv(caminhos_externos.ds(EXTERNOS[0]) / "participants.tsv", sep="\t")
    participantes = participantes.replace("n/a", np.nan)
    participantes["sub"] = participantes["participant_id"].str.replace("sub-", "", regex=False)
    pacientes = (predicoes[predicoes["y"] == 1].merge(amostra[["rec_id", "sub"]], on="rec_id")
                 .merge(participantes, on="sub"))
    linhas = []
    for variavel in VARIAVEIS_CLINICAS:
        valor = pd.to_numeric(pacientes[variavel], errors="coerce")
        ok = valor.notna()
        rho, p = spearmanr(pacientes.loc[ok, MODELO_PRINCIPAL], valor[ok])
        linhas.append({"variavel": variavel, "n": int(ok.sum()), "rho_spearman": rho, "p": p})
    return pd.DataFrame(linhas)


def rodar(caminhos: Caminhos, caminhos_externos: Caminhos, n_jobs: int = 1):
    pasta = caminhos_externos.resultados / "externo"
    pasta.mkdir(parents=True, exist_ok=True)

    comuns = json.loads((caminhos.features / "canais_comuns.json").read_text())
    treino = dados_p1(carregar(caminhos), "ON", False)
    tabela, logs = _features_externas(caminhos_externos, comuns, n_jobs)
    externo = _amostra_externa(tabela)
    assert not set(treino["sujeito"]) & set(externo["sujeito"])
    print(f"treino: {len(treino)} sujeitos de {treino['dataset'].nunique()} centros; "
          f"externo: {len(externo)} ({externo['y'].sum()} PD, {(1 - externo['y']).sum()} HC)")

    espectral = M.colunas(treino, "espectral")
    roi_treino = media_por_regiao(treino, PREFIXOS_ESPECTRAIS, comuns)
    roi_externo = media_por_regiao(externo, PREFIXOS_ESPECTRAIS, comuns)[roi_treino.columns]
    modelos = {
        MODELO_PRINCIPAL: (treino[espectral], externo[espectral]),
        "espectral_roi (secundário)": (roi_treino, roi_externo),
        "so_idade (controle)": (treino[["idade"]], externo[["idade"]]),
    }
    y_treino, grupos, y = treino["y"].to_numpy(), treino["sujeito"].to_numpy(), externo["y"].to_numpy()
    predicoes = externo[["rec_id", "sujeito", "grupo", "idade", "sexo", "y"]].copy()
    linhas = []
    for nome, (X_treino, X_externo) in modelos.items():
        X_treino, X_externo = X_treino.to_numpy(float), X_externo.to_numpy(float)
        completo = ~np.isnan(X_treino).any(1)
        prob = M.ajustar_prever("logistica", X_treino[completo], y_treino[completo], grupos[completo], X_externo)
        predicoes[nome] = prob
        linhas.append({"modelo": nome, "n": len(y), "n_pd": int(y.sum()), **S.resumo(y, prob, N_BOOTSTRAP),
                       "p_permutacao": p_permutacao_auc(y, prob)})
    resumo = pd.DataFrame(linhas)
    p_idade = p_alem_da_idade(predicoes[MODELO_PRINCIPAL].to_numpy(), externo["idade"].to_numpy(float), y)
    resumo["p_alem_da_idade"] = np.where(resumo["modelo"] == MODELO_PRINCIPAL, p_idade, np.nan)
    resumo.to_csv(pasta / "resumo.csv", index=False)
    predicoes.to_csv(pasta / "predicoes.csv", index=False)
    print(resumo[["modelo", "n", "n_pd", "auc", "auc_ic", "acuracia_balanceada", "sensibilidade", "especificidade",
                  "brier", "p_permutacao", "p_alem_da_idade"]].round(4).to_string(index=False))

    qc = _qc_por_grupo(externo, logs)
    qc.to_csv(pasta / "qc_por_grupo.csv", index=False)
    print(qc.round(3).to_string(index=False))

    clinica = _correlacao_clinica(predicoes, externo, caminhos_externos)
    clinica.to_csv(pasta / "clinica_pacientes.csv", index=False)
    print(clinica.round(3).to_string(index=False))
    return resumo
