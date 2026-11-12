"""Modelos clássicos, harmonização por centro, riemanniano e divisões (folds)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import N_FOLDS_INTERNOS, N_FOLDS_SUJEITO, SEMENTE

PREFIXOS = {"espectral": ("rel_", "exp_"), "conectividade": ("wpli_",), "tudo": ("rel_", "exp_", "wpli_")}


def colunas(df: pd.DataFrame, conjunto: str) -> list[str]:
    return [c for c in df.columns if c.startswith(PREFIXOS[conjunto])]


def modelo(nome: str, n_jobs: int = 1):
    """(estimador, grade). As grades são pequenas e foram fixadas antes de rodar."""
    if nome == "logistica":
        return (make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
                {"logisticregression__C": [0.01, 0.1, 1.0]})
    if nome == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                                                          priors=[0.5, 0.5])), None
    if nome == "svm":
        return (make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced",
                                                    random_state=SEMENTE)),
                {"svc__C": [0.1, 1.0, 10.0]})
    if nome == "floresta":
        return (RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=n_jobs,
                                       random_state=SEMENTE),
                {"min_samples_leaf": [1, 5]})
    if nome == "gboost":
        return (HistGradientBoostingClassifier(class_weight="balanced", max_depth=3, min_samples_leaf=10,
                                               random_state=SEMENTE),
                {"learning_rate": [0.05, 0.1]})
    raise ValueError(nome)


# valores fixos usados quando não há busca de hiperparâmetros (P2, com 36 pacientes,
# em que a validação interna seria ruído e tornaria a permutação inviável)
FIXOS = {"logistica": {"logisticregression__C": 0.1}, "svm": {"svc__C": 1.0}}


def ajustar_prever(nome, Xtr, ytr, gtr, Xte, n_jobs=1, saida="proba", sem_busca=False):
    est, grade = modelo(nome, n_jobs)
    if sem_busca:
        est.set_params(**FIXOS.get(nome, {}))
        grade = None
    if saida == "decisao" and nome == "svm":
        est.set_params(svc__probability=False)
    if grade:
        n_int = min(N_FOLDS_INTERNOS, int(np.bincount(ytr).min()))
        cv = StratifiedGroupKFold(n_splits=max(n_int, 2), shuffle=True, random_state=SEMENTE)
        est = GridSearchCV(est, grade, cv=cv, scoring="roc_auc", n_jobs=1)
        est.fit(Xtr, ytr, groups=gtr)
    else:
        est.fit(Xtr, ytr)
    if saida == "decisao" and hasattr(est, "decision_function"):
        return est.decision_function(Xte)
    return est.predict_proba(Xte)[:, 1]


# --------------------------------------------------------------------------- harmonização
def harmonizar(Xtr, sitios_tr, Xte, sitios_te):
    """Padroniza cada feature dentro de cada centro, com estatísticas do treino.
    Centro de teste ausente do treino (protocolo C) usa as próprias estatísticas,
    calculadas sem rótulo."""
    Xtr, Xte = np.asarray(Xtr, float).copy(), np.asarray(Xte, float).copy()
    est = {}
    for s in np.unique(sitios_tr):
        m = sitios_tr == s
        est[s] = (Xtr[m].mean(0), Xtr[m].std(0) + 1e-12)
        Xtr[m] = (Xtr[m] - est[s][0]) / est[s][1]
    for s in np.unique(sitios_te):
        m = sitios_te == s
        mu, sd = est.get(s, (Xte[m].mean(0), Xte[m].std(0) + 1e-12))
        Xte[m] = (Xte[m] - mu) / sd
    return Xtr, Xte


# --------------------------------------------------------------------------- riemanniano
def _recentrar(C, sitios, ref=None):
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann

    C = C.copy()
    medias = dict(ref or {})
    for s in np.unique(sitios):
        m = sitios == s
        if s not in medias:
            medias[s] = mean_riemann(C[m])
        w = invsqrtm(medias[s])
        C[m] = w @ C[m] @ w
    return C, medias


def riemann_prever(Ctr, ytr, gtr, sitios_tr, Cte, sitios_te, recentrar: bool):
    from pyriemann.tangentspace import TangentSpace

    if recentrar:
        Ctr, med = _recentrar(Ctr, sitios_tr)
        # centro de teste novo: média dele mesmo (sem rótulo)
        Cte, _ = _recentrar(Cte, sitios_te, {s: v for s, v in med.items() if s in set(sitios_te)})
    ts = TangentSpace(metric="riemann").fit(Ctr)
    return ajustar_prever("logistica", ts.transform(Ctr), ytr, gtr, ts.transform(Cte))


# --------------------------------------------------------------------------- folds
def folds_sujeito(sujeitos, y, grupos_estrat=None, n: int = N_FOLDS_SUJEITO):
    """Divisão por sujeito, estratificada por rótulo (e centro, se informado)."""
    est = np.asarray(y).astype(str)
    if grupos_estrat is not None:
        est = np.char.add(est, np.asarray(grupos_estrat).astype(str))
    n = min(n, int(pd.Series(est).value_counts().min()))  # amostras pequenas (testes)
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEMENTE)
    return [(np.asarray(sujeitos)[tr].tolist(), np.asarray(sujeitos)[te].tolist())
            for tr, te in cv.split(np.zeros(len(y)), est, groups=sujeitos)]


def folds_centro(sujeitos, centros):
    s, c = np.asarray(sujeitos), np.asarray(centros)
    return [(s[c != k].tolist(), s[c == k].tolist()) for k in np.unique(c)]


def salvar_folds(caminho: Path, folds):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    corpo = [{"fold": i, "treino": sorted(tr), "teste": sorted(te)} for i, (tr, te) in enumerate(folds)]
    h = hashlib.sha1(json.dumps(corpo).encode()).hexdigest()[:12]
    caminho.write_text(json.dumps({"hash": h, "folds": corpo}, indent=1))


def carregar_folds(caminho: Path):
    d = json.loads(caminho.read_text())
    return [(f["treino"], f["teste"]) for f in d["folds"]]


def checar_vazamento(treino, teste):
    comum = set(treino) & set(teste)
    assert not comum, f"vazamento: sujeito(s) no treino e no teste: {sorted(comum)[:5]}"


# --------------------------------------------------------------------------- previsão fora do fold
def imputar_e_residualizar(X_treino, X_teste, Q_treino=None, Q_teste=None):
    """Imputa valores ausentes com a mediana do treino e, se houver covariáveis Q,
    remove de X a parte explicada por elas. Tudo é ajustado só no treino."""
    mediana = np.nanmedian(X_treino, axis=0)
    mediana = np.where(np.isnan(mediana), 0, mediana)
    X_treino = np.where(np.isnan(X_treino), mediana, X_treino)
    X_teste = np.where(np.isnan(X_teste), mediana, X_teste)
    if Q_treino is not None:
        regressao = LinearRegression().fit(Q_treino, X_treino)
        X_treino = X_treino - regressao.predict(Q_treino)
        X_teste = X_teste - regressao.predict(Q_teste)
    return X_treino, X_teste


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _empilhar(blocos_treino, blocos_teste, y, grupos):
    """Stacking: uma logística por bloco de features e uma logística final treinada
    nas predições fora do fold interno (dividido por sujeito)."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    fora_do_fold = np.zeros((len(y), len(blocos_treino)))
    for interno, validacao in cv.split(np.zeros(len(y)), y, groups=grupos):
        for k, X in enumerate(blocos_treino):
            fora_do_fold[validacao, k] = ajustar_prever("logistica", X[interno], y[interno], grupos[interno],
                                                        X[validacao])
    teste = np.column_stack([ajustar_prever("logistica", Xtr, y, grupos, Xte)
                             for Xtr, Xte in zip(blocos_treino, blocos_teste)])
    final = LogisticRegression(class_weight="balanced", max_iter=5000).fit(_logit(fora_do_fold), y)
    return final.predict_proba(_logit(teste))[:, 1]


def prever_fora_do_fold(dados: pd.DataFrame, folds, blocos: list, covariaveis=None) -> pd.DataFrame:
    """Probabilidade de PD de cada sujeito, prevista pelo fold em que ele é teste.

    blocos: uma matriz de features (logística) ou várias (stacking), com as linhas
    na ordem de `dados`. covariaveis: matriz opcional cuja influência é removida das
    features dentro de cada fold. Sujeitos dos folds ausentes em `dados` são ignorados.
    """
    posicao = {s: i for i, s in enumerate(dados["sujeito"])}
    y, sujeitos = dados["y"].to_numpy(), dados["sujeito"].to_numpy()
    linhas = []
    for k, (treino, teste) in enumerate(folds):
        checar_vazamento(treino, teste)
        itr = np.array([posicao[s] for s in treino if s in posicao])
        ite = np.array([posicao[s] for s in teste if s in posicao])
        if len(ite) == 0:
            continue
        Q_tr, Q_te = (covariaveis[itr], covariaveis[ite]) if covariaveis is not None else (None, None)
        preparados = [imputar_e_residualizar(B[itr], B[ite], Q_tr, Q_te) for B in blocos]
        if len(preparados) == 1:
            X_tr, X_te = preparados[0]
            prob = ajustar_prever("logistica", X_tr, y[itr], sujeitos[itr], X_te)
        else:
            prob = _empilhar([a for a, _ in preparados], [b for _, b in preparados], y[itr], sujeitos[itr])
        linhas += [{"sujeito": sujeitos[i], "dataset": dados["dataset"].iat[i], "y": int(y[i]), "prob": float(p),
                    "fold": k} for i, p in zip(ite, prob)]
    return pd.DataFrame(linhas)
