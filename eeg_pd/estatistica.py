"""Métricas, intervalos, testes e correções."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _auc(y, p):
    """AUC pela estatística de Mann-Whitney (igual ao roc_auc_score, bem mais rápida)."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def metricas(y, p, limiar: float = 0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    yh = (p >= limiar).astype(int)
    sens = yh[y == 1].mean() if (y == 1).any() else np.nan
    espec = 1 - yh[y == 0].mean() if (y == 0).any() else np.nan
    return {"acuracia_balanceada": np.nanmean([sens, espec]), "auc": _auc(y, p),
            "sensibilidade": sens, "especificidade": espec, "brier": np.mean((p - y) ** 2)}


def bootstrap(y, p, n: int = 2000, semente: int = 0) -> dict:
    """IC 95% por bootstrap de sujeitos (cada linha é um sujeito)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    rng = np.random.default_rng(semente)
    amostras = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        amostras.append(metricas(y[i], p[i]))
    if not amostras:
        return {}
    df = pd.DataFrame(amostras)
    return {f"{k}_ic": f"[{df[k].quantile(.025):.3f}, {df[k].quantile(.975):.3f}]" for k in df}


def resumo(y, p, n_boot: int = 2000, semente: int = 0) -> dict:
    return {**metricas(y, p), **bootstrap(y, p, n_boot, semente)}


# --------------------------------------------------------------------------- DeLong
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong(y, p1, p2) -> tuple[float, float, float]:
    """Teste de DeLong para duas AUCs pareadas. Devolve (auc1, auc2, p bilateral)."""
    y = np.asarray(y).astype(bool)
    preds = np.vstack([p1, p2])
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    auc = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return auc[0], auc[1], 1.0
    z = (auc[0] - auc[1]) / np.sqrt(var)
    return auc[0], auc[1], 2 * stats.norm.sf(abs(z))


def nadeau_bengio(acc_a, acc_b, n_treino: float, n_teste: float) -> float:
    """t corrigido (Nadeau e Bengio, 2003) para a diferença média entre folds."""
    d = np.asarray(acc_a) - np.asarray(acc_b)
    k = len(d)
    var = d.var(ddof=1) * (1 / k + n_teste / n_treino)
    if var <= 0:
        return 1.0
    t = d.mean() / np.sqrt(var)
    return 2 * stats.t.sf(abs(t), df=k - 1)


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    q = p[ok]
    ordem = np.argsort(q)
    m = len(q)
    aj = np.maximum.accumulate((m - np.arange(m)) * q[ordem])
    res = np.empty(m)
    res[ordem] = np.minimum(aj, 1)
    out[ok] = res
    return out


# --------------------------------------------------------------------------- P2 e pareado
def taxa_pareada(acertos: np.ndarray, n_boot: int = 2000, semente: int = 0) -> dict:
    a = np.asarray(acertos, float)
    rng = np.random.default_rng(semente)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_boot)]
    return {"pacientes": len(a), "acertos": int(a.sum()), "taxa": a.mean(),
            "ic_inf": np.quantile(bs, .025), "ic_sup": np.quantile(bs, .975),
            "p_binomial": stats.binomtest(int(a.sum()), len(a), 0.5, alternative="greater").pvalue}


def contraste(delta) -> dict:
    d = np.asarray(delta, float)
    d = d[~np.isnan(d)]
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(.975, n - 1)
    return {"n_pares": n, "media_delta": m, "ic_inf": m - tc * se, "ic_sup": m + tc * se,
            "t": m / se, "p_t": 2 * stats.t.sf(abs(m / se), n - 1),
            "p_wilcoxon": stats.wilcoxon(d).pvalue if n >= 5 else np.nan, "dz": m / d.std(ddof=1)}


def icc21(x: np.ndarray) -> float:
    """ICC(2,1), concordância absoluta, para uma matriz sujeitos x medidas."""
    x = np.asarray(x, float)
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
