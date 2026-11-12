"""Análise pareada da potência beta em C3/C4: ON vs. OFF, duração e deriva.

Pergunta: quanto a estimativa da diferença ON - OFF muda quando se usa menos
sinal limpo do que a gravação inteira?

Principais escolhas (fixadas em config.py antes de rodar):
- referência principal (REFERENCIA_PRINCIPAL); a outra entra como sensibilidade;
- margem de erro aceitável (MARGEM_ERRO_DELTA);
- durações por centro (DURACOES_S), em múltiplos da janela de 4 s.

Análises:
1. contraste na gravação completa, por centro e agrupado;
2. sorteios (principal para a duração): d segundos sorteados ao acaso dentro de
   cada sessão, N_SORTEIOS vezes. Isola o efeito da duração do efeito da posição
   na gravação. Erro de cada paciente = delta(d) - delta(completa); viés, RMSE e
   proporção de pacientes dentro da margem, com IC por bootstrap de pacientes;
3. primeiros d segundos (secundária): mostra o efeito da posição;
4. confiabilidade do próprio delta: duas metades disjuntas de d segundos da
   mesma sessão, ICC(2,1) do delta entre pacientes. Não depende da gravação
   completa, então não sofre com a sobreposição parte-todo;
5. deriva (blocos): beta por bloco consecutivo, modelo misto com intercepto
   aleatório por paciente.

A "gravação completa" tem duração diferente nos dois centros (~170 s em San
Diego, ~56 s no Novo México). Quanto maior a fração d / completa, mais o delta(d)
se sobrepõe ao delta(completa) e menor fica o erro por construção. Por isso a
fração aparece em todas as tabelas e a comparação entre centros usa a análise 4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import estatistica as S
from .config import (BLOCO_S, DURACOES_S, MARGEM_ERRO_DELTA, N_BOOTSTRAP, N_SORTEIOS, PREPROC,
                     REFERENCIA_PRINCIPAL, SEMENTE, Caminhos)
from .features import carregar

CANAIS = ("C3C4", "C3", "C4")


def _janelas_beta(caminhos, com_excluidas):
    grav = carregar(caminhos)
    g = grav if com_excluidas else grav[grav["principal"]]
    g = g[(g["grupo"] == "PD") & g["med"].isin(["ON", "OFF"]) & g["dataset"].isin(DURACOES_S)]
    ambos = g.groupby("sujeito")["med"].nunique()
    g = g[g["sujeito"].isin(ambos[ambos == 2].index)]
    b = pd.read_parquet(caminhos.features / "janelas_beta.parquet")
    return b.merge(g[["rec_id", "dataset", "sujeito", "med"]], on="rec_id")


def pares_elegiveis(caminhos: Caminhos, com_excluidas: bool) -> pd.DataFrame:
    """Pares em que as duas sessões têm janelas para a maior duração do centro."""
    b = _janelas_beta(caminhos, com_excluidas)
    n = b[b["ref"] == REFERENCIA_PRINCIPAL].groupby(["dataset", "sujeito", "med"]).size().unstack("med")
    n = n.reset_index()
    precisa = n["dataset"].map(lambda ds: int(round(max(DURACOES_S[ds]) / PREPROC.janela_s)))
    n["elegivel"] = (n[["ON", "OFF"]].min(axis=1) >= precisa)
    return n[n["elegivel"]]


def _matriz(b, ref, canal):
    """dict sujeito -> (valores ON em ordem temporal, valores OFF)."""
    out = {}
    for s, g in b[b["ref"] == ref].groupby("sujeito"):
        g = g.sort_values("inicio_s")
        out[s] = (g.loc[g["med"] == "ON", canal].to_numpy(), g.loc[g["med"] == "OFF", canal].to_numpy())
    return out


def _boot(fn, erros, n=N_BOOTSTRAP, semente=SEMENTE):
    rng = np.random.default_rng(semente)
    v = [fn(erros[rng.integers(0, len(erros), len(erros))]) for _ in range(n)]
    return np.quantile(v, .025), np.quantile(v, .975)


def rodar(caminhos: Caminhos, com_excluidas=False):
    pasta = caminhos.resultados / ("pareado" + ("_com_excluidas" if com_excluidas else ""))
    pasta.mkdir(parents=True, exist_ok=True)
    b = _janelas_beta(caminhos, com_excluidas)
    ok = pares_elegiveis(caminhos, com_excluidas)
    b = b[b["sujeito"].isin(ok["sujeito"])]
    inv = ok.groupby("dataset").agg(pares=("sujeito", "size"), janelas_min_ON=("ON", "min"),
                                    janelas_min_OFF=("OFF", "min"), janelas_med_ON=("ON", "median"),
                                    janelas_med_OFF=("OFF", "median"))
    inv.to_csv(pasta / "inventario_pares.csv")
    print(inv.to_string())
    rng = np.random.default_rng(SEMENTE)
    w = PREPROC.janela_s
    contraste, sorteios, primeiros, conf = [], [], [], []
    for ref in ("media", "csd"):
        for canal in CANAIS:
            for ds in sorted(b["dataset"].unique()):
                mat = _matriz(b[b["dataset"] == ds], ref, canal)
                suj = sorted(mat)
                full = np.array([mat[s][0].mean() - mat[s][1].mean() for s in suj])
                dur_full = np.median([min(len(mat[s][0]), len(mat[s][1])) * w for s in suj])
                base = {"ref": ref, "canal": canal, "dataset": ds, "principal": ref == REFERENCIA_PRINCIPAL}
                contraste.append({**base, "duracao": "completa", **S.contraste(full)})
                for d in DURACOES_S[ds]:
                    k = int(round(d / w))
                    # 2. sorteios
                    deltas = np.empty((N_SORTEIOS, len(suj)))
                    for i in range(N_SORTEIOS):
                        deltas[i] = [rng.choice(mat[s][0], k, replace=False).mean()
                                     - rng.choice(mat[s][1], k, replace=False).mean() for s in suj]
                    erro = deltas - full
                    vies_i, rmse_i = erro.mean(0), np.sqrt((erro ** 2).mean(0))
                    dentro_i = (np.abs(erro) <= MARGEM_ERRO_DELTA).mean(0)
                    rmse = np.sqrt((rmse_i ** 2).mean())
                    lo, hi = _boot(lambda x: np.sqrt((x ** 2).mean()), rmse_i)
                    vlo, vhi = _boot(np.mean, vies_i)
                    sorteios.append({**base, "duracao_s": k * w, "fracao_da_completa": k * w / dur_full,
                                     "n_pares": len(suj), "vies": vies_i.mean(), "vies_ic": f"[{vlo:.4f}, {vhi:.4f}]",
                                     "rmse": rmse, "rmse_ic_inf": lo, "rmse_ic_sup": hi,
                                     "pct_dentro_da_margem": 100 * dentro_i.mean(),
                                     "rmse_abaixo_da_margem": hi < MARGEM_ERRO_DELTA,
                                     "media_delta_p2_5": np.quantile(deltas.mean(1), .025),
                                     "media_delta_p97_5": np.quantile(deltas.mean(1), .975),
                                     "pct_sorteios_p_menor_005": 100 * np.mean(
                                         [S.contraste(x)["p_t"] < .05 for x in deltas[:200]])})
                    # 3. primeiros d segundos
                    dp = np.array([mat[s][0][:k].mean() - mat[s][1][:k].mean() for s in suj])
                    primeiros.append({**base, "duracao_s": k * w, **S.contraste(dp),
                                      "vies_vs_completa": (dp - full).mean(),
                                      "rmse_vs_completa": np.sqrt(((dp - full) ** 2).mean()),
                                      "r_vs_completa": np.corrcoef(dp, full)[0, 1]})
                    # 4. confiabilidade do delta (metades disjuntas)
                    tem = [s for s in suj if min(len(mat[s][0]), len(mat[s][1])) >= 2 * k]
                    if len(tem) >= 5:
                        iccs, iccs_abs = [], []
                        for _ in range(200):
                            da, db, aa, ab = [], [], [], []
                            for s in tem:
                                on = rng.permutation(mat[s][0])[: 2 * k]
                                off = rng.permutation(mat[s][1])[: 2 * k]
                                da.append(on[:k].mean() - off[:k].mean())
                                db.append(on[k:].mean() - off[k:].mean())
                                aa += [on[:k].mean(), off[:k].mean()]
                                ab += [on[k:].mean(), off[k:].mean()]
                            iccs.append(S.icc21(np.c_[da, db]))
                            iccs_abs.append(S.icc21(np.c_[aa, ab]))
                        conf.append({**base, "duracao_s": k * w, "n_pares": len(tem),
                                     "icc_delta": np.median(iccs), "icc_delta_p2_5": np.quantile(iccs, .025),
                                     "icc_delta_p97_5": np.quantile(iccs, .975),
                                     "icc_beta_absoluto": np.median(iccs_abs)})
                    else:
                        conf.append({**base, "duracao_s": k * w, "n_pares": len(tem), "icc_delta": np.nan,
                                     "nota": f"menos de 5 pares com {2 * k * w:.0f} s limpos por sessão"})
            # agrupado (só faz sentido na gravação completa; durações diferem por centro)
            mats = [_matriz(b[b["dataset"] == ds], ref, canal) for ds in sorted(b["dataset"].unique())]
            full_all = np.array([m[s][0].mean() - m[s][1].mean() for m in mats for s in m])
            contraste.append({"ref": ref, "canal": canal, "dataset": "todos", "principal": ref == REFERENCIA_PRINCIPAL,
                              "duracao": "completa", **S.contraste(full_all)})
    saidas = {"contraste_on_off": contraste, "duracao_sorteios": sorteios, "duracao_primeiros_segundos": primeiros,
              "confiabilidade_delta": conf}
    for nome, linhas in saidas.items():
        pd.DataFrame(linhas).to_csv(pasta / f"{nome}.csv", index=False)
    s = pd.DataFrame(sorteios)
    print(pd.DataFrame(contraste).query("canal == 'C3C4'").round(4).to_string(index=False))
    print(s.query("canal == 'C3C4'")[["ref", "dataset", "duracao_s", "fracao_da_completa", "vies", "rmse",
                                      "rmse_ic_sup", "pct_dentro_da_margem"]].round(4).to_string(index=False))
    print(pd.DataFrame(conf).query("canal == 'C3C4'").round(3).to_string(index=False))
    return saidas


def blocos(caminhos: Caminhos, com_excluidas=False):
    import statsmodels.formula.api as smf

    pasta = caminhos.resultados / ("pareado" + ("_com_excluidas" if com_excluidas else ""))
    pasta.mkdir(parents=True, exist_ok=True)
    b = _janelas_beta(caminhos, com_excluidas)
    b = b[b["sujeito"].isin(pares_elegiveis(caminhos, com_excluidas)["sujeito"])]
    b["bloco"] = [int(t // BLOCO_S[ds]) for t, ds in zip(b["inicio_s"], b["dataset"])]
    b["tempo_min"] = [(k + 0.5) * BLOCO_S[ds] / 60 for k, ds in zip(b["bloco"], b["dataset"])]
    por = b.groupby(["ref", "dataset", "sujeito", "med", "bloco", "tempo_min"])["C3C4"].mean().reset_index()
    larga = por.pivot_table(index=["ref", "dataset", "sujeito", "bloco", "tempo_min"], columns="med",
                            values="C3C4").dropna().reset_index()
    larga["delta"] = larga["ON"] - larga["OFF"]
    resumo, tend = [], []
    for (ref, ds), g in larga.groupby(["ref", "dataset"]):
        for k, gg in g.groupby("bloco"):
            resumo.append({"ref": ref, "dataset": ds, "bloco": k, "inicio_s": k * BLOCO_S[ds], "ON": gg["ON"].mean(),
                           "OFF": gg["OFF"].mean(), **S.contraste(gg["delta"])})
        lin = {"ref": ref, "dataset": ds, "pares": g["sujeito"].nunique(), "blocos": g["bloco"].nunique()}
        try:
            m1 = smf.mixedlm("delta ~ tempo_min", g, groups=g["sujeito"]).fit(reml=True)
            ci = m1.conf_int().loc["tempo_min"]
            lin.update(delta_inclinacao_por_min=m1.params["tempo_min"], delta_ic_inf=ci[0], delta_ic_sup=ci[1],
                       delta_p=m1.pvalues["tempo_min"])
            pp = por[(por["ref"] == ref) & (por["dataset"] == ds) & por["sujeito"].isin(g["sujeito"])].copy()
            pp["off"] = (pp["med"] == "OFF").astype(int)
            m2 = smf.mixedlm("C3C4 ~ tempo_min * off", pp, groups=pp["sujeito"]).fit(reml=True)
            pp["on"] = 1 - pp["off"]
            m3 = smf.mixedlm("C3C4 ~ tempo_min * on", pp, groups=pp["sujeito"]).fit(reml=True)
            lin.update(ON_inclinacao_por_min=m2.params["tempo_min"], ON_p=m2.pvalues["tempo_min"],
                       OFF_inclinacao_por_min=m3.params["tempo_min"], OFF_p=m3.pvalues["tempo_min"],
                       diferenca_inclinacoes_p=m2.pvalues["tempo_min:off"])
        except Exception as e:
            lin["erro"] = str(e)[:100]
        tend.append(lin)
    pd.DataFrame(resumo).to_csv(pasta / "blocos_resumo.csv", index=False)
    pd.DataFrame(tend).to_csv(pasta / "blocos_tendencia.csv", index=False)
    print(pd.DataFrame(tend).round(4).to_string(index=False))
