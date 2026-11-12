"""P1 (PD vs. controle), controles de vazamento e confusão, P2 (ON vs. OFF),
comparação de modelos, auditoria e fluxograma de participantes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import modelos as M
from . import estatistica as S
from .config import EXTERNOS, N_BOOTSTRAP, SEMENTE, Caminhos
from .features import carregar

COM_OFF = ("ds002778", "ds003490")
REFERENCIA = ("espectral", "logistica", False)


# --------------------------------------------------------------------------- dados
def _base(grav, com_excluidas):
    return grav if com_excluidas else grav[grav["principal"]]


def dados_p1(grav: pd.DataFrame, med: str, com_excluidas: bool) -> pd.DataFrame:
    assert not grav["dataset"].isin(EXTERNOS).any(), "centro externo nos dados de treino"
    g = _base(grav, com_excluidas)
    hc = g[g["grupo"] == "HC"]
    if med == "OFF":  # controles só dos centros que têm pacientes em OFF
        hc = hc[hc["dataset"].isin(COM_OFF)]
    pd_ = g[(g["grupo"] == "PD") & (g["med"] == med)]
    d = pd.concat([hc, pd_]).reset_index(drop=True)
    assert d["sujeito"].is_unique, "mais de uma gravação por sujeito na P1"
    d["y"] = (d["grupo"] == "PD").astype(int)
    return d


def pasta_p1(caminhos, med, com_excluidas):
    return caminhos.resultados / (f"p1_{med}" + ("_com_excluidas" if com_excluidas else ""))


def folds_p1(caminhos, d, med, com_excluidas):
    pasta = pasta_p1(caminhos, med, com_excluidas)
    out = {}
    for prot, fn in (("B_sujeito", lambda: M.folds_sujeito(d["sujeito"], d["y"], d["dataset"])),
                     ("C_centro", lambda: M.folds_centro(d["sujeito"], d["dataset"]))):
        arq = pasta / f"folds_{prot}.json"
        if not arq.exists():
            M.salvar_folds(arq, fn())
        folds = M.carregar_folds(arq)
        todos = sorted(s for _, te in folds for s in te)
        assert todos == sorted(d["sujeito"]), f"{arq.name} não corresponde aos sujeitos atuais; apague-o e rode de novo"
        out[prot] = folds
    return out


def _avaliar(d, folds, X, nome, harm, n_jobs=1, riemann=None):
    linhas = []
    idx = {s: i for i, s in enumerate(d["sujeito"])}
    for k, (tr, te) in enumerate(folds):
        M.checar_vazamento(tr, te)
        itr, ite = [idx[s] for s in tr], [idx[s] for s in te]
        ytr = d["y"].to_numpy()[itr]
        gtr = d["sujeito"].to_numpy()[itr]
        str_, ste = d["dataset"].to_numpy()[itr], d["dataset"].to_numpy()[ite]
        if riemann is not None:
            p = M.riemann_prever(riemann[itr], ytr, gtr, str_, riemann[ite], ste, recentrar=harm)
        else:
            Xtr, Xte = X[itr], X[ite]
            if harm:
                Xtr, Xte = M.harmonizar(Xtr, str_, Xte, ste)
            p = M.ajustar_prever(nome, Xtr, ytr, gtr, Xte, n_jobs=n_jobs)
        linhas += [{"sujeito": d["sujeito"].iat[i], "dataset": d["dataset"].iat[i], "y": int(d["y"].iat[i]),
                    "prob": float(pi), "fold": k} for i, pi in zip(ite, p)]
    return pd.DataFrame(linhas)


def nome_config(prot, conjunto, modelo, harm):
    return f"{prot}__{conjunto}__{modelo}__{'harm' if harm else 'semharm'}"


def resumir_predicoes(pasta: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    geral, centro = [], []
    for f in sorted(pasta.glob("pred__*.csv")):
        prot, conj, mod, harm = f.stem.split("__")[1:5]
        pr = pd.read_csv(f)
        base = {"protocolo": prot, "features": conj, "modelo": mod, "harmonizacao": harm == "harm"}
        geral.append({**base, "n_sujeitos": len(pr), **S.resumo(pr["y"], pr["prob"], N_BOOTSTRAP)})
        for ds, g in pr.groupby("dataset"):
            centro.append({**base, "centro": ds, "n": len(g), "n_pd": int(g["y"].sum()),
                           **S.resumo(g["y"], g["prob"], N_BOOTSTRAP)})
    g, c = pd.DataFrame(geral), pd.DataFrame(centro)
    g.to_csv(pasta / "resumo.csv", index=False)
    c.to_csv(pasta / "resumo_por_centro.csv", index=False)
    return g, c


def rodar_p1(caminhos: Caminhos, med="ON", com_excluidas=False, n_jobs=1, so_referencia=False):
    grav = carregar(caminhos)
    d = dados_p1(grav, med, com_excluidas)
    pasta = pasta_p1(caminhos, med, com_excluidas)
    folds = folds_p1(caminhos, d, med, com_excluidas)
    print(f"P1 {med}: {len(d)} sujeitos ({d['y'].sum()} PD, {(1 - d['y']).sum()} HC) | por centro: "
          f"{d.groupby('dataset')['y'].agg(['size', 'sum']).to_dict('index')}")
    configs = [("espectral", m) for m in ("logistica", "lda", "svm", "floresta", "gboost")]
    configs += [(c, m) for c in ("conectividade", "tudo") for m in ("logistica", "floresta")]
    if so_referencia:
        configs = [("espectral", "logistica")]
    tarefas = [(p, c, m, h) for p in folds for c, m in configs for h in (False, True)]

    cov = None
    arq_cov = caminhos.features / "covariancias.npz"
    if arq_cov.exists() and not so_referencia:
        z = np.load(arq_cov, allow_pickle=True)
        pos = {r: i for i, r in enumerate(z["rec_id"])}
        cov = z["C"][[pos[r] for r in d["rec_id"]]]
        tarefas += [(p, "covariancia", "riemann", h) for p in folds for h in (False, True)]

    def uma(p, c, m, h):
        if m == "riemann":
            pr = _avaliar(d, folds[p], None, m, h, riemann=cov)
        else:
            X = d[M.colunas(d, c)].to_numpy(float)
            pr = _avaliar(d, folds[p], X, m, h)
        pr.to_csv(pasta / f"pred__{nome_config(p, c, m, h)}.csv", index=False)
        return nome_config(p, c, m, h), S.metricas(pr["y"], pr["prob"])

    res = Parallel(n_jobs=n_jobs)(delayed(uma)(*t) for t in tarefas)
    for nome, r in res:
        print(f"{nome}: AB={r['acuracia_balanceada']:.3f}  AUC={r['auc']:.3f}")
    g, c = resumir_predicoes(pasta)
    return g, c


# --------------------------------------------------------------------------- controles
def _logistica_fixa():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))


def _demografia_e_idade(dados, pasta) -> dict:
    """Tabela demográfica por centro e grupo e AUC de um modelo que usa só a idade."""
    feminino = lambda s: 100 * s.astype(str).str.upper().str.startswith("F").mean()
    demografia = dados.groupby(["dataset", "grupo"]).agg(n=("sujeito", "size"), idade_media=("idade", "mean"),
                                                         idade_dp=("idade", "std"), pct_feminino=("sexo", feminino))
    demografia.round(1).to_csv(pasta / "demografia_p1.csv")
    print(demografia.round(1).to_string())
    com_idade = dados.dropna(subset=["idade"]).reset_index(drop=True)
    if len(com_idade) <= 20:
        return {}
    folds = M.folds_sujeito(com_idade["sujeito"], com_idade["y"], com_idade["dataset"])
    pred = _avaliar(com_idade, folds, com_idade[["idade"]].to_numpy(float), "logistica", False)
    auc = roc_auc_score(pred["y"], pred["prob"])
    print(f"AUC usando só a idade: {auc:.3f}")
    return {"auc_so_idade": auc}


def _janela_aleatoria_vs_sujeito(dados, caminhos) -> dict:
    """Acurácia por janela com janelas sorteadas (A, vaza sujeito) e por sujeito (B)."""
    janelas = pd.read_parquet(caminhos.features / "janelas_espectral.parquet")
    janelas = janelas.merge(dados[["rec_id", "sujeito", "y"]], on="rec_id")
    X = janelas[[c for c in janelas.columns if c.startswith("rel_")]].to_numpy(float)
    y, sujeitos = janelas["y"].to_numpy(), janelas["sujeito"].to_numpy()
    classificador = _logistica_fixa()
    saida = {}
    for protocolo, cv, grupos in (
            ("A_janela_aleatoria", StratifiedKFold(10, shuffle=True, random_state=SEMENTE), None),
            ("B_sujeito", StratifiedGroupKFold(10, shuffle=True, random_state=SEMENTE), sujeitos)):
        previsto = np.empty_like(y)
        for treino, teste in cv.split(X, y, grupos):
            previsto[teste] = classificador.fit(X[treino], y[treino]).predict(X[teste])
        saida[f"ab_janela_{protocolo}"] = balanced_accuracy_score(y, previsto)
    print(f"janela: A={saida['ab_janela_A_janela_aleatoria']:.3f}  B={saida['ab_janela_B_sujeito']:.3f}")
    return saida


def _classificador_de_centro(dados, X) -> dict:
    """Quão bem as features identificam o centro (acima do acaso = efeito de centro)."""
    centros = dados["dataset"].to_numpy()
    previsto = np.empty_like(centros)
    classificador = _logistica_fixa()
    cv = StratifiedGroupKFold(10, shuffle=True, random_state=SEMENTE)
    for treino, teste in cv.split(X, centros, dados["sujeito"]):
        previsto[teste] = classificador.fit(X[treino], centros[treino]).predict(X[teste])
    saida = {"ab_centro": balanced_accuracy_score(centros, previsto), "acaso_centro": 1 / len(np.unique(centros))}
    print(f"classificador de centro: AB={saida['ab_centro']:.3f} (acaso {saida['acaso_centro']:.3f})")
    return saida


def _embaralhar_dentro_do_centro(dados, semente):
    rng = np.random.default_rng(semente)
    embaralhado = dados.copy()
    for centro in embaralhado["dataset"].unique():
        m = (embaralhado["dataset"] == centro).to_numpy()
        embaralhado.loc[m, "y"] = rng.permutation(embaralhado.loc[m, "y"].to_numpy())
    return embaralhado


def _teste_de_permutacao(dados, folds, X, n_perm, n_jobs) -> dict:
    """p do modelo de referência com rótulos embaralhados dentro de cada centro."""
    def metricas(d):
        pred = _avaliar(d, folds, X, "logistica", False)
        return S.metricas(pred["y"], pred["prob"])["acuracia_balanceada"], roc_auc_score(pred["y"], pred["prob"])

    ab, auc = metricas(dados)
    nulos = np.array(Parallel(n_jobs=n_jobs)(
        delayed(metricas)(_embaralhar_dentro_do_centro(dados, SEMENTE + i)) for i in range(n_perm)))
    return {"ab": ab, "auc": auc, "media_nula_ab": nulos[:, 0].mean(), "dp_nulo_ab": nulos[:, 0].std(),
            "media_nula_auc": nulos[:, 1].mean(), "p_ab": (1 + (nulos[:, 0] >= ab).sum()) / (n_perm + 1),
            "p_auc": (1 + (nulos[:, 1] >= auc).sum()) / (n_perm + 1), "n_perm": n_perm}


def rodar_controles(caminhos: Caminhos, n_perm=1000, com_excluidas=False, n_jobs=1):
    dados = dados_p1(carregar(caminhos), "ON", com_excluidas)
    pasta = caminhos.resultados / ("controles" + ("_com_excluidas" if com_excluidas else ""))
    pasta.mkdir(parents=True, exist_ok=True)
    folds = folds_p1(caminhos, dados, "ON", com_excluidas)
    X = dados[M.colunas(dados, "espectral")].to_numpy(float)

    resultado = {**_demografia_e_idade(dados, pasta), **_janela_aleatoria_vs_sujeito(dados, caminhos),
                 **_classificador_de_centro(dados, X)}
    for protocolo in ("B_sujeito", "C_centro"):
        resultado[f"perm_{protocolo}"] = _teste_de_permutacao(dados, folds[protocolo], X, n_perm, n_jobs)
        print(f"permutação {protocolo}: {resultado[f'perm_{protocolo}']}")
    (pasta / "controles.json").write_text(json.dumps(resultado, indent=1, default=float))
    return resultado


# --------------------------------------------------------------------------- P2
def dados_p2(grav, com_excluidas):
    g = _base(grav, com_excluidas)
    g = g[(g["grupo"] == "PD") & g["med"].isin(["ON", "OFF"]) & g["dataset"].isin(COM_OFF)]
    ambos = g.groupby("sujeito")["med"].nunique()
    g = g[g["sujeito"].isin(ambos[ambos == 2].index)].sort_values(["sujeito", "med"]).reset_index(drop=True)
    g["y"] = (g["med"] == "OFF").astype(int)
    return g


def _folds_paciente(pacientes, n=10, semente=SEMENTE):
    p = np.random.default_rng(semente).permutation(sorted(set(pacientes)))
    return [(sorted(set(p) - set(te)), sorted(te)) for te in np.array_split(p, min(n, len(p)))]


def _p2_avaliar(d, X, nome, folds, centrar):
    X = X.copy()
    if centrar:  # usa as duas sessões do paciente, sem rótulo
        for s in d["sujeito"].unique():
            m = (d["sujeito"] == s).to_numpy()
            X[m] -= X[m].mean(0)
    saida = "decisao" if nome == "svm" else "proba"
    score = np.full(len(d), np.nan)
    for tr, te in folds:
        M.checar_vazamento(tr, te)
        mtr, mte = d["sujeito"].isin(tr).to_numpy(), d["sujeito"].isin(te).to_numpy()
        score[mte] = M.ajustar_prever(nome, X[mtr], d["y"].to_numpy()[mtr], d["sujeito"].to_numpy()[mtr],
                                      X[mte], saida=saida, sem_busca=True)
    t = d.assign(score=score).pivot_table(index=["sujeito", "dataset"], columns="y", values="score").reset_index()
    t["acerto"] = (t[1] > t[0]).astype(int)       # sessão OFF com escore maior que a ON
    return t


def _trocar_on_off(dados, semente):
    """Troca ON e OFF em metade dos pacientes, sorteados (distribuição nula da P2)."""
    rng = np.random.default_rng(semente)
    pacientes = np.array(sorted(dados["sujeito"].unique()))
    trocados = set(pacientes[rng.random(len(pacientes)) < .5])
    embaralhado = dados.copy()
    m = embaralhado["sujeito"].isin(trocados).to_numpy()
    embaralhado.loc[m, "y"] = 1 - embaralhado.loc[m, "y"]
    return embaralhado


def _protocolos_p2(dados):
    protocolos = {"B_paciente": _folds_paciente(dados["sujeito"])}
    centros = sorted(dados["dataset"].unique())
    if len(centros) == 2:
        a, b = (sorted(dados.loc[dados["dataset"] == c, "sujeito"].unique()) for c in centros)
        protocolos["D_entre_centros"] = [(a, b), (b, a)]
    return protocolos


def _modelos_p2(conjunto):
    # LDA com shrinkage nas ~1200 features de wPLI custa ~1 s por ajuste, o que torna a
    # permutação inviável; na conectividade ficam logística e SVM
    return ("logistica", "lda", "svm") if conjunto == "espectral" else ("logistica", "svm")


def _configuracao_p2(dados, X, modelo, folds, centrar, n_perm, n_jobs):
    """Taxa de pares ordenados corretamente e p por permutação de uma configuração."""
    tabela = _p2_avaliar(dados, X, modelo, folds, centrar)
    observada = tabela["acerto"].mean()
    nulos = np.array(Parallel(n_jobs=n_jobs)(
        delayed(lambda semente: _p2_avaliar(_trocar_on_off(dados, semente), X, modelo, folds, centrar)
                ["acerto"].mean())(SEMENTE + i) for i in range(n_perm)))
    resumo = {**S.taxa_pareada(tabela["acerto"].to_numpy()), "media_nula": nulos.mean(),
              "p_permutacao": (1 + (nulos >= observada).sum()) / (n_perm + 1)}
    return tabela, resumo


def rodar_p2(caminhos: Caminhos, n_perm=1000, com_excluidas=False, n_jobs=1):
    dados = dados_p2(carregar(caminhos), com_excluidas)
    pasta = caminhos.resultados / ("p2" + ("_com_excluidas" if com_excluidas else ""))
    pasta.mkdir(parents=True, exist_ok=True)
    print(f"P2: {dados['sujeito'].nunique()} pacientes com ON e OFF: "
          f"{dados.groupby('dataset')['sujeito'].nunique().to_dict()}")
    linhas = []
    for protocolo, folds in _protocolos_p2(dados).items():
        for conjunto in ("espectral", "conectividade"):
            X = dados[M.colunas(dados, conjunto)].to_numpy(float)
            for centrar in (False, True):
                for modelo in _modelos_p2(conjunto):
                    tabela, resumo = _configuracao_p2(dados, X, modelo, folds, centrar, n_perm, n_jobs)
                    nome = f"{protocolo}__{conjunto}__{'centrado' if centrar else 'bruto'}__{modelo}"
                    tabela.to_csv(pasta / f"pred__{nome}.csv", index=False)
                    linhas.append({"protocolo": protocolo, "features": conjunto, "centrado_por_paciente": centrar,
                                   "modelo": modelo, **resumo})
                    print(f"P2 {nome}: taxa={resumo['taxa']:.3f} [{resumo['ic_inf']:.3f}, {resumo['ic_sup']:.3f}] "
                          f"p_perm={resumo['p_permutacao']:.4f} (nula média {resumo['media_nula']:.3f})", flush=True)
    resultado = pd.DataFrame(linhas)
    resultado["p_permutacao_holm"] = S.holm(resultado["p_permutacao"])
    resultado.to_csv(pasta / "resumo_p2.csv", index=False)
    return resultado


# --------------------------------------------------------------------------- comparação e auditoria
def comparar(caminhos: Caminhos, med="ON", com_excluidas=False):
    pasta = pasta_p1(caminhos, med, com_excluidas)
    resumo, _ = resumir_predicoes(pasta)
    folds = {p: M.carregar_folds(pasta / f"folds_{p}.json") for p in ("B_sujeito", "C_centro")}
    linhas = []
    for prot in ("B_sujeito", "C_centro"):
        ref = pd.read_csv(pasta / f"pred__{nome_config(prot, *REFERENCIA)}.csv").set_index("sujeito")
        for f in sorted(pasta.glob(f"pred__{prot}__*.csv")):
            _, _, conj, mod, harm = f.stem.split("__")
            pr = pd.read_csv(f).set_index("sujeito").loc[ref.index]
            a_ref, a_m, p_dl = S.delong(ref["y"], ref["prob"], pr["prob"])
            lin = {"protocolo": prot, "features": conj, "modelo": mod, "harmonizacao": harm == "harm",
                   "dif_auc_vs_ref": a_m - a_ref, "p_delong": p_dl if f.stem != f"pred__{nome_config(prot, *REFERENCIA)}" else np.nan}
            if prot == "B_sujeito":
                acc = lambda x: [balanced_accuracy_score(g["y"], g["prob"] >= .5) for _, g in x.groupby("fold")]
                n_te = np.mean([len(te) for _, te in folds[prot]])
                lin["p_nadeau_bengio"] = S.nadeau_bengio(acc(pr), acc(ref), len(ref) - n_te, n_te) \
                    if not np.isnan(lin["p_delong"]) else np.nan
            linhas.append(lin)
    comp = pd.DataFrame(linhas)
    for prot in comp["protocolo"].unique():
        m = comp["protocolo"] == prot
        comp.loc[m, "p_delong_holm"] = S.holm(comp.loc[m, "p_delong"])
        if "p_nadeau_bengio" in comp:
            comp.loc[m, "p_nadeau_bengio_holm"] = S.holm(comp.loc[m, "p_nadeau_bengio"])
    chave = ["protocolo", "features", "modelo", "harmonizacao"]
    comp = resumo.merge(comp, on=chave).sort_values(["protocolo", "auc"], ascending=[True, False])
    comp.to_csv(pasta / "comparacao_modelos.csv", index=False)
    cols = chave + ["acuracia_balanceada", "acuracia_balanceada_ic", "auc", "auc_ic", "dif_auc_vs_ref", "p_delong_holm"]
    print(comp[[c for c in cols if c in comp]].round(3).to_string(index=False))
    return comp


def auditoria(caminhos: Caminhos):
    linhas = []
    for pasta in sorted(caminhos.resultados.glob("p1_*")):
        folds = {p: M.carregar_folds(pasta / f"folds_{p}.json") for p in ("B_sujeito", "C_centro")
                 if (pasta / f"folds_{p}.json").exists()}
        for f in sorted(pasta.glob("pred__*.csv")):
            prot = f.stem.split("__")[1]
            pr = pd.read_csv(f)
            linhas.append({"arquivo": f"{pasta.name}/{f.name}", "checagem": "cada sujeito testado uma vez",
                           "ok": pr["sujeito"].is_unique})
            ok = all(sorted(pr.loc[pr["fold"] == k, "sujeito"]) == sorted(te) for k, (_, te) in enumerate(folds[prot]))
            linhas.append({"arquivo": f"{pasta.name}/{f.name}", "checagem": "teste igual ao arquivo de folds", "ok": ok})
        for f in sorted(pasta.glob("execucao__*.json")):
            ex = json.loads(f.read_text())
            prot = f.stem.split("__")[1]
            for k in ex["folds"]:
                tr, va, te = set(k["treino"]), set(k["validacao"]), set(k["teste"])
                tr_fold, te_fold = map(set, folds[prot][k["fold"]])
                linhas.append({"arquivo": f"{pasta.name}/{f.name}",
                               "checagem": f"fold {k['fold']}: validação tirada do treino do fold, longe do teste",
                               "ok": (tr | va) == tr_fold and te == te_fold and not (tr & va) and not ((tr | va) & te)})
    df = pd.DataFrame(linhas)
    caminhos.resultados.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminhos.resultados / "auditoria_vazamento.csv", index=False)
    print(f"{len(df)} checagens em {df['arquivo'].nunique()} arquivos: {df['ok'].sum()} ok, {(~df['ok']).sum()} falha(s)")
    if not df["ok"].all():
        print(df[~df["ok"]].to_string(index=False))
        raise SystemExit("auditoria falhou")
    return df


# --------------------------------------------------------------------------- fluxograma
def fluxo(caminhos: Caminhos):
    from .dados import tabela_gravacoes
    from .pareado import pares_elegiveis
    from .preprocessamento import carregar_logs

    tab = tabela_gravacoes(caminhos)
    logs = carregar_logs(caminhos)
    t = tab.merge(logs[["rec_id", "status", "etapa_exclusao", "motivo"]], on="rec_id", how="left")
    t["status"] = t["status"].fillna("nao_processada")
    feats = set(carregar(caminhos)["rec_id"]) if (caminhos.features / "gravacoes.parquet").exists() else set()
    t["com_features"] = t["rec_id"].isin(feats)
    linhas = []
    for ds, g in t.groupby("dataset"):
        linhas.append({"dataset": ds, "etapa": "1. gravações no BIDS", "n": len(g)})
        for etapa, gg in g[g["status"] != "ok"].groupby(g["etapa_exclusao"].fillna("nao_processada")):
            linhas.append({"dataset": ds, "etapa": f"   excluídas: {etapa}", "n": len(gg)})
        linhas.append({"dataset": ds, "etapa": "2. passaram no controle de qualidade", "n": int((g["status"] == "ok").sum())})
        linhas.append({"dataset": ds, "etapa": "3. com features", "n": int(g["com_features"].sum())})
        fora = g[(g["status"] == "ok") & ~g["principal"]]
        linhas.append({"dataset": ds, "etapa": "   fora da análise principal (pd6/pd16 ON)", "n": len(fora)})
    if feats:
        grav = carregar(caminhos)
        for med in ("ON", "OFF"):
            d = dados_p1(grav, med, False)
            for ds, g in d.groupby("dataset"):
                linhas.append({"dataset": ds, "etapa": f"4. P1 {med}: HC / PD", "n": f"{(g['y'] == 0).sum()} / {g['y'].sum()}"})
        p2 = dados_p2(grav, False)
        for ds, g in p2.groupby("dataset"):
            linhas.append({"dataset": ds, "etapa": "5. P2: pacientes com ON e OFF", "n": g["sujeito"].nunique()})
        if (caminhos.features / "janelas_beta.parquet").exists():
            for ds, n in pares_elegiveis(caminhos, False).groupby("dataset")["sujeito"].nunique().items():
                linhas.append({"dataset": ds, "etapa": "6. beta pareado: pares com sinal para todas as durações", "n": n})
    fl = pd.DataFrame(linhas)
    caminhos.resultados.mkdir(parents=True, exist_ok=True)
    fl.to_csv(caminhos.resultados / "fluxo_participantes.csv", index=False)
    exc = t[t["status"] != "ok"][["rec_id", "dataset", "grupo", "med", "etapa_exclusao", "motivo"]]
    exc.to_csv(caminhos.resultados / "exclusoes.csv", index=False)
    print(fl.to_string(index=False))
    if len(exc):
        print("\nExclusões:")
        print(exc.to_string(index=False))
    return fl, exc
