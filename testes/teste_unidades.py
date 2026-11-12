"""Testes unitários. Uso: python -m pytest testes/teste_unidades.py  (ou python testes/teste_unidades.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne
import numpy as np
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from eeg_pd import modelos as M
from eeg_pd import estatistica as S
from eeg_pd.config import DATASETS
from eeg_pd.preprocessamento import _anotar, _converter_emendas, _rel, janelas_validas, trechos_repouso

mne.set_log_level("ERROR")
rng = np.random.default_rng(0)


def teste_delong_bate_com_sklearn():
    y = rng.integers(0, 2, 200)
    p1, p2 = rng.random(200) + y * .5, rng.random(200) + y * .2
    a1, a2, p = S.delong(y, p1, p2)
    assert abs(a1 - roc_auc_score(y, p1)) < 1e-10 and abs(a2 - roc_auc_score(y, p2)) < 1e-10
    assert 0 <= p < 0.05
    assert S.delong(y, p1, p1)[2] == 1.0


def teste_holm_bate_com_statsmodels():
    p = rng.random(12) / 10
    assert np.allclose(S.holm(p), multipletests(p, method="holm")[1])


def teste_vazamento():
    try:
        M.checar_vazamento(["a", "b"], ["b", "c"])
    except AssertionError:
        return
    raise AssertionError("o assert de vazamento não disparou")


def teste_folds_por_sujeito_e_centro():
    s = [f"s{i}" for i in range(60)]
    y = [i % 2 for i in range(60)]
    c = [f"c{i % 3}" for i in range(60)]
    f = M.folds_sujeito(s, y, c, n=5)
    assert sorted(x for _, te in f for x in te) == sorted(s)
    for tr, te in f + M.folds_centro(s, c):
        assert not set(tr) & set(te)


def teste_icc():
    x = rng.standard_normal(30)
    assert abs(S.icc21(np.c_[x, x]) - 1) < 1e-12
    assert S.icc21(np.c_[x, rng.standard_normal(30)]) < 0.5


def teste_harmonizacao_centro_novo_usa_as_proprias_estatisticas():
    Xtr = rng.standard_normal((40, 3)) + 5
    Xte = rng.standard_normal((10, 3)) * 3 - 7
    _, te = M.harmonizar(Xtr, np.array(["a"] * 40), Xte, np.array(["b"] * 10))
    assert np.allclose(te.mean(0), 0) and np.allclose(te.std(0), 1)


def teste_anotacoes_depois_do_crop():
    for md in (None, 0):
        raw = mne.io.RawArray(np.zeros((1, 25000)), mne.create_info(1, 250.))
        if md is not None:
            raw.set_meas_date(md)
        _anotar(raw, [55.0], [0.0], ["boundary"])
        raw.crop(8, 90)
        assert np.allclose(_rel(raw), [47.0])


def teste_janelas_respeitam_emendas():
    raw = mne.io.RawArray(rng.standard_normal((3, 250 * 60)) * 1e-6,
                          mne.create_info(["Cz", "Pz", "Fz"], 250., "eeg"))
    _anotar(raw, [21.0], [0.0], ["boundary"])
    _converter_emendas(raw)
    jan = janelas_validas(raw, [(0.0, 60.0)], 4.0, 200.0)
    assert 20.0 not in jan and 16.0 in jan and 24.0 in jan and len(jan) == 14


def teste_olhos_abertos_param_no_ultimo_marcador():
    raw = mne.io.RawArray(np.zeros((1, 250 * 300)), mne.create_info(["Cz"], 250., "eeg"))
    on = [5.0 + k for k in range(60)] + [65.0 + k for k in range(60)] + [200.0, 201.3]
    de = ["Eyes Open: Every 1000 ms/S  1"] * 60 + ["Eyes Closed: Every 1000 ms"] * 60 + ["Standard Tone"] * 2
    _anotar(raw, on, [0.0] * len(on), de)
    segs = trechos_repouso(raw, DATASETS["ds003490"])
    assert len(segs) == 1 and np.allclose(segs[0], (7.0, 63.0))   # 5 a 65 s, menos 2 s em cada borda


def teste_p2_recupera_efeito_e_respeita_o_nulo():
    import pandas as pd
    from eeg_pd.experimentos import _folds_paciente, _p2_avaliar

    linhas = []
    for i in range(30):
        base = rng.standard_normal(10) * 2              # diferença grande entre pacientes
        for med in ("ON", "OFF"):
            x = base + rng.standard_normal(10) * 0.3
            if med == "OFF":
                x[0] += 1.0                             # efeito só na primeira feature
            linhas.append({"sujeito": f"p{i}", "dataset": "ds002778" if i < 15 else "ds003490", "med": med,
                           **{f"rel_beta_c{k}": v for k, v in enumerate(x)}})
    d = pd.DataFrame(linhas)
    d["y"] = (d["med"] == "OFF").astype(int)
    X = d[[c for c in d if c.startswith("rel_")]].to_numpy(float).copy()
    folds = _folds_paciente(d["sujeito"])
    for nome in ("logistica", "lda", "svm"):
        assert _p2_avaliar(d, X, nome, folds, centrar=True)["acerto"].mean() >= 0.9, nome
    X[:, 0] = rng.standard_normal(len(X))
    taxa = _p2_avaliar(d, X, "logistica", folds, centrar=True)["acerto"].mean()
    assert 0.15 < taxa < 0.85, taxa


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("teste_"):
            f()
            print("ok", nome)
