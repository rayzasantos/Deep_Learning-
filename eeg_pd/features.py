"""Features por gravação, a partir das janelas válidas gravadas no log.

O critério de inclusão é um só: a gravação com `status == "ok"` no log do
pré-processamento tem features. Esta etapa não descarta nenhuma gravação; se uma
gravação ok falhar aqui, o erro interrompe a execução em vez de sumir em silêncio.

Saídas em derivados/features/:
  gravacoes.parquet       uma linha por gravação: metadados + potência relativa,
                          expoente aperiódico e wPLI (canais comuns aos 3 centros)
  janelas_espectral.parquet  potência relativa por janela (protocolo A)
  janelas_beta.parquet    beta absoluto por janela em C3/C4, referência média e CSD
  janelas_brutas.npz      até 30 janelas de 4 s a 125 Hz por gravação (redes)
  covariancias.npz        covariância OAS por gravação (modelo riemanniano)
  canais_comuns.json      lista e ordem dos canais usados em todas as análises
"""
from __future__ import annotations

import json
import warnings

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import welch

from .config import (APERIODICO_FAIXA, BANDAS, BANDAS_CONECTIVIDADE, BETA, CANAIS_BETA, JANELAS_BRUTAS,
                     PREPROC, Caminhos)
from .preprocessamento import carregar_logs

mne.set_log_level("ERROR")
META = ["rec_id", "dataset", "sujeito", "sub", "ses", "grupo", "med", "idade", "sexo", "principal",
        "n_janelas", "duracao_limpa_s"]


def logs_ok(caminhos: Caminhos) -> pd.DataFrame:
    logs = carregar_logs(caminhos)
    if logs.empty:
        raise RuntimeError("nenhum log de pré-processamento; rode `preprocessar` antes")
    return logs[logs["status"] == "ok"].reset_index(drop=True)


def canais_comuns(logs: pd.DataFrame) -> list[str]:
    """Interseção dos canais finais, na ordem do sistema 10-05 (igual em todos os centros)."""
    comum = set.intersection(*[set(c) for c in logs["canais_finais"]])
    ordem = mne.channels.make_standard_montage("standard_1005").ch_names
    return [c for c in ordem if c in comum]


def recortar_janelas(raw, inicios, janela_s):
    sf = raw.info["sfreq"]
    n = int(round(janela_s * sf))
    x = raw.get_data()
    return np.stack([x[:, int(round(t * sf)): int(round(t * sf)) + n] for t in inicios])


def _aperiodico(freqs, psd):
    from fooof import FOOOF

    out = []
    for p in psd:
        fm = FOOOF(aperiodic_mode="fixed", peak_width_limits=(1, 12), max_n_peaks=6, verbose=False)
        try:
            fm.fit(freqs, p, list(APERIODICO_FAIXA))
            out.append(fm.get_params("aperiodic_params", "exponent"))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def _band_mask(freqs, a, b):
    return (freqs >= a) & (freqs < b)


def features_gravacao(linha: dict, caminhos: Caminhos, comuns: list[str], max_janelas: int | None = None,
                      so_tabela: bool = False) -> dict:
    """max_janelas: usa só as primeiras N janelas válidas (sensibilidade com a mesma
    quantidade de sinal em todos os centros). so_tabela: só potência relativa,
    expoente aperiódico e wPLI."""
    raw = mne.io.read_raw_fif(caminhos.preproc / linha["arquivo_saida"], preload=True)
    sf = raw.info["sfreq"]
    jan = linha["janelas_inicio_s"][:max_janelas] if max_janelas else linha["janelas_inicio_s"]
    nper = int(2 * sf)

    # ---------------- espectral (canais comuns)
    xc = recortar_janelas(raw.copy().pick(comuns), jan, PREPROC.janela_s)          # (n_jan, n_ch, n_t)
    freqs, psd = welch(xc, fs=sf, nperseg=nper, noverlap=nper // 2, axis=-1)
    tot = _band_mask(freqs, 1, 45)
    psd_m = np.median(psd, axis=0)                                         # robusto a janela ruidosa
    feats = {}
    for b, (lo, hi) in BANDAS.items():
        rel = psd_m[:, _band_mask(freqs, lo, hi)].sum(1) / psd_m[:, tot].sum(1)
        feats.update({f"rel_{b}_{c}": v for c, v in zip(comuns, np.log10(rel))})
    feats.update({f"exp_{c}": v for c, v in zip(comuns, _aperiodico(freqs, psd_m))})

    # potência relativa por janela (protocolo A)
    jan_rel = []
    for k in range(len(jan)):
        lin = {"rec_id": linha["rec_id"], "janela": k}
        for b, (lo, hi) in BANDAS.items():
            rel = psd[k][:, _band_mask(freqs, lo, hi)].sum(1) / psd[k][:, tot].sum(1)
            lin.update({f"rel_{b}_{c}": v for c, v in zip(comuns, np.log10(rel))})
        jan_rel.append(lin)

    # ---------------- wPLI (canais comuns)
    from mne_connectivity import spectral_connectivity_epochs

    ep = mne.EpochsArray(xc, mne.create_info(comuns, sf, "eeg"), verbose=False)
    fmin = tuple(v[0] for v in BANDAS_CONECTIVIDADE.values())
    fmax = tuple(v[1] for v in BANDAS_CONECTIVIDADE.values())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        con = spectral_connectivity_epochs(ep, method="wpli", mode="multitaper", sfreq=sf, fmin=fmin,
                                           fmax=fmax, faverage=True, verbose=False)
    m = con.get_data(output="dense")                                       # (n_ch, n_ch, n_bandas)
    iu = np.tril_indices(len(comuns), k=-1)
    for k, b in enumerate(BANDAS_CONECTIVIDADE):
        feats.update({f"wpli_{b}_{comuns[j]}_{comuns[i]}": m[i, j, k] for i, j in zip(*iu)})
    if so_tabela:
        return {"feats": feats}

    # ---------------- beta absoluto em C3/C4, referência média e CSD
    beta = []
    canais_b = [c for c in CANAIS_BETA if c in raw.ch_names]
    if canais_b:
        csd = mne.preprocessing.compute_current_source_density(raw.copy())
        for ref, r in (("media", raw), ("csd", csd)):
            xb = recortar_janelas(r.copy().pick(canais_b), jan, PREPROC.janela_s)
            f, p = welch(xb, fs=sf, nperseg=nper, noverlap=nper // 2, axis=-1)
            pb = np.log10(p[:, :, _band_mask(f, *BETA)].mean(-1))           # (n_jan, n_canais)
            for k, t in enumerate(jan):
                lin = {"rec_id": linha["rec_id"], "ref": ref, "janela": k, "inicio_s": t}
                lin.update({c: pb[k, i] for i, c in enumerate(canais_b)})
                if len(canais_b) == 2:
                    lin["C3C4"] = pb[k].mean()
                beta.append(lin)

    # ---------------- janelas brutas (redes) e covariância (riemanniano)
    r125 = raw.copy().pick(comuns).resample(JANELAS_BRUTAS["sfreq"], method="polyphase")
    xb = recortar_janelas(r125, jan, PREPROC.janela_s)
    k = JANELAS_BRUTAS["max_por_gravacao"]
    if len(xb) > k:
        xb = xb[np.linspace(0, len(xb) - 1, k).round().astype(int)]
    from sklearn.covariance import OAS

    cov = OAS().fit(np.concatenate(list(xc), axis=1).T).covariance_

    return {"feats": feats, "jan_rel": jan_rel, "beta": beta, "brutas": xb.astype(np.float32),
            "cov": cov}


def extrair(caminhos: Caminhos, n_jobs: int = 1):
    logs = logs_ok(caminhos)
    comuns = canais_comuns(logs)
    print(f"{len(logs)} gravações ok; {len(comuns)} canais comuns: {comuns}")
    res = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(features_gravacao)(r, caminhos, comuns) for r in logs.to_dict("records"))
    out = caminhos.features
    out.mkdir(parents=True, exist_ok=True)
    meta = logs[[c for c in META if c in logs.columns]]
    grav = pd.concat([meta, pd.DataFrame([r["feats"] for r in res])], axis=1)
    assert len(grav) == len(logs), "gravação ok sem features"
    grav.to_parquet(out / "gravacoes.parquet", index=False)
    pd.DataFrame([j for r in res for j in r["jan_rel"]]).to_parquet(out / "janelas_espectral.parquet", index=False)
    pd.DataFrame([b for r in res for b in r["beta"]]).to_parquet(out / "janelas_beta.parquet", index=False)
    X = np.concatenate([r["brutas"] for r in res])
    ids = np.concatenate([[rid] * len(r["brutas"]) for rid, r in zip(logs["rec_id"], res)])
    np.savez_compressed(out / "janelas_brutas.npz", X=X, rec_id=ids, canais=np.array(comuns))
    np.savez_compressed(out / "covariancias.npz", C=np.stack([r["cov"] for r in res]),
                        rec_id=logs["rec_id"].to_numpy(), canais=np.array(comuns))
    (out / "canais_comuns.json").write_text(json.dumps(comuns))
    print(f"{len(grav)} gravações, {sum(len(r['jan_rel']) for r in res)} janelas; brutas {X.shape}")
    print(grav.groupby(["dataset", "grupo", "med"], dropna=False).size().rename("gravações").to_string())
    return grav


def carregar(caminhos: Caminhos) -> pd.DataFrame:
    return pd.read_parquet(caminhos.features / "gravacoes.parquet")
