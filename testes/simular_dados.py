"""Gera três datasets BIDS falsos que imitam a estrutura dos reais.

- ds002778: BDF a 512 Hz, 32 canais + EXG1-8 (EXG3 = EOG vertical, EXG4 = EOG
  horizontal, EXG5-8 desconectados, com ruído grande), todos com tipo EEG no
  channels.tsv; sessões hc, on e off.
- ds003490: EEGLAB a 500 Hz, nomes em maiúsculas, VEOG e acelerômetros X/Y/Z,
  tipo n/a no channels.tsv; 60 s de olhos abertos com um marcador por segundo,
  60 s de olhos fechados e tons depois; sessões 01 e 02 com medicação no
  participants.tsv.
- ds004584: EEGLAB a 500 Hz, emenda "boundary" no meio, task-Rest_events.json
  com uma lista vazia na raiz (o que quebra o mne-bids).

Com efeito: PD com mais teta e menos beta; OFF com mais beta em C3/C4.
Sem efeito (--nulo): grupos e sessões idênticos.

Uso: python testes/simular_dados.py --raiz dados_teste [--nulo]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")
MONT = mne.channels.make_standard_montage("standard_1005")
POS = MONT.get_positions()["ch_pos"]

SD_CANAIS = ["Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1",
             "Oz", "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8", "FC6", "FC2", "F4", "F8", "AF4", "Fp2",
             "Fz", "Cz"]
NM_CANAIS = SD_CANAIS[:-2] + ["Fz", "Cz", "FCz", "CPz", "C1", "C2", "C5", "C6", "F1", "F2"]


def _pink(n, sf, rng):
    f = np.fft.rfftfreq(n, 1 / sf)
    s = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.maximum(f, 1) ** 0.9
    return np.fft.irfft(s, n)


def _osc(n, sf, f0, rng, bursts=True):
    t = np.arange(n) / sf
    env = np.ones(n)
    if bursts:
        env = np.convolve(rng.random(n) < 2 / sf, np.hanning(int(0.4 * sf)), "same").clip(0, 1) * 0.7 + 0.3
    return env * np.sin(2 * np.pi * (f0 + 0.5 * rng.standard_normal()) * t + rng.random() * 6)


def _mapa(canais, centro, largura=0.04):
    c = np.array(POS[centro])
    return np.array([np.exp(-np.sum((np.array(POS[ch]) - c) ** 2) / (2 * largura ** 2)) for ch in canais])


def sinal(canais, sf, dur, rng, grupo, sessao, ganho, efeito=True):
    n = int(dur * sf)
    nc = len(canais)
    fundo = np.stack([_pink(n, sf, rng) for _ in range(6)])
    mix = rng.standard_normal((nc, 6))
    x = mix @ fundo
    x = x / x.std() * 8
    x += 0.5 * np.stack([_pink(n, sf, rng) for _ in range(nc)]) / 0.5 * 2
    teta = 1.0 * (2.0 if efeito and grupo == "PD" else 1.0)
    beta = 1.0 * (0.7 if efeito and grupo == "PD" else 1.0) * (1.5 if efeito and sessao == "OFF" else 1.0)
    x += 10 * np.outer(_mapa(canais, "Oz", 0.05), _osc(n, sf, 10, rng))
    x += 4 * teta * np.outer(_mapa(canais, "Fz", 0.05), _osc(n, sf, 6, rng))
    for cc in ("C3", "C4"):
        x += 4 * beta * np.outer(_mapa(canais, cc, 0.03), _osc(n, sf, 20, rng))
    x += 0.5 * np.sin(2 * np.pi * 60 * np.arange(n) / sf)
    # piscadas
    blink = np.zeros(n)
    w = np.hanning(int(0.3 * sf))
    for t in np.arange(2, dur - 2, 4) + rng.random(len(np.arange(2, dur - 2, 4))):
        i = int(t * sf)
        blink[i:i + len(w)] += 100 * w
    x += np.outer(_mapa(canais, "Fpz", 0.06), blink)
    return x * ganho * 1e-6, blink * 1e-6


def _salvar(raw, base: Path, fmt, tipos, eventos, dur):
    base.parent.mkdir(parents=True, exist_ok=True)
    arq = base.with_name(base.name + ("_eeg.bdf" if fmt == "bdf" else "_eeg.set"))
    mne.export.export_raw(arq, raw, fmt=fmt, overwrite=True)
    (base.with_name(base.name + "_eeg.json")).write_text(json.dumps(
        {"SamplingFrequency": raw.info["sfreq"], "RecordingDuration": dur, "PowerLineFrequency": 60}))
    pd.DataFrame({"name": raw.ch_names, "type": tipos}).to_csv(base.with_name(base.name + "_channels.tsv"),
                                                              sep="\t", index=False)
    eventos.to_csv(base.with_name(base.name + "_events.tsv"), sep="\t", index=False)


def ds002778(raiz, rng, efeito, n=6):
    d = raiz / "ds002778"
    part = []
    for i in range(1, 2 * n + 1):
        grupo = "HC" if i <= n else "PD"
        sub = f"{'hc' if grupo == 'HC' else 'pd'}{i}"
        part.append({"participant_id": f"sub-{sub}", "age": int(rng.integers(55, 75)), "gender": rng.choice(["f", "m"]),
                     "notes": "n/a"})
        for ses in (["hc"] if grupo == "HC" else ["on", "off"]):
            sf, dur = 512.0, 130.0
            x, blink = sinal(SD_CANAIS, sf, dur, rng, grupo, ses.upper(), 1.0, efeito)
            nt = x.shape[1]
            exg = np.vstack([rng.standard_normal((2, nt)) * 10e-6,                 # mastoides
                             blink[None] * 1.1 + rng.standard_normal((1, nt)) * 3e-6,  # VEOG
                             rng.standard_normal((1, nt)) * 8e-6,                  # HEOG
                             rng.standard_normal((4, nt)) * 100e-6])                # desconectados
            dados = np.vstack([x, exg]) + rng.standard_normal((40, 1)) * 5e-3      # offset DC do BDF
            raw = mne.io.RawArray(dados, mne.create_info(SD_CANAIS + [f"EXG{k}" for k in range(1, 9)], sf, "eeg"))
            ev = pd.DataFrame({"onset": [0.0, 6.5], "duration": [0, 0], "value": [65536, 1]})
            _salvar(raw, d / f"sub-{sub}" / f"ses-{ses}" / "eeg" / f"sub-{sub}_ses-{ses}_task-rest", "bdf",
                    ["EEG"] * 40, ev, dur)
    pd.DataFrame(part).to_csv(d / "participants.tsv", sep="\t", index=False)


def ds003490(raiz, rng, efeito, n=6):
    d = raiz / "ds003490"
    nomes = [c.upper() for c in NM_CANAIS] + ["VEOG", "X", "Y", "Z"]
    part = []
    for i in range(1, 2 * n + 1):
        grupo = "CTL" if i <= n else "PD"
        sub = f"{i:03d}"
        meds = ["ON", "OFF"] if rng.random() < .5 else ["OFF", "ON"]
        part.append({"participant_id": f"sub-{sub}", "Group": grupo, "sess1_Med": meds[0] if grupo == "PD" else "OFF",
                     "sess2_Med": meds[1] if grupo == "PD" else "no s2", "sex": rng.choice(["Female", "Male"]),
                     "age": int(rng.integers(55, 75))})
        for s in ([1] if grupo == "CTL" else [1, 2]):
            sf, dur = 500.0, 175.0
            med = meds[s - 1] if grupo == "PD" else None
            x, blink = sinal(NM_CANAIS, sf, dur, rng, "PD" if grupo == "PD" else "HC", med, 1.3, efeito)
            nt = x.shape[1]
            dados = np.vstack([x, blink[None] + rng.standard_normal((1, nt)) * 3e-6, rng.standard_normal((3, nt)) * 1e-3])
            raw = mne.io.RawArray(dados, mne.create_info(nomes, sf, "eeg"))
            on = [5.0 + k for k in range(60)] + [65.0 + k for k in range(60)] + list(np.arange(130, 170, 1.3))
            tt = (["Eyes Open: Every 1000 ms"] * 60 + ["Eyes Closed: Every 1000 ms"] * 60
                  + ["Standard Tone"] * len(np.arange(130, 170, 1.3)))
            ev = pd.DataFrame({"onset": on, "duration": 0, "trial_type": tt})
            raw.set_annotations(mne.Annotations(on, 0, tt))
            _salvar(raw, d / f"sub-{sub}" / f"ses-{s:02d}" / "eeg" / f"sub-{sub}_ses-{s:02d}_task-Rest", "eeglab",
                    ["n/a"] * len(nomes), ev, dur)
    pd.DataFrame(part).to_csv(d / "participants.tsv", sep="\t", index=False)


def ds004584(raiz, rng, efeito, n_hc=6, n_pd=10):
    d = raiz / "ds004584"
    part = []
    for i in range(1, n_hc + n_pd + 1):
        grupo = "Control" if i <= n_hc else "PD"
        sub = f"{i:03d}"
        part.append({"participant_id": f"sub-{sub}", "GROUP": grupo, "AGE": int(rng.integers(55, 75)),
                     "GENDER": rng.choice(["F", "M"]), "TYPE": int(grupo == "PD")})
        sf, dur = 500.0, 110.0
        x, _ = sinal(NM_CANAIS, sf, dur, rng, "PD" if grupo == "PD" else "HC", "ON", 0.8, efeito)
        raw = mne.io.RawArray(x, mne.create_info(NM_CANAIS, sf, "eeg"))
        raw.set_annotations(mne.Annotations([55.0], [0.0], ["boundary"]))
        ev = pd.DataFrame({"onset": [55.0], "duration": [0.0], "value": ["boundary"]})
        _salvar(raw, d / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-Rest", "eeglab", ["n/a"] * len(NM_CANAIS), ev, dur)
    pd.DataFrame(part).to_csv(d / "participants.tsv", sep="\t", index=False)
    (d / "task-Rest_events.json").write_text("[]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default="dados_teste")
    ap.add_argument("--nulo", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    raiz = Path(a.raiz) / "bids"
    rng = np.random.default_rng(1)
    ds002778(raiz, rng, not a.nulo, a.n)
    ds003490(raiz, rng, not a.nulo, a.n)
    ds004584(raiz, rng, not a.nulo, a.n, int(1.7 * a.n))
    print("dados simulados em", raiz)
