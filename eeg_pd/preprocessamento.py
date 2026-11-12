"""Pré-processamento de uma gravação, com log JSON.

Ordem das etapas (é a ordem que vai para os métodos):

 1. Conferência de integridade do arquivo (duração e cabeçalho).
 2. Leitura direta do arquivo; eventos tirados do events.tsv do BIDS.
 3. Seleção de canais: só nomes do sistema 10-05 viram EEG; os canais listados em
    `eog` viram EOG; todo o resto (EXG desconectados, mastoides, acelerômetros,
    Status) sai ANTES de qualquer referência.
 4. Trechos de repouso (olhos abertos), corte das bordas e emendas "boundary".
 5. Duas cópias filtradas na taxa original e reamostradas para 250 Hz:
    a) 1-45 Hz FIR, o sinal analisado;
    b) notch na frequência da rede + 1-100 Hz, usada só para ajustar o ICA e
       rodar o ICLabel (faixa em que o ICLabel foi treinado).
 6. Canais ruins: z robusto (MAD) do desvio padrão e maior correlação com
    outro canal, em janelas de 2 s, com referência na mediana dos canais.
 7. Referência média (sem os canais ruins) nas duas cópias.
 8. ICA infomax estendido na cópia 1-100 Hz, número de componentes igual ao
    posto dos dados (limitado a 40); ICLabel com limiar fixo.
 9. Remoção dos mesmos componentes na cópia 1-45 Hz, interpolação dos canais
    ruins e nova referência média.
10. Janelas de 4 s dentro do repouso, sem emenda e com pico a pico abaixo do
    limite. A gravação entra se tiver o mínimo de sinal limpo do dataset.
"""
from __future__ import annotations

import json
import re
import time
import traceback
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .config import DATASETS, PREPROC, Caminhos, hash_preproc
from .dados import Gravacao, checar_integridade

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", message=".*boundary.*")


class Exclusao(Exception):
    """Motivo de exclusão de uma gravação (vai para o log e para o fluxograma)."""

    def __init__(self, etapa: str, motivo: str):
        super().__init__(motivo)
        self.etapa, self.motivo = etapa, motivo


# --------------------------------------------------------------------------- anotações
def _rel(raw) -> np.ndarray:
    """Início das anotações em segundos, na escala de raw.times."""
    # Com ou sem orig_time, o MNE conta o início a partir da amostra 0 da gravação
    # original; depois de um crop, raw.times recomeça em 0 e first_time guarda o deslocamento.
    return np.asarray(raw.annotations.onset, float) - raw.first_time


def _anotar(raw, inicios, duracoes, descricoes):
    if len(inicios) == 0:
        return
    on = np.asarray(inicios, float) + raw.first_time
    raw.set_annotations(raw.annotations + mne.Annotations(on, duracoes, descricoes,
                                                          orig_time=raw.annotations.orig_time))


def _limpa(code: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(code).lower())


def _eventos_bids(g: Gravacao, raw):
    """Troca as anotações pelas do events.tsv (fonte oficial no BIDS), mantendo as emendas."""
    ev = g.arquivo.with_name(g.arquivo.name.replace("_eeg" + g.arquivo.suffix, "_events.tsv"))
    if not ev.exists():
        return
    df = pd.read_csv(ev, sep="\t")
    col = "trial_type" if "trial_type" in df.columns else "value"
    df = df[pd.to_numeric(df["onset"], errors="coerce").notna()]
    desc = df[col].astype(str).tolist()
    dur = pd.to_numeric(df.get("duration", 0), errors="coerce").fillna(0).clip(lower=0).to_numpy()
    on = df["onset"].astype(float).to_numpy()
    emendas = [(o, d) for o, d, s in zip(_rel(raw), raw.annotations.duration, raw.annotations.description)
               if _limpa(s) == "boundary"]
    raw.set_annotations(mne.Annotations([], [], [], orig_time=raw.annotations.orig_time))
    _anotar(raw, on, dur, desc)
    ja = {round(o, 2) for o, s in zip(_rel(raw), raw.annotations.description) if _limpa(s) == "boundary"}
    novas = [(o, d) for o, d in emendas if round(o, 2) not in ja]
    _anotar(raw, [o for o, _ in novas], [d for _, d in novas], ["boundary"] * len(novas))


def _converter_emendas(raw) -> int:
    """'boundary' do EEGLAB vira 'EDGE boundary' (o filtro trata cada lado separado)
    e 'BAD boundary' (nenhuma janela cruza a emenda)."""
    ann = raw.annotations
    idx = [i for i, s in enumerate(ann.description) if _limpa(s) == "boundary"]
    if not idx:
        return 0
    on = _rel(raw)[idx]
    novas = ann.copy()
    novas.delete(idx)
    raw.set_annotations(novas)
    _anotar(raw, on, [0.0] * len(on), ["BAD boundary"] * len(on))
    _anotar(raw, on, [0.0] * len(on), ["EDGE boundary"] * len(on))
    return len(idx)


# --------------------------------------------------------------------------- canais
def selecionar_canais(raw, eog_map: dict) -> dict:
    mont = mne.channels.make_standard_montage("standard_1005")
    canon = {n.lower(): n for n in mont.ch_names}
    renomear, tipos, descartar = {}, {}, []
    # EGI: o canal de referência (VREF) fica no vértex. Gravado como zeros, ele É o
    # Cz; mantido, a referência média o reconstrói corretamente.
    if "VREF" in raw.ch_names and "Cz" not in raw.ch_names:
        raw.rename_channels({"VREF": "Cz"})
    for ch in raw.ch_names:
        if ch in eog_map:
            renomear[ch] = eog_map[ch]
            tipos[eog_map[ch]] = "eog"
        elif ch.lower() in canon:
            renomear[ch] = canon[ch.lower()]
            tipos[canon[ch.lower()]] = "eeg"
        else:
            descartar.append(ch)
    raw.drop_channels(descartar)
    raw.rename_channels(renomear)
    raw.set_channel_types(tipos)
    raw.set_montage(mont, on_missing="ignore")
    return {"canais_descartados": descartar, "canais_eog": sorted(k for k, v in tipos.items() if v == "eog"),
            "n_canais_eeg": sum(v == "eeg" for v in tipos.values())}


def canais_ruins(raw, z_lim: float, corr_min: float, janela_s: float = 2.0) -> list[str]:
    """Canais ruins por dois critérios, calculados em janelas de 2 s do repouso e
    com referência na mediana dos canais (o BDF vem sem referência):

    - desvio: z robusto (MAD) da mediana, entre janelas, do log do desvio padrão.
      Usar a mediana entre janelas evita marcar Fp1/Fp2 só por causa das piscadas,
      que são transitórias e ficam para o ICA;
    - correlação: mediana, entre janelas, da maior correlação absoluta com qualquer
      outro canal. Um canal solto não se correlaciona com nenhum.
    """
    picks = mne.pick_types(raw.info, eeg=True)
    nomes = np.array([raw.ch_names[i] for i in picks])
    x = raw.get_data(picks=picks, reject_by_annotation="omit")
    x = x - np.median(x, axis=0, keepdims=True)
    n = int(janela_s * raw.info["sfreq"])
    w = x[:, : x.shape[1] // n * n].reshape(len(nomes), -1, n)
    sd = w.std(axis=2)
    lsd = np.median(np.log(np.maximum(sd, 1e-12)), axis=1)
    mad = 1.4826 * np.median(np.abs(lsd - np.median(lsd))) or 1e-6
    z = (lsd - np.median(lsd)) / mad
    maxcorr = []
    for k in range(w.shape[1]):
        c = np.abs(np.corrcoef(w[:, k]))
        np.fill_diagonal(c, 0)
        maxcorr.append(np.nan_to_num(c).max(axis=1))
    maxcorr = np.median(np.array(maxcorr), axis=0)
    ruim = (np.abs(z) > z_lim) | (np.median(sd, axis=1) < 0.5e-6) | (maxcorr < corr_min)
    return sorted(nomes[ruim].tolist())


# --------------------------------------------------------------------------- repouso e janelas
def trechos_repouso(raw, ds) -> list[tuple[float, float]]:
    fim = raw.times[-1]
    if ds.marcadores_olhos_abertos is None:
        segs = [(0.0, fim)]
    else:
        pref = [_limpa(p) for p in ds.marcadores_olhos_abertos]
        t = np.sort([o for o, s in zip(_rel(raw), raw.annotations.description)
                     if any(_limpa(s).startswith(p) for p in pref)])
        segs = []
        if len(t):
            # marcadores repetidos (um por segundo): cada sequência com intervalos de
            # até 2 s é um trecho; o trecho termina 1 s depois do último marcador
            quebras = np.where(np.diff(t) > 2.0)[0]
            ini = np.r_[0, quebras + 1]
            fins = np.r_[quebras, len(t) - 1]
            segs = [(t[a], min(t[b] + 1.0, fim)) for a, b in zip(ini, fins)]
    c = ds.corte_bordas_s
    return [(a + c, b - c) for a, b in segs if b - a > 2 * c]


def marcar_fora_do_repouso(raw, segs):
    fim = raw.times[-1]
    bordas = [0.0] + [v for s in segs for v in s] + [fim]
    on, du = [], []
    for a, b in zip(bordas[::2], bordas[1::2]):
        if b - a > 1e-3:
            on.append(a); du.append(b - a)
    _anotar(raw, on, du, ["BAD_fora_repouso"] * len(on))


def janelas_validas(raw, segs, janela_s: float, ptp_max: float) -> list[float]:
    # anotações BAD; as de duração zero só contam se forem emendas (o crop deixa
    # restos de duração zero nas bordas do trecho de repouso)
    bad = [(o, o + d) for o, d, s in zip(_rel(raw), raw.annotations.duration, raw.annotations.description)
           if s.upper().startswith("BAD") and (d > 0 or "boundary" in s.lower())]
    sf = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True)
    x = raw.get_data(picks=picks)
    n = int(round(janela_s * sf))
    out = []
    for a, b in segs:
        t = a
        while t + janela_s <= b + 1e-9:
            cruza = any((o < t + janela_s and e > t) or (o == e and t <= o < t + janela_s) for o, e in bad)
            if not cruza:
                i = int(round(t * sf))
                w = x[:, i:i + n]
                if w.shape[1] == n and np.ptp(w, axis=1).max() < ptp_max * 1e-6:
                    out.append(round(float(t), 4))
            t += janela_s
    return out


# --------------------------------------------------------------------------- ICA
def ajustar_ica(raw_ica, cfg):
    picks = mne.pick_types(raw_ica.info, eeg=True, exclude="bads")
    # referência média sobre os canais bons: posto = n_bons - 1
    n_comp = int(min(len(picks) - 1, cfg.ica_max_componentes))
    ica = mne.preprocessing.ICA(n_components=n_comp, method="infomax", fit_params=dict(extended=True),
                                max_iter="auto", random_state=cfg.semente)
    try:
        ica.fit(raw_ica, picks=picks, reject_by_annotation=True, reject=dict(eeg=500e-6), tstep=2.0)
    except Exception as e:
        x = raw_ica.get_data(picks=picks, reject_by_annotation="omit")
        v = x.var(axis=1)
        i = int(np.argmax(v))
        raise Exclusao("ica", f"{type(e).__name__}: {str(e)[:120]} | canal com maior variância: "
                              f"{raw_ica.ch_names[picks[i]]} ({100 * v[i] / v.sum():.1f}% do total)")
    return ica


ICLABEL_CLASSES = ["brain", "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise", "other"]


def _corr_eog(raw_ica, ica, canais) -> np.ndarray:
    """|r| entre cada componente e o(s) canal(is) de referência ocular, no sinal
    filtrado em 1-10 Hz (padrão do find_bads_eog). Com vários canais, o máximo."""
    _, sc = ica.find_bads_eog(raw_ica, ch_name=canais, measure="correlation", threshold=0.99)
    sc = np.abs(np.atleast_2d(np.asarray(sc)))
    return sc.max(axis=0)


def rotular_e_remover(raw_ica, ica, cfg) -> dict:
    from mne_icalabel.iclabel import iclabel_label_components

    prob = np.asarray(iclabel_label_components(raw_ica, ica, inplace=False))      # (n_comp, 7)
    labels = [ICLABEL_CLASSES[k] for k in prob.argmax(1)]
    pmax = prob.max(1)
    remover = [i for i, (l, p) in enumerate(zip(labels, pmax)) if l in cfg.iclabel_remover and p >= cfg.iclabel_limiar]
    info = {"n_componentes_ica": int(ica.n_components_),
            "removidos_por_tipo": pd.Series([labels[i] for i in remover]).value_counts().to_dict()}

    # rede de segurança para piscadas: componente fortemente correlacionado com o EOG
    # (ou, sem EOG, com Fp1/Fp2) que o ICLabel não passou do limiar.
    if "VEOG" in raw_ica.ch_names:
        ref, proxy = "VEOG", False
    else:
        ref = [c for c in ("Fp1", "Fp2") if c in raw_ica.ch_names and c not in raw_ica.info["bads"]]
        proxy = True
    info["eog_referencia"] = ref if isinstance(ref, str) else "/".join(ref) or None
    extra = []
    if ref:
        try:
            sc = _corr_eog(raw_ica, ica, ref)
            p_olho = prob[:, ICLABEL_CLASSES.index("eye blink")]
            for i in range(len(sc)):
                if i in remover or sc[i] < cfg.eog_corr_limiar:
                    continue
                # com o canal frontal como substituto, exige que o ICLabel também veja olho
                if proxy and p_olho[i] < cfg.eog_proxy_prob_olho_min:
                    continue
                extra.append(i)
            remover = sorted(remover + extra)
            info["corr_eog_max_removido"] = float(sc[remover].max()) if remover else None
            info["corr_eog_max_mantido"] = float(np.delete(sc, remover).max()) if len(sc) > len(remover) else None
            info["removidos_por_correlacao_eog"] = [{"componente": int(i), "r": round(float(sc[i]), 3),
                                                     "p_olho_iclabel": round(float(p_olho[i]), 3)} for i in extra]
        except Exception as e:
            info["erro_corr_eog"] = f"{type(e).__name__}: {e}"[:200]
    ica.exclude = remover
    info["componentes_removidos"] = len(remover)
    if extra:
        info["removidos_por_tipo"]["eye blink (correlacao EOG)"] = len(extra)
    return info


# --------------------------------------------------------------------------- principal
ANOTACOES_SEM_FILTRO = ("edge", "bad_acq_skip")
MIN_CANAIS_EEG = 16


def _ler_gravacao(gravacao: Gravacao, dataset, log: dict):
    """Etapas 1-3: integridade, leitura, eventos do BIDS, canais e emendas."""
    problema = checar_integridade(gravacao)
    if problema:
        raise Exclusao("integridade", problema)
    raw = mne.io.read_raw(gravacao.arquivo, preload=True, verbose=False)
    log["sfreq_original"] = float(raw.info["sfreq"])
    _eventos_bids(gravacao, raw)
    log.update(selecionar_canais(raw, dataset.eog))
    if log["n_canais_eeg"] < MIN_CANAIS_EEG:
        raise Exclusao("canais", f"só {log['n_canais_eeg']} canais EEG reconhecidos")
    log["emendas"] = _converter_emendas(raw)
    return raw


def _separar_repouso(raw, dataset, cfg, log: dict):
    """Etapa 4: trechos de repouso. Devolve a cópia usada no ICA e os trechos.

    A cópia do ICA é feita antes de marcar o que fica fora do repouso, para que o ICA
    seja ajustado na gravação inteira (só emendas e trechos acima de 500 µV ficam de
    fora). No ds003490 o repouso de olhos abertos tem ~56 s, pouco para 40 componentes.
    """
    trechos = trechos_repouso(raw, dataset)
    log["duracao_total_s"] = round(raw.times[-1], 2)
    log["duracao_repouso_s"] = round(sum(fim - inicio for inicio, fim in trechos), 2)
    if log["duracao_repouso_s"] < dataset.min_limpo_s:
        raise Exclusao("repouso", f"repouso curto demais ({log['duracao_repouso_s']:.0f} s)")
    raw_ica = raw.copy() if cfg.ica_gravacao_inteira else None
    marcar_fora_do_repouso(raw, trechos)
    log["duracao_ica_s"] = round(raw.times[-1] if cfg.ica_gravacao_inteira else log["duracao_repouso_s"], 1)
    return (raw_ica if raw_ica is not None else raw.copy()), trechos


def _filtrar_e_reamostrar(raw, raw_ica, dataset, cfg):
    """Etapa 5: notch e 1-100 Hz na cópia do ICA, 1-45 Hz no sinal analisado, na taxa
    original; depois as duas cópias vão para a taxa comum."""
    nyquist = raw.info["sfreq"] / 2
    harmonicos = list(np.arange(dataset.frequencia_rede,
                                min(nyquist - 1, cfg.banda_ica[1] + dataset.frequencia_rede),
                                dataset.frequencia_rede))
    if harmonicos:
        raw_ica.notch_filter(harmonicos, picks=["eeg", "eog"], skip_by_annotation=ANOTACOES_SEM_FILTRO)
    raw_ica.filter(*cfg.banda_ica, picks=["eeg", "eog"], skip_by_annotation=ANOTACOES_SEM_FILTRO)
    raw.filter(*cfg.banda, picks=["eeg", "eog"], skip_by_annotation=ANOTACOES_SEM_FILTRO)
    for copia in (raw, raw_ica):
        copia.resample(cfg.sfreq, method="polyphase")


def _marcar_ruins_e_referenciar(raw, raw_ica, cfg, log: dict) -> list[str]:
    """Etapas 6-7: canais ruins e referência média sem eles, nas duas cópias."""
    ruins = canais_ruins(raw, cfg.z_canal_ruim, cfg.corr_max_min)
    log["canais_ruins"] = ruins
    if len(ruins) > cfg.max_frac_canais_ruins * log["n_canais_eeg"]:
        raise Exclusao("canais_ruins", f"{len(ruins)} canais ruins de {log['n_canais_eeg']}")
    for copia in (raw, raw_ica):
        copia.info["bads"] = list(ruins)
        copia.set_eeg_reference("average", projection=False)
    return ruins


def _remover_artefatos(raw, raw_ica, ruins, cfg, log: dict):
    """Etapas 8-9: ICA na cópia 1-100 Hz, remoção na cópia 1-45 Hz, interpolação e
    nova referência média. Ao final só restam os canais EEG."""
    ica = ajustar_ica(raw_ica, cfg)
    log.update(rotular_e_remover(raw_ica, ica, cfg))
    ica.apply(raw)
    if ruins:
        raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference("average", projection=False)
    raw.drop_channels([c for c, tipo in zip(raw.ch_names, raw.get_channel_types()) if tipo != "eeg"])


def _janelar(raw, trechos, dataset, cfg, log: dict):
    """Etapa 10: recorta o trecho que contém o repouso e registra as janelas válidas."""
    inicio, fim = trechos[0][0], trechos[-1][1]
    raw.crop(inicio, min(fim, raw.times[-1]))
    trechos = [(a - inicio, b - inicio) for a, b in trechos]
    janelas = janelas_validas(raw, trechos, cfg.janela_s, cfg.pico_a_pico_max_uv)
    log["n_janelas"] = len(janelas)
    log["duracao_limpa_s"] = len(janelas) * cfg.janela_s
    log["janelas_inicio_s"] = janelas
    log["segmentos_repouso_s"] = [[round(a, 3), round(b, 3)] for a, b in trechos]
    log["canais_finais"] = raw.ch_names
    if log["duracao_limpa_s"] < dataset.min_limpo_s:
        raise Exclusao("sinal_limpo",
                       f"{log['duracao_limpa_s']:.0f} s limpos (mínimo {dataset.min_limpo_s:.0f} s)")


def _gravar_log(caminhos: Caminhos, log: dict):
    caminhos.preproc.mkdir(parents=True, exist_ok=True)
    (caminhos.preproc / f"{log['rec_id']}_log.json").write_text(
        json.dumps(log, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def preprocessar_gravacao(gravacao: Gravacao, caminhos: Caminhos) -> dict:
    """Pré-processa uma gravação, salva o sinal limpo (se aprovada) e o log JSON."""
    dataset, cfg = DATASETS[gravacao.dataset], PREPROC
    inicio = time.time()
    log = {**{k: v for k, v in gravacao.como_dict().items() if k != "arquivo"},
           "hash_config": hash_preproc(dataset), "status": "excluido", "etapa_exclusao": None, "motivo": None}
    saida = caminhos.preproc / f"{gravacao.rec_id}_raw.fif"
    try:
        raw = _ler_gravacao(gravacao, dataset, log)
        raw_ica, trechos = _separar_repouso(raw, dataset, cfg, log)
        _filtrar_e_reamostrar(raw, raw_ica, dataset, cfg)
        ruins = _marcar_ruins_e_referenciar(raw, raw_ica, cfg, log)
        _remover_artefatos(raw, raw_ica, ruins, cfg, log)
        del raw_ica
        _janelar(raw, trechos, dataset, cfg, log)
        saida.parent.mkdir(parents=True, exist_ok=True)
        raw.save(saida, fmt="single", overwrite=True)
        log.update(status="ok", arquivo_saida=saida.name)
    except Exclusao as e:
        log.update(etapa_exclusao=e.etapa, motivo=e.motivo)
    except Exception as e:
        log.update(etapa_exclusao="erro", motivo=f"{type(e).__name__}: {e}"[:300],
                   traceback=traceback.format_exc()[-1500:])
    log["tempo_s"] = round(time.time() - inicio, 1)
    if log["status"] != "ok" and saida.exists():
        saida.unlink()
    _gravar_log(caminhos, log)
    return log


def ler_log(p: Path) -> dict:
    """Lê um log em UTF-8, aceitando também logs gravados na codificação do Windows."""
    b = p.read_bytes()
    try:
        return json.loads(b.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(b.decode("cp1252"))


def carregar_logs(caminhos: Caminhos) -> pd.DataFrame:
    linhas = [ler_log(p) for p in sorted(caminhos.preproc.glob("*_log.json"))]
    return pd.DataFrame(linhas)
