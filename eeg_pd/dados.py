"""Download, conferência de integridade, inspeção e listagem das gravações."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASETS, EXCLUIR_DA_PRINCIPAL, PRINCIPAIS, Caminhos

EXTENSOES = (".bdf", ".set", ".vhdr", ".edf")


# --------------------------------------------------------------------------- download
def baixar(caminhos: Caminhos, datasets=None, paralelo: int = 2, rodadas: int = 6):
    import openneuro

    for ds_id in datasets or PRINCIPAIS:
        ds = DATASETS[ds_id]
        alvo = caminhos.ds(ds_id)
        alvo.mkdir(parents=True, exist_ok=True)
        print(f"\n===== baixando {ds_id} (versão {ds.versao})", flush=True)
        for r in range(rodadas):
            try:
                openneuro.download(dataset=ds_id, tag=ds.versao, target_dir=alvo,
                                   max_concurrent_downloads=paralelo, max_retries=10)
                break
            except Exception as e:  # rede instável: espera e tenta de novo
                print(f"rodada {r + 1} falhou ({type(e).__name__}); nova tentativa em 60 s", flush=True)
                time.sleep(60)
        (alvo / "versao_openneuro.txt").write_text(f"{ds_id} {ds.versao}\n")


# --------------------------------------------------------------------------- listagem
@dataclass
class Gravacao:
    dataset: str
    sub: str
    ses: str | None
    arquivo: Path
    grupo: str           # PD ou HC
    med: str | None      # ON, OFF ou None
    idade: float | None
    sexo: str | None

    @property
    def sujeito(self) -> str:
        return f"{self.dataset}_sub-{self.sub}"

    @property
    def rec_id(self) -> str:
        return self.sujeito + (f"_ses-{self.ses}" if self.ses else "")

    @property
    def principal(self) -> bool:
        return self.rec_id not in EXCLUIR_DA_PRINCIPAL

    def como_dict(self) -> dict:
        return {"rec_id": self.rec_id, "dataset": self.dataset, "sujeito": self.sujeito, "sub": self.sub,
                "ses": self.ses, "grupo": self.grupo, "med": self.med, "idade": self.idade,
                "sexo": self.sexo, "principal": self.principal, "arquivo": str(self.arquivo)}


def _participantes(raiz_ds: Path) -> pd.DataFrame:
    p = raiz_ds / "participants.tsv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, sep="\t", dtype=str).replace("n/a", np.nan)
    df["sub"] = df["participant_id"].str.replace("sub-", "", regex=False)
    return df.set_index("sub")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _metadados(ds_id: str, sub: str, ses: str | None, part: pd.DataFrame):
    linha = part.loc[sub] if sub in part.index else pd.Series(dtype=object)
    if ds_id == "ds002778":
        grupo = "PD" if sub.startswith("pd") else "HC"
        med = ses.upper() if grupo == "PD" and ses in ("on", "off") else None
        return grupo, med, _num(linha.get("age")), linha.get("gender")
    if ds_id == "ds003490":
        grupo = "PD" if str(linha.get("Group", "")).upper() == "PD" else "HC"
        med = None
        if grupo == "PD" and ses:
            v = str(linha.get(f"sess{int(ses)}_Med", "")).upper()
            med = v if v in ("ON", "OFF") else None
        return grupo, med, _num(linha.get("age")), linha.get("sex")
    if ds_id == "ds004584":
        grupo = "PD" if str(linha.get("GROUP", "")).upper() == "PD" else "HC"
        # Anjum et al. (2024), npj Parkinson's Disease 10:6: EEG coletado em ON
        med = "ON" if grupo == "PD" else None
        return grupo, med, _num(linha.get("AGE")), linha.get("GENDER")
    if ds_id == "ds007526":
        grupo = "PD" if str(linha.get("group", "")).upper() == "PD" else "HC"
        # O dataset não documenta o estado de medicação; os pacientes estavam em
        # tratamento (LEDD registrado em 115 de 116) e são tratados como ON
        med = "ON" if grupo == "PD" else None
        return grupo, med, _num(linha.get("age")), linha.get("sex")
    raise ValueError(ds_id)


_PAD = re.compile(r"sub-(?P<sub>[^_]+)(?:_ses-(?P<ses>[^_]+))?_task-(?P<task>[^_]+)_eeg$")


def listar(caminhos: Caminhos, datasets=None) -> list[Gravacao]:
    out = []
    for ds_id in datasets or PRINCIPAIS:
        raiz = caminhos.ds(ds_id)
        part = _participantes(raiz)
        arquivos = sorted(p for ext in EXTENSOES for p in raiz.glob(f"sub-*/**/eeg/*_eeg{ext}"))
        for f in arquivos:
            m = _PAD.match(f.name[: -len(f.suffix)])
            if not m or m["task"].lower() != DATASETS[ds_id].task.lower():
                continue
            grupo, med, idade, sexo = _metadados(ds_id, m["sub"], m["ses"], part)
            out.append(Gravacao(ds_id, m["sub"], m["ses"], f, grupo, med, idade, sexo))
    return out


def tabela_gravacoes(caminhos: Caminhos, datasets=None) -> pd.DataFrame:
    return pd.DataFrame([g.como_dict() for g in listar(caminhos, datasets)])


# --------------------------------------------------------------------------- integridade
def _bdf_bytes(f: Path) -> tuple[int, int]:
    with open(f, "rb") as fh:
        h = fh.read(256)
        hb, nrec, ns = int(h[184:192]), int(h[236:244]), int(h[252:256])
        fh.seek(256 + ns * 216)
        spr = [int(fh.read(8)) for _ in range(ns)]
    largura = 3 if f.suffix.lower() == ".bdf" else 2
    return hb + nrec * sum(spr) * largura, f.stat().st_size


def checar_integridade(g: Gravacao, tolerancia_s: float = 2.0) -> str | None:
    """Devolve None se o arquivo está íntegro, ou o motivo do problema.

    O MNE lê arquivo truncado sem dar erro (lê só os registros completos), por isso
    a duração é comparada com o JSON do BIDS e, em BDF/EDF, o cabeçalho é comparado
    com o tamanho real do arquivo.
    """
    import mne
    import warnings
    warnings.filterwarnings("ignore", message=".*boundary.*")

    f = g.arquivo
    if not f.exists() or f.stat().st_size == 0:
        return "arquivo ausente ou vazio"
    if f.suffix.lower() in (".bdf", ".edf"):
        esperado, real = _bdf_bytes(f)
        if real < esperado:
            return f"arquivo truncado: faltam {(esperado - real) / 1e6:.2f} MB"
    js =f.with_name(f.name[: -len(f.suffix)] + ".json")
    if not js.exists():
        return "JSON do BIDS ausente"
    dur_json = json.loads(js.read_text()).get("RecordingDuration")
    try:
        raw = mne.io.read_raw(f, preload=False, verbose=False)
    except Exception as e:
        return f"falha na leitura: {type(e).__name__}: {e}"[:200]
    dur = raw.n_times / raw.info["sfreq"]
    if dur_json and dur + tolerancia_s < float(dur_json):
        return f"duração lida ({dur:.0f} s) menor que a do JSON ({float(dur_json):.0f} s)"
    return None


def verificar(caminhos: Caminhos, datasets=None, apagar: bool = False) -> pd.DataFrame:
    """Confere todos os arquivos. Com apagar=True, apaga os arquivos com problema
    (o `baixar` seguinte baixa de novo só o que falta)."""
    linhas = []
    for g in listar(caminhos, datasets):
        linhas.append({"rec_id": g.rec_id, "dataset": g.dataset, "problema": checar_integridade(g),
                       "arquivo": str(g.arquivo)})
    df = pd.DataFrame(linhas)
    if df.empty:
        print("nenhuma gravação encontrada")
        return df
    for ds_id, d in df.groupby("dataset"):
        n_prob = d["problema"].notna().sum()
        print(f"{ds_id}: {len(d)} gravações, " + ("completo" if n_prob == 0 else f"{n_prob} com problema"))
    prob = df[df["problema"].notna()]
    if len(prob):
        print(prob[["rec_id", "problema"]].to_string(index=False))
        if apagar:
            for f in prob["arquivo"]:
                for x in (Path(f), Path(f).with_suffix(".fdt")):
                    if x.exists():
                        x.unlink()
            print(f"\n{len(prob)} arquivo(s) apagado(s); rode `baixar` de novo para baixá-los")
        else:
            print("\nRode `verificar --apagar-problemas` e depois `baixar` de novo")
    return df


# --------------------------------------------------------------------------- inspeção
def inspecionar(caminhos: Caminhos, datasets=None):
    for ds_id in datasets or PRINCIPAIS:
        raiz = caminhos.ds(ds_id)
        print(f"=== {ds_id} ({DATASETS[ds_id].centro}) em {raiz}")
        tab = tabela_gravacoes(caminhos, [ds_id])
        if tab.empty:
            print("  nenhuma gravação")
            continue
        print(tab.groupby(["grupo", "med"], dropna=False).size().rename("gravações").to_string())
        print(f"  sujeitos: {tab['sujeito'].nunique()}  |  idade sem valor: {tab['idade'].isna().sum()}")
        pj = raiz / "participants.json"
        if pj.exists():
            for k, v in json.loads(pj.read_text()).items():
                desc = v.get("Description", "") if isinstance(v, dict) else v
                print(f"  {k}: {desc}")
        part = _participantes(raiz)
        if "notes" in part.columns:
            for sub, nota in part["notes"].dropna().items():
                print(f"  NOTA sub-{sub}: {nota}")
        ch = sorted(raiz.glob("sub-*/**/eeg/*_channels.tsv"))
        if ch:
            c = pd.read_csv(ch[0], sep="\t")
            print(f"  {ch[0].name}: {len(c)} canais; tipos {c['type'].value_counts().to_dict()}")
