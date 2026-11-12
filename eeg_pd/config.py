"""Tudo que muda o resultado fica aqui.

Regra: qualquer mudança neste arquivo que afete o pré-processamento muda o hash
gravado no log de cada gravação, e o `preprocessar` refaz as gravações com hash
diferente. Assim não sobra derivado de uma versão antiga da configuração.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Entra no hash do pré-processamento: incremente ao mudar o código de preprocessamento.py
VERSAO_PREPROC = "2.1"


# --------------------------------------------------------------------------- centros
@dataclass
class Dataset:
    id: str
    centro: str
    versao: str
    task: str
    # trechos de repouso: None = gravação inteira; senão, prefixos dos marcadores
    # de olhos abertos (comparados sem espaços e sem pontuação)
    marcadores_olhos_abertos: tuple | None = None
    corte_bordas_s: float = 8.0
    min_limpo_s: float = 60.0
    # canais que não são do sistema 10-05 e devem virar EOG (nome original -> novo)
    eog: dict = field(default_factory=dict)
    frequencia_rede: float = 60.0


DATASETS = {
    "ds002778": Dataset(
        id="ds002778", centro="San Diego", versao="1.0.5", task="rest",
        # EXG3 acompanha Fp1/Fp2 (r = 0,98 nos dados brutos): EOG vertical.
        # EXG4 acompanha F7: EOG horizontal. EXG1-2 (mastoides) e EXG5-8
        # (desconectados) são descartados antes de qualquer referência.
        eog={"EXG3": "VEOG", "EXG4": "HEOG"},
    ),
    "ds003490": Dataset(
        id="ds003490", centro="Novo México", versao="1.1.0", task="Rest",
        marcadores_olhos_abertos=("Eyes Open",),
        # só ~60 s de olhos abertos por sessão: corte e mínimo menores
        corte_bordas_s=2.0, min_limpo_s=45.0,
        eog={"VEOG": "VEOG"},
    ),
    "ds004584": Dataset(
        id="ds004584", centro="Iowa", versao="1.0.0", task="Rest",
    ),
    # 4º centro, só para teste externo (raiz separada, dados_externo/): nunca entra
    # no treino nem na escolha de nada. EGI 64 canais, referência em Cz (VREF),
    # repouso sentado de olhos abertos (~4 min). EOG2 fica abaixo do olho direito
    # (infraorbital): faz o papel do VEOG; EOG1, lateral, do HEOG.
    "ds007526": Dataset(
        id="ds007526", centro="Tel Aviv", versao="1.0.2", task="rest",
        eog={"EOG2": "VEOG", "EOG1": "HEOG"}, frequencia_rede=50.0,
    ),
}
EXTERNOS = ("ds007526",)
# Datasets usados quando nenhum é indicado: só os de treino. O centro externo é sempre
# pedido explicitamente, numa raiz separada, para nunca entrar no treino por engano.
PRINCIPAIS = tuple(d for d in DATASETS if d not in EXTERNOS)

# Sessões ON de pd6 e pd16 vieram de arquivos já pré-processados no EEGLAB
# (nota dos curadores no participants.tsv). Ficam fora da análise principal.
EXCLUIR_DA_PRINCIPAL = {
    "ds002778_sub-pd6_ses-on": "sessão ON vinda de dado já pré-processado no EEGLAB (nota no participants.tsv)",
    "ds002778_sub-pd16_ses-on": "sessão ON vinda de dado já pré-processado no EEGLAB (nota no participants.tsv)",
}


# --------------------------------------------------------------------------- pré-processamento
@dataclass
class Preproc:
    sfreq: float = 250.0
    banda: tuple = (1.0, 45.0)          # sinal analisado
    banda_ica: tuple = (1.0, 100.0)     # cópia usada para ajustar o ICA e o ICLabel
    # canais ruins: z robusto (MAD) do desvio padrão e maior correlação com outro canal
    z_canal_ruim: float = 5.0
    corr_max_min: float = 0.25
    max_frac_canais_ruins: float = 0.2
    # ICA + ICLabel
    ica_max_componentes: int = 40
    iclabel_limiar: float = 0.8
    iclabel_remover: tuple = ("eye blink", "muscle artifact", "heart beat", "line noise", "channel noise")
    # ICA ajustado na gravação inteira (não só no trecho de repouso analisado)
    ica_gravacao_inteira: bool = True
    # rede de segurança para piscadas que o ICLabel deixou abaixo do limiar:
    # |r| >= limiar com o VEOG; sem EOG (Iowa), com Fp1/Fp2 e P(olho) do ICLabel >= mínimo
    eog_corr_limiar: float = 0.7
    eog_proxy_prob_olho_min: float = 0.3
    # janelas
    janela_s: float = 4.0
    pico_a_pico_max_uv: float = 200.0
    semente: int = 97


PREPROC = Preproc()


def hash_preproc(ds: Dataset) -> str:
    """Hash da configuração que define o pré-processamento de um dataset."""
    d = {"versao": VERSAO_PREPROC, "preproc": asdict(PREPROC), "dataset": asdict(ds)}
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- features
BANDAS = {"delta": (1, 4), "teta": (4, 8), "alfa": (8, 13), "beta": (13, 30), "gama": (30, 45)}
BANDAS_CONECTIVIDADE = {"teta": (4, 8), "alfa": (8, 13), "beta": (13, 30)}
APERIODICO_FAIXA = (2.0, 40.0)
BETA = (13.0, 30.0)
CANAIS_BETA = ("C3", "C4")
JANELAS_BRUTAS = {"sfreq": 125.0, "max_por_gravacao": 30}

# --------------------------------------------------------------------------- avaliação
N_FOLDS_SUJEITO = 10
N_FOLDS_INTERNOS = 3
N_BOOTSTRAP = 2000
SEMENTE = 2026

# --------------------------------------------------------------------------- análise pareada
# Parâmetros fixados antes da análise.
REFERENCIA_PRINCIPAL = "csd"   # "csd" (Laplaciano de superfície) ou "media"
# Erro aceitável na diferença ON - OFF, em log10 (0,02 log10 corresponde a ~5% de potência)
MARGEM_ERRO_DELTA = 0.02
# múltiplos da janela de 4 s; no Novo México o sinal limpo tem ~56 s, por isso
# 24 s é o maior valor com metade-metade (2 x 24 s) disponível
DURACOES_S = {"ds002778": (16, 32, 60, 92), "ds003490": (16, 24)}
N_SORTEIOS = 500
BLOCO_S = {"ds002778": 30.0, "ds003490": 15.0}


# --------------------------------------------------------------------------- caminhos
@dataclass
class Caminhos:
    raiz: Path

    @property
    def bids(self) -> Path: return self.raiz / "bids"
    @property
    def deriv(self) -> Path: return self.raiz / "derivados"
    @property
    def preproc(self) -> Path: return self.deriv / "preproc"
    @property
    def features(self) -> Path: return self.deriv / "features"
    @property
    def resultados(self) -> Path: return self.raiz / "resultados"

    def ds(self, ds_id: str) -> Path:
        return self.bids / ds_id
