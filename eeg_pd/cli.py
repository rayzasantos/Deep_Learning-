"""Linha de comando: python -m eeg_pd.cli <etapa> [opções]."""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from .config import DATASETS, PRINCIPAIS, Caminhos, hash_preproc

warnings.filterwarnings("ignore")


def _precisa_processar(gravacao, caminhos, refazer) -> bool:
    """Falta o log, a configuração mudou ou a última tentativa falhou por falta de memória."""
    from .preprocessamento import ler_log

    arquivo = caminhos.preproc / f"{gravacao.rec_id}_log.json"
    if refazer or not arquivo.exists():
        return True
    log = ler_log(arquivo)
    faltou_memoria = "MemoryError" in str(log.get("motivo") or "")
    return log.get("hash_config") != hash_preproc(DATASETS[gravacao.dataset]) or faltou_memoria


def preprocessar(caminhos, args):
    from joblib import Parallel, delayed

    from .dados import listar
    from .preprocessamento import carregar_logs, preprocessar_gravacao

    todas = listar(caminhos, args.datasets)
    pendentes = [g for g in todas if _precisa_processar(g, caminhos, args.refazer)]
    print(f"{len(todas)} gravações; {len(todas) - len(pendentes)} já processadas com a configuração atual; "
          f"{len(pendentes)} a processar", flush=True)
    logs = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(preprocessar_gravacao)(g, caminhos) for g in pendentes)
    for log in logs:
        if log["status"] != "ok":
            print(f"  {log['rec_id']}: excluída ({log['etapa_exclusao']}) {log['motivo']}")
    print(carregar_logs(caminhos).groupby(["dataset", "status"]).size().to_string())


def baixar(caminhos, args):
    from .dados import baixar as baixar_datasets
    baixar_datasets(caminhos, args.datasets, args.paralelo)


def verificar(caminhos, args):
    from .dados import verificar as verificar_arquivos
    verificar_arquivos(caminhos, args.datasets, args.apagar_problemas)


def inspecionar(caminhos, args):
    from .dados import inspecionar as inspecionar_datasets
    inspecionar_datasets(caminhos, args.datasets)


def features(caminhos, args):
    from .features import extrair
    extrair(caminhos, args.n_jobs)


def fluxo(caminhos, args):
    from .experimentos import fluxo as gerar_fluxo
    gerar_fluxo(caminhos)


def controles(caminhos, args):
    from .experimentos import rodar_controles
    rodar_controles(caminhos, args.n_perm, args.com_excluidas, args.n_jobs)


def p1(caminhos, args):
    from .experimentos import rodar_p1
    rodar_p1(caminhos, args.med, args.com_excluidas, args.n_jobs, args.so_referencia)


def profundos(caminhos, args):
    from .profundos import rodar
    rodar(caminhos, args.modelo, args.med, args.com_excluidas, args.max_epocas)


def comparar(caminhos, args):
    from .experimentos import comparar as comparar_modelos
    comparar_modelos(caminhos, args.med, args.com_excluidas)


def auditoria(caminhos, args):
    from .experimentos import auditoria as auditar
    auditar(caminhos)


def p2(caminhos, args):
    from .experimentos import rodar_p2
    rodar_p2(caminhos, args.n_perm, args.com_excluidas, args.n_jobs)


def pareado(caminhos, args):
    from .pareado import rodar
    rodar(caminhos, args.com_excluidas)


def blocos(caminhos, args):
    from .pareado import blocos as analisar_blocos
    analisar_blocos(caminhos, args.com_excluidas)


def sensibilidade(caminhos, args):
    from .sensibilidade import rodar
    rodar(caminhos, args.n_jobs)


def variacoes(caminhos, args):
    from .variacoes import rodar
    rodar(caminhos, args.n_jobs)


def externo(caminhos, args):
    from .externo import rodar
    rodar(caminhos, Caminhos(Path(args.raiz_externa)), args.n_jobs)


def figuras(caminhos, args):
    from .figuras import gerar
    gerar(caminhos, args.med)


def artigo(caminhos, args):
    from .figuras_artigo import gerar
    gerar(caminhos, Caminhos(Path(args.raiz_externa)))


ETAPAS = {f.__name__: f for f in (baixar, verificar, inspecionar, preprocessar, features, fluxo, controles, p1,
                                  profundos, comparar, auditoria, p2, pareado, blocos, sensibilidade, variacoes,
                                  externo, figuras, artigo)}


def _argumentos():
    parser = argparse.ArgumentParser(prog="eeg_pd")
    parser.add_argument("etapa", choices=list(ETAPAS))
    parser.add_argument("--raiz", default="dados", help="pasta com bids/, derivados/ e resultados/")
    parser.add_argument("--raiz-externa", default="dados_externo", help="pasta do centro de teste externo")
    parser.add_argument("--datasets", nargs="+", default=list(PRINCIPAIS), choices=list(DATASETS))
    parser.add_argument("--med", default="ON", choices=["ON", "OFF"])
    parser.add_argument("--com-excluidas", action="store_true", help="inclui pd6 e pd16 ON (sensibilidade)")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--paralelo", type=int, default=2, help="downloads simultâneos")
    parser.add_argument("--refazer", action="store_true", help="refaz o pré-processamento de todas as gravações")
    parser.add_argument("--apagar-problemas", action="store_true", help="no verificar: apaga arquivos truncados")
    parser.add_argument("--modelo", nargs="+", default=["gcn", "gcn_sem_arestas", "shallownet", "eegnet", "conformer"])
    parser.add_argument("--max-epocas", type=int, default=100)
    parser.add_argument("--so-referencia", action="store_true", help="P1 só com o modelo de referência")
    return parser


def main(argv=None):
    args = _argumentos().parse_args(argv)
    ETAPAS[args.etapa](Caminhos(Path(args.raiz)), args)


if __name__ == "__main__":
    main()
