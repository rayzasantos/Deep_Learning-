"""Executa a pipeline completa, do pré-processamento às figuras do artigo.

Pré-requisito: os quatro datasets baixados (ver README). O tempo registrado de cada
etapa vai para <raiz>/registro_execucao.csv.

Uso: python rodar_tudo.py [--raiz dados] [--raiz-externa dados_externo] [--n-jobs 11]
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from eeg_pd import cli

# O pré-processamento usa ~1-1,5 GB por processo (ICA na gravação inteira).
N_JOBS_PREPROCESSAMENTO = "3"


def _etapas(n_jobs: str, raiz_externa: str) -> list[list[str]]:
    return [
        ["preprocessar", "--n-jobs", N_JOBS_PREPROCESSAMENTO],
        ["features", "--n-jobs", "6"],
        ["fluxo"],
        ["controles", "--n-perm", "1000", "--n-jobs", n_jobs],
        ["p1", "--med", "ON", "--n-jobs", n_jobs],
        ["p1", "--med", "OFF", "--n-jobs", n_jobs],
        ["profundos", "--modelo", "gcn", "gcn_sem_arestas"],
        ["comparar"],
        ["auditoria"],
        ["p2", "--n-perm", "1000", "--n-jobs", n_jobs],
        ["pareado"],
        ["blocos"],
        ["sensibilidade", "--n-jobs", "6"],
        ["variacoes", "--n-jobs", "6"],
        ["preprocessar", "--datasets", "ds007526", "--n-jobs", N_JOBS_PREPROCESSAMENTO, "--raiz", raiz_externa],
        ["externo", "--n-jobs", "6"],
        ["figuras"],
        ["artigo"],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raiz", default="dados")
    parser.add_argument("--raiz-externa", default="dados_externo")
    parser.add_argument("--n-jobs", default="11")
    args = parser.parse_args()

    registro = []
    for etapa in _etapas(args.n_jobs, args.raiz_externa):
        inicio = time.time()
        print(f">>> {' '.join(etapa)}  ({time.strftime('%H:%M')})", flush=True)
        raiz = [] if "--raiz" in etapa else ["--raiz", args.raiz]
        cli.main(etapa + raiz + ["--raiz-externa", args.raiz_externa])
        registro.append({"etapa": " ".join(etapa), "inicio": time.strftime("%Y-%m-%d %H:%M", time.localtime(inicio)),
                         "minutos": round((time.time() - inicio) / 60, 1)})
        pd.DataFrame(registro).to_csv(Path(args.raiz) / "registro_execucao.csv", index=False)
    print("fim", flush=True)


if __name__ == "__main__":
    main()
