"""Roda a pipeline inteira nos dados simulados (cerca de 30 min em 2 núcleos).

Uso: python testes/rodar_teste.py [--nulo] [--raiz dados_teste]
"""
import argparse
import subprocess
import sys
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_PROJETO))

from eeg_pd import cli  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=str(RAIZ_PROJETO / "dados_teste"))
ap.add_argument("--nulo", action="store_true")
ap.add_argument("--n-jobs", default="2")
a = ap.parse_args()

if not (Path(a.raiz) / "bids").exists():
    cmd = [sys.executable, str(RAIZ_PROJETO / "testes" / "simular_dados.py"), "--raiz", a.raiz]
    subprocess.run(cmd + (["--nulo"] if a.nulo else []), check=True)

etapas = [
    ["verificar"], ["preprocessar", "--n-jobs", a.n_jobs], ["features", "--n-jobs", a.n_jobs], ["fluxo"],
    ["controles", "--n-perm", "40", "--n-jobs", a.n_jobs], ["p1", "--n-jobs", a.n_jobs],
    ["p1", "--med", "OFF", "--n-jobs", a.n_jobs], ["profundos", "--max-epocas", "5"], ["comparar"], ["auditoria"],
    ["p2", "--n-perm", "100", "--n-jobs", a.n_jobs], ["pareado"], ["blocos"], ["figuras"],
]
for e in etapas:
    print(f"\n>>> {' '.join(e)}", flush=True)
    cli.main(e + ["--raiz", a.raiz])
print("\nteste completo")
