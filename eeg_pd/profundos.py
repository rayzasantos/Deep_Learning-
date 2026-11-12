"""Redes (ShallowNet, EEGNet, EEGConformer) e GCN, nos mesmos folds da P1.

Sem busca de hiperparâmetros: valores publicados de cada arquitetura, fixados
antes de rodar. O único ajuste é o early stopping, numa validação de ~20% dos
sujeitos do treino. Escala dos canais e padronização das features da GCN usam
só o treino. A previsão de uma gravação é a média das probabilidades das janelas.
"""
from __future__ import annotations

import json
import time

import mne
import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import SEMENTE, Caminhos
from .experimentos import dados_p1, folds_p1, nome_config, pasta_p1
from .features import carregar

LR = {"shallownet": 6.25e-4, "eegnet": 1e-3, "conformer": 2e-4, "gcn": 1e-2, "gcn_sem_arestas": 1e-2}
BRUTOS = ("shallownet", "eegnet", "conformer")


def _rede(nome, n_ch, n_t):
    from braindecode.models import EEGConformer, EEGNet, ShallowFBCSPNet

    if nome == "shallownet":
        return ShallowFBCSPNet(n_chans=n_ch, n_outputs=2, n_times=n_t, final_conv_length="auto")
    if nome == "eegnet":
        return EEGNet(n_chans=n_ch, n_outputs=2, n_times=n_t)
    if nome == "conformer":
        return EEGConformer(n_chans=n_ch, n_outputs=2, n_times=n_t, final_fc_length="auto")
    raise ValueError(nome)


class GCN(nn.Module):
    """GCN de 2 camadas com grafo fixo (4 vizinhos mais próximos no escalpo)."""

    def __init__(self, a_norm: torch.Tensor, n_in: int, oculta: int = 32):
        super().__init__()
        self.register_buffer("a", a_norm)
        self.l1, self.l2 = nn.Linear(n_in, oculta), nn.Linear(oculta, oculta)
        self.saida = nn.Linear(oculta, 2)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):  # x: (lote, canais, features)
        h = torch.relu(self.a @ self.l1(x))
        h = torch.relu(self.a @ self.l2(self.drop(h)))
        return self.saida(h.mean(1))


def grafo(canais, k=4, sem_arestas=False):
    if sem_arestas:
        return torch.eye(len(canais))
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    p = np.array([pos[c] for c in canais])
    d = np.linalg.norm(p[:, None] - p[None], axis=2)
    a = np.zeros_like(d)
    for i in range(len(canais)):
        a[i, np.argsort(d[i])[1:k + 1]] = 1
    a = np.maximum(a, a.T) + np.eye(len(canais))
    dinv = np.diag(1 / np.sqrt(a.sum(1)))
    return torch.tensor(dinv @ a @ dinv, dtype=torch.float32)


def _treinar(rede, Xtr, ytr, Xva, yva, lr, max_epocas, paciencia, lote=64, betas=(0.9, 0.999)):
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    rede = rede.to(disp)
    opt = torch.optim.Adam(rede.parameters(), lr=lr, betas=betas)
    peso = torch.tensor(len(ytr) / (2 * np.bincount(ytr, minlength=2).clip(1)), dtype=torch.float32, device=disp)
    perda = nn.CrossEntropyLoss(weight=peso)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t, yva_t = torch.tensor(Xva).to(disp), torch.tensor(yva).to(disp)
    melhor, estado, sem_melhora, ep = np.inf, None, 0, 0
    g = torch.Generator().manual_seed(SEMENTE)
    for ep in range(1, max_epocas + 1):
        rede.train()
        for idx in torch.randperm(len(Xtr_t), generator=g).split(lote):
            opt.zero_grad()
            perda(rede(Xtr_t[idx].to(disp)), ytr_t[idx].to(disp)).backward()
            opt.step()
        rede.eval()
        with torch.no_grad():
            pv = sum(perda(rede(Xva_t[i]), yva_t[i]).item() * len(i)
                     for i in torch.arange(len(Xva_t)).split(256)) / len(Xva_t)
        if pv < melhor - 1e-4:
            melhor, estado, sem_melhora = pv, {k: v.detach().clone() for k, v in rede.state_dict().items()}, 0
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                break
    rede.load_state_dict(estado)
    return rede, ep, melhor


def _prever(rede, X):
    disp = next(rede.parameters()).device
    rede.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(rede(torch.tensor(X[i]).to(disp)), 1)[:, 1].cpu().numpy()
                               for i in np.array_split(np.arange(len(X)), max(1, len(X) // 256))])


def rodar(caminhos: Caminhos, modelos, med="ON", com_excluidas=False, max_epocas=100, paciencia=15):
    torch.manual_seed(SEMENTE)
    grav = carregar(caminhos)
    d = dados_p1(grav, med, com_excluidas)
    pasta = pasta_p1(caminhos, med, com_excluidas)
    folds = folds_p1(caminhos, d, med, com_excluidas)
    lab = dict(zip(d["rec_id"], d["y"]))
    suj = dict(zip(d["rec_id"], d["sujeito"]))
    canais = json.loads((caminhos.features / "canais_comuns.json").read_text())

    if any(m in BRUTOS for m in modelos):
        z = np.load(caminhos.features / "janelas_brutas.npz", allow_pickle=True)
        m = np.isin(z["rec_id"], d["rec_id"])
        Xw, rid = z["X"][m] * 1e6, z["rec_id"][m]          # microvolts
        sw = np.array([suj[r] for r in rid])
        yw = np.array([lab[r] for r in rid])
    feats_no = np.stack([d[[f"rel_{b}_{c}" for b in ("delta", "teta", "alfa", "beta", "gama")] + [f"exp_{c}"]]
                         .to_numpy(float) for c in canais], axis=1)       # (sujeitos, canais, 6)

    for nome in modelos:
        print(f"===== {nome}", flush=True)
        for prot, fl in folds.items():
            linhas, execs = [], []
            for k, (tr, te) in enumerate(fl):
                rng = np.random.default_rng(SEMENTE + k)
                tr = np.array(tr)
                ytr_s = d.set_index("sujeito").loc[tr, "y"].to_numpy()
                va = np.concatenate([rng.choice(tr[ytr_s == c], max(1, int(.2 * (ytr_s == c).sum())), replace=False)
                                     for c in (0, 1)])
                tr2 = np.setdiff1d(tr, va)
                t0 = time.time()
                if nome in BRUTOS:
                    mtr, mva, mte = np.isin(sw, tr2), np.isin(sw, va), np.isin(sw, te)
                    esc = Xw[mtr].std(axis=(0, 2), keepdims=True)[0] + 1e-6      # escala por canal, só treino
                    f = lambda X: (X / esc).astype(np.float32)
                    rede = _rede(nome, Xw.shape[1], Xw.shape[2])
                    rede, ep, pv = _treinar(rede, f(Xw[mtr]), yw[mtr], f(Xw[mva]), yw[mva], LR[nome], max_epocas,
                                            paciencia, betas=(0.5, 0.999) if nome == "conformer" else (0.9, 0.999))
                    p = _prever(rede, f(Xw[mte]))
                    pr = pd.DataFrame({"sujeito": sw[mte], "p": p}).groupby("sujeito")["p"].mean()
                else:
                    idx = {s: i for i, s in enumerate(d["sujeito"])}
                    itr, iva, ite = ([idx[s] for s in g] for g in (tr2, va, te))
                    mu, sd = feats_no[itr].mean(0), feats_no[itr].std(0) + 1e-6
                    f = lambda i: ((feats_no[i] - mu) / sd).astype(np.float32)
                    rede = GCN(grafo(canais, sem_arestas=nome == "gcn_sem_arestas"), feats_no.shape[2])
                    y = d["y"].to_numpy()
                    rede, ep, pv = _treinar(rede, f(itr), y[itr], f(iva), y[iva], LR[nome], max_epocas, paciencia, lote=32)
                    pr = pd.Series(_prever(rede, f(ite)), index=list(te))
                for s in te:
                    linhas.append({"sujeito": s, "dataset": d.set_index("sujeito").at[s, "dataset"],
                                   "y": int(d.set_index("sujeito").at[s, "y"]), "prob": float(pr[s]), "fold": k})
                execs.append({"fold": k, "treino": sorted(tr2.tolist()), "validacao": sorted(va.tolist()),
                              "teste": sorted(te), "epocas": ep, "perda_val": pv})
                print(f"  {prot} fold {k + 1}/{len(fl)}: {ep} épocas, perda val {pv:.3f} [{time.time() - t0:.0f} s]",
                      flush=True)
            tag = nome_config(prot, "bruto" if nome in BRUTOS else "espectral", nome, False)
            pd.DataFrame(linhas).to_csv(pasta / f"pred__{tag}.csv", index=False)
            (pasta / f"execucao__{tag}.json").write_text(json.dumps({"folds": execs}, indent=1))
