import sys as _sys
from pathlib import Path as _P
_F = _P(__file__).resolve()
_R = _F.parent
while _R != _R.parent and not ((_R / "src").is_dir() and (_R / "experiments").is_dir()):
    _R = _R.parent
for _d in (_R / "src", _R / "tasks" / "rna", _R / "tasks" / "protein_dna",
           _R / "tasks" / "membrane"):
    if _d.is_dir() and str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
for _e in sorted((_R / "experiments").glob("exp*")):
    if _e.is_dir() and str(_e) not in _sys.path:
        _sys.path.insert(0, str(_e))
#!/usr/bin/env python3
"""ProS-GNN grouped CV on M223: train from scratch on 6 protein groups,
predict held-out group. Their architecture/hyperparams, zero pretrained
weights. Outputs prosgnn_predictions.csv."""
import argparse
import json
import sys
import time
from pathlib import Path

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

REPO = Path(r'H:\2026try\11.3GSPH\GSPH_H3\external\ProS-GNN')
sys.path.insert(0, str(REPO))
from dataset import Dataset, collate_fn  # noqa: E402
from models import gnn  # noqa: E402
import utils  # noqa: E402

DATA = Path(r'H:\2026try\11.3GSPH\GSPH_H3\baselines_ext\prosgnn_data')
OUT_CSV = DATA.parent / 'prosgnn_predictions.csv'
EPOCHS = 400
SEED = 42


def make_args():
    a = argparse.Namespace(
        batch_size=16, dropout_rate=0.5, weight_decay=5e-5,
        n_graph_layer=4, d_graph_layer=1120, n_FC_layer=4,
        d_FC_layer=1024, ngpu=0)
    return a


def run_fold(train_keys, test_keys, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args = make_args()
    model = utils.initialize_model(gnn(args), device)
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    tr = DataLoader(Dataset(train_keys, 'mutation_pdb', 'ddg', 'wild_pdb'),
                    batch_size=16, shuffle=True, num_workers=0,
                    collate_fn=collate_fn)
    te = DataLoader(Dataset(test_keys, 'mutation_pdb', 'ddg', 'wild_pdb'),
                    batch_size=16, shuffle=False, num_workers=0,
                    collate_fn=collate_fn)
    for ep in range(EPOCHS):
        model.train()
        for H1, H2, A1, A2, P1, P2, labels, key in tr:
            opt.zero_grad()
            labels = torch.Tensor(labels).to(device)
            pred = model.train_model((H1.to(device), H2.to(device),
                                      A1.to(device), A2.to(device),
                                      P1.to(device), P2.to(device)))
            loss = loss_fn(pred, labels)
            loss.backward()
            opt.step()
    model.eval()
    preds = {}
    with torch.no_grad():
        for H1, H2, A1, A2, P1, P2, labels, key in te:
            pred = model.train_model((H1.to(device), H2.to(device),
                                      A1.to(device), A2.to(device),
                                      P1.to(device), P2.to(device)))
            for k, p in zip(key, pred.cpu().numpy().ravel()):
                preds[k] = float(p)
    return preds


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    keys_by_prot = json.load(open(DATA / 'keys_by_protein.json'))
    prots = sorted(keys_by_prot)
    all_rows = []
    import os
    os.chdir(DATA)
    done = set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        all_rows = prev.to_dict('records')
        done = set(prev.protein)
        print('resume:', len(prev), 'rows, folds done:', sorted(done), flush=True)
    for held in prots:
        if held in done:
            continue
        t0 = time.time()
        test_keys = keys_by_prot[held]
        train_keys = [k for p in prots if p != held for k in keys_by_prot[p]]
        preds = run_fold(train_keys, test_keys, device)
        for k, p in preds.items():
            all_rows.append({'key': k, 'protein': held, 'pred': p})
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        print(f'{held}: {len(preds)} preds in {time.time()-t0:.0f}s',
              flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print('wrote', OUT_CSV, len(df))
    # quick metrics vs ddG from manifest table
    mut = pd.read_csv(DATA.parent / 'm223_mutations.csv')
    mut['key'] = (mut['pdbid'] + '_' + mut['resid'].astype(str) + '_' +
                  mut['wt_aa'] + mut['resid'].astype(str) + mut['mut_aa'])
    df = df.merge(mut[['key', 'ddG']], on='key')
    sp_all = spearmanr(df['pred'], df['ddG']).statistic
    print(f'pooled spearman={sp_all:.4f} n={len(df)}')
    for p, g in df.groupby('protein'):
        print(f'  {p}: n={len(g)} spearman={spearmanr(g.pred, g.ddG).statistic:.3f}')


if __name__ == '__main__':
    main()
