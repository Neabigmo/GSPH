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
"""Evaluate T-MP ESPH features vs GSPH on M223 under identical protocol."""
import glob, json, sys
import numpy as np
import pandas as pd

ROOT = r'H:\2026try\11.3GSPH'
sys.path.insert(0, ROOT + r'\GSPH_H3\evaluation')
import train_eval as TE

df = pd.concat([pd.read_parquet(f) for f in
                glob.glob(ROOT + r'\GSPH_H3\_feat\membrane\membrane_s*.parquet')])
t = pd.read_csv(ROOT + r'\GSPH_H3\baselines_ext\tmp_esph_features.csv')
t['mut_aa'] = t['mutation'].str[-1]
tmp_cols = [c for c in t.columns if c.startswith('tmp_')]
df = df.merge(t[['pdbid','resid','mut_aa'] + tmp_cols],
              on=['pdbid','resid','mut_aa'])
df = df.rename(columns={c: 'base_' + c for c in tmp_cols})
if 'group' not in df.columns:
    df['group'] = df['protein']
if 'structure_id' not in df.columns:
    df['structure_id'] = df['pdbid'] + '_' + df['resid'].astype(str) + '_' + df['mut_aa']
df = df.copy()
df = df.replace([np.inf, -np.inf], np.nan)
df['quality'] = TE.rank_quality(df, 'ddG', False)
df = df.dropna(subset=['quality', 'group'])
targets = sorted(df['group'].unique())
folds = TE.target_grouped_folds(targets, n_splits=5)
df['fold'] = df['group'].map(folds)

gsph_cols = [c for c in TE.feature_columns(df) if not c.startswith('base_')]
tmp_cols_b = [c for c in df.columns if c.startswith('base_tmp_')]
models = {'M0_gsph': gsph_cols, 'TMP_esph': tmp_cols_b,
          'M2_union': gsph_cols + tmp_cols_b}
res = {}
for name, cols in models.items():
    fold_metrics, preds_all = [], []
    for f in sorted(set(folds.values())):
        tr, te = df[df.fold != f], df[df.fold == f]
        if not len(tr) or not len(te):
            continue
        te = te.copy()
        te['score'] = TE.train_predict(tr, te, cols, 'quality')
        preds_all.append(te)
        fold_metrics.append(TE.target_metrics(
            [{'target_id': r.group, 'candidate_id': r.structure_id,
              'score': r.score, 'quality': r.quality, 'irmsd': 1e9}
             for r in te.itertuples()]))
    pooled = pd.concat(preds_all)
    pooled_m = TE.target_metrics(
        [{'target_id': r.group, 'candidate_id': r.structure_id,
          'score': r.score, 'quality': r.quality, 'irmsd': 1e9}
         for r in pooled.itertuples()])
    per_t = {t_: TE._target_spearman(pooled[pooled.group == t_])
             for t_ in pooled.group.unique()}
    res[name] = {'n_features': len(cols),
                 'fold_mean': {k: float(np.mean([m[k] for m in fold_metrics]))
                               for k in fold_metrics[0]},
                 'pooled': pooled_m, 'per_target_spearman': per_t}
    print(name, 'pooled_spearman', pooled_m['spearman'],
          'pearson-ish', pooled_m.get('pearson'))
with open(ROOT + r'\GSPH_H3\reports\eval_membrane_tmp.json', 'w') as f:
    json.dump({'task': 'membrane', 'baseline': 'T-MP_ESPH_gudhi_alphacomplex',
               'metrics': res}, f, indent=2, default=float)
print('saved')
