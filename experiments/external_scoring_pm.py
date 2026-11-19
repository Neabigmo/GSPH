"""Evaluate PM energies with the EXP-06 decoy-only convention."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def evaluate(scores):
    required = {'target', 'method', 'structure', 'score', 'rmsd'}
    if not required.issubset(scores.columns) or scores.empty:
        raise ValueError('Nonempty PM scores with target, method, structure, score, rmsd required')
    if scores.duplicated(['target', 'method', 'structure']).any():
        raise ValueError('Duplicate candidate scores')
    if not np.isfinite(scores[['score', 'rmsd']].to_numpy()).all():
        raise ValueError('Nonfinite score or RMSD')
    rows = []
    for (target, method), group in scores.groupby(['target', 'method'], sort=True):
        native = group.structure.str.removesuffix('.pdb').str.casefold().eq(str(target).casefold())
        decoys = group.loc[~native].sort_values('structure', kind='stable')
        if len(decoys) < 2 or decoys.rmsd.nunique() < 2 or decoys.score.nunique() < 2:
            raise ValueError(f'{target}/{method}: correlation is undefined')
        top = decoys.loc[decoys.score.idxmin()]
        rows.append(dict(target=target, method=method, n=len(decoys), n_with_native=len(group),
                         spearman=float(spearmanr(-decoys.score, -decoys.rmsd).statistic),
                         spearman_with_native=float(spearmanr(group.score, group.rmsd).statistic),
                         top1_structure=top.structure, top1_rmsd=float(top.rmsd)))
    return pd.DataFrame(rows)


def paired_bootstrap(differences, n_boot=2000, seed=0):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or n_boot < 1:
        raise ValueError('Finite nonempty target differences and positive resample count required')
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    lo, hi = np.percentile(sample, [2.5, 97.5])
    return dict(n_targets=len(values), mean_diff=float(values.mean()),
                ci_lo=float(lo), ci_hi=float(hi), n_boot=n_boot, seed=seed)


def saved_single_split(tables, targets):
    rows = []
    specifications = {
        'metrics_rna.json': ['A1_P', 'A7_PTH', 'CleanFull_PTHS__nolj'],
        'metrics_v2_rna.json': ['B9_PcT2H2'],
    }
    for name, arms in specifications.items():
        payload = json.loads((tables / name).read_text())
        for arm in arms:
            values = payload[arm]['per_target_spearman']
            if len(values) != len(targets) or not np.isfinite(values).all():
                raise ValueError(f'{name}/{arm}: target order cannot be reconstructed')
            if not np.isclose(np.mean(values), payload[arm]['target_macro_spearman']):
                raise ValueError(f'{name}/{arm}: inconsistent saved macro')
            rows.extend(dict(target=t, gsph_arm=arm, gsph_spearman=v,
                             reference_scope='saved_single_split_not_20split_mean', source=name)
                        for t, v in zip(targets, values))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=Path, required=True)
    args = parser.parse_args()
    tables = args.tables
    scores = pd.read_csv(tables / 'external_scoring_pm_full_scores.csv')
    metrics = evaluate(scores)
    splits = json.loads((tables / 'cv_splits_rna.json').read_text())
    targets = sorted(t for group in splits.values() for t in group)
    if len(set(targets)) != 20 or set(metrics.target) != set(targets):
        raise ValueError('PM targets do not match saved EXP-06 folds')
    for method, group in metrics.groupby('method'):
        if len(group) != 20 or group.n.sum() != 806 or group.n_with_native.sum() != 826:
            raise ValueError(f'{method}: incomplete PM candidate pool')
    metrics.to_csv(tables / 'external_scoring_pm_full_per_target.csv', index=False)
    reference = saved_single_split(tables, targets)
    reference.to_csv(tables / 'external_scoring_pm_gsph_saved_per_target.csv', index=False)
    differences = metrics.merge(reference, on='target', validate='many_to_many')
    differences['gsph_minus_external'] = differences.gsph_spearman - differences.spearman
    differences.to_csv(tables / 'external_scoring_pm_paired_single_split.csv', index=False)
    intervals = []
    for (arm, method), group in differences.groupby(['gsph_arm', 'method'], sort=True):
        result = paired_bootstrap(group.sort_values('target').gsph_minus_external)
        intervals.append(dict(gsph_arm=arm, external_method=method,
                              reference_scope=group.reference_scope.iloc[0], **result))
    pd.DataFrame(intervals).to_csv(tables / 'external_scoring_pm_bootstrap_single_split.csv', index=False)
    print(metrics.groupby('method').agg(n_targets=('target', 'size'), n=('n', 'sum'),
                                       macro_spearman=('spearman', 'mean')).to_string())
    print(pd.DataFrame(intervals).to_string(index=False))
    print('20-split target means: unavailable; these paired intervals use saved single-split values only.')


if __name__ == '__main__':
    main()
