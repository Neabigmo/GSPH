"""Compare VoroMQA and the official FTDMP ranking with GSPH on the same protein-DNA pool.

The pool is the GSPH candidate list (`ftdmp_candidates.csv`); external scores are
joined one-to-one by task, target and model ID. Metrics follow EXP-06: per-target
Spearman between score and interface CAD-score, macro-averaged over targets, and
the best CAD-score among the top 1, 5 and 10 ranked candidates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from experiments.external_scoring_pm import paired_bootstrap
except ImportError:
    from external_scoring_pm import paired_bootstrap

TASKS = {'pd_bound': 'Bound_docking', 'pd_unbound': 'Unbound_docking'}
ARMS = ['A1_P', 'A7_PTH', 'CleanFull_PTHS']


def load_pool(tables):
    pool = pd.read_csv(tables / 'ftdmp_candidates.csv')
    if pool.duplicated(['task', 'group', 'structure_id']).any() or pool.label_cadscore.isna().any():
        raise ValueError('candidate list must be unique and labelled')
    voro = pd.DataFrame(json.loads(line) for line in (tables / 'pd_voromqa_scores.jsonl').open() if line.strip())
    parts = voro.pdb.str.replace('\\', '/', regex=False).str.split('/')
    voro = pd.DataFrame(dict(task=parts.str[0].map({v: k for k, v in TASKS.items()}), group=parts.str[1],
                             structure_id=parts.str[-1].str.removesuffix('.pdb'), voromqa=voro.voromqa))
    official = pd.read_csv(tables / 'external_scoring_ftdmp_official_ranks.csv')
    joined = (pool.merge(voro, on=['task', 'group', 'structure_id'], how='left', validate='one_to_one')
                  .merge(official, on=['task', 'group', 'structure_id'], how='left', validate='one_to_one'))
    if joined[['voromqa', 'relaxed_RJS_rank']].isna().any().any():
        raise ValueError('an external score is missing for a pool candidate')
    outside = len(voro) - len(joined)
    scores = pd.concat([
        joined.assign(method='VoroMQA', score=joined.voromqa),
        joined.assign(method='FTDMP official rank', score=-joined.relaxed_RJS_rank),
    ])
    return scores[['task', 'group', 'structure_id', 'label_cadscore', 'method', 'score']], outside


def evaluate(scores):
    rows = []
    for (task, method, target), group in scores.groupby(['task', 'method', 'group'], sort=True):
        ranked = group.sort_values(['score', 'structure_id'], ascending=[False, True], kind='stable')
        rows.append(dict(task=task, method=method, target=target, n=len(group),
                         spearman=float(spearmanr(group.score, group.label_cadscore).statistic),
                         **{f'top{k}_label': float(ranked.label_cadscore.iloc[:k].max()) for k in (1, 5, 10)}))
    return pd.DataFrame(rows)


def gsph_reference(tables, task):
    splits = json.loads((tables / f'cv_splits_{task}.json').read_text())
    targets = sorted(t for fold in splits.values() for t in fold)
    payload = json.loads((tables / f'metrics_{task}.json').read_text())
    rows = []
    for arm in ARMS:
        values = payload[arm]['per_target_spearman']
        if len(values) != len(targets) or not np.isclose(np.mean(values), payload[arm]['target_macro_spearman']):
            raise ValueError(f'{task}/{arm}: target order cannot be reconstructed')
        rows.extend(dict(task=task, target=t, gsph_arm=arm, gsph_spearman=v) for t, v in zip(targets, values))
    summary = [dict(task=task, method=f'GSPH {arm}', n_targets=len(targets),
                    macro_spearman=payload[arm]['target_macro_spearman'],
                    **{f'mean_top{k}_label': payload[arm][f'top{k}_label'] for k in (1, 5, 10)}) for arm in ARMS]
    return pd.DataFrame(rows), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=Path, required=True)
    args = parser.parse_args()
    tables = args.tables
    scores, outside = load_pool(tables)
    metrics = evaluate(scores)
    metrics.to_csv(tables / 'external_scoring_pd_per_target.csv', index=False)
    summary = (metrics.groupby(['task', 'method'], sort=True)
               .agg(n_targets=('target', 'size'), n_candidates=('n', 'sum'), macro_spearman=('spearman', 'mean'),
                    mean_top1_label=('top1_label', 'mean'), mean_top5_label=('top5_label', 'mean'),
                    mean_top10_label=('top10_label', 'mean')).reset_index())
    intervals, gsph_rows = [], []
    for task in TASKS:
        reference, rows = gsph_reference(tables, task)
        gsph_rows.extend(rows)
        paired = metrics[metrics.task == task].merge(reference, on=['task', 'target'], validate='many_to_many')
        if paired.groupby(['gsph_arm', 'method']).size().ne(47).any():
            raise ValueError(f'{task}: targets do not pair')
        for (arm, method), group in paired.groupby(['gsph_arm', 'method'], sort=True):
            result = paired_bootstrap((group.gsph_spearman - group.spearman).to_numpy())
            intervals.append(dict(task=task, gsph_arm=arm, external_method=method,
                                  reference_scope='saved_single_split', **result))
    summary = pd.concat([summary, pd.DataFrame(gsph_rows)], ignore_index=True)
    summary.to_csv(tables / 'external_scoring_pd_summary.csv', index=False)
    pd.DataFrame(intervals).to_csv(tables / 'external_scoring_pd_bootstrap.csv', index=False)
    print(f'{outside} scored model(s) outside the GSPH pool were not used')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(pd.DataFrame(intervals).to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
