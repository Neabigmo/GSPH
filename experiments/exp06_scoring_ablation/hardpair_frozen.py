"""Frozen, target-held-out nested-feature hard-pair comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

try:
    from .feature_groups import classify
except ImportError:
    from feature_groups import classify


def select_tiers(columns, config):
    tiers, selected = {}, []
    for tier, rule in config['tiers'].items():
        added = []
        for col in sorted(columns):
            group, _ = classify(col)
            extension = col.startswith(('barcode_', 'persistence_'))
            if group in ('META', 'DIAG', 'S') or (group == 'EXCL' and not extension):
                continue
            if col not in selected and any(re.search(p, col) for p in rule['add']):
                added.append(col)
        if not added:
            raise ValueError(f'{tier} has zero new columns; supply the required feature tier')
        selected = selected + added
        tiers[tier] = selected.copy()
        print(f'{tier}: {len(selected)} columns (+{len(added)})', flush=True)
    return tiers


def load_task(directory, config, task_config, barcode_table=None):
    files = sorted(directory.glob(config['input_pattern']))
    if not files:
        raise ValueError(f'No feature tables in {directory}')
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    good = set()
    for path in sorted(directory.glob(config['audit_pattern'])):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get('status') in ('complete', 'ok'):
                good.add(str(row.get('structure_id') or row.get('row')))
    key = 'structure_id' if 'structure_id' in frame else 'row'
    frame = frame[frame[key].astype(str).isin(good)].copy()
    if key != 'structure_id':
        frame = frame.rename(columns={key: 'structure_id'})
    frame['structure_id'] = frame.structure_id.astype(str)
    if frame.empty:
        raise ValueError('No candidates with successful audit records')
    if frame.structure_id.duplicated().any():
        raise ValueError('Duplicate structure_id in feature tables')
    if barcode_table:
        path = Path(barcode_table)
        barcode = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
        barcode['structure_id'] = barcode.structure_id.astype(str)
        overlap = (set(barcode) & set(frame)) - {'structure_id'}
        if overlap:
            raise ValueError(f'Barcode columns already present: {sorted(overlap)}')
        frame = frame.merge(barcode, on='structure_id', how='left', validate='one_to_one')
    label, group = task_config['label'], task_config['group']
    if frame[group].isna().any() or not np.isfinite(frame[label].to_numpy(float)).all():
        raise ValueError('Target and finite label required for every audited candidate')
    return frame.sort_values([group, 'structure_id']).reset_index(drop=True)


def load_v2(directory, task, manifest_path=None):
    """Join compact exports by candidate identity, never by row position.

    build_v2_features preserves structure_id/group but does not export labels.
    The original runner gets labels from the raw table; its OOF y_true is the
    available lossless label source when only compact exports are supplied.
    """
    directory = Path(directory)
    keys = ['structure_id', 'group']
    frame = None
    for name in ('physical_compact_v2', 'topology_features_v2', 'hodge_features_v2'):
        extra = pd.read_parquet(directory / f'{name}_{task}.parquet')
        if not set(keys) <= set(extra) or extra[keys].isna().any().any():
            raise ValueError(f'{name}: nonmissing structure_id/group required')
        extra[keys] = extra[keys].astype(str)
        if extra.duplicated(keys).any():
            raise ValueError(f'{name}: duplicate candidate keys')
        if frame is None:
            frame = extra
            continue
        if set(map(tuple, frame[keys].to_numpy())) != set(map(tuple, extra[keys].to_numpy())):
            raise ValueError(f'{name}: candidate pool differs from physical table')
        shared = [c for c in extra if c in frame and c not in keys]
        joined = frame.merge(extra, on=keys, validate='one_to_one', suffixes=('', '__other'))
        for col in shared:
            left, right = joined[col], joined[col + '__other']
            if not (left.eq(right) | (left.isna() & right.isna())).all():
                raise ValueError(f'{name}: conflicting shared column {col}')
        frame = joined.drop(columns=[c + '__other' for c in shared])
    sources = [directory / f'{name}_{task}.parquet' for name in ('oof_v2', 'oof')]
    labels = None
    for source in sources:
        if not source.exists():
            continue
        oof = pd.read_parquet(source).rename(columns={'group_id': 'group', 'y_true': 'label_cadscore'})
        needed = keys + ['label_cadscore']
        if not set(needed) <= set(oof):
            raise ValueError(f'{source.name}: need structure_id, group_id/group, y_true')
        oof = oof[needed].copy()
        oof[keys] = oof[keys].astype(str)
        oof = oof.drop_duplicates()
        if oof.duplicated(keys).any():
            raise ValueError(f'{source.name}: inconsistent labels across models')
        if labels is None:
            labels = oof
        else:
            overlap = labels.merge(oof, on=keys, suffixes=('', '__other'))
            if not overlap.label_cadscore.eq(overlap.label_cadscore__other).all():
                raise ValueError('Conflicting labels between OOF files')
    if labels is None:
        raise ValueError('Compact exports omit labels: need oof_v2_<task>.parquet or oof_<task>.parquet')
    if 'label_cadscore' in frame:
        check = frame.merge(labels, on=keys, how='left', validate='one_to_one', suffixes=('', '__other'))
        if not check.label_cadscore.eq(check.label_cadscore__other).all():
            raise ValueError('Compact and OOF labels disagree')
    else:
        frame = frame.merge(labels, on=keys, how='left', validate='one_to_one')
    if frame.empty or not np.isfinite(frame.label_cadscore.to_numpy(float)).all():
        raise ValueError('Every compact candidate must have a finite OOF label')
    manifest_path = manifest_path or Path(__file__).resolve().parents[2] / 'results/tables/feature_manifest_v2.csv'
    manifest = pd.read_csv(manifest_path)
    registered = set(manifest.loc[manifest.scenario == task, 'feature_name'])
    # Metadata is retained; only registered columns can enter a feature tier.
    metadata = set(keys + ['row', 'protein', 'kind', 'label_cadscore'])
    frame = frame[[c for c in frame if c in registered or c in metadata]]
    if not registered:
        raise ValueError(f'No feature manifest entries for {task}')
    return frame.sort_values(keys).reset_index(drop=True)


def predict_oof(frame, columns, task, config, seed):
    import xgboost as xgb

    groups = frame[task['group']].astype(str).to_numpy()
    unique = np.sort(pd.unique(groups))
    folds = config['cv']['folds']
    if len(unique) < folds or folds < 2:
        raise ValueError('Need at least one target per fold and at least two folds')
    shuffled = np.random.RandomState(seed).permutation(unique)
    x = frame[columns].to_numpy(dtype=np.float32, copy=True)
    x[~np.isfinite(x)] = np.nan
    y = frame[task['label']].to_numpy(float)
    if not task['higher_better']:
        y = -y
    predictions = np.full(len(frame), np.nan)
    for fold in range(folds):
        test = np.isin(groups, shuffled[fold::folds])
        positions = np.flatnonzero(~test)
        order = positions[np.lexsort((frame.structure_id.to_numpy()[positions], groups[positions]))]
        train = x[order]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            median = np.nanmedian(train, axis=0)
        median = np.where(np.isfinite(median), median, 0)
        train = np.where(np.isnan(train), median, train)
        query = xgb.DMatrix(train, label=y[order])
        query.set_group(pd.Series(groups[order]).groupby(groups[order], sort=False).size().tolist())
        model = xgb.train(config['rank_params'], query, num_boost_round=config['rank_rounds'])
        predictions[test] = model.predict(xgb.DMatrix(np.where(np.isnan(x[test]), median, x[test])))
    return predictions


def frozen_pairs(y, baseline, rule):
    if len(y) < rule['min_candidates'] or np.ptp(y) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    left, right = np.triu_indices(len(y), 1)
    dy = np.abs(y[left] - y[right])
    ds = np.abs(baseline[left] - baseline[right])
    keep = (dy >= np.quantile(dy, rule['q_hard'])) & (ds <= np.quantile(ds, rule['q_amb']))
    if keep.sum() < rule['min_pairs']:
        keep[:] = False
    return left[keep], right[keep]


def evaluate(frame, predictions, task, rule):
    rows = []
    for target, sub in frame.groupby(task['group']):
        idx = sub.index.to_numpy()
        y = sub[task['label']].to_numpy(float) * (1 if task['higher_better'] else -1)
        left, right = frozen_pairs(y, predictions['T1'][idx], rule)
        row = dict(target=target, n_pairs=len(left))
        # This single index set is reused for every tier, never recomputed.
        for tier, scores in predictions.items():
            s = scores[idx]
            row[f'win_{tier}'] = float(np.mean(np.sign(s[left] - s[right]) == np.sign(y[left] - y[right]))) if len(left) else np.nan
            row[f'spearman_{tier}'] = float(spearmanr(y, s).statistic) if np.ptp(y) and np.ptp(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(table, config):
    rng = np.random.default_rng(config['inference']['seed'])
    rows = []
    for task, task_table in table.groupby('task'):
        samples = [(str(seed), part) for seed, part in task_table.groupby('seed')]
        pooled = task_table.groupby('target', as_index=False).mean(numeric_only=True)
        samples.append(('pooled', pooled))
        for seed, part in samples:
            start = len(rows)
            for high, low in [('T2', 'T1'), ('T3', 'T2'), ('T3', 'T1')]:
                d = (part[f'win_{high}'] - part[f'win_{low}']).dropna().to_numpy()
                row = dict(task=task, seed=seed, contrast=f'{high}-{low}', n_targets=len(d),
                           mean_delta_win=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_signflip=np.nan, p_holm=np.nan)
                if len(d):
                    boot = d[rng.integers(len(d), size=(config['inference']['bootstrap'], len(d)))].mean(axis=1)
                    flips = (rng.choice([-1, 1], size=(config['inference']['sign_flips'], len(d))) * d).mean(axis=1)
                    row.update(mean_delta_win=d.mean(), ci_lo=np.quantile(boot, .025), ci_hi=np.quantile(boot, .975),
                               p_signflip=(1 + (np.abs(flips) >= abs(d.mean())).sum()) / (1 + len(flips)))
                row.update({f'macro_spearman_{tier}': part[f'spearman_{tier}'].mean() for tier in config['tiers']})
                rows.append(row)
            primary = [r for r in rows[start:] if r['contrast'] in config['inference']['primary_contrasts']]
            running = 0.
            for rank, row in enumerate(sorted(primary, key=lambda r: r['p_signflip'])):
                if np.isfinite(row['p_signflip']):
                    running = max(running, min(1., (len(primary) - rank) * row['p_signflip']))
                    row['p_holm'] = running
    return pd.DataFrame(rows)


def run(config_path, feat_dir, out, barcode_table=None, v2_dir=None):
    raw = Path(config_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    config = yaml.safe_load(raw)
    tables, selected = [], {}
    for task, task_config in config['tasks'].items():
        # Multiple tasks use task-named subdirectories; a flat directory is
        # accepted only for a single-task configuration to avoid reusing a pool.
        if v2_dir is not None:
            if barcode_table:
                raise ValueError('--barcode-table currently requires --feat-dir')
            frame = load_v2(v2_dir, task)
        else:
            directory = Path(feat_dir) / task if len(config['tasks']) > 1 else Path(feat_dir)
            frame = load_task(directory, config, task_config, barcode_table)
        tiers = select_tiers(frame.columns, config)
        selected[task] = tiers
        for seed in config['cv']['seeds']:
            predictions = {tier: predict_oof(frame, cols, task_config, config, seed) for tier, cols in tiers.items()}
            result = evaluate(frame, predictions, task_config, config['hard_pairs'])
            result['task'], result['seed'] = task, seed
            tables.append(result)
    per_target = pd.concat(tables, ignore_index=True)
    summary = summarize(per_target, config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in [('per_target', per_target), ('summary', summary[summary.seed != 'pooled'].copy()),
                        ('summary_pooled', summary[summary.seed == 'pooled'].copy())]:
        table['config_sha256'] = digest
        table.to_csv(out / f'{name}.csv', index=False)
    (out / 'feature_columns.json').write_text(json.dumps(dict(config_sha256=digest, tiers=selected), indent=2))
    print(summary.to_string(index=False))
    return per_target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--feat-dir', type=Path)
    inputs.add_argument('--v2-dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--barcode-table', type=Path)
    args = parser.parse_args()
    run(args.config, args.feat_dir, args.out, args.barcode_table, args.v2_dir)


if __name__ == '__main__':
    main()
