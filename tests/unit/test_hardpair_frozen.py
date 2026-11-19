import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / 'experiments' / 'exp06_scoring_ablation'
sys.path.insert(0, str(MODULE_DIR))
spec = importlib.util.spec_from_file_location('hardpair_frozen', MODULE_DIR / 'hardpair_frozen.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


def synthetic_inputs(directory):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 360
    signal = rng.normal(size=n)
    frame = pd.DataFrame({
        'structure_id': [f'candidate_{i}' for i in range(n)],
        'group': np.repeat([f'target_{i:02}' for i in range(12)], 30),
        'label_cadscore': signal + rng.normal(scale=.03, size=n),
        'patch_energy_mean': rng.normal(size=n),
        'tau_2_mean': rng.normal(size=n),
        'eta_3_mean': signal,
        'persistence_interval_count_k1': rng.normal(size=n),
        'patch_contact_count': np.ones(n),
        'barcode_solver_residual': np.ones(n),
    })
    frame.loc[::19, 'patch_energy_mean'] = np.nan
    frame.to_parquet(directory / 'pd_ftdmp_synthetic.parquet', index=False)
    (directory / 'pd_ftdmp_synthetic.audit.jsonl').write_text('\n'.join(
        json.dumps(dict(structure_id=s, status='complete')) for s in frame.structure_id))
    config = yaml.safe_load((ROOT / 'configs/experiments/exp06_hardpair_frozen.yaml').read_text())
    config['tasks'] = {'pd_bound': config['tasks']['pd_bound']}
    config['cv']['seeds'] = [0, 1]
    config['rank_rounds'] = 12
    config['rank_params']['n_jobs'] = 1
    config['inference']['bootstrap'] = 200
    config['inference']['sign_flips'] = 200
    path = directory / 'synthetic_config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return frame, config, path


def test_planted_signal(tmp_path):
    frame, config, path = synthetic_inputs(tmp_path / 'features')
    tiers = frozen.select_tiers(frame.columns, config)
    assert set(tiers['T1']) < set(tiers['T2']) < set(tiers['T3'])
    assert 'patch_contact_count' not in tiers['T1']
    assert 'barcode_solver_residual' not in tiers['T3']
    per_target, summary = frozen.run(path, path.parent, tmp_path / 'out')
    assert {'mean_delta_win', 'ci_lo', 'ci_hi', 'p_signflip', 'p_holm', 'macro_spearman_T1'} <= set(summary)
    assert (summary.loc[summary.contrast == 'T2-T1', 'mean_delta_win'] > 0).all()
    for name in ['per_target', 'summary', 'summary_pooled']:
        output = pd.read_csv(tmp_path / 'out' / f'{name}.csv')
        assert output.config_sha256.str.len().eq(64).all()
    assert per_target.n_pairs.gt(0).all()


def test_pair_set_shared_and_empty(tmp_path):
    frame, config, _ = synthetic_inputs(tmp_path / 'features')
    baseline = np.random.default_rng(4).normal(size=len(frame))
    predictions = {'T1': baseline, 'T2': -baseline, 'T3': baseline.copy()}
    table = frozen.evaluate(frame, predictions, config['tasks']['pd_bound'], config['hard_pairs'])
    for _, sub in frame.groupby('group'):
        y = sub.label_cadscore.to_numpy()
        left, right = frozen.frozen_pairs(y, baseline[sub.index], config['hard_pairs'])
        expected = (np.sign(baseline[sub.index][left] - baseline[sub.index][right]) == np.sign(y[left] - y[right])).mean()
        row = table[table.target == sub.group.iloc[0]].iloc[0]
        assert row.n_pairs == len(left)
        assert row.win_T1 == row.win_T3 == expected
        assert row.win_T2 == pytest.approx(1 - expected)
    left, _ = frozen.frozen_pairs(np.ones(30), np.ones(30), config['hard_pairs'])
    assert len(left) == 0
    empty = table.copy()
    empty[['win_T1', 'win_T2', 'win_T3']] = np.nan
    empty['task'], empty['seed'] = 'pd_bound', 0
    assert frozen.summarize(empty, config).n_targets.eq(0).all()
    with pytest.raises(ValueError, match='zero new columns'):
        frozen.select_tiers(['patch_energy', 'eta_3_mean'], config)
    with pytest.raises(ValueError, match='No feature tables'):
        frozen.load_task(tmp_path, config, config['tasks']['pd_bound'])


def test_barcode_merge_and_audit_errors(tmp_path):
    frame, config, _ = synthetic_inputs(tmp_path / 'features')
    barcode = pd.DataFrame({'structure_id': frame.structure_id[:2], 'barcode_lifetime': [1., 2.]})
    path = tmp_path / 'barcode.csv'
    barcode.to_csv(path, index=False)
    merged = frozen.load_task(tmp_path / 'features', config, config['tasks']['pd_bound'], path)
    assert merged.barcode_lifetime.notna().sum() == 2
    assert merged.barcode_lifetime.isna().sum() == len(frame) - 2
    pd.concat([barcode, barcode]).to_csv(path, index=False)
    with pytest.raises(pd.errors.MergeError):
        frozen.load_task(tmp_path / 'features', config, config['tasks']['pd_bound'], path)
    (tmp_path / 'features' / 'pd_ftdmp_synthetic.audit.jsonl').write_text('')
    with pytest.raises(ValueError, match='No candidates'):
        frozen.load_task(tmp_path / 'features', config, config['tasks']['pd_bound'])
