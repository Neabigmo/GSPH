import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiments import baseline_checks, ion_calibration, union_audit_batch
from experiments.exp06_scoring_ablation import hardpair_frozen as frozen

ROOT = Path(__file__).resolve().parents[2]


def compact_inputs(root, task='pd_bound', n=90):
    import numpy as np

    rng = np.random.default_rng(5)
    keys = pd.DataFrame({'structure_id': [f'candidate_{i}' for i in range(n)],
                         'group': [f'target_{i // 18}' for i in range(n)], 'row': range(n)})
    for name, col in [('physical_compact_v2', 'patch_global_patch_geometry_anisotropy_sum'),
                      ('topology_features_v2', 'eta_3_mean'),
                      ('hodge_features_v2', 'persistence_interval_count_k1')]:
        table = keys.assign(**{col: rng.normal(size=n)})
        table.sample(frac=1, random_state=2).to_parquet(root / f'{name}_{task}.parquet', index=False)
    labels = keys[['structure_id', 'group']].rename(columns={'group': 'group_id'})
    labels['y_true'] = keys.row.to_numpy() / n
    pd.concat([labels.assign(model='first'), labels.assign(model='second')]).to_parquet(root / f'oof_v2_{task}.parquet', index=False)
    return keys, labels


def test_compact_loader_keys_labels_and_tiers(tmp_path):
    keys, labels = compact_inputs(tmp_path)
    frame = frozen.load_v2(tmp_path, 'pd_bound')
    expected = labels.set_index('structure_id').y_true
    assert frame.label_cadscore.tolist() == frame.structure_id.map(expected).tolist()
    assert frame.group.nunique() == 5
    config = yaml.safe_load((ROOT / 'configs/experiments/exp06_hardpair_frozen.yaml').read_text())
    tiers = frozen.select_tiers(frame.columns, config)
    assert set(tiers['T1']) < set(tiers['T2']) < set(tiers['T3'])
    assert 'row' not in tiers['T1']
    path = tmp_path / 'topology_features_v2_pd_bound.parquet'
    table = pd.read_parquet(path)
    table.loc[0, 'row'] = -1
    table.to_parquet(path)
    with pytest.raises(ValueError, match='conflicting shared column row'):
        frozen.load_v2(tmp_path, 'pd_bound')


@pytest.mark.parametrize('problem', ['missing_label', 'duplicate', 'empty', 'pool', 'manifest', 'conflict'])
def test_compact_loader_rejects_inconsistent_data(tmp_path, problem):
    _, labels = compact_inputs(tmp_path)
    source = tmp_path / 'physical_compact_v2_pd_bound.parquet'
    table = pd.read_parquet(source)
    if problem == 'missing_label':
        labels.iloc[:-1].to_parquet(tmp_path / 'oof_v2_pd_bound.parquet')
    elif problem == 'duplicate':
        pd.concat([table, table.iloc[:1]]).to_parquet(source)
    elif problem in ('empty', 'pool'):
        table.iloc[:0 if problem == 'empty' else -1].to_parquet(source)
    elif problem == 'manifest':
        pd.DataFrame({'scenario': ['other'], 'feature_name': ['eta_3_mean']}).to_csv(tmp_path / 'manifest.csv', index=False)
    else:
        pd.concat([labels, labels.assign(y_true=-1)]).to_parquet(tmp_path / 'oof_v2_pd_bound.parquet')
    with pytest.raises(ValueError):
        frozen.load_v2(tmp_path, 'pd_bound', tmp_path / 'manifest.csv' if problem == 'manifest' else None)


def make_dumps(root):
    # A triangle with a delayed 2-path has a different H1 lifetime in the two filtrations.
    records = []
    supports = [(str(i), 0.) for i in range(3)] + [('0,1', .1), ('0,2', .1), ('1,2', .1)]
    for graph in ('dir', 'und'):
        for path, tau in supports + [('0,1,2', .2)]:
            records.append(dict(graph=graph, path=path, tau=tau, valid=True))
    pd.DataFrame(records).to_parquet(root / 'example.complex_paths.parquet')
    pd.DataFrame([dict(canon=key, tau=tau) for key, tau in supports]).to_parquet(root / 'example.tau_table.parquet')
    pd.DataFrame([dict(patch_id=i, contact_count=1, r_min_angstrom=1.5, r_max_angstrom=1.5,
                       delta_angstrom=0.) for i in range(3)]).to_parquet(root / 'example.patch_geometry.parquet')


def test_union_driver_both_graphs_ratios_barcodes_and_errors(tmp_path):
    make_dumps(tmp_path)
    summary = union_audit_batch.run([tmp_path], tmp_path / 'out')
    assert summary['complete'] and summary['n_graphs_completed'] == 2
    assert len(summary['single_contact_positive']) == 2
    assert all(row['ratio'] > 1 for row in summary['single_contact_positive'])
    bars = pd.read_csv(tmp_path / 'out/barcode_summary.csv')
    assert set(bars.order) == {0, 1, 2} and bars.different.any()
    assert bars.common_infinite.sum() > 0
    assert not union_audit_batch.run([], tmp_path / 'empty')['complete']
    (tmp_path / 'example.patch_geometry.parquet').unlink()
    assert len(union_audit_batch.run([tmp_path], tmp_path / 'bad')['errors']) == 2


def test_baseline_counts_and_di(tmp_path):
    compact_inputs(tmp_path)
    rows = baseline_checks.inspect_tables(tmp_path, expected_groups=5)
    assert len(rows) == 4 and all(r['status'] == 'ok' for r in rows)
    assert all(r['status'] == 'group_count_mismatch' for r in baseline_checks.inspect_tables(tmp_path))
    for directory, suffix in [('RMSD', '.rmsd'), ('DI', '.DI')]:
        (tmp_path / directory).mkdir()
        (tmp_path / directory / ('target' + suffix)).write_text('native.pdb 0\na.pdb 1\nb.pdb 2\n')
    assert baseline_checks.check_di(tmp_path)[0]['rmsd_di_spearman'] == 1
    (tmp_path / 'DI/target.DI').write_text('native.pdb 2\na.pdb 1\nb.pdb 0\n')
    assert baseline_checks.check_di(tmp_path)[0]['status'] == 'direction_conflict'
    (tmp_path / 'DI/target.DI').unlink()
    assert baseline_checks.check_di(tmp_path)[0]['status'] == 'missing_or_ambiguous_di'
    with pytest.raises(ValueError, match='No RMSD'):
        baseline_checks.check_di(tmp_path / 'absent')
    assert baseline_checks.inspect_tables(tmp_path / 'absent') == []


def test_ion_selection_preserves_censored_and_size_extremes():
    table = pd.read_csv(ROOT / 'results/tables/exp09_metal_site_pilot_per_structure.csv')
    chosen = ion_calibration.select_structures(table)
    assert chosen.groupby('cohort').size().to_dict() == {'mg': 10, 'water': 20, 'zn': 20}
    assert set(table.loc[table.status != 'complete', 'pdb_id']) <= set(chosen.pdb_id)
    for cohort, group in table.groupby('cohort'):
        part = chosen[chosen.cohort == cohort]
        assert part.atoms.min() == group.atoms.min() and part.atoms.max() == group.atoms.max()
        assert (part.population_count / part.stratum_sample_count).sum() == pytest.approx(len(group))
    with pytest.raises(ValueError):
        ion_calibration.select_structures(table.iloc[:0])


def test_ion_summary_weighting_censoring_and_missing(tmp_path):
    source = ROOT / 'results/tables/exp09_metal_site_pilot_per_structure.csv'
    config = ROOT / 'configs/experiments/exp09_metal_site_pilot.yaml'
    selection = ion_calibration.prepare(tmp_path, source, config)
    cfg = yaml.safe_load(config.read_text())
    (tmp_path / 'structures').mkdir()
    for row in selection.itertuples(index=False):
        result = dict(status=row.status, config_hash=ion_calibration.fingerprint(cfg), cpu_seconds=3.6,
                      peak_rss_kib=100, process_wall_seconds=4,
                      metrics=[dict(sampled=2, pairwise_passing=20, candidate_seconds=1.8)])
        (tmp_path / 'structures' / f'{row.cohort}_{row.pdb_id}.json').write_text(json.dumps(result))
    summary = ion_calibration.summarize(tmp_path)['summary']
    assert all(r['sampled_core_hours_per_1000'] == pytest.approx(1.) for r in summary)
    assert all(r['exhaustive_candidate_core_hours_per_1000_complete'] == pytest.approx(5.) for r in summary)
    assert next(r for r in summary if r['cohort'] == 'mg')['censored_fraction'] == .4
    next((tmp_path / 'structures').glob('*.json')).unlink()
    with pytest.raises(RuntimeError, match='incomplete'):
        ion_calibration.summarize(tmp_path)


def test_ion_download_only_selected_and_failures(tmp_path, monkeypatch):
    import requests

    selection = ion_calibration.prepare(tmp_path,
        ROOT / 'results/tables/exp09_metal_site_pilot_per_structure.csv',
        ROOT / 'configs/experiments/exp09_metal_site_pilot.yaml')
    seen = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        def iter_content(self, size):
            yield b'synthetic'
    class Session:
        def mount(self, *args):
            pass
        def get(self, url, **kwargs):
            seen.append(url.split('/')[-1].split('.')[0].lower())
            return Response()
    monkeypatch.setattr(requests, 'Session', Session)
    ion_calibration.download(tmp_path)
    assert set(seen) == set(selection.pdb_id)
    seen.clear()
    ion_calibration.download(tmp_path)
    assert not seen
    for path in (tmp_path / 'data').glob('*.gz'):
        path.unlink()
    def unavailable(*args, **kwargs):
        raise RuntimeError('offline')
    monkeypatch.setattr(Session, 'get', unavailable)
    with pytest.raises(RuntimeError, match='Downloads incomplete'):
        ion_calibration.download(tmp_path)


def test_ion_worker_dispatch_offline_and_error_records(tmp_path, monkeypatch):
    selection = ion_calibration.prepare(tmp_path,
        ROOT / 'results/tables/exp09_metal_site_pilot_per_structure.csv',
        ROOT / 'configs/experiments/exp09_metal_site_pilot.yaml')
    cfg = yaml.safe_load((tmp_path / 'config.yaml').read_text())
    (tmp_path / 'data').mkdir()
    for identifier in selection.pdb_id:
        (tmp_path / 'data' / f'{identifier}.cif.gz').write_bytes(b'synthetic')
    calls = []
    monkeypatch.setattr(ion_calibration.subprocess, 'run', lambda command, **kw: calls.append(command))
    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
            assert command[:2] == ['/usr/bin/time', '-v']
            assert command[command.index('--stage') + 1] == 'worker'
            assert kwargs['env']['OMP_NUM_THREADS'] == '1'
            output = Path(command[command.index('--output') + 1])
            output.write_text(json.dumps(dict(config_hash=ion_calibration.fingerprint(cfg),
                status='error', error='synthetic failure')))
        def wait(self, **kwargs):
            return 0
        def poll(self):
            return 0
    monkeypatch.setattr(ion_calibration.subprocess, 'Popen', Process)
    ion_calibration.run_workers(tmp_path, 2)
    assert len(calls) == 51
    assert all(json.loads(path.read_text())['status'] == 'worker_error'
               for path in (tmp_path / 'structures').glob('*.json'))
    with pytest.raises(RuntimeError, match='incomplete'):
        ion_calibration.summarize(tmp_path)
