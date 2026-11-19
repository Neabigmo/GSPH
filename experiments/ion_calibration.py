"""Frozen size-stratified ion calibration; download on login, compute offline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'experiments/exp09_metal_site_pilot/run.py'
COUNTS = {'zn': 20, 'water': 20, 'mg': 10}


def select_structures(table, counts=None, seed=0):
    counts = COUNTS if counts is None else counts
    rng = random.Random(seed)
    rows = []
    for cohort, count in counts.items():
        group = table.loc[table.cohort == cohort].sort_values(['atoms', 'pdb_id']).reset_index(drop=True).copy()
        if len(group) < count or group.pdb_id.duplicated().any() or group.atoms.isna().any():
            raise ValueError(f'{cohort}: need {count} distinct structures with atom counts')
        group['atom_bin'] = [min(4, i * 5 // len(group)) for i in range(len(group))]
        group['stratum'] = group.atom_bin.astype(str) + ':' + group.status.ne('complete').map({True: 'censored', False: 'complete'})
        group.loc[0, 'stratum'] = 'certainty:smallest'
        group.loc[len(group) - 1, 'stratum'] = 'certainty:largest'
        selected = set(group.index[group.status.ne('complete')]) | {0, len(group) - 1}
        for _, sub in group.groupby('stratum', sort=True):
            if not selected.intersection(sub.index):
                selected.add(rng.choice(list(sub.index)))
        if len(selected) > count:
            raise ValueError(f'{cohort}: quota cannot preserve censored cases and size strata')
        while len(selected) < count:
            available = [i for i in group.index if i not in selected]
            strata = sorted({group.loc[i, 'stratum'] for i in available},
                            key=lambda s: (sum(group.loc[i, 'stratum'] == s for i in selected), s))
            choices = [i for i in available if group.loc[i, 'stratum'] == strata[0]]
            selected.add(rng.choice(choices))
        for index in sorted(selected):
            row = group.loc[index].to_dict()
            row['population_count'] = int(group.stratum.eq(row['stratum']).sum())
            row['stratum_sample_count'] = sum(group.loc[i, 'stratum'] == row['stratum'] for i in selected)
            row['full_cohort_size'] = len(group)
            rows.append({k: row[k] for k in ('pdb_id', 'cohort', 'atoms', 'status', 'atom_bin', 'stratum',
                                            'population_count', 'stratum_sample_count', 'full_cohort_size')})
    return pd.DataFrame(rows)


def prepare(out, source, config):
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(source)
    cfg = yaml.safe_load(Path(config).read_text())
    selection = select_structures(table, seed=cfg['seed'])
    path = out / 'selection.csv'
    text = selection.to_csv(index=False)
    if path.exists() and path.read_text() != text:
        raise ValueError('Selection changed; use a new calibration output directory')
    path.write_text(text)
    provenance = dict(config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
                      source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                      selection_sha256=hashlib.sha256(text.encode()).hexdigest())
    previous = out / 'provenance.json'
    if previous.exists() and json.loads(previous.read_text()) != provenance:
        raise ValueError('Calibration provenance changed; use a new output directory')
    previous.write_text(json.dumps(provenance, indent=2))
    (out / 'config.yaml').write_text(Path(config).read_text())
    print(selection.groupby('cohort').size().to_string())
    return selection


def read_selection(out):
    provenance = json.loads((out / 'provenance.json').read_text())
    for name, key in [('selection.csv', 'selection_sha256'), ('config.yaml', 'config_sha256')]:
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != provenance[key]:
            raise ValueError(f'{name} changed after prepare; use a new output directory')
    table = pd.read_csv(out / 'selection.csv')
    if table.empty:
        raise ValueError('No selected calibration structures')
    return table


def download(out):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    selection = read_selection(out)
    data = out / 'data'
    data.mkdir(exist_ok=True)
    ledger_path = out / 'downloads.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    cfg = yaml.safe_load((out / 'config.yaml').read_text())
    used = sum(r['bytes'] for r in ledger)
    failed = []
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504])))
    for identifier in sorted(set(selection.pdb_id)):
        path = data / f'{identifier}.cif.gz'
        if any(r['pdb_id'] == identifier and r['status'] == 'complete' for r in ledger) and path.is_file() and path.stat().st_size:
            continue
        size = 0
        partial = path.with_suffix('.partial')
        try:
            with session.get(f'https://files.rcsb.org/download/{identifier.upper()}.cif.gz', stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                with partial.open('wb') as handle:
                    for block in response.iter_content(65536):
                        size += len(block)
                        used += len(block)
                        if used > cfg['download_limit_bytes']:
                            raise RuntimeError('Calibration download budget exceeded')
                        handle.write(block)
            if not size:
                raise ValueError('Empty download')
            partial.replace(path)
            status = 'complete'
        except Exception as exc:
            status = f'{type(exc).__name__}: {exc}'
            failed.append(identifier)
        ledger.append(dict(pdb_id=identifier, bytes=size, status=status))
        ledger_path.write_text(json.dumps(ledger, indent=2))
        print(f'{identifier}: {status}, {size} bytes', flush=True)
        if used > cfg['download_limit_bytes']:
            break
    if failed:
        raise RuntimeError(f'Downloads incomplete: {failed}')


def fingerprint(cfg):
    return hashlib.sha256(json.dumps({'config': cfg, 'region_radius': 0.}, sort_keys=True).encode()).hexdigest()


def run_workers(out, workers):
    cfg = yaml.safe_load((out / 'config.yaml').read_text())
    selection = read_selection(out)
    results = out / 'structures'
    results.mkdir(exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run([sys.executable, str(PILOT), '--stage', 'crosscheck', '--config', str(out / 'config.yaml'),
                    '--data', str(out / 'data')], check=True, env=environment)
    def launch(row):
        name = f'{row.cohort}_{row.pdb_id}'
        output = results / f'{name}.json'
        if output.exists():
            old = json.loads(output.read_text())
            if old.get('config_hash') == fingerprint(cfg) and old.get('status') in ('complete', 'size_censored', 'enumeration_censored', 'timeout_censored'):
                return
        if not (out / 'data' / f'{row.pdb_id}.cif.gz').is_file():
            raise FileNotFoundError(f'{row.pdb_id}: run download stage on login node first')
        command = ['/usr/bin/time', '-v', '-o', str(results / f'{name}.time.txt'), sys.executable, str(PILOT),
                   '--stage', 'worker', '--data', str(out / 'data'), '--config', str(out / 'config.yaml'),
                   '--cohort', row.cohort, '--pdb-id', row.pdb_id, '--output', str(output)]
        started = time.perf_counter()
        with (results / f'{name}.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment)
            try:
                code = process.wait(timeout=cfg['timeout_seconds'] + cfg['worker_grace_seconds'])
                if code or not output.exists():
                    raise RuntimeError(f'worker exit {code}; see {name}.log')
                result = json.loads(output.read_text())
                if result.get('status') == 'error':
                    raise RuntimeError(result.get('error', 'worker error'))
            except Exception as exc:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                result = dict(pdb_id=row.pdb_id, cohort=row.cohort, config_hash=fingerprint(cfg),
                              status='worker_error', error=str(exc))
            result['process_wall_seconds'] = time.perf_counter() - started
            output.write_text(json.dumps(result, indent=2, allow_nan=False))
            print(f'{name}: {result["status"]}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(launch, selection.itertuples(index=False)))


def summarize(out):
    selection = read_selection(out)
    cfg = yaml.safe_load((out / 'config.yaml').read_text())
    records, missing = [], []
    for row in selection.itertuples(index=False):
        path = out / 'structures' / f'{row.cohort}_{row.pdb_id}.json'
        result = json.loads(path.read_text()) if path.exists() else {}
        if not result or result.get('config_hash') != fingerprint(cfg):
            missing.append(f'{row.cohort}:{row.pdb_id}')
            continue
        metrics = result.get('metrics', [])
        projection = sum(m['candidate_seconds'] * m['pairwise_passing'] / m['sampled'] for m in metrics if m['sampled'])
        records.append(dict(cohort=row.cohort, pdb_id=row.pdb_id, stratum=row.stratum, status=result['status'],
            weight=row.population_count / row.stratum_sample_count, full_cohort_size=row.full_cohort_size,
            cpu_seconds=result.get('cpu_seconds'), process_wall_seconds=result.get('process_wall_seconds'),
            peak_rss_kib=result.get('peak_rss_kib'), exhaustive_candidate_seconds=projection if result['status'] == 'complete' else None))
    per_structure = pd.DataFrame(records)
    per_structure.to_csv(out / 'per_structure.csv', index=False)
    rows = []
    if records:
        for cohort, group in per_structure.groupby('cohort'):
            complete = group.status.eq('complete')
            cpu_known = group.cpu_seconds.notna()
            complete_weight = group.loc[complete, 'weight'].sum()
            weight = group.weight.sum()
            full = int(group.full_cohort_size.iloc[0])
            sampled = (group.loc[cpu_known, 'cpu_seconds'] * group.loc[cpu_known, 'weight']).sum() / weight / 3600 if cpu_known.all() else None
            projected = (group.loc[complete, 'exhaustive_candidate_seconds'] * group.loc[complete, 'weight']).sum() / complete_weight / 3600 if complete_weight else None
            selected_count = int(selection.cohort.eq(cohort).sum())
            if len(group) != selected_count or group.status.eq('worker_error').any():
                sampled = projected = None
            rows.append(dict(cohort=cohort, selected=selected_count, measured=len(group),
                full_cohort_size=full, complete=int(complete.sum()),
                censored_fraction=float(group.status.str.endswith('_censored').mean()),
                weighted_censored_fraction=float(group.loc[group.status.str.endswith('_censored'), 'weight'].sum() / weight),
                worker_error_count=int(group.status.eq('worker_error').sum()), max_rss_kib=float(group.peak_rss_kib.max()) if group.peak_rss_kib.notna().any() else None,
                sampled_core_hours_per_1000=sampled * 1000 if sampled is not None else None,
                sampled_core_hours_full_cohort=sampled * full if sampled is not None else None,
                exhaustive_candidate_core_hours_per_1000_complete=projected * 1000 if projected is not None else None,
                exhaustive_candidate_core_hours_full_cohort_complete_only=projected * full * complete_weight / weight if projected is not None else None))
    pd.DataFrame(rows).to_csv(out / 'summary.csv', index=False)
    report = dict(summary=rows, missing=missing, selection=json.loads((out / 'provenance.json').read_text()),
        interpretation='Size-stratum inverse-inclusion weights; full cohort sizes from source table. Exhaustive candidate-only projections exclude censored structures and all non-solver overhead; they are not full-run ETAs.')
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
    if missing or any(r['worker_error_count'] for r in rows):
        raise RuntimeError('Calibration incomplete; inspect missing/error records')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'download', 'run', 'summarize'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/tables/exp09_metal_site_pilot_per_structure.csv')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/exp09_metal_site_pilot.yaml')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.stage == 'prepare':
        prepare(out, args.source, args.config)
    elif args.stage == 'download':
        download(out)
    elif args.stage == 'run':
        run_workers(out, args.workers)
    else:
        summarize(out)


if __name__ == '__main__':
    main()
