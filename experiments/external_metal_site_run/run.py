"""Run Metal3D or AllMetal3D on one time-split Zn structure, or collect all runs.

One Slurm array task handles one structure. Each task writes a few kilobytes:
predicted sites, a runtime record and, for AllMetal3D, identity probabilities.
"""
import argparse
import csv
import gzip
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IDS = (HERE / 'zn_time_split_32.txt').read_text().split()
THRESHOLD = 7
P_THRESHOLD = {'metal3d': 0.10, 'allmetal3d': 0.25}
RUNTIME_FIELDS = ['method', 'pdb_id', 'status', 'seconds', 'n_sites', 'cpus', 'batch_size', 'peak_rss_mb', 'torch']


def read_sites(path):
    """Both tools write site probability in the occupancy column of their probe PDB."""
    sites = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith(('HETATM', 'ATOM')):
                sites.append(dict(element=line[17:20].strip(), x=float(line[30:38]), y=float(line[38:46]),
                                  z=float(line[46:54]), probability=float(line[54:60])))
    return sites


def unpack(identifier, scratch):
    pdb = scratch / f'{identifier}.pdb'
    pdb.write_bytes(gzip.decompress((HERE / 'inputs' / f'{identifier}.pdb.gz').read_bytes()))
    return pdb


def run_metal3d(pdb, scratch, tools, batch_size):
    probes = scratch / 'probes.pdb'
    p = subprocess.run([sys.executable, 'metal3d.py', '--pdb', str(pdb), '--metalbinding', '--softexit',
                        '--writeprobes', '--probefile', str(probes), '--batch-size', str(batch_size),
                        '--threshold', str(THRESHOLD), '--pthreshold', str(P_THRESHOLD['metal3d'])],
                       cwd=tools / 'Metal3D', capture_output=True, text=True)
    (scratch / 'metal3d.log').write_text(p.stdout + p.stderr)
    if p.returncode:
        return f'error_{p.returncode}', [], None
    return 'success', read_sites(probes), None


def run_allmetal3d(pdb, scratch, tools, batch_size):
    os.environ['HOME'] = str(tools / 'home')
    import allmetal3d
    probefile, _, _, _, details = allmetal3d.predict(
        input_pdb=str(pdb), models='allmetal3d', mode='fast', batch_size=batch_size,
        threshold=THRESHOLD, pthreshold=P_THRESHOLD['allmetal3d'], output_dir=str(scratch))
    return 'success', read_sites(Path(probefile)), details


def run_one(args):
    identifier = IDS[args.index] if args.pdb_id is None else args.pdb_id
    scratch = args.out / args.method / identifier
    scratch.mkdir(parents=True, exist_ok=True)
    import torch
    cpus = os.environ.get('SLURM_CPUS_PER_TASK', '')
    if cpus:
        torch.set_num_threads(int(cpus))
    runner = run_metal3d if args.method == 'metal3d' else run_allmetal3d
    start = time.perf_counter()
    try:
        status, sites, details = runner(unpack(identifier, scratch), scratch, args.tools, args.batch_size)
    except Exception as error:
        status, sites, details = f'error_{type(error).__name__}: {error}', [], None
    peak_kb = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    record = dict(method=args.method, pdb_id=identifier, status=status, seconds=round(time.perf_counter() - start, 1),
                  n_sites=len(sites), cpus=cpus, batch_size=args.batch_size, peak_rss_mb=round(peak_kb / 1024),
                  torch=torch.__version__)
    (scratch / 'result.json').write_text(json.dumps(dict(record=record, sites=sites, details=details), default=str))
    print(json.dumps(record), flush=True)
    return 0 if status == 'success' else 1


def write(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def collect(args):
    runtime, sites, details = [], [], []
    for method in ('metal3d', 'allmetal3d'):
        for identifier in IDS:
            path = args.out / method / identifier / 'result.json'
            if not path.exists():
                runtime.append({**dict.fromkeys(RUNTIME_FIELDS, ''), 'method': method, 'pdb_id': identifier,
                                'status': 'missing', 'n_sites': 0})
                continue
            result = json.loads(path.read_text())
            runtime.append(result['record'])
            for rank, site in enumerate(result['sites'], 1):
                sites.append(dict(method=method, pdb_id=identifier, rank=rank, **site))
            if result['details']:
                details.append(json.dumps(dict(pdb_id=identifier, details=result['details']), default=str))
    tables = REPO / 'results' / 'tables'
    write(tables / 'external_metal_cpu_runtime.csv', runtime)
    if sites:
        write(tables / 'external_metal_cpu_predictions.csv', sites)
    if details:
        (tables / 'external_allmetal_cpu_details.jsonl').write_text('\n'.join(details) + '\n')
    done = sum(r['status'] == 'success' for r in runtime)
    print(f'{done}/{len(runtime)} runs succeeded; tables written to {tables}')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='output folder outside the repository')
    parser.add_argument('--tools', type=Path, help='folder made by setup_env.sh')
    parser.add_argument('--method', choices=['metal3d', 'allmetal3d'])
    parser.add_argument('--index', type=int, help='array index into zn_time_split_32.txt')
    parser.add_argument('--pdb-id')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--collect', action='store_true', help='merge all result.json files into results/tables')
    args = parser.parse_args()
    if args.collect:
        return collect(args)
    if args.tools is None or args.method is None or (args.index is None and args.pdb_id is None):
        parser.error('--tools, --method and --index or --pdb-id are required')
    return run_one(args)


if __name__ == '__main__':
    sys.exit(main())
