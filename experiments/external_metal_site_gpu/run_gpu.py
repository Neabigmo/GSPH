"""Run Metal3D and AllMetal3D on the 32 time-split Zn structures with a CUDA GPU.

Official code and weights are used unchanged; only the device is CUDA.
One command does setup, inference and writes three small CSV tables.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
METAL3D_COMMIT = '09cbfb3c30e71e4fbea7e535072506c66af97cd9'
METAL3D_ZIP = f'https://github.com/lcbc-epfl/metal-site-prediction/archive/{METAL3D_COMMIT}.zip'
ALLMETAL3D_VERSION = '0.7.3'
HF = 'https://huggingface.co/spaces/simonduerr/allmetal3d/resolve/main/'
ALLMETAL3D_WEIGHTS = {
    'identity_vacancy_geometry_model_2024-03-01_train_lessclasses_skipconnect_hyperparametertuned_geometry_identity_all_g09_300epoch_p01_epoch300.pth':
        'da49b26e8a4eb918dca3b144961312f66eb3218ee0eae1a0d1548feb12d727cc',
    'water_0.5A_allmetal_one3dchannel_epoch9.pth': '6b762d94c66fe098a0c63696f5b1c87d955a7465a433628f42896536b9ba1310',
    'metal_0.5A_allmetal_one3dchannel_16Abox_filter6_epoch6.pth': '38889a0d0f53a0dae8eb8b07268829fe959e346111f9739ad3b3e47194de0aab',
}
METAL3D_DEPS = ['moleculekit', 'py3Dmol', 'scikit-learn', 'scipy', 'tqdm', 'biopython']
ALLMETAL3D_DEPS = METAL3D_DEPS + ['halo', 'tabulate', 'requests']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pip_target(packages, target):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(target), *packages], check=True)


def ensure_deps(modules, packages, target):
    missing = []
    for module, package in zip(modules, packages):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        pip_target(missing, target)


def setup_metal3d(work):
    root = work / 'metal3d'
    if not (root / 'Metal3D' / 'weights' / 'metal_0.5A_v3_d0.2_16Abox.pth').exists():
        data = urllib.request.urlopen(METAL3D_ZIP).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            prefix = f'metal-site-prediction-{METAL3D_COMMIT}/Metal3D/'
            for name in z.namelist():
                if name.startswith(prefix) and not name.endswith('/'):
                    out = root / 'Metal3D' / name[len(prefix):]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(name))
    return root / 'Metal3D'


def setup_allmetal3d(work):
    home = work / 'home'
    weights = home / '.local' / 'share' / 'allmetal3d' / 'weights'
    weights.mkdir(parents=True, exist_ok=True)
    for name, digest in ALLMETAL3D_WEIGHTS.items():
        path = weights / name
        if not path.exists() or sha256(path) != digest:
            urllib.request.urlretrieve(HF + name, path)
        if sha256(path) != digest:
            raise RuntimeError(f'checksum mismatch for {name}')
    return home


def unpack_inputs(work):
    folder = work / 'inputs'
    folder.mkdir(parents=True, exist_ok=True)
    ids = (HERE / 'zn_time_split_32.txt').read_text().split()
    for identifier in ids:
        target = folder / f'{identifier}.pdb'
        if not target.exists():
            target.write_bytes(gzip.decompress((HERE / 'inputs' / f'{identifier}.pdb.gz').read_bytes()))
    return ids, folder


def read_probes(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith('HETATM'):
                f = line.split()
                rows.append(dict(x=float(f[6]), y=float(f[7]), z=float(f[8]), confidence=float(f[9])))
    return rows


def run_metal3d(ids, inputs, work, env):
    code = setup_metal3d(work)
    out = work / 'out_metal3d'
    out.mkdir(exist_ok=True)
    for identifier in ids:
        probes = out / f'{identifier}_probes.pdb'
        start = time.perf_counter()
        p = subprocess.run([sys.executable, 'metal3d.py', '--pdb', str(inputs / f'{identifier}.pdb'), '--metalbinding',
                            '--softexit', '--writeprobes', '--probefile', str(probes), '--batch-size', '32',
                            '--threshold', '7', '--pthreshold', '0.10'],
                           cwd=code, env=env, capture_output=True, text=True)
        (out / f'{identifier}.log').write_text(p.stdout + p.stderr)
        ok = p.returncode == 0
        rows = read_probes(probes) if ok else []
        yield 'Metal3D', identifier, time.perf_counter() - start, 'success' if ok else f'error_{p.returncode}', rows, None


def run_allmetal3d(ids, inputs, work):
    import allmetal3d
    out = work / 'out_allmetal3d'
    out.mkdir(exist_ok=True)
    for identifier in ids:
        start = time.perf_counter()
        try:
            probefile, _, _, _, details = allmetal3d.predict(
                input_pdb=str(inputs / f'{identifier}.pdb'), models='allmetal3d', mode='fast',
                batch_size=32, threshold=7, pthreshold=0.25, output_dir=str(out))
            rows, status = read_probes(Path(probefile)), 'success'
        except Exception as error:
            rows, status, details = [], f'error_{type(error).__name__}: {error}', None
        yield 'AllMetal3D', identifier, time.perf_counter() - start, status, rows, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True, help='scratch folder for code, weights and raw outputs')
    parser.add_argument('--method', choices=['metal3d', 'allmetal3d', 'both'], default='both')
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)

    import torch
    if not torch.cuda.is_available():
        sys.exit('CUDA is not available in this Python environment')
    device = torch.cuda.get_device_name(0)

    deps = work / 'pydeps'
    deps.mkdir(exist_ok=True)
    sys.path.insert(0, str(deps))
    ensure_deps(['moleculekit', 'py3Dmol', 'sklearn', 'scipy', 'tqdm', 'Bio'], METAL3D_DEPS, deps)
    if args.method != 'metal3d':
        ensure_deps(['halo', 'tabulate', 'requests'], ['halo', 'tabulate', 'requests'], deps)
        try:
            import allmetal3d  # noqa: F401
        except ImportError:
            pip_target([f'allmetal3d=={ALLMETAL3D_VERSION}'], deps)
        home = setup_allmetal3d(work)
        os.environ['HOME'] = os.environ['USERPROFILE'] = str(home)

    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(deps), os.environ.get('PYTHONPATH', '')])}
    ids, inputs = unpack_inputs(work)

    tables = REPO / 'results' / 'tables'
    predictions, runtimes, details_out = [], [], []
    runs = []
    if args.method in ('metal3d', 'both'):
        runs.append(run_metal3d(ids, inputs, work, env))
    if args.method in ('allmetal3d', 'both'):
        runs.append(run_allmetal3d(ids, inputs, work))
    for run in runs:
        for method, identifier, seconds, status, rows, details in run:
            print(method, identifier, status, f'{seconds:.1f}s', len(rows), flush=True)
            runtimes.append(dict(method=method, pdb_id=identifier, seconds=round(seconds, 3), status=status,
                                 n_predictions=len(rows), device=device))
            for rank, row in enumerate(sorted(rows, key=lambda r: -r['confidence']), 1):
                predictions.append(dict(method=method, pdb_id=identifier, rank=rank, **row))
            if details is not None:
                details_out.append(json.dumps(dict(pdb_id=identifier, details=details), default=str))

    def write(name, rows):
        with open(tables / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write('external_metal_gpu_runtime.csv', runtimes)
    if predictions:
        write('external_metal_gpu_predictions.csv', predictions)
    if details_out:
        (tables / 'external_allmetal_gpu_details.jsonl').write_text('\n'.join(details_out) + '\n')
    versions = [dict(package='python', version=sys.version.split()[0]), dict(package='torch', version=torch.__version__),
                dict(package='cuda_device', version=device), dict(package='Metal3D', version=METAL3D_COMMIT),
                dict(package='allmetal3d', version=ALLMETAL3D_VERSION)]
    write('external_metal_gpu_software.csv', versions)
    print('wrote tables to', tables)


if __name__ == '__main__':
    main()
