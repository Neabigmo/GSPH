"""Download Metal3D and AllMetal3D weights and make both run on CPU.

Run once on the login node, which has network access. The only source edits
are device selection, CPU-mapped weight loading, limiting worker processes
to the CPUs Slurm allocated, running inference without autograd and dropping
an unused turtle import that needs Tk; model definitions and thresholds are untouched.
"""
import argparse
import hashlib
import importlib.util
import io
import urllib.request
import zipfile
from pathlib import Path

METAL3D_COMMIT = '09cbfb3c30e71e4fbea7e535072506c66af97cd9'
METAL3D_ZIP = f'https://github.com/lcbc-epfl/metal-site-prediction/archive/{METAL3D_COMMIT}.zip'
METAL3D_WEIGHTS = 'weights/metal_0.5A_v3_d0.2_16Abox.pth'
HF = 'https://huggingface.co/spaces/simonduerr/allmetal3d/resolve/main/'
ALLMETAL3D_WEIGHTS = {
    'identity_vacancy_geometry_model_2024-03-01_train_lessclasses_skipconnect_hyperparametertuned_geometry_identity_all_g09_300epoch_p01_epoch300.pth':
        'da49b26e8a4eb918dca3b144961312f66eb3218ee0eae1a0d1548feb12d727cc',
    'water_0.5A_allmetal_one3dchannel_epoch9.pth': '6b762d94c66fe098a0c63696f5b1c87d955a7465a433628f42896536b9ba1310',
    'metal_0.5A_allmetal_one3dchannel_16Abox_filter6_epoch6.pth': '38889a0d0f53a0dae8eb8b07268829fe959e346111f9739ad3b3e47194de0aab',
}
ALLOCATED = 'int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))'
CUDA_BUFFER = 'device="cuda")'
ANY_BUFFER = 'device="cuda" if torch.cuda.is_available() else "cpu")'
NO_GRAD = ('with warnings.catch_warnings():\n', 'with warnings.catch_warnings(), torch.no_grad():\n')
CUDA_CHECK = ('if torch.cuda.is_available()==False:\n'
              '    raise ImportError("No CUDA detected by PyTorch, please install a CUDA compatible PyTorch build")\n')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def edit(path, old, new, count):
    """Replace exactly `count` copies of `old`; a second run is a no-op."""
    text = path.read_text()
    if old not in text and (not new or new in text):
        return
    if text.count(old) != count:
        raise RuntimeError(f'{path.name}: expected {count} copies of {old!r}, found {text.count(old)}')
    path.write_text(text.replace(old, new))


def limit_workers(path):
    text = path.read_text()
    if 'import os\n' not in text:
        path.write_text(text.replace('import multiprocessing\n', 'import multiprocessing\nimport os\n', 1))
    edit(path, 'multiprocessing.cpu_count()\n', ALLOCATED + '\n', 1)


def prepare_metal3d(tools):
    code = tools / 'Metal3D'
    if not (code / METAL3D_WEIGHTS).exists():
        with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(METAL3D_ZIP).read())) as z:
            prefix = f'metal-site-prediction-{METAL3D_COMMIT}/Metal3D/'
            for name in z.namelist():
                if name.startswith(prefix) and not name.endswith('/'):
                    out = code / name[len(prefix):]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(name))
    patch_metal3d(code)


def patch_metal3d(code):
    edit(code / 'metal3d.py', f'torch.load(f"{METAL3D_WEIGHTS}")',
         f'torch.load(f"{METAL3D_WEIGHTS}", map_location=device)', 1)
    edit(code / 'utils' / 'voxelization.py', CUDA_BUFFER, ANY_BUFFER, 1)
    limit_workers(code / 'utils' / 'voxelization.py')
    limit_workers(code / 'utils' / 'helpers.py')
    edit(code / 'utils' / 'helpers.py', 'from turtle import width\n', '', 1)
    edit(code / 'metal3d.py', *NO_GRAD, 1)


def patch_allmetal3d(package):
    edit(package / '__init__.py', CUDA_CHECK, '', 1)
    edit(package / 'utils' / 'main.py', 'weights_only=True\n', 'weights_only=True, map_location=device\n', 6)
    edit(package / 'utils' / 'voxelizationNew.py', CUDA_BUFFER, ANY_BUFFER, 1)
    edit(package / 'utils' / 'main.py', *NO_GRAD, 2)
    limit_workers(package / 'utils' / 'voxelizationNew.py')
    limit_workers(package / 'utils' / 'helpersNew.py')


def prepare_allmetal3d(tools):
    weights = tools / 'home' / '.local' / 'share' / 'allmetal3d' / 'weights'
    weights.mkdir(parents=True, exist_ok=True)
    for name, digest in ALLMETAL3D_WEIGHTS.items():
        path = weights / name
        if not path.exists() or sha256(path) != digest:
            urllib.request.urlretrieve(HF + name, path)
        if sha256(path) != digest:
            raise RuntimeError(f'checksum mismatch for {name}')
    patch_allmetal3d(Path(importlib.util.find_spec('allmetal3d').origin).parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools', type=Path, required=True, help='folder for Metal3D code and AllMetal3D weights')
    args = parser.parse_args()
    args.tools.mkdir(parents=True, exist_ok=True)
    prepare_metal3d(args.tools)
    prepare_allmetal3d(args.tools)
    print('tools ready in', args.tools)


if __name__ == '__main__':
    main()
