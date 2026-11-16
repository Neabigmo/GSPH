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
#!/usr/bin/env python3
"""Batch VoroMQA scoring of FTDMP protein-DNA decoys. Checkpointed, resumable.
Env: PD_DECOY_ROOT, PD_OUT_JSONL, PD_TOOL_PATH (dir containing voronota+scripts)."""
import json, os, subprocess, sys, time
from pathlib import Path
from multiprocessing import Pool

DECOY_ROOT = Path(os.environ.get('PD_DECOY_ROOT', '/root/pd_decoys/Protein-DNA'))
OUT_JSONL = Path(os.environ.get(
    'PD_OUT_JSONL',
    '/mnt/h/2026try/11.3GSPH/GSPH_H3/baselines_ext/pd_voromqa_scores.jsonl'))
TOOL_PATH = os.environ.get(
    'PD_TOOL_PATH',
    '/root/voro2:/mnt/h/2026try/11.3GSPH/GSPH_H3/external/voronota')
ENV = dict(os.environ)
ENV['PATH'] = TOOL_PATH + ':' + ENV['PATH']


def score_one(args):
    pdb, rel = args
    t0 = time.time()
    try:
        r = subprocess.run(['voronota-voromqa', '-i', pdb], capture_output=True,
                           text=True, env=ENV, timeout=300)
        toks = r.stdout.strip().split()
        if len(toks) >= 4:
            return {'pdb': rel, 'voromqa': float(toks[1]), 'nres': int(toks[2]),
                    'natoms': int(toks[3]), 'sec': round(time.time() - t0, 2),
                    'err': None}
        return {'pdb': rel, 'voromqa': None,
                'err': (r.stderr or r.stdout).strip()[:200]}
    except Exception as e:
        return {'pdb': rel, 'voromqa': None, 'err': str(e)[:200]}


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    pdbs = sorted(str(p) for p in DECOY_ROOT.rglob('*.pdb'))
    done = set()
    if OUT_JSONL.exists():
        for line in OUT_JSONL.open():
            try:
                done.add(json.loads(line)['pdb'])
            except Exception:
                pass
    todo = []
    for p in pdbs:
        rel = str(Path(p).relative_to(DECOY_ROOT))
        if rel not in done:
            todo.append((p, rel))
    print(f'total={len(pdbs)} done={len(done)} todo={len(todo)}', flush=True)
    out = OUT_JSONL.open('a', buffering=1)
    n = 0
    with Pool(nproc) as pool:
        for rec in pool.imap_unordered(score_one, todo, chunksize=4):
            out.write(json.dumps(rec) + '\n')
            n += 1
            if n % 200 == 0:
                print(f'scored={n}/{len(todo)}', flush=True)
    out.close()
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
