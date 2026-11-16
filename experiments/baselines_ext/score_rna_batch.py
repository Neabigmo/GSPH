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
"""Batch score rsRNASP PM decoys with weight-free statistical potentials.

Methods (all re-run on OUR exact 826 structures, no pretrained weights):
  3dRNAscore (dist+torsion stat potential), DFIRE-RNA, RASP (rasp_fd), RNA-BRiQ.
Output: JSONL checkpoint + merged CSV keyed by manifest structure_id.
"""
import json, os, re, subprocess, sys, tempfile, time
from multiprocessing import Pool
from pathlib import Path

PM = Path(os.environ.get('RNA_PM_DIR', '/mnt/h/2026try/11.3GSPH/GSPH_H3/external/rsRNASP/PM_decoy_set'))
IDS_CSV = Path(os.environ.get('RNA_IDS_CSV', '/mnt/h/2026try/11.3GSPH/GSPH_H3/baselines_ext/rna_structure_ids.csv'))
OUT_JSONL = Path(os.environ.get('RNA_OUT_JSONL', '/mnt/h/2026try/11.3GSPH/GSPH_H3/baselines_ext/rna_statpot_scores.jsonl'))
OUT_CSV = Path(os.environ.get('RNA_OUT_CSV', '/mnt/h/2026try/11.3GSPH/GSPH_H3/baselines_ext/rna_statpot_scores.csv'))
TOOLS = os.environ.get('RNA_TOOLS_DIR', '/root/rna_tools')

ENV = dict(os.environ,
           RNAscore=f'{TOOLS}/3dRNAscore/3dRNAscore',
           RNA=f'{TOOLS}/3dRNAscore/3dRNAscore/lib/',
           DFIRE_RNA_HOME=f'{TOOLS}/DFIRE-RNA/dfire',
           RASP=f'{TOOLS}/RASP/rasp-fd-1.0',
           BRiQ_DATAPATH=f'{TOOLS}/BRiQ_data/BRiQ_data')

BINS = dict(
    fmt_pl=f'{TOOLS}/3dRNAscore/3dRNAscore/lib/format.pl',
    rnascore=f'{TOOLS}/3dRNAscore/3dRNAscore/bin/3dRNAscore',
    dfire=f'{TOOLS}/DFIRE-RNA/dfire/bin/DFIRE_RNA',
    rasp=f'{TOOLS}/RASP/rasp-fd-1.0/bin/rasp_fd',
    briq_ss=f'{TOOLS}/BRiQ/RNA-BRiQ/build/bin/BRiQ_AssignSS',
    briq_e=f'{TOOLS}/BRiQ/RNA-BRiQ/build/bin/BRiQ_Energy')


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env=ENV,
                          timeout=kw.pop('timeout', 300), **kw)


def score_3drnascore(pdb):
    with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as f:
        fmt = f.name
    try:
        r = subprocess.run(['perl', BINS['fmt_pl'], str(pdb)], capture_output=True,
                           text=True, env=ENV, timeout=60)
        Path(fmt).write_text(r.stdout)
        r = run([BINS['rnascore'], '-s', fmt])
        toks = r.stdout.strip().split()
        return float(toks[-1]) if toks else float('nan')
    finally:
        try: os.unlink(fmt)
        except OSError: pass


def score_dfire(pdb):
    r = run([BINS['dfire'], str(pdb)])
    toks = r.stdout.strip().split()
    return float(toks[1]) if len(toks) > 1 else float('nan')


def score_rasp(pdb):
    r = run([BINS['rasp'], '-p', str(pdb)])
    toks = r.stdout.strip().split()
    return float(toks[0]) if toks else float('nan')


def score_briq(pdb):
    ss = tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False)
    ss.close()
    ss2 = tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False)
    try:
        run([BINS['briq_ss'], str(pdb), ss.name], timeout=600)
        content = Path(ss.name).read_text()
        ss2.write(f'pdb {pdb}\n' + content)
        ss2.close()
        r = run([BINS['briq_e'], ss2.name], timeout=600)
        toks = r.stderr.strip().split()
        return float(toks[1]) if len(toks) > 1 else float('nan')
    finally:
        for f_ in (ss.name, ss2.name):
            try: os.unlink(f_)
            except OSError: pass


SCORERS = {'score_3drnascore': score_3drnascore, 'score_dfire': score_dfire,
           'score_rasp': score_rasp, 'score_briq': score_briq}


def find_pdb(sid):
    target = sid.split('-')[-1].upper() if '-' in sid else sid.upper()
    for tgt_dir in PM.iterdir():
        if not tgt_dir.is_dir():
            continue
        p = tgt_dir / f'{sid}.pdb'
        if p.is_file():
            return p
    return None


def worker(job):
    sid, pdb = job
    rec = {'structure_id': sid, 'pdb': pdb}
    t0 = time.time()
    for name, fn in SCORERS.items():
        try:
            rec[name] = fn(Path(pdb))
        except Exception as e:
            rec[name] = float('nan')
            rec.setdefault('errors', {})[name] = str(e)[:200]
    rec['elapsed_s'] = round(time.time() - t0, 2)
    return rec


def main():
    ids = [l.strip() for l in IDS_CSV.read_text().splitlines()
           if l.strip() and l.strip() != 'structure_id']
    done = set()
    if OUT_JSONL.exists():
        for l in OUT_JSONL.read_text().splitlines():
            try:
                done.add(json.loads(l)['structure_id'])
            except Exception:
                pass
    jobs = []
    missing_pdb = []
    for sid in ids:
        if sid in done:
            continue
        p = find_pdb(sid)
        if p is None:
            missing_pdb.append(sid)
        else:
            jobs.append((sid, str(p)))
    print(f'total={len(ids)} done={len(done)} todo={len(jobs)} missing_pdb={missing_pdb[:10]}')
    n = 0
    with OUT_JSONL.open('a') as out, Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as pool:
        for rec in pool.imap_unordered(worker, jobs):
            out.write(json.dumps(rec) + '\n')
            out.flush()
            n += 1
            if n % 25 == 0:
                print(f'  {n}/{len(jobs)}', flush=True)
    print('all done')


if __name__ == '__main__':
    main()
