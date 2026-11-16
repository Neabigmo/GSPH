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
"""Generate TopologyNet/T-MP .mutinfo files for all 223 M223 mutations.

Preprocessing is reimplemented (their preprocessing code is not public):
all atoms within 10 A of the mutated residue CA — matching their paper's
"near the mutation site (within 10 A)" definition.
Mutant sidechains are built with OUR mutation_builder (same idealized
geometry used by the GSPH membrane pipeline) for consistency.

Output: mutinfo/{pdbid}_{resid}_{mut}/wild.mutinfo + mut.mutinfo
Format per line: <resid> <element> <x> <y> <z>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

import os
ROOT = Path(os.environ.get('GSPH_ROOT', '/mnt/h/2026try/11.3GSPH'))
sys.path.insert(0, str(ROOT / 'GSPH_H3' / 'gsph' / 'task_adapters'))
sys.path.insert(0, str(ROOT))

import intramolecular as I  # noqa: E402
import mutation_builder as MB  # noqa: E402

TABLE = ROOT / 'GSPH_H3' / 'data' / 'membrane' / '../../baselines_ext/m223_mutations.csv'
PDB_DIR = ROOT / 'GSPH_H3' / 'data' / 'membrane' / 'pdb_structures'
OUT = Path(os.environ.get('MUTINFO_OUT', '/root/tmp_mutinfo'))
RADII = 10.0

_AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
_AA1 = {v: k for k, v in _AA3.items()}


def resolve_chain(atoms, resid):
    chains = sorted({a.chain for a in atoms})
    for ch in chains:
        if any(a.chain == ch and int(a.resid) == int(resid)
               and a.name == 'CA' for a in atoms):
            return ch
    return chains[0]


def neighborhood(atoms, center, radius=RADII):
    return [a for a in atoms
            if np.linalg.norm(np.asarray(a.xyz) - center) <= radius]


def write_mutinfo(path, atoms):
    with open(path, 'w') as f:
        for a in atoms:
            el = (a.element or a.name[0]).strip().upper()[:1]
            f.write(f"{int(a.resid)} {el} {a.xyz[0]:.3f} {a.xyz[1]:.3f} {a.xyz[2]:.3f}\n")


def main():
    df = list(csv.DictReader(open(TABLE)))
    OUT.mkdir(exist_ok=True)
    cache, manifest = {}, []
    for r in df:
        pdbid, resid = r['pdbid'], int(r['resid'])
        wt1, mut1 = r['wt_aa'].strip().upper(), r['mut_aa'].strip().upper()
        mut3 = _AA1.get(mut1)
        if mut3 is None:
            manifest.append(dict(pdbid=pdbid, resid=resid, error='bad_mut_aa'))
            continue
        if pdbid not in cache:
            cache[pdbid] = I.parse_structure(str(PDB_DIR / f'{pdbid}.cif'))
        atoms = cache[pdbid]
        ch = resolve_chain(atoms, resid)
        site = [a for a in atoms if a.chain == ch and int(a.resid) == resid]
        cas = [a for a in site if a.name == 'CA']
        if not cas:
            manifest.append(dict(pdbid=pdbid, resid=resid, error='no_CA'))
            continue
        center = np.asarray(cas[0].xyz)
        try:
            mut_atoms, audit = MB.build_mutant_atoms(
                atoms, ch, resid, '', mut3)
        except Exception as e:
            manifest.append(dict(pdbid=pdbid, resid=resid, error=str(e)[:120]))
            continue
        tag = f'{pdbid}_{resid}_{wt1}{resid}{mut1}'
        d = OUT / tag
        d.mkdir(exist_ok=True)
        write_mutinfo(d / 'wild.mutinfo', neighborhood(atoms, center))
        write_mutinfo(d / 'mut.mutinfo', neighborhood(mut_atoms, center))
        manifest.append(dict(pdbid=pdbid, resid=resid, chain=ch,
                             mutation=f'{wt1}{resid}{mut1}', tag=tag,
                             wt_atoms=len(neighborhood(atoms, center)),
                             mut_atoms=len(neighborhood(mut_atoms, center))))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    ok = sum(1 for m in manifest if 'tag' in m)
    print(f'generated {ok}/{len(df)} mutinfo pairs; errors: '
          f'{[m for m in manifest if "error" in m][:5]}')


if __name__ == '__main__':
    main()
