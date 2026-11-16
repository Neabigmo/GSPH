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
"""Generate ProS-GNN inputs for all 223 M223 mutations.

Faithful to their pipeline: mutation-site window +/-3 residues extracted
as {key}_mutation.pdb (mutant structure) and {key}_wild.pdb (same residues
from WT chain). Mutant sidechains built with OUR mutation_builder.

Outputs under baselines_ext/prosgnn_data/:
  wild_pdb/{key}_wild.pdb, mutation_pdb/{key}_mutation.pdb,
  ddg/{key} (pickle float), keys_by_protein.json, manifest.csv
"""
import csv
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(r'H:\2026try\11.3GSPH')
sys.path.insert(0, str(ROOT / 'GSPH_H3' / 'gsph' / 'task_adapters'))
sys.path.insert(0, str(ROOT))

import intramolecular as I  # noqa: E402
import mutation_builder as MB  # noqa: E402

TABLE = ROOT / 'GSPH_H3' / 'baselines_ext' / 'm223_mutations.csv'
PDB_DIR = ROOT / 'GSPH_H3' / 'data' / 'membrane' / 'pdb_structures'
OUT = ROOT / 'GSPH_H3' / 'baselines_ext' / 'prosgnn_data'

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


def window_atoms(atoms, chain, resid, win=3):
    lo, hi = int(resid) - win, int(resid) + win
    return [a for a in atoms
            if a.chain == chain and lo <= int(a.resid) <= hi]


def write_pdb(path, atoms):
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = 0
    with open(path, 'w') as f:
        for a in atoms:
            serial += 1
            el = (a.element or a.name[0]).strip().upper()[:2].rjust(2)
            f.write(
                f"ATOM  {serial:>5d} {a.name:^4s} {a.resname[:3]:>3s} "
                f"{a.chain:1s}{int(a.resid):>4d}    "
                f"{a.xyz[0]:8.3f}{a.xyz[1]:8.3f}{a.xyz[2]:8.3f}"
                f"  1.00  0.00          {el}\n")
        f.write("END\n")


def main():
    (OUT / 'wild_pdb').mkdir(parents=True, exist_ok=True)
    (OUT / 'mutation_pdb').mkdir(parents=True, exist_ok=True)
    (OUT / 'ddg').mkdir(parents=True, exist_ok=True)
    cache = {}
    manifest, keys_by_prot = [], {}
    for r in csv.DictReader(open(TABLE)):
        pdbid, resid = r['pdbid'], int(r['resid'])
        mut1 = r['mut_aa'].strip().upper()
        wt1 = r['wt_aa'].strip().upper()
        try:
            mut3 = _AA1[mut1]
        except KeyError:
            manifest.append(dict(pdbid=pdbid, resid=resid, error='bad_mut_aa'))
            continue
        if pdbid not in cache:
            cache[pdbid] = I.parse_structure(str(PDB_DIR / f'{pdbid}.cif'))
        atoms = cache[pdbid]
        ch = resolve_chain(atoms, resid)
        try:
            mut_atoms, audit = MB.build_mutant_atoms(
                atoms, ch, resid, '', mut3)
        except Exception as e:
            manifest.append(dict(pdbid=pdbid, resid=resid, error=str(e)[:120]))
            continue
        key = f'{pdbid}_{resid}_{wt1}{resid}{mut1}'
        w_win = window_atoms(atoms, ch, resid)
        m_win = window_atoms(mut_atoms, ch, resid)
        if not w_win or not m_win:
            manifest.append(dict(pdbid=pdbid, resid=resid, error='empty_window'))
            continue
        write_pdb(OUT / 'wild_pdb' / f'{key}_wild.pdb', w_win)
        write_pdb(OUT / 'mutation_pdb' / f'{key}_mutation.pdb', m_win)
        with open(OUT / 'ddg' / key, 'wb') as f:
            pickle.dump(float(r['ddG']), f)
        keys_by_prot.setdefault(r['protein'], []).append(key)
        manifest.append(dict(pdbid=pdbid, resid=resid, chain=ch, key=key,
                             n_w=len(w_win), n_m=len(m_win), error=''))
    with open(OUT / 'keys_by_protein.json', 'w') as f:
        json.dump(keys_by_prot, f, indent=1)
    with open(OUT / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pdbid', 'resid', 'chain', 'key',
                                        'n_w', 'n_m', 'error'])
        w.writeheader()
        w.writerows(manifest)
    ok = sum(1 for m in manifest if not m.get('error'))
    errs = [m for m in manifest if m.get('error')]
    print(f'generated {ok}/{len(manifest)} prosgnn inputs; errors: {len(errs)}')
    for m in errs[:10]:
        print(' ERR', m)


if __name__ == '__main__':
    main()
