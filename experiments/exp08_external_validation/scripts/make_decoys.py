# -*- coding: utf-8 -*-
"""EXP-08: independent protein-DNA targets from Dockground non-redundant set.

Selects PDB complexes disjoint from the 47-case FTDMP/Prot-DNABenchmark
development set, downloads bound structures, and generates rigid-body
decoys of the DNA duplex with ligand-RMSD labels (native vs decoy).

Outputs per target: exp08/decoys/<pdb>/models/model_XXX.pdb + manifest.csv
Decoy protocol (documented in experiment_card): the two DNA chains are
kept as a rigid duplex and perturbed by random rotation (axis uniform on
sphere, angle U[15,180] deg) about the DNA centroid plus translation
U[2,15] A; a second tier perturbs each DNA chain independently to mimic
partial unbinding. Label: ligand-RMSD of DNA heavy atoms after Kabsch
superposition of the PROTEIN chains (protein-fixed frame), native=0.
"""
import csv, gzip, os, sys, urllib.request, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PDB_DIR = HERE / "pdb"
OUT_DIR = HERE / "decoys"
BENCH_DIR = Path(r"H:\2026try\11.3GSPH\GSPH_H3\data\protein_dna\ftdmp_labels\Protein-DNA\Bound_docking")
NR_CSV = HERE / "dna_nr_seq.csv.gz"
N_DECOYS = 24
rng = np.random.default_rng(20261117)

AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
       "MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE","SEC","PYL"}
DNA_RES = {"DA","DC","DG","DT","DU","DI","A","C","G","T","U","I",
           "ADE","CYT","GUA","THY","URA"}


def parse_pdb(path):
    atoms = []  # (serial, name, resn, chain, x,y,z, element, line)
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resn = line[17:20].strip()
        atoms.append(dict(
            serial=line[6:11].strip(), name=line[12:16].strip(),
            resn=resn, chain=line[21].strip(),
            xyz=np.array([float(line[30:38]), float(line[38:46]),
                          float(line[46:54])]),
            elem=line[76:78].strip() or line[12:16].strip()[0],
            het=line.startswith("HETATM"), line=line.rstrip("\n")))
    return atoms


def is_dna(a):
    return a["resn"] in DNA_RES or (a["het"] and a["resn"] not in AA3
                                    and a["chain"].isdigit() is False
                                    and a["resn"] in DNA_RES)


def kabsch(P, Q):
    """Return R,t minimizing |P@R + t - Q|; P,Q (n,3)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, Qc - Pc @ R


def rand_rot(rng):
    v = rng.normal(size=3); v /= np.linalg.norm(v)
    ang = np.deg2rad(rng.uniform(15, 180))
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def write_pdb(atoms, path):
    with open(path, "w") as f:
        for i, a in enumerate(atoms, 1):
            l = a["line"]
            x, y, z = a["xyz"]
            l = l[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + l[54:]
            l = l[:6] + f"{i:5d}" + l[11:]
            f.write(l + "\n")
        f.write("END\n")


def ligand_rmsd(native_dna_xyz, dec_dna_xyz, prot_xyz):
    """Superpose decoy protein onto native protein, then DNA RMSD."""
    R, t = kabsch(dec_dna_xyz[:0] if False else prot_xyz[1], prot_xyz[0])
    moved = dec_dna_xyz @ R + t
    return float(np.sqrt(((moved - native_dna_xyz) ** 2).sum(1).mean()))


def main():
    bench = {d.upper() for d in os.listdir(BENCH_DIR)} if BENCH_DIR.exists() else set()
    rows = [x for x in csv.DictReader(gzip.open(NR_CSV, "rt"))
            if x["PDB Code"].upper() not in bench]
    targets_file = HERE / "exp08_targets.csv"
    picks = []
    if targets_file.exists():
        picks = [r["pdb"] for r in csv.DictReader(open(targets_file))]
    else:
        k = max(1, len(rows) // 20)
        picks = [x["PDB Code"] for x in rows[::k][:20]]
    meta = {}
    for x in rows:
        meta[x["PDB Code"].lower()] = x
    OUT_DIR.mkdir(exist_ok=True)
    man_rows = []
    for pdb in picks:
        pdb = pdb.lower()
        src = PDB_DIR / f"{pdb}.pdb"
        if not src.exists():
            try:
                urllib.request.urlretrieve(
                    f"https://files.rcsb.org/download/{pdb}.pdb", src)
                time.sleep(0.3)
            except Exception as e:
                print(pdb, "download FAIL", e); continue
        atoms = parse_pdb(src)
        dna_chains = set()
        for a in atoms:
            if is_dna(a):
                dna_chains.add(a["chain"])
        if len(dna_chains) < 2:
            print(pdb, "skip: dna chains", dna_chains); continue
        dna_chains = sorted(dna_chains)[:2]
        prot = [a for a in atoms if not is_dna(a) and a["chain"] not in dna_chains]
        dna = [a for a in atoms if a["chain"] in dna_chains]
        if not prot or not dna:
            print(pdb, "skip: empty prot/dna"); continue
        # native model
        tdir = OUT_DIR / pdb
        (tdir / "models").mkdir(parents=True, exist_ok=True)
        write_pdb(atoms, tdir / "models" / "model_native.pdb")
        prot_xyz = np.array([a["xyz"] for a in prot])
        dna_xyz = np.array([a["xyz"] for a in dna])
        man_rows.append(dict(pdb=pdb, candidate_id=f"{pdb}_native",
                             rmsd=0.0, label="native", model="model_native.pdb",
                             n_prot=len(prot), n_dna=len(dna),
                             dna_chains=",".join(dna_chains)))
        # decoys: rigid duplex perturbation
        centroid = dna_xyz.mean(0)
        for i in range(N_DECOYS):
            tier = i % 3
            new_atoms = [dict(a) for a in atoms]
            idx = {id(a): j for j, a in enumerate(new_atoms)}
            if tier == 0:
                # whole-duplex rigid transform
                R = rand_rot(rng)
                ang_t = rng.uniform(2, 15)
                tv = rng.normal(size=3); tv /= np.linalg.norm(tv)
                for a in new_atoms:
                    if a["chain"] in dna_chains:
                        a["xyz"] = (a["xyz"] - centroid) @ R + centroid + ang_t * tv
            else:
                # per-chain independent perturbation (partial unbinding)
                for ch in dna_chains:
                    R = rand_rot(rng)
                    c = np.array([a["xyz"] for a in new_atoms if a["chain"] == ch]).mean(0)
                    tv = rng.normal(size=3); tv /= np.linalg.norm(tv)
                    shift = rng.uniform(3, 20)
                    for a in new_atoms:
                        if a["chain"] == ch:
                            a["xyz"] = (a["xyz"] - c) @ R + c + shift * tv
            # protein-fixed-frame ligand RMSD
            dec_prot = np.array([a["xyz"] for a in new_atoms
                                 if a["chain"] not in dna_chains and not is_dna(a)])
            dec_dna = np.array([a["xyz"] for a in new_atoms if a["chain"] in dna_chains])
            nat_prot = prot_xyz
            R2, t2 = kabsch(dec_prot, nat_prot)
            moved = dec_dna @ R2 + t2
            rmsd = float(np.sqrt(((moved - dna_xyz) ** 2).sum(1).mean()))
            label = "native_like" if rmsd < 4.0 else ("medium" if rmsd < 10.0 else "decoy")
            name = f"model_{i:03d}.pdb"
            write_pdb(new_atoms, tdir / "models" / name)
            man_rows.append(dict(pdb=pdb, candidate_id=f"{pdb}_{i:03d}",
                                 rmsd=round(rmsd, 3), label=label,
                                 model=name, n_prot=len(prot), n_dna=len(dna),
                                 dna_chains=",".join(dna_chains)))
        print(pdb, "done", len(dna_chains), "dna chains")
    with open(OUT_DIR / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man_rows[0].keys()))
        w.writeheader(); w.writerows(man_rows)
    print("manifest rows:", len(man_rows))


if __name__ == "__main__":
    main()
