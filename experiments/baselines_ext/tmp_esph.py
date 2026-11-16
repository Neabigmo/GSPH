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
"""T-MP (TopologyNet) ESPH features, reimplemented with GUDHI AlphaComplex
(dionysus 1.x alpha bindings are unavailable; alpha-complex PH is identical).
Faithful to Protein-mutation/MutationFeature.py: 48 bins x 45 cols."""
import json, math, sys
from pathlib import Path
import numpy as np
import gudhi

ElementList = ["C", "N", "O"]
Cut, thr, rs, small = 12.0, 12.0, 0.25, 0.00001
lth = int(Cut / rs)


def EleID(AtomType, l):
    for i in range(len(l)):
        if l[i] == AtomType:
            return i
    return -1


class Protein:
    def __init__(self, fname):
        self.AtomPos, self.ResID, self.AtomType = [], [], []
        for line in open(fname):
            p = line.split()
            if len(p) < 5:
                continue
            self.ResID.append(int(p[0]))
            self.AtomType.append(p[1])
            self.AtomPos.append([float(p[2]), float(p[3]), float(p[4])])
        self.AtomPos = np.array(self.AtomPos)


def alpha_pairs(pts):
    ac = gudhi.AlphaComplex(points=pts)
    st = ac.create_simplex_tree()
    out = []
    for dim, (b, d) in st.persistence():
        out.append((dim, math.sqrt(max(b, 0.0)),
                    math.sqrt(d) if math.isfinite(d) else float("inf")))
    return out


def features(wild, mut, res_num):
    X = np.zeros([lth, 45])
    for j, Prot in enumerate((wild, mut)):
        MutSite = [(Prot.AtomPos[k], Prot.AtomType[k])
                   for k in range(len(Prot.AtomPos))
                   if Prot.ResID[k] == res_num and Prot.AtomType[k] != "H"]
        ProtDigit0 = np.zeros([lth, 9])
        for pos, st_ in MutSite:
            Is = EleID(st_, ElementList)
            if Is < 0:
                continue
            for k in range(len(Prot.AtomPos)):
                if Prot.ResID[k] == res_num or Prot.AtomType[k] not in ElementList:
                    continue
                Dis = np.linalg.norm(Prot.AtomPos[k] - pos)
                if Dis > Cut:
                    continue
                Ip = EleID(Prot.AtomType[k], ElementList)
                Ib = int(math.floor(Dis / rs))
                if Ib >= lth:
                    continue
                ProtDigit0[Ib, Is * 3 + Ip] += 1.0
        X[:, j * 9:j * 9 + 9] = ProtDigit0
    X[:, 18:27] = X[:, 9:18] - X[:, 0:9]

    for j, Prot in enumerate((wild, mut)):
        pts = [Prot.AtomPos[i].tolist() for i in range(len(Prot.AtomPos))
               if Prot.AtomType[i] != "H"]
        ProtDigit12 = np.zeros([lth, 6])
        for dim, b, d in alpha_pairs(pts):
            if dim == 0 or dim > 2 or b > thr or (d - b) < small:
                continue
            intid = min(int(math.floor(b / rs)), lth - 1)
            fid = (dim - 1) * 3
            ProtDigit12[intid, fid] += 1.0
            if math.isfinite(d):
                did = min(int(math.floor(d / rs)), lth - 1)
                ProtDigit12[did, fid + 1] += 1.0
            else:
                did = lth - 1
            ProtDigit12[intid:did + 1, fid + 2] += 1.0
        X[:, 27 + j * 6:27 + j * 6 + 6] = ProtDigit12
    X[:, 39:45] = X[:, 33:39] - X[:, 27:33]
    return X


def main():
    import os
    base = Path(os.environ.get("MUTINFO_DIR", "/work/mutinfo"))
    rows = []
    for m in json.loads((base / "manifest.json").read_text()):
        if "tag" not in m:
            continue
        wild = Protein(base / m["tag"] / "wild.mutinfo")
        mut = Protein(base / m["tag"] / "mut.mutinfo")
        X = features(wild, mut, m["resid"])
        rows.append(dict(pdbid=m["pdbid"], resid=m["resid"],
                         mutation=m["mutation"], X=X.flatten()))
        print(m["tag"], X.shape, flush=True)
    import csv
    out = Path(os.environ.get("TMP_FEAT_OUT", "/work/tmp_esph_features.csv"))
    ncol = rows[0]["X"].size
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdbid", "resid", "mutation"] + [f"tmp_{i}" for i in range(ncol)])
        for r in rows:
            w.writerow([r["pdbid"], r["resid"], r["mutation"]] +
                       [f"{v:.6g}" for v in r["X"]])
    print("saved", len(rows), "rows x", ncol, "features ->", out)


if __name__ == "__main__":
    main()
