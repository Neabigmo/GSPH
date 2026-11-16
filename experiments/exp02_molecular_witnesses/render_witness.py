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
# -*- coding: utf-8 -*-
"""Render common-witness figures from witness_dump.py JSON output.

Usage: python render_witness.py <sid.witness.json> [out_prefix]
Produces <out_prefix>.png (multi-panel) and <out_prefix>.pdf.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PATCH_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd",
                "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
ELEMENT_COLORS = {"N": "#3050ff", "O": "#ff4040", "P": "#ff8c00",
                  "C": "#555555", "S": "#ffff00"}


def _sphere(center, radius, color, alpha=0.08, n=14):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2 + 1)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    polys = []
    for i in range(len(u) - 1):
        for j in range(len(v) - 1):
            polys.append([list(zip(x[i, j:j + 2], y[i, j:j + 2],
                                   z[i, j:j + 2]))[0],
                          ])
    # simpler: use plot_surface-like via Poly3DCollection on quads
    verts = []
    for i in range(len(u) - 1):
        for j in range(len(v) - 1):
            quad = [[x[i, j], y[i, j], z[i, j]],
                    [x[i + 1, j], y[i + 1, j], z[i + 1, j]],
                    [x[i + 1, j + 1], y[i + 1, j + 1], z[i + 1, j + 1]],
                    [x[i, j + 1], y[i, j + 1], z[i, j + 1]]]
            verts.append(quad)
    return Poly3DCollection(verts, facecolor=color, edgecolor="none",
                            alpha=alpha)


def render(doc_path: Path, out_prefix: Path):
    doc = json.loads(doc_path.read_text())
    examples = doc["examples"]
    n = len(examples)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(5.2 * ncol, 4.6 * nrow))
    for ei, ex in enumerate(examples):
        ax = fig.add_subplot(nrow, ncol, ei + 1, projection="3d")
        atoms = ex["atoms"]
        # assign atom -> first patch containing it (via contacts)
        atom_patch = {}
        for pi, p in enumerate(ex["patches"]):
            ids = set()
            for c in p["contacts"]:
                ids.add(c["donor"]); ids.add(c["acceptor"])
            for aid in ids:
                atom_patch.setdefault(str(aid), pi)
        # draw all atoms as small grey context, patch atoms colored
        for aid, info in atoms.items():
            xyz = info["xyz"]
            el = info["element"]
            pi = atom_patch.get(aid)
            if pi is None:
                ax.scatter(*xyz, s=6, c="#cccccc", alpha=0.5, depthshade=False)
            else:
                ax.scatter(*xyz, s=22, c=PATCH_COLORS[pi % len(PATCH_COLORS)],
                           alpha=0.9, depthshade=False,
                           edgecolors="k", linewidths=0.3)
        # contact edges within each patch
        for pi, p in enumerate(ex["patches"]):
            col = PATCH_COLORS[pi % len(PATCH_COLORS)]
            for c in p["contacts"]:
                a = atoms.get(str(c["donor"])); b = atoms.get(str(c["acceptor"]))
                if a is None or b is None:
                    continue
                ax.plot([a["xyz"][0], b["xyz"][0]],
                        [a["xyz"][1], b["xyz"][1]],
                        [a["xyz"][2], b["xyz"][2]],
                        c=col, lw=0.9, alpha=0.55)
        # witness point + support ball
        w = ex["witness"]
        cen = np.asarray(w["center"], dtype=float)
        ax.scatter(*cen, s=60, c="k", marker="*", depthshade=False,
                   label=f"witness tau={w['tau']:.2f}")
        ax.add_collection3d(_sphere(cen, w["tau"], "#ff00ff", alpha=0.10))
        ax.set_title(f"{ex['canon']}  gap={ex['gap']:.2f} A", fontsize=8)
        ax.set_axis_off()
        # equal aspect
        allxyz = np.array([i["xyz"] for i in atoms.values()])
        if len(allxyz):
            mid = allxyz.mean(0); rng = (allxyz.max(0) - allxyz.min(0)).max() / 2 * 1.15
            ax.set_xlim(mid[0] - rng, mid[0] + rng)
            ax.set_ylim(mid[1] - rng, mid[1] + rng)
            ax.set_zlim(mid[2] - rng, mid[2] + rng)
    fig.suptitle(f"{doc['structure_id']} ({doc['task']}) — "
                 f"common-witness examples", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[render] {out_prefix}.png/.pdf")


if __name__ == "__main__":
    render(Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2
                                    else sys.argv[1].replace(".witness.json", "")))
