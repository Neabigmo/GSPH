# -*- coding: utf-8 -*-
"""PyMOL renderer for witness_strat / witness JSON dumps.

Run inside PyMOL:
    pymol -cq render_witness_pymol.py -- <witness.json> <complex.pdb> <out.png> [example_index]

Matches the actual witness JSON schema produced by
witness_dump_stratified.py:
  examples[i]:
    atoms:    {atom_id: {name, chain, resseq, resname, element, xyz}}
    patches:  [{patch_id, contacts: [{donor, acceptor, distance, tau, ...}]}]
    witness:  {center: [x,y,z], tau, certificate_passed, route}
    path, order, gap, selection_rule
"""
import json
import sys

from pymol import cmd, cgo

PATCH_COLORS = ["firebrick", "marine", "forest", "purple",
                "orange", "cyan"]


def _pick_example(data, index):
    exs = data.get("examples", [])
    if index is not None:
        return exs[int(index)]
    # default: largest tau-vs-pairwise gap
    return max(exs, key=lambda e: e.get("gap", 0.0))


def render(witness_json, pdb_path, out_png, index=None):
    data = json.load(open(witness_json))
    ex = _pick_example(data, index)
    atoms = {str(k): v for k, v in ex["atoms"].items()}
    wit = ex["witness"]
    center, tau = wit["center"], float(wit["tau"])

    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.load(pdb_path, "complex")
    cmd.hide("everything", "complex")
    cmd.show("cartoon", "complex")
    cmd.color("gray70", "complex and polymer.protein")
    cmd.color("wheat", "complex and polymer.nucleic")
    cmd.set("cartoon_transparency", 0.45, "complex")

    # involved atoms as spheres, colored per patch
    for i, p in enumerate(ex["patches"]):
        col = PATCH_COLORS[i % len(PATCH_COLORS)]
        for c in p.get("contacts", []):
            for aid in (str(c["donor"]), str(c["acceptor"])):
                a = atoms.get(aid)
                if a is None:
                    continue
                name = f"pa{i}_{aid}"
                cmd.pseudoatom(name, pos=a["xyz"],
                               vdw=1.0, color=col)
                cmd.show("spheres", name)
                cmd.group(f"patch_{i}", name)
            # contact dash
            d, a = atoms.get(str(c["donor"])), atoms.get(str(c["acceptor"]))
            if d and a:
                n1, n2 = f"pa{i}_{c['donor']}", f"pa{i}_{c['acceptor']}"
                cmd.distance(f"con{i}_{c['donor']}_{c['acceptor']}",
                             n1, n2)
                cmd.color(col,
                          f"con{i}_{c['donor']}_{c['acceptor']}")
    cmd.hide("labels")

    # certified common witness: center point + tau-radius support ball
    cmd.pseudoatom("witness_center", pos=center, vdw=0.0,
                   color="black")
    cmd.show("nb_spheres", "witness_center")
    cmd.set("nb_spheres_size", 0.5, "witness_center")
    ball = [cgo.ALPHA, 0.18, cgo.COLOR, 1.0, 0.0, 1.0,
            cgo.SPHERE, center[0], center[1], center[2], tau]
    cmd.load_cgo(ball, "support_ball")

    # frame on involved atoms + witness
    sel = " or ".join(f"patch_{i}" for i in range(len(ex["patches"])))
    cmd.zoom(f"({sel}) or witness_center", buffer=6)
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    cmd.ray(1600, 1200)
    cmd.png(out_png, dpi=300)
    print(f"[render] {data.get('structure_id')} example gap={ex.get('gap')}"
          f" tau={tau:.3f} cert={wit.get('certificate_passed')} -> {out_png}")


if __name__ == "__main__":
    args = [a for a in sys.argv if not a.endswith(".py")]
    args = args[1:] if args and args[0].startswith("-") else args
    render(args[0], args[1], args[2],
           index=(int(args[3]) if len(args) > 3 else None))

