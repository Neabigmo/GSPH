# EXP-02 witness visualization - PyMOL rendering
# Usage: pymol -cq render_witness.pml
# Expects witness JSONs in results/witness/*.json (V33 format:
# atoms[{name,resn,chain,x,y,z,element}], contacts, witness{x,y,z,tau},
# patches[{patch_id,contacts[[i,j]]}]).

def load_witness(path):
    import json
    return json.load(open(path))

def render(path, out_png):
    import json
    w = json.load(open(path))
    cmd.reinitialize()
    # load source structure if pdb field present
    pdb = w.get("pdb") or w.get("source_pdb")
    if pdb:
        cmd.load(pdb)
    atoms = w.get("atoms", [])
    # show only atoms involved in witness contacts
    involved = set()
    for c in w.get("contacts", []):
        involved.add(c.get("atom_i", c.get("i")))
        involved.add(c.get("atom_j", c.get("j")))
    cmd.hide("everything")
    for a in atoms:
        sel = f"id {a.get('serial', a.get('name',''))}"
    # pseudo: create atoms from coordinates if no pdb
    if not pdb:
        for k, a in enumerate(atoms):
            cmd.load_cgo([a["x"], a["y"], a["z"], 0.4],
                         [1, 0, 0], 1, name=f"atom_{k}")
    wit = w.get("witness", {})
    tau = float(wit.get("tau", w.get("tau", 2.0)))
    cmd.load_cgo([wit.get("x", 0), wit.get("y", 0), wit.get("z", 0), tau],
                 [1, 0, 1], 16, name="support_ball")
    cmd.show("spheres", "support_ball")
    cmd.color("magenta", "support_ball")
    cmd.set("sphere_scale", 0.3, "support_ball")
    cmd.orient()
    cmd.png(out_png, dpi=300, ray=1)
    cmd.save(out_png.replace(".png", ".pse"))
    print("rendered", out_png)

import glob, os
for jf in glob.glob("results/witness/*.json"):
    render(jf, jf.replace(".json", ".png"))
