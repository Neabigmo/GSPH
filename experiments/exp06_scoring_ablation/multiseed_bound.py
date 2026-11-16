# -*- coding: utf-8 -*-
"""Multi-seed robustness for the PD-bound T×H interaction.

Re-runs ONLY the four interaction-relevant v1 models (A1_P, A4_PT, A5_PH,
A7_PTH) on pd_bound with alternative fold seeds (1, 2).  Seed-0 folds are
the canonical cv_splits used everywhere else; the canonical splits file is
restored after each run.  OOF saved as oof_ms<seed>_pd_bound.parquet.
"""
from __future__ import annotations
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
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ablation as RA  # noqa: E402
from feature_groups import build_manifest  # noqa: E402

OUT = RA.OUT
KEEP = ["A1_P", "A4_PT", "A5_PH", "A7_PTH"]
SEEDS = [1, 2]

df = RA.load_task("pd_bound")
man = build_manifest(df.columns, "pd_bound")
sets = {k: v for k, v in RA.model_sets(man).items() if k in KEEP}
print(f"[ms] sets={ {k: len(v) for k, v in sets.items()} }", flush=True)

cv = OUT / "cv_splits_pd_bound.json"
orig = cv.read_bytes()
try:
    for seed in SEEDS:
        oof, met = RA.run_rank("pd_bound", df, sets, seed=seed)
        oof.to_parquet(OUT / f"oof_ms{seed}_pd_bound.parquet", index=False)
        (OUT / f"metrics_ms{seed}_pd_bound.json").write_text(
            json.dumps(met, indent=1, default=float))
        print(f"[ms] seed={seed} done", flush=True)
finally:
    cv.write_bytes(orig)
print("[ms] all done; canonical cv_splits restored", flush=True)
