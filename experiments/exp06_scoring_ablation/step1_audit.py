# -*- coding: utf-8 -*-
"""Step-1 data audit (no training):
  results/ablation/feature_group_manifest.csv   (all tasks stacked)
  results/ablation/feature_health.csv
  results/ablation/candidate_coverage.json
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_groups import build_manifest, group_columns  # noqa: E402
from feature_health import audit_frame                     # noqa: E402

H3 = Path(r"H:\2026try\11.3GSPH\GSPH_H3")
OUT = H3 / "results" / "ablation"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "rna":        dict(dir=H3 / "_feat/rna_new",     pat="rna_quality_s*.parquet",
                       audit_glob="rna_quality_s*.audit.jsonl", decoy_only=True),
    "membrane":   dict(dir=H3 / "_feat/membrane",    pat="membrane_s*.parquet",
                       audit_glob="membrane_s*.audit.jsonl", decoy_only=False),
    "pd_bound":   dict(dir=H3 / "_feat/pd_bound",    pat="pd_ftdmp_s*.parquet",
                       audit_dir=H3 / "_feat/protein_dna",
                       audit_glob="pd_ftdmp_s*.audit.jsonl",
                       decoy_only=False, protocol="bound"),
    "pd_unbound": dict(dir=H3 / "_feat/pd_unbound",  pat="pd_ftdmp_s*.parquet",
                       audit_dir=H3 / "_feat/protein_dna",
                       audit_glob="pd_ftdmp_s*.audit.jsonl",
                       decoy_only=False, protocol="unbound"),
}

manifests, healths, coverage = [], [], {}
for task, cfg in TASKS.items():
    files = sorted(cfg["dir"].glob(cfg["pat"]))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    n_raw = len(df)

    # audit filtering
    ok_ids, audit_n = None, 0
    if cfg["audit_glob"]:
        good = set()
        for af in cfg.get("audit_dir", cfg["dir"]).glob(cfg["audit_glob"]):
            for line in af.read_text().splitlines():
                r = json.loads(line)
                audit_n += 1
                # audit bound_protocol flag is unreliable (all true);
                # protocol separation comes from structure_id match itself.
                if r.get("status") in ("complete", "ok"):
                    sid = r.get("structure_id") or r.get("row")
                    if sid is not None:
                        good.add(str(sid))
        key = "structure_id" if "structure_id" in df.columns else "row"
        ok_ids = good
        df = df[df[key].astype(str).isin(good)].reset_index(drop=True)
    n_after_audit = len(df)

    # RNA: decoys only (native excluded from candidate pool)
    n_native = 0
    if cfg["decoy_only"] and "kind" in df.columns:
        n_native = int((df["kind"] != "decoy").sum())
        df = df[df["kind"] == "decoy"].reset_index(drop=True)
    n_final = len(df)

    man = build_manifest(df.columns, task)
    manifests.append(man)
    gc = group_columns(man)
    feat_cols = [c for c in man.feature_name[man.included]]
    healths.append(audit_frame(df, feat_cols, task))

    gid = "group" if "group" in df.columns else ("protein" if "protein" in df.columns else None)
    coverage[task] = dict(
        parquet_files=len(files), raw_rows=n_raw,
        audit_records=audit_n, audit_ok=len(ok_ids) if ok_ids else None,
        rows_after_audit=n_after_audit, native_excluded=n_native,
        final_rows=n_final,
        groups=int(df[gid].nunique()) if gid else None, group_col=gid,
        n_P=len(gc["P"]), n_T=len(gc["T"]), n_H=len(gc["H"]), n_S=len(gc["S"]),
        n_meta=int((man.assigned_group == "META").sum()),
        n_diag=int((man.assigned_group == "DIAG").sum()),
        n_excl=int((man.assigned_group == "EXCL").sum()),
    )
    man.to_csv(OUT / f"feature_group_manifest_{task}.csv", index=False)
    print(f"[{task}] rows={n_final} groups={coverage[task]['groups']} "
          f"P={len(gc['P'])} T={len(gc['T'])} H={len(gc['H'])} S={len(gc['S'])} "
          f"meta={coverage[task]['n_meta']} diag={coverage[task]['n_diag']} excl={coverage[task]['n_excl']}")

pd.concat(manifests).to_csv(OUT / "feature_group_manifest.csv", index=False)
h = pd.concat(healths)
h.to_csv(OUT / "feature_health.csv", index=False)
(OUT / "candidate_coverage.json").write_text(json.dumps(coverage, indent=2))
print("\nextreme features (max_abs>1e5):", int(h.extreme_flag.sum()),
      "| constants:", int(h.constant.sum()))
print(h[h.extreme_flag].nlargest(10, "max_abs")[["task", "feature_name", "max_abs"]].to_string())
