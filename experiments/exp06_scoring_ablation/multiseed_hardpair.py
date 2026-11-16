# -*- coding: utf-8 -*-
"""Multi-seed hard-pair robustness for PD bound.

Retrains ONLY A1_P and A7_PTH (raw variant) on pd_bound under fold seeds
0..9 and evaluates the hard-pair win-rate contrast per seed.

Seed 0 reuses the canonical oof_v2_pd_bound.parquet (identical protocol).
Seeds 1..9 write oof_msX_pd_bound.parquet / metrics_msX_pd_bound.json.
The canonical cv_splits_pd_bound.json is restored afterwards.
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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ablation as RA  # noqa: E402
from feature_groups import build_manifest  # noqa: E402
from hard_pair import per_target_diffs, summarize_diffs  # noqa: E402

OUT = RA.OUT
SEEDS = list(range(1, 10))          # seed 0 = canonical oof_v2
N_BOOT = 10_000
rng = np.random.default_rng(7)


def run_seed(task, df, cols, seed):
    """Minimal rank fit identical to RA.run_rank (raw variant only).

    NumPy-only inner loop: the full df is ~9400x3677 and pandas .loc copies
    OOM under concurrent jobs.  Median imputation, group sorting and the
    rel-label convention are identical to run_rank.
    """
    import xgboost as xgb
    cfg = RA.TASKS[task]
    groups = df[cfg["group"]].astype(str).to_numpy()
    y_all = df[cfg["label"]].astype(float).to_numpy()
    key = df[[cfg["group"], "structure_id"]].astype(str)
    Xdf = df[list(cols)]
    X = np.array(Xdf.to_numpy(dtype=np.float32), copy=True)
    del Xdf
    X[~np.isfinite(X)] = np.nan
    folds = RA.make_folds(groups, 5, seed)
    pred = np.full(len(df), np.nan, dtype=np.float64)
    for test_groups in folds:
        te = np.isin(groups, test_groups)
        tr_pos = np.where(~te)[0]
        # sort train rows by (group, structure_id) — stable, same as run_rank
        ktr = key.iloc[tr_pos]
        order = ktr.assign(_i=tr_pos).sort_values(
            [cfg["group"], "structure_id"], kind="stable")["_i"].to_numpy()
        Xtr = X[order]
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xte = X[te]
        Xte = np.where(np.isnan(Xte), med, Xte)
        g_sorted = groups[order]
        gsz = pd.Series(g_sorted).groupby(g_sorted, sort=False).size().tolist()
        assert sum(gsz) == len(Xtr)
        # contiguous-block count via boundary positions
        blocks = int((g_sorted[1:] != g_sorted[:-1]).sum()) + 1
        assert blocks == len(gsz), "ranker query groups not contiguous"
        lab_sorted = y_all[order]
        rel = -lab_sorted if not cfg["higher_better"] else lab_sorted
        dm = xgb.DMatrix(Xtr, label=rel)
        dm.set_group(gsz)
        m = xgb.train(RA.RANK_PARAMS, dm, num_boost_round=RA.RANK_ROUNDS)
        pred[te] = m.predict(xgb.DMatrix(Xte))
        del Xtr, Xte, dm, m
    base = df[["structure_id", cfg["group"]]].copy()
    base.columns = ["structure_id", "group_id"]
    base["y_true"] = y_all
    part = base.copy()
    part["model"] = "RAW"
    part["y_pred"] = pred
    return part


def main():
    task = "pd_bound"
    df = RA.load_task(task)
    man = build_manifest(df.columns, task)
    sets = RA.model_sets(man)
    cols_p = sets["A1_P"]; cols_f = sets["A7_PTH"]
    cv = OUT / f"cv_splits_{task}.json"
    orig = cv.read_bytes()
    rows = []
    done_seeds = set()
    for seed in SEEDS:
        f = OUT / f"oof_mshp{seed}_{task}.parquet"
        if f.exists():
            done_seeds.add(seed)
            print(f"[ms-hp] seed={seed} oof exists, reuse", flush=True)
    try:
        for seed in SEEDS:
            if seed in done_seeds:
                oof = pd.read_parquet(OUT / f"oof_mshp{seed}_{task}.parquet")
            else:
                parts = []
                for name, cols in (("A1_P", cols_p), ("A7_PTH", cols_f)):
                    part = run_seed(task, df, cols, seed)
                    part["model"] = name
                    parts.append(part)
                    met = RA.rank_metrics(part, "group_id",
                                          RA.TASKS[task]["higher_better"])
                    print(f"[ms-hp] seed={seed} {name}: "
                          f"spear={met['target_macro_spearman']:.3f}", flush=True)
                oof = pd.concat(parts, ignore_index=True)
                oof.to_parquet(OUT / f"oof_mshp{seed}_{task}.parquet",
                               index=False)
            per_t = per_target_diffs(oof, "A1_P", "A7_PTH", task)
            s = summarize_diffs(per_t, rng, N_BOOT)
            s.update(seed=seed, task=task, contrast="A7_PTH-vs-A1_P")
            rows.append(s)
            print(f"[ms-hp] seed={seed}: d={s['mean_d']:+.4f} "
                  f"CI[{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}] p={s['p_perm']:.4f}",
                  flush=True)
    finally:
        cv.write_bytes(orig)
    # seed 0 from canonical OOF
    oof0 = pd.read_parquet(OUT / f"oof_v2_{task}.parquet")
    per_t = per_target_diffs(oof0, "A1_P", "A7_PTH", task)
    s0 = summarize_diffs(per_t, rng, N_BOOT)
    s0.update(seed=0, task=task, contrast="A7_PTH-vs-A1_P")
    rows.insert(0, s0)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "hardpair_multiseed_pd_bound.csv", index=False)
    print(df_out[["seed", "mean_d", "ci_lo", "ci_hi", "p_perm"]].to_string(index=False))
    d = df_out["mean_d"].to_numpy()
    print(f"[ms-hp] mean d={d.mean():+.4f}  frac_positive={np.mean(d > 0):.2f} "
          f"sign_range=[{d.min():+.4f},{d.max():+.4f}]")


if __name__ == "__main__":
    main()
