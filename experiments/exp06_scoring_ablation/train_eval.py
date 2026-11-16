#!/usr/bin/env python3
"""Unified M0/M1/M2 training + evaluation harness.

M0: GSPH features only
M1: baseline-score features only (e.g. rsRNASP/cgRNASP for RNA)
M2: M0 + M1 combined

Protocol per task:
  - identical candidate pools across methods (same structure rows)
  - strict target-grouped k-fold CV (no decoy of a test target in train)
  - identical predictor budget (XGBRanker, fixed hyperparams from
    configs/fair_methods.yaml ranker block)
  - no zero-fill: NaN features stay NaN; rows with audit status != complete
    are excluded from both train and test
  - audit columns (status/labels/timing) never enter feature sets

Metrics per fold + pooled: spearman, top1/top5 quality, success rates,
paired target-level bootstrap for method deltas.
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

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from evaluation.grouped_cv import target_grouped_folds  # noqa: E402
from evaluation.ranking_metrics import target_metrics  # noqa: E402
from evaluation.paired_bootstrap import paired_target_bootstrap  # noqa: E402

META_COLS = {
    "structure_id", "group", "kind", "target_id", "candidate_id",
    "status", "wt_status", "mut_status", "wt_topology_valid",
    "mut_topology_valid", "pdbid", "protein", "wt", "mut", "row",
    "wt_aa", "mut_aa", "resid", "bound_protocol", "gsph_mode",
    "protocol", "fold", "quality", "score", "_audit_status",
}
LABEL_PREFIXES = ("label_", "ddG")


def load_shards(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(pattern)):
        if path.endswith(".parquet"):
            frames.append(pd.read_parquet(path))
        elif path.endswith(".csv"):
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"no shard files match {pattern}")
    return pd.concat(frames, ignore_index=True)


def load_audit_status(pattern: str) -> dict:
    status = {}
    for path in sorted(glob.glob(pattern)):
        for line in open(path):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = rec.get("structure_id") or rec.get("row")
            if key is None:
                continue
            status[str(key)] = str(rec.get("status", "unknown"))
    return status


def feature_columns(df: pd.DataFrame, extra_exclude=()) -> list:
    drop = set(META_COLS) | set(extra_exclude)
    cols = []
    for c in df.columns:
        if c in drop or any(str(c).startswith(p) for p in LABEL_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def merge_baseline_scores(df: pd.DataFrame, score_csvs: list) -> pd.DataFrame:
    out = df.copy()
    for path, prefix in score_csvs:
        sc = pd.read_csv(path)
        sc = sc.rename(columns={"structure": "structure_id",
                                "score_kbt": f"base_{prefix}"})
        sc["structure_id"] = sc["structure_id"].str.replace(".pdb", "", regex=False)
        keep = [c for c in sc.columns if c in {"structure_id", f"base_{prefix}"}]
        out = out.merge(sc[keep], on="structure_id", how="left")
    return out


def rank_quality(df: pd.DataFrame, label_col: str, label_lower_better: bool):
    q = -df[label_col] if label_lower_better else df[label_col]
    return q


def train_predict(train_df, test_df, cols, qcol, gcol="group",
                  predictor="ranker", seed=20261103):
    import xgboost as xgb
    from sklearn.impute import SimpleImputer
    # NOTE: imputation uses train-fold medians only; applied to X, never to labels.
    imp = SimpleImputer(strategy="median")
    Xte = imp.fit(train_df[cols]).transform(test_df[cols])
    if predictor == "ranker":
        # XGBRanker query groups must be CONTIGUOUS in the training matrix.
        train_df = train_df.sort_values([gcol], kind="stable")
        Xtr_sorted = imp.transform(train_df[cols])
        sizes = train_df.groupby(gcol, sort=False).size().tolist()
        # assert contiguity: number of contiguous blocks == number of groups
        blocks = int((train_df[gcol] != train_df[gcol].shift()).sum())
        assert blocks == len(sizes) == train_df[gcol].nunique(), (
            f"query groups not contiguous: {blocks} blocks vs "
            f"{train_df[gcol].nunique()} groups")
        dtrain = xgb.DMatrix(Xtr_sorted, label=train_df[qcol].values)
        dtrain.set_group(sizes)
        params = {
            "objective": "rank:pairwise", "eta": 0.05, "max_depth": 4,
            "subsample": 0.9, "colsample_bytree": 0.9, "seed": seed,
            "nthread": 4, "verbosity": 0,
        }
        booster = xgb.train(params, dtrain, num_boost_round=80)
        return booster.predict(xgb.DMatrix(Xte))
    if predictor == "ridge":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=1.0)
    elif predictor == "rf":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=200, max_depth=8,
                                  random_state=seed, n_jobs=4)
    elif predictor == "xgb_reg":
        m = xgb.XGBRegressor(objective="reg:squarederror", eta=0.05,
                             max_depth=4, subsample=0.9,
                             colsample_bytree=0.9, seed=seed,
                             n_estimators=200, nthread=4, verbosity=0)
    else:
        raise ValueError(predictor)
    m.fit(imp.transform(train_df[cols]), train_df[qcol].values)
    return m.predict(Xte)


def _reg_metrics(pooled, gcol):
    from scipy.stats import spearmanr, pearsonr
    per = {}
    for t, g in pooled.groupby(gcol):
        if len(g) < 3:
            continue
        per[t] = {
            "n": int(len(g)),
            "spearman": float(spearmanr(g["score"], g["quality"]).statistic),
            "pearson": float(pearsonr(g["score"], g["quality"]).statistic),
            "rmse": float(np.sqrt(np.mean((g["score"] - g["quality"]) ** 2))),
            "mae": float(np.mean(np.abs(g["score"] - g["quality"]))),
        }
    agg = {k: float(np.nanmean([v[k] for v in per.values()]))
           for k in ("spearman", "pearson", "rmse", "mae")}
    agg["targets"] = float(len(per))
    return agg, per


def evaluate_task(df, label_col, label_lower_better, base_cols, n_splits=5,
                  gcol="group", predictor="ranker", lopo=False,
                  rmsd_col=None):
    df = df.copy()
    # non-finite feature values -> NaN -> median imputation (train-fold only)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df["quality"] = rank_quality(df, label_col, label_lower_better)
    df = df.dropna(subset=["quality", gcol])
    targets = sorted(df[gcol].unique())
    if lopo:
        folds = {t: i for i, t in enumerate(targets)}
    else:
        folds = target_grouped_folds(targets, n_splits=n_splits)
    df["fold"] = df[gcol].map(folds)

    gsph_cols = [c for c in feature_columns(df) if not c.startswith("base_")]
    m_cols = {"M0_gsph": gsph_cols,
              "M1_baseline": base_cols,
              "M2_union": gsph_cols + base_cols}

    metrics = {}
    for name, cols in m_cols.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            metrics[name] = {"status": "no_features"}
            continue
        fold_metrics = []
        preds_all = []
        for f in sorted(set(folds.values())):
            tr, te = df[df.fold != f], df[df.fold == f]
            if not len(tr) or not len(te):
                continue
            te = te.copy()
            te["score"] = train_predict(tr, te, cols, "quality", gcol=gcol,
                                        predictor=predictor)
            preds_all.append(te)
            fold_metrics.append(te)
        if not fold_metrics:
            metrics[name] = {"status": "no_folds"}
            continue
        pooled = pd.concat(preds_all)
        cand = (pooled["structure_id"] if "structure_id" in pooled.columns
                else pooled.get("row", pooled.index).astype(str))
        if predictor == "ranker":
            pooled_m = target_metrics(
                [{"target_id": r[0], "candidate_id": str(r[1]),
                  "score": r[2], "quality": r[3],
                  "irmsd": (r[4] if r[4] == r[4] else 1e9)}
                 for r in zip(pooled[gcol], cand,
                              pooled["score"], pooled["quality"],
                              pooled[rmsd_col] if rmsd_col else [1e9] * len(pooled))])
            metrics[name] = {
                "n_features": len(cols),
                "pooled": pooled_m,
                "per_target_spearman": {
                    t: _target_spearman(pooled[pooled[gcol] == t])
                    for t in pooled[gcol].unique()},
            }
        else:
            agg, per = _reg_metrics(pooled, gcol)
            metrics[name] = {"n_features": len(cols), "pooled": agg,
                             "per_target": per}
    # Table A: raw baseline scores directly (no predictor).
    # Convention: energy-style scores (lower=better) -> report rho(-score, quality).
    from scipy.stats import spearmanr as _spr
    raw = {}
    for bc in base_cols:
        g = df.dropna(subset=[bc])
        per = {}
        for t, gg in g.groupby(gcol):
            if len(gg) < 5:
                continue
            per[t] = float(_spr(-gg[bc], gg["quality"]).statistic)
        raw[bc] = {"target_macro_spearman": float(np.nanmean(list(per.values()))),
                   "per_target": per, "n_rows": int(len(g)),
                   "note": "energy direction assumed: lower score = better"}
    # paired bootstrap: M2 vs M1 and M0 vs M1 on per-target spearman
    boot = {}
    if predictor == "ranker":
        key = "per_target_spearman"
    else:
        key = "per_target"
    if key in metrics.get("M2_union", {}):
        for a, b in (("M0_gsph", "M1_baseline"), ("M2_union", "M1_baseline")):
            if key not in metrics.get(a, {}) or key not in metrics.get(b, {}):
                continue
            ta = metrics[a][key]
            tb = metrics[b][key]
            if predictor != "ranker":
                ta = {t: v["spearman"] for t, v in ta.items()}
                tb = {t: v["spearman"] for t, v in tb.items()}
            common = sorted(set(ta) & set(tb))
            common = [t for t in common
                      if ta[t] == ta[t] and tb[t] == tb[t]]
            delta = {t: ta[t] - tb[t] for t in common}
            if delta:
                mean, lo, hi = paired_target_bootstrap(delta)
                boot[f"{a}_minus_{b}"] = {"mean": mean, "ci95": [lo, hi],
                                           "n_targets": len(common)}
    return metrics, boot, raw


def _target_spearman(g):
    if len(g) < 5:
        return np.nan
    from scipy.stats import spearmanr
    return float(spearmanr(g["score"], g["quality"]).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["rna", "membrane", "protein_dna"])
    ap.add_argument("--features", required=True,
                    help="glob for feature shards")
    ap.add_argument("--audit", required=True, help="glob for audit jsonl")
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--label-lower-better", action="store_true")
    ap.add_argument("--baseline-scores", nargs="*", default=[],
                    help="path[:name] baseline score csvs")
    ap.add_argument("--group-col", default="group")
    ap.add_argument("--predictor", default="ranker",
                    choices=["ranker", "ridge", "rf", "xgb_reg"])
    ap.add_argument("--lopo", action="store_true",
                    help="leave-one-group-out (one fold per group)")
    ap.add_argument("--rmsd-col", default=None,
                    help="column with real iRMSD values for success metrics")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = load_shards(args.features)
    status = load_audit_status(args.audit)
    key = "structure_id" if "structure_id" in df.columns else "row"
    df["_audit_status"] = df[key].astype(str).map(status)
    complete = df["_audit_status"].isin({"complete", "ok"})
    dropped = int((~complete).sum())
    df = df[complete]
    # decoy ranking must never train/evaluate on native reference rows
    n_native = 0
    if "kind" in df.columns:
        n_native = int((df["kind"] == "native").sum())
        df = df[df["kind"] != "native"]

    score_csvs = []
    for spec in args.baseline_scores:
        if ":" in spec:
            p, n = spec.rsplit(":", 1)
        else:
            p, n = spec, os.path.splitext(os.path.basename(spec))[0].replace(
                "_scores", "")
        score_csvs.append((p, n))
    if score_csvs:
        df = merge_baseline_scores(df, score_csvs)
    base_cols = [c for c in df.columns if c.startswith("base_")]

    metrics, boot, raw = evaluate_task(
        df, args.label_col, args.label_lower_better, base_cols,
        n_splits=args.n_splits, gcol=args.group_col,
        predictor=args.predictor, lopo=args.lopo,
        rmsd_col=args.rmsd_col)
    result = {"task": args.task, "label": args.label_col,
              "n_rows_used": int(len(df)),
              "n_rows_dropped_incomplete": dropped,
              "n_native_excluded": n_native,
              "group_col": args.group_col, "predictor": args.predictor,
              "lopo": bool(args.lopo),
              "metrics": metrics, "raw_baseline_tableA": raw,
              "bootstrap": boot}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(json.dumps({k: (v.get("pooled") or v) for k, v in metrics.items()},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
