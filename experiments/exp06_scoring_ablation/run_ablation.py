# -*- coding: utf-8 -*-
"""Step-2/3: run the A1-A9 ablation matrix on a fixed candidate pool.

Outputs per task:
  results/ablation/oof_<task>.parquet      (structure_id, group, y_true, model, y_pred)
  results/ablation/metrics_<task>.json     (per-model metrics incl. per-group values)
  results/ablation/cv_splits_<task>.json

No feature extraction — reads existing _feat parquets + Step-1 manifest.
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
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_groups import build_manifest  # noqa: E402

H3 = Path(r"H:\2026try\11.3GSPH\GSPH_H3")
OUT = H3 / "results" / "ablation"

TASKS = {
    "rna":        dict(dir=H3 / "_feat/rna_clash06", pat="rna_quality_s*.parquet",
                       audit_glob="rna_quality_s*.audit.jsonl", label="label_rmsd_official",
                       group="group", kind="rank", higher_better=False,
                       decoy_only=True),
    "membrane":   dict(dir=H3 / "_feat/membrane",    pat="membrane_s*.parquet",
                       audit_glob="membrane_s*.audit.jsonl", label="ddG",
                       group="protein", kind="reg", higher_better=True,
                       decoy_only=False),
    "pd_bound":   dict(dir=H3 / "_feat/pd_bound",    pat="pd_ftdmp_s*.parquet",
                       audit_dir=H3 / "_feat/protein_dna",
                       audit_glob="pd_ftdmp_s*.audit.jsonl", label="label_cadscore",
                       group="group", kind="rank", higher_better=True,
                       decoy_only=False),
    "pd_unbound": dict(dir=H3 / "_feat/pd_unbound",  pat="pd_ftdmp_s*.parquet",
                       audit_dir=H3 / "_feat/protein_dna",
                       audit_glob="pd_ftdmp_s*.audit.jsonl", label="label_cadscore",
                       group="group", kind="rank", higher_better=True,
                       decoy_only=False),
}

RANK_PARAMS = dict(objective="rank:pairwise", max_depth=4,
                   learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                   tree_method="hist", n_jobs=-1)
RANK_ROUNDS = 80


def load_task(task: str) -> pd.DataFrame:
    cfg = TASKS[task]
    df = pd.concat([pd.read_parquet(f) for f in sorted(cfg["dir"].glob(cfg["pat"]))],
                   ignore_index=True)
    good = set()
    for af in cfg.get("audit_dir", cfg["dir"]).glob(cfg["audit_glob"]):
        for line in af.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") in ("complete", "ok"):
                good.add(str(r.get("structure_id") or r.get("row")))
    key = "structure_id" if "structure_id" in df.columns else "row"
    df = df[df[key].astype(str).isin(good)].reset_index(drop=True)
    if cfg["decoy_only"]:
        df = df[df["kind"] == "decoy"].reset_index(drop=True)
    return df


def model_sets(man: pd.DataFrame):
    g = {k: man.loc[man.assigned_group == k, "feature_name"].tolist()
         for k in "PTHS"}
    p_compact = [c for c in g["P"] if c.startswith(("patch_global_patch_", "d_patch_global_patch_"))]
    all_num = [c for c in man.feature_name
               if man.loc[man.feature_name == c, "assigned_group"].iloc[0]
               in ("P", "T", "H", "S", "DIAG", "EXCL")]
    sets = {
        "A1_P": g["P"], "A2_T": g["T"], "A3_H": g["H"],
        "A4_PT": g["P"] + g["T"], "A5_PH": g["P"] + g["H"], "A6_TH": g["T"] + g["H"],
        "A7_PTH": g["P"] + g["T"] + g["H"],
        "A8_S": g["S"], "A9_Pcompact": p_compact,
        "CleanFull_PTHS": g["P"] + g["T"] + g["H"] + g["S"],
        "LegacyFull_allnum": all_num,
    }
    return sets


def signed_log1p_cols(df, cols):
    out = df[cols].copy()
    for c in cols:
        if "_lj_" in c or c.endswith(("_lj_att", "_lj_rep", "_lj_net", "_lj_abs")) \
           or "_lj_" in c:
            out[c] = np.sign(out[c]) * np.log1p(np.abs(out[c]))
    return out


def variant_cols(cols, variant):
    if variant == "nolj":
        return [c for c in cols if "_lj_" not in c]
    return cols


def make_folds(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    uniq = np.sort(pd.unique(groups))
    perm = np.random.RandomState(seed).permutation(uniq)
    return [perm[i::k] for i in range(k)]


def spearman_macro(df, group_col, y="y_oriented", s="y_pred"):
    from scipy.stats import spearmanr
    vals = []
    for _, g in df.groupby(group_col):
        if g[y].nunique() > 1 and g[s].nunique() > 1:
            r = spearmanr(g[y], g[s]).statistic
            if np.isfinite(r):
                vals.append(r)
    return float(np.mean(vals)), vals


def rank_metrics(oof: pd.DataFrame, group_col: str, higher_better: bool):
    """Per-target: spearman of oriented label vs score (positive = good)."""
    oof = oof.copy()
    oof["y_oriented"] = oof["y_true"] if higher_better else -oof["y_true"]
    recs = {}
    sp_mean, per = spearman_macro(oof, group_col)
    for k in (1, 5, 10):
        vals = []
        for _, g in oof.groupby(group_col):
            top = g.nlargest(min(k, len(g)), "y_pred")
            lab = top["y_true"]
            vals.append(lab.max() if higher_better else lab.min())
        recs[f"top{k}_label"] = float(np.mean(vals))
    recs["target_macro_spearman"] = sp_mean
    recs["per_target_spearman"] = per
    return recs


def run_rank(task, df, sets, seed=0, n_folds=5):
    import xgboost as xgb
    cfg = TASKS[task]
    label = df[cfg["label"]].astype(float)
    folds = make_folds(df[cfg["group"]].to_numpy(), n_folds, seed)
    splits = {f"fold{i}": sorted(map(str, f.tolist())) for i, f in enumerate(folds)}
    (OUT / f"cv_splits_{task}.json").write_text(json.dumps(splits, indent=1))

    oof_parts, metrics = [], {}
    base = df[["structure_id", cfg["group"]]].copy()
    base.columns = ["structure_id", "group_id"]
    base["y_true"] = label

    for name, cols in sets.items():
        if not cols:
            continue
        for phys in ("raw",) if name not in ("A1_P", "A7_PTH", "CleanFull_PTHS") else ("raw", "log", "nolj"):
            mname = name if phys == "raw" else f"{name}__{phys}"
            vcols = variant_cols(cols, phys)
            if not vcols:
                continue
            X = df[vcols].replace([np.inf, -np.inf], np.nan)
            if phys == "log":
                X = signed_log1p_cols(X, vcols)
            pred = np.full(len(df), np.nan)
            for fold_i, test_groups in enumerate(folds):
                te = df[cfg["group"]].isin(test_groups).to_numpy()
                tr = ~te
                tr_df = df.loc[tr].sort_values([cfg["group"], "structure_id"],
                                              kind="stable")  # keep original index for alignment
                med = X.loc[tr].median()
                Xtr = X.loc[tr].loc[tr_df.index].fillna(med)
                Xte = X.loc[te].fillna(med)
                # train-fold median imputation, aligned to group-sorted frame
                med = X.loc[tr].median()
                Xtr = X.loc[tr].loc[tr_df.index].fillna(med)
                Xte = X.loc[te].fillna(med)
                gsz = tr_df.groupby(cfg["group"], sort=False).size().tolist()
                assert sum(gsz) == len(Xtr)
                # verify contiguity: each group's rows are consecutive in tr_df
                blocks = (tr_df[cfg["group"]] != tr_df[cfg["group"]].shift()).sum()
                assert int(blocks) == len(gsz), "ranker query groups not contiguous"
                lab_sorted = df.loc[tr_df.index, cfg["label"]].astype(float).to_numpy()
                rel_sorted = -lab_sorted if not cfg["higher_better"] else lab_sorted
                dm = xgb.DMatrix(Xtr.to_numpy(), label=rel_sorted)
                dm.set_group(gsz)
                m = xgb.train(RANK_PARAMS, dm, num_boost_round=RANK_ROUNDS)
                pred[te] = m.predict(xgb.DMatrix(Xte.to_numpy()))
            part = base.copy()
            part["model"] = mname
            part["y_pred"] = pred
            oof_parts.append(part)
            metrics[mname] = rank_metrics(part.dropna(subset=["y_pred"]), "group_id",
                                          cfg["higher_better"])
            metrics[mname]["n_features"] = len(vcols)
            print(f"  [{task}] {mname}: spear={metrics[mname]['target_macro_spearman']:.3f} "
                  f"top1={metrics[mname]['top1_label']:.3f} ({len(vcols)}f)", flush=True)
    return pd.concat(oof_parts), metrics


def run_reg(task, df, sets, seed=0):
    from scipy.stats import pearsonr
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    import xgboost as xgb

    clipper = FunctionTransformer(lambda X: np.clip(X, -10.0, 10.0))

    cfg = TASKS[task]
    y = df[cfg["label"]].astype(float).to_numpy()
    proteins = np.sort(df[cfg["group"]].unique())
    splits = {f"fold{i}": [str(p)] for i, p in enumerate(proteins)}
    (OUT / f"cv_splits_{task}.json").write_text(json.dumps(splits, indent=1))

    predictors = {
        "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                       VarianceThreshold(0.0), StandardScaler(),
                                       clipper, Ridge(alpha=1.0)),
        "enet":  lambda: make_pipeline(SimpleImputer(strategy="median"),
                                       VarianceThreshold(0.0), StandardScaler(),
                                       clipper,
                                       ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=20000)),
        "rf":    lambda: make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                                             random_state=seed)),
        "xgb":   lambda: make_pipeline(SimpleImputer(strategy="median"),
                                       xgb.XGBRegressor(n_estimators=300, max_depth=4,
                                                        learning_rate=0.05, subsample=0.9,
                                                        colsample_bytree=0.9, n_jobs=-1)),
    }
    oof_parts, metrics = [], {}
    base = df[["structure_id" if "structure_id" in df.columns else "row", cfg["group"]]].copy()
    base.columns = ["structure_id", "group_id"]
    base["y_true"] = y

    for name, cols in sets.items():
        if not cols or name in ("LegacyFull_allnum",):
            continue
        for phys in ("raw",) if name not in ("A1_P", "A7_PTH", "CleanFull_PTHS") else ("raw", "log", "nolj"):
            mname = name if phys == "raw" else f"{name}__{phys}"
            vcols = variant_cols(cols, phys)
            if not vcols:
                continue
            X = df[vcols].replace([np.inf, -np.inf], np.nan)
            if phys == "log":
                X = signed_log1p_cols(X, vcols)
            for pname, mk in predictors.items():
                pred = np.full(len(df), np.nan)
                for p in proteins:
                    te = (df[cfg["group"]] == p).to_numpy()
                    pipe = mk()
                    pipe.fit(X.loc[~te], y[~te])
                    pred[te] = pipe.predict(X.loc[te])
                part = base.copy()
                part["model"] = f"{mname}::{pname}"
                part["y_pred"] = pred
                oof_parts.append(part)
                per_sp, per_pr = [], []
                for p in proteins:
                    g = part[part.group_id == p]
                    if g.y_true.nunique() > 1 and g.y_pred.nunique() > 1:
                        from scipy.stats import spearmanr
                        sp = spearmanr(g.y_true, g.y_pred).statistic
                        pr = pearsonr(g.y_true, g.y_pred).statistic
                        if np.isfinite(sp): per_sp.append(sp)
                        if np.isfinite(pr): per_pr.append(pr)
                err = pred - y
                sign_acc = float(np.mean(np.sign(pred) == np.sign(y)))
                metrics[f"{mname}::{pname}"] = dict(
                    macro_spearman=float(np.mean(per_sp)),
                    macro_pearson=float(np.mean(per_pr)),
                    per_protein_spearman=per_sp, per_protein_pearson=per_pr,
                    rmse=float(np.sqrt(np.mean(err**2))), mae=float(np.mean(np.abs(err))),
                    sign_accuracy=sign_acc, n_features=len(vcols))
                print(f"  [{task}] {mname}::{pname}: sp={np.mean(per_sp):.3f} "
                      f"rmse={np.sqrt(np.mean(err**2)):.3f} ({len(vcols)}f)", flush=True)
    return pd.concat(oof_parts), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    df = load_task(a.task)
    man = build_manifest(df.columns, a.task)
    sets = model_sets(man)
    if a.only:
        keep = set(a.only.split(","))
        sets = {k: v for k, v in sets.items() if k in keep}
    print(f"[{a.task}] rows={len(df)} sets={list(sets)}")
    if TASKS[a.task]["kind"] == "rank":
        oof, met = run_rank(a.task, df, sets, seed=a.seed)
    else:
        oof, met = run_reg(a.task, df, sets, seed=a.seed)
    oof.to_parquet(OUT / f"oof_{a.task}.parquet", index=False)
    (OUT / f"metrics_{a.task}.json").write_text(json.dumps(met, indent=1, default=float))


if __name__ == "__main__":
    main()
