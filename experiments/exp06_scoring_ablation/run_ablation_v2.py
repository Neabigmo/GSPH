# -*- coding: utf-8 -*-
"""Step-5: run the B-matrix (v2 feature groups) on identical candidate pools/folds.

  New models trained here:
    B2_PcV2, B4_T2, B6_H2, B7_PcT2, B8_PcH2, B9_PcT2H2,
    B10_Pc_T2res_H2res (train-fold residualization on S),
    P_PCA{142,40,20}  (train-fold PCA on full P — dimension controls)
  Reused from v1 OOF (identical protocol): B1=A1_P, B3=A2_T, B5=A3_H, B11=A8_S.

  Rank tasks: XGBRanker same budget, folds from cv_splits_<task>.json (v1).
  Membrane: strict LOPO, Ridge/ENet (impute->varthresh->scale->clip±10).
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
from run_ablation import (load_task, TASKS, RANK_PARAMS, RANK_ROUNDS,   # noqa: E402
                          make_folds, rank_metrics)

H3 = Path(r"H:\2026try\11.3GSPH\GSPH_H3")
OUT = H3 / "results" / "ablation"


def group_cols(task):
    """v2 group columns taken from the built parquet schemas."""
    import pyarrow.parquet as pq
    get = lambda n: [c for c in pq.read_schema(OUT / f"{n}_{task}.parquet").names
                     if c not in ("structure_id", "row", "group", "protein", "kind")]
    return dict(P2=get("physical_compact_v2"), T2=get("topology_features_v2"),
                H2=get("hodge_features_v2"))


def p_cols(task):
    """v1 physical group from manifest."""
    man = pd.read_csv(OUT / f"feature_group_manifest_{task}.csv")
    return man.loc[man.assigned_group == "P", "feature_name"].tolist()


def s_cols(task):
    """v2 S list (STRICT structure-size controls) via manifest_v2."""
    man = pd.read_csv(OUT / "feature_manifest_v2.csv")
    return man.loc[(man.scenario == task) & (man.assigned_group == "S"),
                   "feature_name"].tolist()


def v1_cols(task, grp):
    """v1 group columns (P/T/H/S) from the v1 manifest."""
    man = pd.read_csv(OUT / f"feature_group_manifest_{task}.csv")
    return man.loc[man.assigned_group == grp, "feature_name"].tolist()


def load_folds(task, df, n_folds=5, seed=0):
    f = OUT / f"cv_splits_{task}.json"
    if f.exists():
        sp = json.loads(f.read_text())
        return [np.array(v) for k, v in sorted(sp.items(),
                key=lambda kv: int(kv[0].replace("fold", "")))]
    return make_folds(df[TASKS[task]["group"]].to_numpy(), n_folds, seed)


def rank_topk(oof, higher_better, task=None):
    """per-target top-k best label + regret (+ RNA success@2/5A lists)."""
    out = dict(top1=[], top5=[], top10=[], regret=[])
    succ = {k: {2.0: [], 5.0: []} for k in (1, 5, 10)}
    for _, g in oof.groupby("group_id"):
        gl = g["y_true"]
        best = float(gl.max() if higher_better else gl.min())
        for k, key in ((1, "top1"), (5, "top5"), (10, "top10")):
            t = g.nlargest(min(k, len(g)), "y_pred")["y_true"]
            bl = float(t.max() if higher_better else t.min())
            out[key].append(bl)
            for thr in (2.0, 5.0):
                succ[k][thr].append(float(bl < thr) if not higher_better else np.nan)
        out["regret"].append(float(best - out["top1"][-1]) if higher_better
                             else float(out["top1"][-1] - best))
    if task == "rna":
        for k in (1, 5, 10):
            for thr in (2.0, 5.0):
                out[f"success@{k}_rmsd_lt_{int(thr)}"] = succ[k][thr]
    return out


def run_rank_v2(task, df, sets, folds, resid=False):
    import xgboost as xgb
    cfg = TASKS[task]
    label = df[cfg["label"]].astype(float).to_numpy()
    oof_parts, metrics = [], {}
    base = df[["structure_id", cfg["group"]]].copy()
    base.columns = ["structure_id", "group_id"]
    base["y_true"] = label
    S = s_cols(task)

    for name, cols in sets.items():
        res_cols, pca_n = None, None
        if isinstance(cols, dict):
            res_cols = cols.get("residualize_on_S")
            pca_n = cols.get("pca")
            cols = cols["cols"]
        if not cols:
            continue
        X = df[cols].replace([np.inf, -np.inf], np.nan)
        XS = df[S].replace([np.inf, -np.inf], np.nan) if res_cols else None
        pred = np.full(len(df), np.nan)
        for test_groups in folds:
            te = df[cfg["group"]].isin(test_groups).to_numpy()
            tr = ~te
            tr_df = df.loc[tr].sort_values([cfg["group"], "structure_id"], kind="stable")
            med = X.loc[tr].median()
            Xtr = X.loc[tr].loc[tr_df.index].fillna(med)
            Xte = X.loc[te].fillna(med)
            if res_cols:  # train-fold residualization on S
                from sklearn.linear_model import Ridge as _R
                Sm = XS.loc[tr].median()
                Str = XS.loc[tr].loc[tr_df.index].fillna(Sm).to_numpy()
                Ste = XS.loc[te].fillna(Sm).to_numpy()
                for rc in res_cols:
                    if rc in Xtr.columns:
                        mdl = _R(alpha=1.0).fit(Str, Xtr[rc].fillna(0).to_numpy())
                        Xtr[rc] = Xtr[rc].to_numpy() - mdl.predict(Str)
                        Xte[rc] = Xte[rc].to_numpy() - mdl.predict(Ste)
            if pca_n:  # train-fold PCA on full-P matrix
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler().fit(Xtr)
                pc = PCA(n_components=pca_n, random_state=0).fit(sc.transform(Xtr))
                Xtr = pd.DataFrame(pc.transform(sc.transform(Xtr)))
                Xte = pd.DataFrame(pc.transform(sc.transform(Xte)))
            gsz = tr_df.groupby(cfg["group"], sort=False).size().tolist()
            assert sum(gsz) == len(Xtr)
            assert int((tr_df[cfg["group"]] != tr_df[cfg["group"]].shift()).sum()) == len(gsz)
            lab = df.loc[tr_df.index, cfg["label"]].astype(float).to_numpy()
            dm = xgb.DMatrix(np.asarray(Xtr),
                             label=(-lab if not cfg["higher_better"] else lab))
            dm.set_group(gsz)
            m = xgb.train(RANK_PARAMS, dm, num_boost_round=RANK_ROUNDS)
            pred[te] = m.predict(xgb.DMatrix(np.asarray(Xte)))
        part = base.copy(); part["model"] = name; part["y_pred"] = pred
        oof_parts.append(part)
        mm = rank_metrics(part.dropna(subset=["y_pred"]), "group_id", cfg["higher_better"])
        mm.update(rank_topk(part.dropna(subset=["y_pred"]), cfg["higher_better"], task))
        mm["n_features"] = pca_n or len(cols)  # post-PCA dims for PCA models
        metrics[name] = mm
        print(f"  [{task}] {name}: spear={mm['target_macro_spearman']:.3f} "
              f"top1={np.mean(mm['top1']):.3f} ({mm['n_features']}f)", flush=True)
    return pd.concat(oof_parts), metrics


def run_reg_v2(task, df, sets):
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    cfg = TASKS[task]
    y = df[cfg["label"]].astype(float).to_numpy()
    proteins = sorted(df[cfg["group"]].unique())
    clip = FunctionTransformer(lambda X: np.clip(X, -10.0, 10.0))
    preds = {
        "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                       VarianceThreshold(0.0), StandardScaler(),
                                       clip, Ridge(alpha=1.0)),
        "enet": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                      VarianceThreshold(0.0), StandardScaler(), clip,
                                      ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=20000)),
    }
    oof_parts, metrics = [], {}
    base = df[["row" if "row" in df.columns else "structure_id", cfg["group"]]].copy()
    base.columns = ["structure_id", "group_id"]; base["y_true"] = y
    for name, cols in sets.items():
        if not cols: continue
        X = df[cols].replace([np.inf, -np.inf], np.nan)
        for pname, mk in preds.items():
            pred = np.full(len(df), np.nan)
            for p in proteins:
                te = (df[cfg["group"]] == p).to_numpy()
                pipe = mk(); pipe.fit(X.loc[~te], y[~te]); pred[te] = pipe.predict(X.loc[te])
            part = base.copy(); part["model"] = f"{name}::{pname}"; part["y_pred"] = pred
            oof_parts.append(part)
            from scipy.stats import pearsonr, spearmanr
            per = [spearmanr(g.y_true, g.y_pred).statistic
                   for _, g in part.groupby("group_id")
                   if g.y_true.nunique() > 1 and g.y_pred.nunique() > 1]
            err = pred - y
            metrics[f"{name}::{pname}"] = dict(
                macro_spearman=float(np.nanmean(per)), per_protein_spearman=per,
                rmse=float(np.sqrt(np.mean(err**2))), mae=float(np.mean(np.abs(err))),
                sign_accuracy=float(np.mean(np.sign(pred) == np.sign(y))),
                n_features=len(cols))
            print(f"  [{task}] {name}::{pname}: sp={np.nanmean(per):.3f} "
                  f"rmse={np.sqrt(np.mean(err**2)):.3f} ({len(cols)}f)", flush=True)
    return pd.concat(oof_parts), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    a = ap.parse_args()
    df = load_task(a.task)
    # merge derived v2 columns (e.g. __normO) stored in the v2 parquets —
    # row order is identical to load_task output (1:1 positional join)
    for n in ("topology_features_v2", "hodge_features_v2", "physical_compact_v2"):
        extra = pd.read_parquet(OUT / f"{n}_{a.task}.parquet")
        new_cols = [c for c in extra.columns if c not in df.columns]
        for c in new_cols:
            df[c] = extra[c].to_numpy()
    g = group_cols(a.task)
    P = p_cols(a.task)
    S = s_cols(a.task)
    p2_nolj = [c for c in g["P2"] if "_lj_" not in c]
    if TASKS[a.task]["kind"] == "rank":
        T1, H1, S1 = v1_cols(a.task, "T"), v1_cols(a.task, "H"), v1_cols(a.task, "S")
        sets = {
            "B2_PcV2": g["P2"], "B4_T2": g["T2"], "B6_H2": g["H2"],
            "B7_PcT2": g["P2"] + g["T2"], "B8_PcH2": g["P2"] + g["H2"],
            "B9_PcT2H2": g["P2"] + g["T2"] + g["H2"],
            "B10_Pc_T2res_H2res": {"cols": g["P2"] + g["T2"] + g["H2"],
                                   "residualize_on_S": g["T2"] + g["H2"]},
            "B11_Sv2": S,                    # strict size-only (was A8_S mixed)
            "B12_PcS": g["P2"] + S,
            "B13_PcT2H2S": g["P2"] + g["T2"] + g["H2"] + S,
            "B14_PcNoLJ": p2_nolj,
            "B15_PfullS": P + S1,            # needed for CleanFull-vs-PS attribution
            "P_PCA142": {"cols": P, "pca": 142},
            "P_PCA40": {"cols": P, "pca": 40},
            "P_PCA20": {"cols": P, "pca": 20},
        }
        # single-module cross-compression -> localises the source of the
        # attenuated PD-bound T×H interaction (pd_bound primary; rna cheap)
        if a.task in ("pd_bound", "rna"):
            sets.update({
                "X_P_H2": P + g["H2"],           # compress H only
                "X_P_T_H2": P + T1 + g["H2"],
                "X_P_T2": P + g["T2"],           # compress T only
                "X_P_T2_H": P + g["T2"] + H1,
                "X_Pc_T": g["P2"] + T1,          # compress P only
                "X_Pc_H": g["P2"] + H1,
                "X_Pc_T_H": g["P2"] + T1 + H1,
            })
        folds = load_folds(a.task, df)
        oof, met = run_rank_v2(a.task, df, sets, folds)
    else:
        sets = {
            "B2_PcV2": g["P2"], "B4_T2": g["T2"], "B6_H2": g["H2"],
            "B7_PcT2": g["P2"] + g["T2"], "B8_PcH2": g["P2"] + g["H2"],
            "B9_PcT2H2": g["P2"] + g["T2"] + g["H2"],
            "B11_Sv2": S, "B12_PcS": g["P2"] + S, "B14_PcNoLJ": p2_nolj,
        }
        oof, met = run_reg_v2(a.task, df, sets)
    # merge v1 anchor models for paired comparisons
    f1 = OUT / f"oof_{a.task}.parquet"
    if f1.exists():
        o1 = pd.read_parquet(f1)
        keep = {"rna": ["A1_P", "A2_T", "A3_H", "A8_S", "A7_PTH", "A4_PT", "A5_PH",
                        "A6_TH", "A1_P__nolj", "A7_PTH__nolj", "CleanFull_PTHS",
                        "CleanFull_PTHS__nolj"],
                "pd_bound": ["A1_P", "A2_T", "A3_H", "A8_S", "A7_PTH", "A4_PT", "A5_PH",
                             "A6_TH", "A1_P__nolj", "A7_PTH__nolj", "CleanFull_PTHS",
                             "CleanFull_PTHS__nolj"],
                "pd_unbound": ["A1_P", "A2_T", "A3_H", "A8_S", "A7_PTH", "A4_PT", "A5_PH",
                               "A6_TH", "A1_P__nolj", "A7_PTH__nolj", "CleanFull_PTHS",
                               "CleanFull_PTHS__nolj"],
                "membrane": []}.get(a.task, [])
        if keep:
            o1 = o1[o1.model.isin(keep)]
            # hard alignment check: v1/v2 OOF must cover identical rows
            s_new = set(oof.structure_id.astype(str))
            s_old = set(o1.structure_id.astype(str))
            assert s_new == s_old, (f"{a.task}: OOF id mismatch "
                                    f"new-only={len(s_new - s_old)} "
                                    f"old-only={len(s_old - s_new)}")
            o1["structure_id"] = o1["structure_id"].astype(str)
            oof = pd.concat([oof.assign(structure_id=oof.structure_id.astype(str)), o1])
    oof.to_parquet(OUT / f"oof_v2_{a.task}.parquet", index=False)
    (OUT / f"metrics_v2_{a.task}.json").write_text(json.dumps(met, indent=1, default=float))


if __name__ == "__main__":
    main()
