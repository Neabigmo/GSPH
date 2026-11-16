# -*- coding: utf-8 -*-
"""Step-5: assemble summary tables + ablation_report.md from metrics/oof/bootstrap."""
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
import json
from pathlib import Path
import pandas as pd

H3 = Path(r"H:\2026try\11.3GSPH\GSPH_H3")
OUT = H3 / "results" / "ablation"

RANK_METRIC = "target_macro_spearman"


def load(task):
    m = json.loads((OUT / f"metrics_{task}.json").read_text())
    oof = pd.read_parquet(OUT / f"oof_{task}.parquet")
    return m, oof


def main():
    cov = json.loads((OUT / "candidate_coverage.json").read_text())
    boot = pd.read_csv(OUT / "paired_bootstrap.csv") if (OUT / "paired_bootstrap.csv").exists() else pd.DataFrame()

    L = []
    L.append("# GSPH Feature Attribution Ablation — 报告\n")
    L.append("协议：固定候选池 + 固定 fold + 固定 XGBRanker(80 rounds, depth4, lr0.05) 或 LOPO 回归；"
             "特征分组 P(物理)/T(GLMY+路径+几何支撑+持久同调)/H(Shared-Hodge 谱读出)/S(规模控制) 互斥。\n")

    L.append("\n## 0. 候选池\n")
    L.append("| 任务 | 行数 | 分组 | n_P | n_T | n_H | n_S |\n|---|---|---|---|---|---|---|\n")
    for t, c in cov.items():
        L.append(f"| {t} | {c['final_rows']} | {c['groups']} | {c['n_P']} | {c['n_T']} | {c['n_H']} | {c['n_S']} |\n")

    for task in ("rna", "pd_bound", "pd_unbound"):
        if not (OUT / f"metrics_{task}.json").exists():
            continue
        m, _ = load(task)
        L.append(f"\n## 1. {task} — target-macro Spearman / top-1 label\n")
        L.append("| 模型 | n_feat | Spearman | top1 | top5 | top10 |\n|---|---|---|---|---|---|\n")
        for k in sorted(m):
            v = m[k]
            L.append(f"| {k} | {v['n_features']} | {v['target_macro_spearman']:.3f} | "
                     f"{v['top1_label']:.3f} | {v['top5_label']:.3f} | {v['top10_label']:.3f} |\n")

    if (OUT / "metrics_membrane.json").exists():
        m, _ = load("membrane")
        L.append("\n## 2. membrane — 严格 LOPO（macro per-protein Spearman）\n")
        L.append("| 模型 | ridge | enet | rf | xgb | best-rmse |\n|---|---|---|---|---|---|\n")
        base = {}
        for k, v in m.items():
            name, pred = k.split("::")
            base.setdefault(name, {})[pred] = v
        for name in sorted(base):
            d = base[name]
            rmses = {p: d[p]["rmse"] for p in d}
            row = "| " + name + " | " + " | ".join(
                f"{d[p]['macro_spearman']:+.3f}" if p in d else "—" for p in ("ridge", "enet", "rf", "xgb"))
            row += f" | {min(rmses.values()):.3f} |\n"
            L.append(row)

    if len(boot):
        L.append("\n## 3. 配对 bootstrap（Holm 校正）\n")
        L.append("| 任务 | 对比 | Δ均值 | 95%CI | p_holm | n_groups |\n|---|---|---|---|---|---|\n")
        for _, r in boot.iterrows():
            L.append(f"| {r.task} | {r.comparison} ({r.model_a}−{r.model_b}) | "
                     f"{r['mean']:+.4f} | [{r['ci95']}] | {r.p_holm:.3f} | {int(r.n)} |\n")

    L.append("\n## 4. 归因结论（按修正后协议）\n")
    L.append("见正文讨论；一切以 paired bootstrap 的 CI/Holm-p 为准，单点差值不作结论。\n")

    (OUT / "ablation_report.md").write_text("".join(L), encoding="utf-8")
    print("wrote", OUT / "ablation_report.md")


if __name__ == "__main__":
    main()
