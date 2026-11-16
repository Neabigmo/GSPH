import csv
import math
import os
from collections import defaultdict

from .io import read_rows, result_path, write_rows
from .paths import REPORTS_DIR, ensure_layout


def _rank_metrics(scores):
    valid = [x for x in scores if x.get("status") == "ok" and x.get("raw_score") not in (None, "")]
    if not valid:
        return {"n_candidates": 0, "top1_candidate": "", "top5_candidates": "",
                "top1_success": "", "top5_success": "", "spearman": "",
                "ranking_regret": "", "bootstrap_ci": "", "status": "no_valid_scores"}
    valid.sort(key=lambda x: float(x["raw_score"]), reverse=True)
    # Quality labels are intentionally optional at this stage. They are not
    # inferred from scores; the benchmark-specific reference mapping must be
    # populated before success, correlation, regret, or bootstrap values are
    # reported.
    return {
        "n_candidates": len(valid),
        "top1_candidate": valid[0].get("candidate_id", ""),
        "top5_candidates": ";".join(x.get("candidate_id", "") for x in valid[:5]),
        "top1_success": "",
        "top5_success": "",
        "spearman": "",
        "ranking_regret": "",
        "bootstrap_ci": "",
        "status": "scores_only_quality_labels_pending",
    }


def evaluate_scenario(scenario):
    ensure_layout()
    rows = read_rows(result_path(scenario, "scores"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("method", ""), row.get("target_id", ""))].append(row)
    out = []
    for (method, target), values in sorted(grouped.items()):
        metric = _rank_metrics(values)
        metric.update({"scenario": scenario, "method": method, "target_id": target})
        out.append(metric)
    path = os.path.join(REPORTS_DIR, "%s_ranking.csv" % scenario)
    write_rows(path, ["scenario", "method", "target_id", "n_candidates", "top1_candidate", "top5_candidates",
                      "top1_success", "top5_success", "spearman", "ranking_regret", "bootstrap_ci", "status"], out)
    return out
