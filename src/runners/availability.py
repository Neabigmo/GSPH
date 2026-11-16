import csv
import json
import os
import time

import yaml

from .paths import CONFIG_DIR, REPORTS_DIR, ROOT, ensure_layout


def build_report():
    ensure_layout()
    with open(os.path.join(CONFIG_DIR, "methods.yaml"), encoding="utf-8") as handle:
        methods = yaml.safe_load(handle).get("methods", {})
    rows = []
    for name, spec in methods.items():
        if spec.get("implemented"):
            status = "implemented"
            reason = spec.get("implemented_note", "runnable adapter validated")
        else:
            status = "blocked_pending_validation"
            reason = spec.get("blocked_reason", "official code/weights/input contract not yet validated")
        upstream = spec.get("upstream", "")
        if upstream and not spec.get("implemented"):
            repo_name = upstream.rstrip("/").split("/")[-1].replace(".git", "")
            manifest = os.path.join(ROOT, "third_party", "manifests", repo_name + "_tree.json")
            if os.path.exists(manifest):
                reason = reason + " Source tree manifest is recorded; runtime adapter remains pending."
            elif upstream:
                reason = reason + " Source tree/API manifest is unavailable."
        rows.append({
            "method": name,
            "category": spec.get("category", ""),
            "implemented": str(bool(spec.get("implemented"))).lower(),
            "status": status,
            "direction": spec.get("direction", ""),
            "scenarios": ";".join(spec.get("scenarios", [])),
            "alias_of": spec.get("alias_of", ""),
            "upstream": upstream,
            "reason": reason,
            "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    path = os.path.join(REPORTS_DIR, "method_availability.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
