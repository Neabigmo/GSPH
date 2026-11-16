import os
import signal
import time

import yaml

from .paths import CONFIG_DIR
from .io import SCORE_FIELDS, log_event, read_rows, result_path, write_rows
from .structures import buried_surface_area, interface_features


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("baseline sample exceeded 30 second timeout")


def _run_bsa_with_timeout(path, chain_a, chain_b):
    atom_count = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                atom_count += 1
    # FreeSASA's exact all-atom calculation is allowed for the present
    # protein-nucleic-acid archives (the largest is about 15k atoms). The
    # explicit timeout remains a guard against malformed or unexpectedly huge
    # future inputs; RNA-RNA is blocked at the protocol level in YAML.
    if atom_count > 50000:
        raise ValueError("bsa skipped for resource safety: %d atoms > 50000" % atom_count)
    previous = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(30)
    try:
        return buried_surface_area(path, chain_a, chain_b)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def method_specs():
    with open(os.path.join(CONFIG_DIR, "methods.yaml"), encoding="utf-8") as handle:
        return yaml.safe_load(handle).get("methods", {})


def methods_for_scenario(scenario):
    """Return every registered method applicable to *scenario* in YAML order."""
    return [name for name, spec in method_specs().items()
            if scenario in spec.get("scenarios", [])]


def _blocked_reason(spec, scenario):
    by_scenario = spec.get("blocked_reason_by_scenario", {})
    return by_scenario.get(scenario) or spec.get("blocked_reason") or (
        "official runtime, weights, input contract, or benchmark labels are unavailable")


def score_structural_methods(scenario, methods):
    samples = read_rows(result_path(scenario, "samples"))
    specs = method_specs()
    external_scores = {}
    if "ftdmp" in methods:
        ftdmp_path = os.path.join(
            os.path.dirname(result_path(scenario, "samples")),
            "%s_ftdmp_scores.csv" % scenario)
        if os.path.exists(ftdmp_path):
            external_scores["ftdmp"] = {
                row.get("candidate_id", ""): row
                for row in read_rows(ftdmp_path)
                if row.get("status") == "ok"
            }
    if scenario == "protein_dna" and "pdiscore" in methods:
        pdiscore_path = os.path.join(
            os.path.dirname(result_path(scenario, "samples")), "pdiscore_scores.csv")
        if os.path.exists(pdiscore_path):
            external_scores["pdiscore"] = {
                row.get("candidate_id", ""): row
                for row in read_rows(pdiscore_path)
                if row.get("status") == "ok"
            }
    out = []
    for sample in samples:
        for method in methods:
            started = time.time()
            spec = specs.get(method, {})
            direction = spec.get("direction", "higher_better")
            row = {
                "scenario": scenario,
                "dataset": sample.get("dataset", ""),
                "target_id": sample.get("target_id", ""),
                "candidate_id": sample.get("candidate_id", ""),
                "method": method,
                "raw_score": "",
                "score_direction": direction,
                "status": "failed",
                "elapsed_seconds": "",
                "error": "",
            }
            try:
                if not spec:
                    row["status"] = "blocked"
                    row["error"] = "unknown method; add it to configs/methods.yaml"
                elif scenario not in spec.get("scenarios", []):
                    row["status"] = "not_applicable"
                    row["error"] = "method is not in the declared scenario scope"
                elif scenario in spec.get("blocked_scenarios", []):
                    row["status"] = "blocked"
                    row["error"] = _blocked_reason(spec, scenario)
                elif not spec.get("implemented", False):
                    row["status"] = "blocked"
                    row["error"] = _blocked_reason(spec, scenario)
                else:
                    path = sample.get("structure_path", "")
                    if not path or not os.path.exists(path):
                        raise FileNotFoundError(path or "empty structure_path")
                    if method in ("pdiscore", "ftdmp"):
                        cached = external_scores.get(method, {}).get(
                            sample.get("candidate_id", ""))
                        if not cached:
                            raise ValueError(
                                "official %s result missing; run its scripts/run_* adapter" % method)
                        score = _float(cached.get("raw_score"))
                    else:
                        features = interface_features(path)
                    if method == "contact_count":
                        score = features["contact_count"]
                    elif method == "bsa":
                        score = _run_bsa_with_timeout(path, features["chain_a"], features["chain_b"])
                    elif method == "interface_area_proxy":
                        score = features["interface_area_proxy"]
                    elif method in ("pdiscore", "ftdmp"):
                        pass
                    else:
                        raise ValueError("method adapter not implemented: %s" % method)
                    if score is None:
                        raise ValueError("structure has fewer than two chains or no interface")
                    row["raw_score"] = score
                    row["status"] = "ok"
            except Exception as exc:
                row["error"] = str(exc)
            row["elapsed_seconds"] = round(time.time() - started, 6)
            out.append(row)
            log_event({"action": "score", "scenario": scenario, "method": method,
                       "target_id": row["target_id"], "candidate_id": row["candidate_id"],
                       "status": row["status"], "error": row["error"]})
    write_rows(result_path(scenario, "scores"), SCORE_FIELDS, out)
    return out
