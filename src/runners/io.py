import csv
import json
import os
import time

from .paths import LOG_DIR, RESULTS_DIR, ensure_layout


SAMPLE_FIELDS = [
    "scenario", "dataset", "target_id", "candidate_id", "structure_path",
    "chain_a", "chain_b", "candidate_source", "reference_path", "quality_irmsd",
    "quality_contact_recovery", "quality_dockq", "quality_status",
]
SCORE_FIELDS = [
    "scenario", "dataset", "target_id", "candidate_id", "method", "raw_score",
    "score_direction", "status", "elapsed_seconds", "error",
]


def result_path(scenario, kind):
    ensure_layout()
    return os.path.join(RESULTS_DIR, "%s_%s.csv" % (scenario, kind))


def write_rows(path, fieldnames, rows):
    ensure_layout()
    rows = list(rows)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def log_event(event):
    ensure_layout()
    event = dict(event)
    event.setdefault("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(LOG_DIR, "run_events.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
