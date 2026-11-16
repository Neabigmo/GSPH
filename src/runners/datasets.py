import csv
import os
import subprocess
import time
from collections import Counter

from .io import SAMPLE_FIELDS, result_path, write_rows
from .paths import CONFIG_DIR, DATASETS_DIR, REPORTS_DIR, ensure_layout


SCENARIOS = {
    "protein_rna": os.path.join(DATASETS_DIR, "protein_rna", "PRDBv3"),
    "rna_rna": os.path.join(DATASETS_DIR, "rna_rna", "RRDB"),
    "protein_dna": os.path.join(DATASETS_DIR, "protein_dna", "protein_dna_benchmark"),
}


DATASET_EXPECTATIONS = {
    "protein_rna": {
        "primary_expected": "197 PRDBv3 docking cases",
        "external_expected": "21 available CASP16 hybrid target archives",
        "coverage_status": "primary_and_external_models_present",
        "coverage_note": "All 197 PRDBv3 targets are normalized for the common pool; the official PRDBv3 bound/unbound source tree is retained. All 21 CASP16 hybrid archives were validated and extracted under datasets/protein_rna/CASP16_hybrid for separate external evaluation.",
    },
    "rna_rna": {
        "primary_expected": "123 RRDB targets",
        "external_expected": "60 DITScoreRR/HNADOCK test cases",
        "coverage_status": "primary_complete_historical_subset_identified",
        "coverage_note": "The official RRDB package contains all 123 targets with bound and unbound RNA partners; 123 native complexes are normalized for scoring. All 60 historical DITScoreRR/HNADOCK IDs are present.",
    },
    "protein_dna": {
        "primary_expected": "47 Protein-DNA Docking Benchmark cases",
        "external_expected": "10 pyDockDNA external cases",
        "coverage_status": "primary_complete_external_structures_present",
        "coverage_note": "The official 47-case Protein-DNA archive contains complex plus bound/unbound protein and DNA partners; all 10 pyDockDNA external PDB structures are present. The common scoring pool contains the 47 benchmark complexes plus 10 external structures.",
    },
}


def scenario_root(scenario):
    root = SCENARIOS[scenario]
    # Prefer normalized full-complex structures when an external benchmark
    # package has been unpacked. Fall back to the official RRDB partner files
    # so preparation remains useful before normalization finishes.
    if scenario == "rna_rna":
        normalized = os.path.join(root, "native_complexes")
        if os.path.isdir(normalized) and any(name.lower().endswith(".pdb") for name in os.listdir(normalized)):
            return normalized
    return root


def _candidate_rows(scenario):
    root = scenario_root(scenario)
    has_prdb_cases = scenario == "protein_rna" and os.path.isdir(
        os.path.join(root, "prdbv3_cases"))
    has_pdna_cases = scenario == "protein_dna" and os.path.isdir(
        os.path.join(root, "ftdmp_structures"))
    rows = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if not name.lower().endswith((".pdb", ".ent", ".cif", ".mmcif")):
                continue
            stem = os.path.splitext(name)[0].lower()
            # Benchmark archives also contain isolated bound/unbound partners
            # and transformed helper files. The common-pool ranking table uses
            # combined complexes only; the original files remain untouched.
            excluded_tokens = ("_prot_u", "_prot_b", "_rna_u", "_rna_b",
                               "_dna_u", "_dna_b", "_full", "_mod",
                               "_rna1", "_rna2", "_bound")
            if any(token in stem for token in excluded_tokens):
                continue
            path = os.path.join(dirpath, name)
            if has_prdb_cases and (
                    "ftdmp_structures" in path or
                    os.sep + "smoke" + os.sep in path or
                    os.sep + "PRDBv3.0" + os.sep in path or
                    os.sep + "PRDBv3_dataset-main" + os.sep in path):
                # The Zenodo subset overlaps the official PRDBv3 cases. Keep
                # raw archives on disk for provenance, but use the normalized
                # official 197-case pool for the common candidate table.
                continue
            if has_pdna_cases and os.sep + "smoke" + os.sep in path:
                # A development fixture must never enter the final benchmark
                # once the official 47-case structure archive is available.
                continue
            parent = os.path.basename(os.path.dirname(path)) or "unknown_target"
            if scenario == "protein_rna" and parent.lower() not in ("prdbv3", "smoke", "prdbv3_cases"):
                # PRDBv3 directories contain bound, unbound, transformed and
                # helper structures. The common candidate pool uses the file
                # whose stem is the directory/complex identifier.
                if stem != parent.lower():
                    continue
            if (os.path.abspath(root).endswith("native_complexes") or
                    parent.lower() in ("prdbv3_cases", "pydockdna_external")):
                target = stem.upper()
            else:
                target = parent
            if scenario == "protein_dna" and parent.lower() not in ("protein_dna_benchmark", "smoke", "pydockdna_external"):
                if stem != parent.lower():
                    continue
            if "pydockdna_external" in path:
                source = "pydockdna_external_rcsb"
            elif "native_complexes" in path:
                source = "rrdb_official_native"
            elif "ftdmp_structures" in path:
                source = "zenodo_ftdmp_structure_archive"
            elif "prdbv3_cases" in path:
                source = "prdbv3_official_rcsb_bound"
            elif os.sep + "smoke" + os.sep in path:
                source = "smoke_fixture"
            elif scenario == "protein_rna":
                source = "prdbv3_official_candidate"
            elif scenario == "protein_dna":
                source = "protein_dna_benchmark_candidate"
            else:
                source = "official_or_downloaded"
            rows.append({
                "scenario": scenario,
                "dataset": os.path.basename(root),
                "target_id": target,
                "candidate_id": os.path.splitext(name)[0],
                "structure_path": path,
                "chain_a": "",
                "chain_b": "",
                "candidate_source": source,
                "reference_path": "",
                "quality_irmsd": "",
                "quality_contact_recovery": "",
                "quality_dockq": "",
                "quality_status": "not_computed",
            })
    return rows


def prepare(scenario):
    ensure_layout()
    scenarios = list(SCENARIOS) if scenario == "all" else [scenario]
    for item in scenarios:
        os.makedirs(SCENARIOS[item], exist_ok=True)
        rows = _candidate_rows(item)
        write_rows(result_path(item, "samples"), SAMPLE_FIELDS, rows)
        if not rows:
            with open(os.path.join(SCENARIOS[item], "SOURCE_STATUS.txt"), "w", encoding="utf-8") as handle:
                handle.write("No structure files discovered yet. Add official benchmark data, then rerun prepare.\n")
        print("%s: %d candidate structures" % (item, len(rows)))


def dataset_report(scenario="all"):
    """Write a provenance/coverage inventory without fabricating missing labels."""
    ensure_layout()
    scenarios = list(SCENARIOS) if scenario == "all" else [scenario]
    rows = []
    for item in scenarios:
        root = SCENARIOS[item]
        samples = _candidate_rows(item)
        target_ids = sorted({row["target_id"] for row in samples})
        source_counts = Counter(row["candidate_source"] for row in samples)
        structure_files = 0
        for dirpath, _, filenames in os.walk(root):
            structure_files += sum(name.lower().endswith((".pdb", ".ent", ".cif", ".mmcif"))
                                   for name in filenames)
        expectation = DATASET_EXPECTATIONS[item]
        historical_ids = []
        if item == "rna_rna":
            id_path = os.path.join(CONFIG_DIR, "rna_rna_ditscorerr_test_ids.txt")
            with open(id_path, encoding="utf-8") as handle:
                historical_ids = [token for line in handle if not line.lstrip().startswith("#")
                                  for token in line.split()]
            present_ids = {row["target_id"] for row in samples}
            missing_ids = sorted(set(historical_ids) - present_ids)
        else:
            missing_ids = []
        rows.append({
            "scenario": item,
            "dataset_root": root,
            "structure_files_in_root": structure_files,
            "candidate_structures": len(samples),
            "unique_targets": len(target_ids),
            "candidate_source_counts": ";".join(
                "%s=%d" % (key, source_counts[key]) for key in sorted(source_counts)),
            "primary_expected": expectation["primary_expected"],
            "external_expected": expectation["external_expected"],
            "historical_test_targets": len(historical_ids),
            "historical_test_missing": ";".join(missing_ids),
            "coverage_status": expectation["coverage_status"] if samples else "missing",
            "coverage_note": expectation["coverage_note"],
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    path = os.path.join(REPORTS_DIR, "dataset_availability.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def fetch_prdbv3():
    dest = SCENARIOS["protein_rna"]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isdir(os.path.join(dest, ".git")):
        subprocess.check_call(["git", "-C", dest, "pull", "--ff-only"])
    elif not os.path.exists(dest) or not os.listdir(dest):
        subprocess.check_call(["git", "clone", "https://github.com/shrikantcombio/PRDBv3_dataset.git", dest])
