"""Audit the union-support predictions using only committed results."""

import argparse
from collections import Counter
from itertools import combinations
import json
from math import isfinite, sqrt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cspph.chemistry.patch_builder import InteractionPatch, MicroContact
from cspph.core.gsph1 import VDW_RADII
from cspph.geometry.patch_support import compute_patch_support


ROOT = Path(__file__).resolve().parents[1]
JUNG_BOUND = sqrt(1.5)
CERTIFICATE_ATOL = 4e-6  # Å, the atom solver's feasibility tolerance


def patch_spread(patch, atoms):
    return max(
        (float(np.linalg.norm(atoms[i].xyz - atoms[j].xyz))
         for members in (patch.donor_members, patch.acceptor_members)
         for i, j in combinations(members, 2)),
        default=0.0,
    )


def audit_path(example):
    if not example["witness"]["certificate_passed"]:
        raise ValueError("Uncertified archived witness")
    atoms = {
        int(key): SimpleNamespace(xyz=np.asarray(atom["xyz"]), element=atom["element"])
        for key, atom in example["atoms"].items()
    }
    patches = []
    for row in example["patches"]:
        contacts = [MicroContact(**contact) for contact in row["contacts"]]
        if not contacts:
            raise ValueError("Empty contact patch")
        patches.append(InteractionPatch(
            patch_id=row["patch_id"], donor_site=tuple(row["donor_site"]),
            acceptor_site=tuple(row["acceptor_site"]), contacts=contacts,
            donor_members=tuple(sorted({contact.donor for contact in contacts})),
            acceptor_members=tuple(sorted({contact.acceptor for contact in contacts})),
            geometry={},
        ))
    canon = tuple(map(int, example["canon"].split(",")))
    if tuple(sorted(patch.patch_id for patch in patches)) != canon or len(set(canon)) < 3:
        raise ValueError("Witness patches do not match a canonical path with at least three patches")
    cache = {}

    def support(family):
        result = compute_patch_support(family, atoms, support_cache=cache)
        if not (result.complete and result.certificate_passed and isfinite(result.tau)):
            raise ValueError(f"Incomplete or uncertified support for {example['canon']}")
        return result.tau

    tau = support(patches)
    pairwise = max(support(pair) for pair in combinations(patches, 2))
    residual = max(abs(tau - example["tau"]), abs(pairwise - example["pairwise_max"]))
    if residual > CERTIFICATE_ATOL:
        raise ValueError(f"Support does not reproduce archived values for {example['canon']}")
    delta = max(patch_spread(patch, atoms) for patch in patches)
    radii = [
        VDW_RADII.get(atoms[i].element, VDW_RADII["X"])
        for patch in patches for i in patch.atom_ids
    ]
    numerator, denominator = min(radii) + tau, max(radii) + pairwise + delta
    eta = tau - pairwise
    if eta < -CERTIFICATE_ATOL:
        raise ValueError(f"Pairwise support exceeds common support for {example['canon']}")
    return dict(
        path=example["path"], selection_rule=example["selection_rule"],
        n_patches=len(patches), single_contact=all(p.contact_count == 1 for p in patches),
        r_min_angstrom=min(radii), r_max_angstrom=max(radii),
        tau_angstrom=tau, tau_pw_angstrom=pairwise, delta_angstrom=delta,
        eta_angstrom=0.0 if abs(eta) <= CERTIFICATE_ATOL else eta,
        ratio=numerator / denominator,
        violates_bound=bool(numerator > JUNG_BOUND * denominator + CERTIFICATE_ATOL),
        archived_support_max_residual_angstrom=residual,
    )


def summarize_witnesses(rows):
    if not rows:
        raise ValueError("No archived witness paths to summarize")
    ratios = [row["ratio"] for row in rows]
    delta = [row["delta_angstrom"] for row in rows]
    eta = [row["eta_angstrom"] for row in rows]
    quantiles = {"min": 0, "q25": 0.25, "median": 0.5, "q75": 0.75, "max": 1}
    varied = len(set(delta)) > 1 and len(set(eta)) > 1
    return dict(
        n_paths=len(rows), n_bound_violations=sum(row["violates_bound"] for row in rows),
        n_single_contact=sum(row["single_contact"] for row in rows),
        n_positive_eta_at_zero_delta=sum(d == 0 and e > 0 for d, e in zip(delta, eta)),
        ratio_quantiles=dict(zip(quantiles, np.quantile(ratios, list(quantiles.values())).tolist())),
        mean_delta_angstrom=float(np.mean(delta)),
        eta_delta_pearson=float(np.corrcoef(delta, eta)[0, 1]) if varied else None,
        eta_delta_spearman=float(spearmanr(delta, eta).statistic) if varied else None,
    )


def count_cohort_paths(paths):
    if paths["order"].max() >= 4:
        raise ValueError("At least one order needs explicit four-patch subset supports")
    summaries = []
    for (task, graph), group in paths.groupby(["task", "graph"]):
        summaries.append(dict(
            task=task, graph=graph, n_structures=group["structure_id"].nunique(),
            n_paths=int(group["n_paths"].sum()), n_certified_paths=int(group["n_valid"].sum()),
            max_patches_per_path=int(group["order"].max()) + 1,
            n_strict_four_patch_excess=0,
        ))
    return summaries


def barcode_signature(arm):
    intervals = arm["intervals"]
    return {d: Counter(map(tuple, intervals[f"intervals_k{d}"])) for d in range(3)}


def audit_barcodes(tables, paths):
    summary = pd.read_csv(tables / "barcode_comparison_16structure.csv")
    if not summary["tau_pairwise_identical"].isin([0, 1]).all():
        raise ValueError("Barcode summary contains unknown equality flags")
    details = {}
    for name in ("filtration_compare.json", "filtration_compare_rna_clash06.json"):
        details.update({
            row["structure_id"]: row for row in json.loads((tables / name).read_text())
        })
    counts = paths.groupby(["structure_id", "graph"])["n_paths"].sum()
    if summary.duplicated(["structure_id", "graph"]).any() or set(counts.index) != set(
        zip(summary.structure_id, summary.graph)
    ):
        raise ValueError("Barcode summary does not match the path cohort")
    task_of = paths.drop_duplicates("structure_id").set_index("structure_id")["task"]
    rows = []
    for entry in summary.itertuples(index=False):
        sid, graph = entry.structure_id, entry.graph
        different = not bool(entry.tau_pairwise_identical)
        raw = details.get(sid)
        verified = False
        if raw is not None and raw[f"n_{graph}_paths"] == counts[sid, graph]:
            common, pair = (raw[f"{graph}_{rule}"] for rule in ("tau", "pairwise"))
            if raw["scales"] == [0.0, 0.5, 2.0] and all(
                arm.get("chain_map_valid") and "intervals" in arm for arm in (common, pair)
            ):
                verified = True
                if (barcode_signature(common) != barcode_signature(pair)) != different:
                    raise ValueError(f"Raw and summary barcodes disagree: {sid}, {graph}")
        rows.append(dict(
            structure_id=sid, task=task_of[sid], graph=graph, different=different,
            verified_from_intervals=verified,
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    tables = args.results / "tables"
    witnesses = {}
    for path in sorted((args.results / "figures" / "pymol_witness").glob("*.witness_strat.json")):
        dump = json.loads(path.read_text())
        rows = [audit_path(example) for example in dump["examples"]]
        witnesses[dump["structure_id"]] = dict(
            task=dump["task"], summary=summarize_witnesses(rows), paths=rows,
        )
    paths = pd.read_parquet(tables / "path_landscape.parquet")
    print(json.dumps(dict(
        jung_bound=JUNG_BOUND,
        witnesses=witnesses,
        witness_summary=summarize_witnesses([row for study in witnesses.values() for row in study["paths"]]),
        cohort_paths=count_cohort_paths(paths),
        archived_barcodes=audit_barcodes(tables, paths),
    ), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
