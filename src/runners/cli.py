import argparse
import json
import os
import subprocess

from .baselines import methods_for_scenario, score_structural_methods
from .datasets import SCENARIOS, dataset_report, fetch_prdbv3, prepare
from .evaluate import evaluate_scenario
from .availability import build_report
from .io import read_rows, result_path
from .paths import ROOT, ensure_layout
from .fair_pipeline import SCENARIO_GROUPS, run_stage, make_folds, merge_shard_labels, read_table


def _status():
    ensure_layout()
    print("workspace=%s" % ROOT)
    for scenario, dataset_root in SCENARIOS.items():
        sample_path = result_path(scenario, "samples")
        score_path = result_path(scenario, "scores")
        print("%-14s data=%s samples=%d scores=%d" % (
            scenario, "present" if os.path.exists(dataset_root) and os.listdir(dataset_root) else "missing",
            len(read_rows(sample_path)), len(read_rows(score_path))))
    for scenario in SCENARIOS:
        sample = os.path.join(ROOT, "results", scenario + "_samples.parquet")
        labels = os.path.join(ROOT, "results", scenario + "_labels.parquet")
        feature_files = 0
        feature_root = os.path.join(ROOT, "features", scenario)
        for base, _, names in os.walk(feature_root):
            feature_files += sum(1 for name in names if name.endswith(".parquet"))
        print("fair %-11s samples=%d labels=%d feature_files=%d" % (
            scenario, len(read_table(sample)), len(read_table(labels)), feature_files))


def main():
    parser = argparse.ArgumentParser(description="GSPH experiment workspace CLI")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status")
    sub.add_parser("method-report")
    p_dataset = sub.add_parser("dataset-report")
    p_dataset.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    p_prepare.add_argument("--fetch-prdbv3", action="store_true")
    p_score = sub.add_parser("score")
    p_score.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    p_score.add_argument("--methods", default="all", help="comma-separated names or all")
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    p_summary = sub.add_parser("summarize")
    p_summary.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    for command in ["prepare-decoys", "label", "extract", "train", "resume"]:
        p = sub.add_parser(command)
        p.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
        p.add_argument("--count", type=int, default=100,
                       help="decoys per target for prepare-decoys/resume")
        p.add_argument("--limit-targets", type=int, default=None)
        p.add_argument("--targets-file", default=None,
                       help="newline-separated target IDs for source-disjoint extraction shards")
    p_folds = sub.add_parser("folds")
    p_folds.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    p_merge = sub.add_parser("merge-labels")
    p_merge.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    p_gsph1 = sub.add_parser("gsph1-extract", help="extract one independent GSPH1 feature record")
    p_gsph1.add_argument("--pdb", required=True)
    p_gsph1.add_argument("--scenario", choices=list(SCENARIOS) + ["unknown"], default="unknown")
    p_gsph1.add_argument("--chain-a", default=None)
    p_gsph1.add_argument("--chain-b", default=None)
    p_gsph1.add_argument("--mode", choices=("debug", "benchmark"), default="debug")
    p_gsph1.add_argument("--max-events", type=int, default=32)
    p_gsph1.add_argument("--max-paths-per-order", type=int, default=64)
    p_gsph1.add_argument("--max-local-pairs-per-event", type=int, default=3000)
    p_gsph1.add_argument("--interaction-backend", choices=("auto", "cpu", "cuda"), default="auto")
    p_gsph1.add_argument("--gpu-min-pairs", type=int, default=250000)
    p_gsph1.add_argument("--gpu-batch-pairs", type=int, default=500000)
    p_gsph1.add_argument("--gpu-device", default="cuda:0")
    p_gsph1.add_argument("--support-workers", type=int, default=4)
    p_gsph1.add_argument("--path-backend", choices=("auto", "compact_csr", "python_stream"), default="auto")
    p_gsph1.add_argument("--surface-backend", choices=("auto", "none"), default="auto")
    p_gsph1.add_argument("--hodge-backend", choices=("auto", "cpu", "cuda"), default="auto")
    p_gsph1.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.command == "status":
        _status()
    elif args.command == "method-report":
        print(build_report())
    elif args.command == "dataset-report":
        print(dataset_report(args.scenario))
    elif args.command == "prepare":
        if args.fetch_prdbv3:
            fetch_prdbv3()
        prepare(args.scenario)
    elif args.command == "score":
        prepare(args.scenario)
        scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        for scenario in scenarios:
            methods = methods_for_scenario(scenario) if args.methods == "all" else [
                x.strip() for x in args.methods.split(",") if x.strip()
            ]
            score_structural_methods(scenario, methods)
    elif args.command == "evaluate":
        scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        for scenario in scenarios:
            fair_predictions = os.path.join(ROOT, "results", scenario + "_predictions.parquet")
            if os.path.exists(fair_predictions):
                print(run_stage("evaluate", scenario))
            else:
                evaluate_scenario(scenario)
    elif args.command == "summarize":
        scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        for scenario in scenarios:
            path = os.path.join(ROOT, "reports", "%s_ranking.csv" % scenario)
            print("%s: %s" % (scenario, path if os.path.exists(path) else "not evaluated"))
    elif args.command in {"prepare-decoys", "label", "extract", "train", "resume"}:
        scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        target_filter = None
        if getattr(args, "targets_file", None):
            with open(args.targets_file, encoding="utf-8") as handle:
                target_filter = [line.strip() for line in handle if line.strip()]
        for scenario in scenarios:
            if args.command == "extract":
                from .fair_pipeline import extract_features
                print(extract_features(scenario, limit_targets=args.limit_targets, target_filter=target_filter))
            elif args.command == "label" and target_filter:
                from .fair_pipeline import label_decoys
                suffix = os.path.basename(args.targets_file)
                out = os.path.join(ROOT, "results", "%s_labels_%s.parquet" % (scenario, suffix))
                print(label_decoys(scenario, target_filter=target_filter, output_path=out))
            else:
                print(run_stage(args.command, scenario, count=args.count, limit_targets=args.limit_targets))
    elif args.command == "folds":
        scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        for scenario in scenarios:
            print(make_folds(scenario))
    elif args.command == "merge-labels":
        scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        for scenario in scenarios:
            print(merge_shard_labels(scenario))
    elif args.command == "gsph1-extract":
        from .gsph1 import GSPH1Config, extract_gsph1
        record = extract_gsph1(
            args.pdb, args.scenario, args.chain_a, args.chain_b,
            GSPH1Config(
                mode=args.mode, max_events=args.max_events,
                max_paths_per_order=args.max_paths_per_order,
                max_local_pairs_per_event=args.max_local_pairs_per_event,
                interaction_backend=args.interaction_backend,
                gpu_min_pairs=args.gpu_min_pairs,
                gpu_batch_pairs=args.gpu_batch_pairs,
                gpu_device=args.gpu_device,
                support_workers=args.support_workers,
                path_backend=args.path_backend,
                surface_backend=args.surface_backend,
                hodge_backend=args.hodge_backend,
            ),
        )
        payload = json.dumps(record, ensure_ascii=False, indent=2)
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
            print(args.output)
        else:
            print(payload)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
