"""Audit complete complex dumps and compare same-path Å barcodes in both graphs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from experiments.union_support_audit import audit_dumps, check_archived_path_counts
from experiments.exp03_filtration_comparison.compare_exact import load_births
from cspph.topology.event_persistence import compare_event_barcodes


def barcode_metrics(common, pairwise):
    rows = []
    for order in range(3):
        left, right = common.get(order, []), pairwise.get(order, [])
        row = {'order': order, 'different': Counter(map(tuple, left)) != Counter(map(tuple, right))}
        for name, bars in [('common', left), ('pairwise', right)]:
            row[name + '_count'] = len(bars)
            row[name + '_infinite'] = sum(death is None for _, death in bars)
            row[name + '_finite_total_persistence_A'] = float(sum(death - birth for birth, death in bars if death is not None))
        row['count_delta'] = len(left) - len(right)
        row['finite_total_persistence_delta_A'] = row['common_finite_total_persistence_A'] - row['pairwise_finite_total_persistence_A']
        rows.append(row)
    return rows


def run(directories, out, cohort=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sources, errors = {}, []
    for directory in directories:
        for path in sorted(Path(directory).glob('*.complex_paths.parquet')):
            sid = path.name.removesuffix('.complex_paths.parquet')
            if sid in sources and sources[sid] != path.parent.resolve():
                raise ValueError(f'Duplicate structure {sid}; choose one extraction, not both RNA variants')
            sources[sid] = path.parent.resolve()
    census = pd.read_parquet(cohort) if cohort else None
    if census is not None:
        expected = set(census.structure_id)
        if set(sources) != expected:
            errors.append({'error': 'cohort mismatch', 'missing': sorted(expected - set(sources)),
                           'unexpected': sorted(set(sources) - expected)})
    paths, barcodes = [], []
    for sid, directory in sorted(sources.items()):
        for graph in ('dir', 'und'):
            try:
                rows = audit_dumps(directory, sid, graph)
                if census is not None:
                    check_archived_path_counts(rows, census, sid, graph)
                table = pd.DataFrame(rows)
                table.to_parquet(out / f'{sid}.{graph}.path_metrics.parquet', index=False)
                paths.append(table)
                common, pairwise = load_births(directory, sid, graph)
                scales, cb, pb = compare_event_barcodes(common, pairwise)
                metrics = barcode_metrics(cb, pb)
                barcodes.extend(dict(structure_id=sid, graph=graph, **row) for row in metrics)
                (out / f'{sid}.{graph}.barcodes.json').write_text(json.dumps(
                    dict(structure_id=sid, graph=graph, scales_A=scales, common_A=cb, pairwise_A=pb), indent=2))
                print(f'{sid} {graph}: {len(rows)} paths, barcode differences {[m["different"] for m in metrics]}', flush=True)
            except Exception as exc:
                error = dict(structure_id=sid, graph=graph, error=f'{type(exc).__name__}: {exc}')
                errors.append(error)
                print(json.dumps(error), flush=True)
    distributions = []
    positive = []
    violations = []
    if paths:
        table = pd.concat(paths, ignore_index=True)
        table.to_parquet(out / 'path_metrics.parquet', index=False)
        for (graph, order), sub in table.groupby(['graph', 'order']):
            distributions.append(dict(graph=graph, order=int(order), n_paths=len(sub),
                bound_violations=int(sub.violates_bound.sum()),
                **{name: float(sub.ratio.quantile(q)) for name, q in
                   [('min', 0), ('q25', .25), ('median', .5), ('q75', .75), ('max', 1)]}))
        positive = table.loc[table.single_contact & table.delta_angstrom.eq(0) & table.eta_angstrom.gt(0)].to_dict('records')
        violations = table.loc[table.violates_bound].to_dict('records')
    pd.DataFrame(distributions).to_csv(out / 'ratio_distribution.csv', index=False)
    pd.DataFrame(barcodes).to_csv(out / 'barcode_summary.csv', index=False)
    pd.DataFrame(positive).to_csv(out / 'single_contact_positive.csv', index=False)
    pd.DataFrame(violations).to_csv(out / 'bound_violations.csv', index=False)
    if not sources:
        errors.append({'error': 'No complex_paths dumps found; summary tables cannot replace full dumps'})
    summary = dict(complete=not errors, n_structures=len(sources), n_graphs_completed=len(barcodes) // 3,
                   ratio_distribution=distributions, bound_violations=violations,
                   single_contact_positive=positive, barcodes=barcodes, errors=errors)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in summary.items() if k != 'single_contact_positive'}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dumps', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, help='Require this archived path census, including per-order counts')
    args = parser.parse_args()
    result = run(args.dumps, args.out, args.cohort)
    if not result['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
