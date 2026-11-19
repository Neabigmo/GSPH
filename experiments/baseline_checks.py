"""Read-only candidate counts and paired RNA RMSD/DI direction diagnostics."""
import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr


def read_labels(path):
    labels = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if parts[0] in labels:
            raise ValueError(f'Duplicate label in {path.name}: {parts[0]}')
        labels[parts[0]] = value
    return labels


def check_di(root):
    rows = []
    for rmsd in sorted((Path(root) / 'RMSD').glob('*')):
        if rmsd.suffix.lower() != '.rmsd':
            continue
        target = rmsd.stem
        matches = [p for p in (Path(root) / 'DI').glob('*') if p.stem == target and p.suffix.lower() == '.di']
        if len(matches) != 1:
            rows.append(dict(target=target, status='missing_or_ambiguous_di'))
            continue
        a, b = read_labels(rmsd), read_labels(matches[0])
        common = sorted(set(a) & set(b))
        x, y = [a[k] for k in common], [b[k] for k in common]
        valid = len(common) >= 3 and len(set(x)) > 1 and len(set(y)) > 1
        rho = float(spearmanr(x, y).statistic) if valid else None
        status = 'insufficient' if not valid else ('ok' if rho > 0 else 'direction_conflict')
        rows.append(dict(target=target, status=status, n_pairs=len(common),
                         n_rmsd=len(a), n_di=len(b), rmsd_di_spearman=rho,
                         evidence='DI lower agrees with RMSD lower' if rho is not None and rho > 0 else 'inspect label definition'))
    if not rows:
        raise ValueError('No RMSD/*.rmsd files; copy RMSD and DI label directories, not PDBs')
    return rows


def inspect_tables(root, expected_groups=47):
    rows = []
    for path in sorted(Path(root).rglob('*.parquet')):
        if not any(token in path.name for token in ('physical_compact_v2', 'topology_features_v2', 'hodge_features_v2', 'oof', 'predictions')):
            continue
        try:
            schema = pq.read_schema(path)
            key = next((c for c in ('group', 'group_id') if c in schema.names), None)
            columns = [c for c in (key, 'structure_id', 'task', 'model') if c and c in schema.names]
            frame = pd.read_parquet(path, columns=columns)
            tasks = frame.groupby('task') if 'task' in frame else [(None, frame)]
            for task, part in tasks:
                task = str(task) if task is not None else next((t for t in ('pd_bound', 'pd_unbound') if t in path.name), 'unknown')
                count = int(part[key].nunique()) if key else None
                rows.append(dict(file=str(path.relative_to(root)), task=task, rows=len(part), columns=len(schema),
                    key_columns=columns, groups=count,
                    candidates=int(part.structure_id.nunique()) if 'structure_id' in part else None,
                    expected_groups=expected_groups if task.startswith('pd_') else None,
                    status='missing_group' if key is None else ('group_count_mismatch' if task.startswith('pd_') and count != expected_groups else 'ok')))
        except Exception as exc:
            rows.append(dict(file=str(path.relative_to(root)), status='error', error=f'{type(exc).__name__}: {exc}'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-root', type=Path)
    parser.add_argument('--pm-root', type=Path)
    parser.add_argument('--expected-groups', type=int, default=47)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    if args.tables_root:
        report['tables'] = inspect_tables(args.tables_root, args.expected_groups)
    if args.pm_root:
        report['di'] = check_di(args.pm_root)
    (args.out / 'baseline_checks.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
    if not report or any(not rows for rows in report.values()) or any(r['status'] != 'ok' for rows in report.values() for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
