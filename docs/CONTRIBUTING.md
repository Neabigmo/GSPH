# Contributing

## Workflow

1. Branch per experiment: `exp/expXX-name` (see plan Phase D).
2. Every PR must pass `pytest tests/unit tests/property` and produce a
   small smoke result.
3. Never commit raw data, credentials, or machine-specific absolute paths.
4. Config thresholds (clash ratio, radii scale, scales, seeds) live in
   `configs/` YAML — not hardcoded.
5. New result tables go to `results/tables/` with an
   `experiment_metadata.json` (git_commit, config_hash,
   data_manifest_hash, seeds, command).

## Module rules (plan §2.1)

- `geometry/` — lenses, patch union support, tau/eta, witnesses,
  certificates. No PDB IO, no model training.
- `chemistry/` — radii, roles, clash filter, patch membership, transfer.
- `graph/` — directed graph + admissible paths only.
- `filtration/` — birth rules as separate functions.
- `topology/` — GLMY admissible chains, boundary, persistence.
- `spectral/` — shared Hodge only.
- `features/` — P/T/H groups kept separate until evaluation.
- `evaluation/` — grouped CV, hard pairs, bootstrap, permutation.
- `core/` — orchestration engine (`gsph1` legacy pipeline, cache,
  checkpoint, incremental).

## Compatibility

`src/gsph6/` and `src/runners/` are alias shims. New code must import
`cspph.*`; shims exist only so legacy experiment scripts keep running.
