# Data Dictionary

## Manifests

- `manifests/datasets.yaml` — frozen dataset registry (clash06 RNA).
- `manifests/targets_rna.csv` — `target_id,n_structures,task`.
- `manifests/targets_pd.csv` — `target_id,n_decoys,task`.
- `manifests/deep_subset_v33.csv` — 16 deep-dump structures with labels,
  strata, path counts.
- `manifests/candidate_manifest.csv` — schema-frozen candidate registry
  (populated per task by `tasks/*/prepare.py`).
- `manifests/legacy_file_map.csv` — old_path -> new_path migration map.
- `manifests/checksums.sha256` — checksums of tracked manifests/configs.

## Result schemas (see plan §6)

- `path_support.parquet` — per-path tau_cs, tau_pw, eta, witness, solver
  status, config/git hashes.
- `filtration_comparison.parquet` — per-path births under each rule.
- `barcode_intervals.parquet` — per-dim birth/death intervals.
- `oof_predictions.parquet` — per-candidate, per-seed, per-arm OOF.
- `hardpair_by_target_seed.csv` — per-target hard-pair accuracy.

## Frozen tables in this repo

`results/tables/` contains only small derived CSV/JSON (metrics, hard-pair
summaries, filtration summary, barcode comparison, feature manifests).
Large Parquet feature/OOF matrices live in `results/processed/`
(gitignored or DVC-tracked) and are reproducible via EXP-05/EXP-06.

## Units & conventions

- All distances in Angstrom; tau/birth values in Angstrom.
- Discrete scales `[0.0, 0.5, 2.0]`; fine grid defined in
  `configs/filtration.yaml`.
- RNA RMSD label is lower-is-better; unified score sign = -1.
