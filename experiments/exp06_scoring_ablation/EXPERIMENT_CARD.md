# EXP-06: Scoring Ablation + 20-seed Grouped CV

- **Purpose**: Evaluate GSPH feature groups (P/T/H/S) for decoy ranking
  on the development tasks with proper grouped CV.
- **Data**: RNA quality (rsRNASP PM decoys), PD bound + unbound
  (FTDMP); feature matrices `_feat/{rna,pd_bound,pd_unbound}`.
- **Method**: `run_ablation.py --tag <seed>`; 20 seeds x 3 tasks x arms
  (A1-A7, B-variants, CleanFull/LegacyFull); XGBoost ranker with
  target-grouped folds; `merge_oof.py` -> oof_predictions.parquet.
  `accept_incomplete` flag exists ONLY for exp08 (schema-metadata
  workaround, documented).
- **Key results** (20 seeds, macro Spearman):
  - RNA: best CleanFull_PTHS__nolj 0.550 +/- 0.021
  - PD bound: ~0.243 +/- 0.008; PD unbound: ~0.190 +/- 0.009
  - Small seed variance; clash06 rerun left RNA conclusions unchanged
    (robust negative vs rsRNASP ~0.58).
- **Status**: DONE.
- **Artifacts**: `results/tables/exp06_20seed_metrics.csv`,
  `exp06_20seed_summary.csv`, `cv_splits_*.json`,
  `feature_group_manifest_*.csv`, `hardpair_*`, `interaction_*`.
