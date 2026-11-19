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

## Frozen hard-pair test

`hardpair_frozen.py --config configs/experiments/exp06_hardpair_frozen.yaml
--feat-dir <features> --out <outputs>` compares nested physical/pairwise,
support-difference, and barcode tiers on unseen targets. Required inputs are
real FTDMP feature parquets and successful audit JSONL records, in
`<features>/pd_bound` and `<features>/pd_unbound` (a single-task config accepts
a flat directory). Optional `--barcode-table` accepts CSV/parquet keyed uniquely
by `structure_id`; unmatched candidates retain NaNs. These real inputs are not
distributed here. Pair indices are frozen once per target/seed from labels and
Tier-1 OOF scores and shared by all tiers. Five grouped folds and ten seeds,
train-fold median imputation, target bootstrap, two-sided sign flips, and Holm
over the two adjacent primary contrasts are fixed in YAML; the nonadjacent
contrast is secondary. Per-seed and target-averaged seed-pooled summaries include
macro Spearman (all evaluable targets), while hard-pair inference excludes
targets without enough pairs. Every output records the config SHA256; retain
the exact config used. Synthetic tests override rounds/seeds only for speed and
are not evidence of real-data performance.

Compact exports may instead be supplied with `--v2-dir`: the three per-task
feature tables are joined by candidate identity, with labels restored from the
supplied OOF `y_true` column, never its predictions. Task-specific manifest
validation and tier nonemptiness checks apply; no tier regex or statistical rule
changed. See `docs/HPC_EVALUATION.md` for required sidecars and offline CPU jobs.

### Frozen hard-pair result (v2 per-candidate features, 47 targets, seeds 0–9)

Pooled hard-pair win-rate differences (`results/tables/hardpair_frozen_summary_pooled.csv`; per seed and per target in the sibling files):

| Task | Contrast | Δ win rate [95% CI] | Holm p |
|---|---|---|---|
| pd_bound | T2 − T1 (+ support difference) | +0.0006 [−0.0054, 0.0064] | 0.84 |
| pd_bound | T3 − T2 (+ barcode) | +0.0269 [0.0129, 0.0399] | 0.0006 |
| pd_unbound | T2 − T1 | −0.0059 [−0.0107, −0.0014] | 0.025 |
| pd_unbound | T3 − T2 | −0.0022 [−0.0086, 0.0043] | 0.52 |

Candidate pool: `results/tables/ftdmp_candidates.csv`; VoroMQA per-candidate scores on the same pool: `results/tables/pd_voromqa_scores.jsonl`.
