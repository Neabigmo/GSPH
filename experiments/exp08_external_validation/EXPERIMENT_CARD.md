# EXP-08: Independent Protein-DNA External Validation

- **Purpose**: Test representation transfer on protein-DNA targets with
  NO overlap with the 47-target development set.
- **Data**: 20 nonredundant Dockground PD targets (30% sequence
  redundancy filter; all dev targets excluded) + 500 rigid-perturbation
  decoys with ligand-RMSD labels; FTDMP-compatible layout.
  `exp08_targets.csv` lists selection + provenance.
- **Method**: `scripts/make_decoys.py`, `to_ftdmp_layout.py`; feature
  extraction on HPC (10-shard array, GSPH_C_BUILD_DIR C backend);
  grouped CV via run_ablation with explicit accept_incomplete flag.
- **Key results**:
  - Extraction: ~490 rows x 3663 cols, ~0.2% NaN; all numerics valid
    (support certificates pass, persistence valid).
  - Caveat: rows were status=incomplete ONLY due to missing
    schemas/protein_dna_v6_features.json on the HPC clone (metadata
    issue, now deployed to ~/gsph_deep/schemas and repo schemas/).
  - 20-seed grouped CV: macro Spearman 0.21-0.24 (dev parity ~0.24);
    topology arms lead (A2_T 0.236, A6_TH 0.234, PTHS__nolj 0.235);
    natives within top-5 for most arms.
  - Interpretation caveat: rigid perturbation decoys are easier than
    true docking poses -- positive transfer evidence, not a hard
    docking benchmark.
- **Status**: DONE.
- **Artifacts**: `results/tables/exp08_cv_metrics.csv`,
  `exp08_targets.csv`, `README.md` (dataset provenance + caveats).
