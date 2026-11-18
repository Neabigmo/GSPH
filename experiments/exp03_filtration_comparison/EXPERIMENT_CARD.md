# EXP-03: Same-Graph Filtration Comparison (tau vs pairwise)

- **Purpose**: Prove GSPH differs from ordinary pairwise filtration on
  the SAME directed graph, SAME path set, SAME GLMY boundary matrices --
  only the path birth function changes.
- **Data**: 16 deep-subset structures (8 PD complex dumps +
  8 RNA clash06 complex dumps).
- **Method**: `filtration_compare.py` (3-scale grid, V33);
  `exp03_fine.py --mode grid|eventvalue` (fine grid 9 scales;
  exact event-value scales from distinct birth values, capped at 1500
  with subsample flag). Birth rules: tau, pairwise, facetmax, edgew,
  shuffled. Certified rank-invariant persistence
  (`persistent_intervals_from_filtration`) -- never snapshot Betti.
- **Key results**:
  - V33 16-structure run: RNA 14/16 structure x graph combos have
    DIFFERENT tau-vs-pairwise barcodes (87.5%); PD 4/16 (25%); all 32
    chain maps valid, residuals ~0.
  - Fine-grid: 16/16 complete; event-value: running (RNA ~1500 event
    scales, O(S^2) rank-invariant is dominant cost -- real slowness,
    verified by CPU profiling).
- **Status**: grid DONE; event-value in progress (aggregate via
  `aggregate_exp03.py`).
- **Artifacts**: `results/reports/filtration_compare*.json`,
  `results/tables/filtration_compare_summary.csv`,
  `results/tables/barcode_comparison_16structure.csv`.

