# EXP-07: Physical Sensitivity + Patch-Representation Ablation

- **Purpose**: (a) Sensitivity of path births/barcodes to physical
  parameters (clash threshold, vdW radius scale); (b) test whether the
  union-of-lenses patch support is replaceable by simpler surrogates.
- **Data**: 16 deep-subset structures; patch dumps from
  `exp07_patch_dump.py` (atoms + patch records).
- **Method**: HPC array 144 extraction units (clash 0.0-0.7 x 16;
  radius 0.90-1.10 x 16) via `aggregate_sens.py`; representation ablation
  via `patch_repr.py` / `patch_repr_solve.py` re-solving path support
  under centroid / enclosing-ball / single-contact rules.
- **Key results**:
  - Clash threshold: flat (tau-median 1.561->1.560) -> robust.
  - Radius: monotone (tau-median 1.698->1.416; intervals 117->140) ->
    smooth continuous sensitivity.
  - Representation: alternatives shift 82-97% of births by >0.1 A;
    Spearman vs union-tau only 0.18-0.58 -> union representation is
    NOT replaceable (supports framework design).
- **Status**: DONE.
- **Artifacts**: `results/tables/sensitivity_summary.csv`,
  `sensitivity_curves.json`, `exp07_repr_summary.csv`.
