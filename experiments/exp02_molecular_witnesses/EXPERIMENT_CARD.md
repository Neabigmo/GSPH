# EXP-02: Molecular Common-Witness Visualization + Residual Audit

- **Purpose**: Show the common-support witness is a real geometric object
  on true molecular structures, and verify union-lens tau semantics.
- **Data**: 16 deep-subset structures (8 PD + 8 RNA clash06).
- **Method**: `witness_dump.py` (top-gap selection), `witness_dump_
  stratified.py` (gap quartile / tau median / eta3 / chemistry-type
  strata), `render_witness.py` + `render_witness.pml` (PyMOL),
  `postprocess_residuals.py` (per-contact residual decomposition).
- **Key results**:
  - 18/18 primary witness points: per-contact residuals reproduce tau
    exactly; every selected microcontact contributes BOTH donor and
    acceptor atom balls as constraints (intersection lens).
  - Stratified extension: 16/16 structures, 105/105 certified examples,
    tau reproduced to max residual 6e-15.
  - 3 certified witness figures rendered (1FOK_ftd158 PD;
    SimRNA-3IVN-5, 3dRNA-3IVN-3 RNA); all certificate_passed=true.
- **Status**: DONE.
- **Artifacts**: `results/tables/witness_contact_residuals.csv`,
  `witness_strat_residuals.csv`, `results/witness/fig_witness_*.{png,pdf,json}`.
