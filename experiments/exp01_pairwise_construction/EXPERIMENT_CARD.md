# EXP-01: Pairwise vs Higher-order Support Construction

- **Purpose**: Demonstrate that identical pairwise minima can correspond
  to different higher-order common supports (the motivating construction).
- **Data**: Synthetic idealized geometry constructions + certified
  random triplets.
- **Method**: `geometry_props.py`; monotonicity, rigid-motion invariance,
  pairwise-same / triple-different exact constructions.
- **Key results**: 60/60 monotonicity tests; 40 rigid-transform tests;
  300 random triple-support constructions; exact equality in idealized
  geometry vs approximate matching in atom-lens numerics (documented
  distinction).
- **Status**: DONE (V31/V33 frozen results).
- **Artifacts**: `results/tables/geometry_props.json`,
  `results/reports/p0_audits.json`.
