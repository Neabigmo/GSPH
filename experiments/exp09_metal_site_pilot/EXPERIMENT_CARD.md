# EXP-09: Metal and water common-witness pilot

- **Purpose:** measure pairwise-compatible but jointly incompatible ligand
  patches and calibrate CPU cost. No training or persistence computation.
- **Inputs:** seeded BioLiP nonredundant Zn entries (resolution >0 and <=2,
  at least three annotated binding residues); seeded RCSB X-ray protein
  entries <=1.5 Angstrom; seeded RNA/Mg entries <=3 Angstrom, including
  a ribosome-sized entry. Selected identifiers are recorded in CSV files.
  Initial release dates use the first mmCIF audit revision date;
  `time_split_test` is true for release on/after 2024-05-01. No fitting or
  tuning uses either side of this descriptive split.
- **Geometry:** union of alternative atoms in each ligand group; triples
  and quadruples spanning at least two residues; delta=0.3 Angstrom.
  Signed weighted support uses the repository's active-set solver and
  float64 certificate, with an exact common radius shift. An independent
  SciPy epigraph solve checks 50 seeded random cases.
- **Sampling:** enumerate pairwise graph cliques using atom KD-tree pruning.
  Maintain uniform reservoirs separately by order and classic-Zn stratum.
  Large candidate populations have sampled, not exhaustive, support solves.
  Counts, inclusion weights, enumeration censoring, timeouts and size caps
  are explicit. Population hard-negative fractions use inverse-inclusion
  weights; they are descriptive pilot estimates, not confidence intervals.
- **Labels:** first deposited model, highest-occupancy atom altloc; only
  protein or RNA polymer atoms remain as ligand inputs. All ions, waters,
  other nonpolymer molecules and hydrogens are excluded from the input.
  No biological assembly or crystal-symmetry expansion is performed.
  Water groups use polar side chains, backbone N and backbone O separately.
  Occupancy and local partner criteria follow the runner's site filters.
- **Evaluation:** geometry-only coverage at 1/2 Angstrom, nearest-witness
  localization, true-ligand-set signed eta, certificate acceptance, and
  witness-cluster count per true site. Coverage is per-site (not a precision
  claim). Connected components of the 1-Angstrom witness graph define clusters.
  Clash filtering excludes all atoms in selected ligand patches, then rejects
  any other retained polymer heavy atom within 1.8 Angstrom.
- **Limits:** compressed downloads have a cumulative byte budget, including
  partial/error responses. Each structure has a wall timeout and atom,
  patch and candidate-enumeration caps. One fresh single-thread process per
  structure makes ru_maxrss structure-specific rather than pool-lifetime RSS.
  An optional separate RNA regional calibration keeps complete patches inside
  a seeded ligand-centered sphere and a six-Angstrom heavy-atom clash shell.
  Its output must not be pooled with whole-structure hard-negative estimates.
- **Run:** install NumPy, SciPy, gemmi, requests and pytest; use
  `PYTHONPATH=src python experiments/exp09_metal_site_pilot/run.py --help`.
  Supply input/data locations as CLI arguments; data and raw JSON logs stay
  outside the repository. Checked-in outputs contain no machine paths.
- **Interpretation:** sampled-candidate recall is a lower bound on exhaustive
  geometry recall. Full-run core-hour extrapolation excludes download time
  and assumes the observed per-solve cost transfers to unsampled candidates.
  Censored structures are excluded from pooled geometric fractions, but
  retained in per-structure and timing outputs. This pilot does not implement
  all quality filters proposed for a future definitive benchmark.
