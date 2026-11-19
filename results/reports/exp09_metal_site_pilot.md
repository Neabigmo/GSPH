# EXP-09: common-witness metal/water pilot

## Scope and execution

No training and no persistence computation. Seed 0; delta 0.3 Angstrom;
eight logical CPUs, eight independent single-thread workers. Each structure
uses a fresh process: peak RSS is Linux `resource.getrusage` KiB, not a
long-lived pool's cumulative high-water mark. Wall times below exclude Python
imports/process startup; `process_wall_seconds` includes startup where logged.

Selected 300 Zn entries from 1,321 eligible BioLiP nonredundant entries,
300 water entries from 27,776 eligible RCSB X-ray protein entries, and 20
RNA/Mg entries from 1,287 eligible entries. All 620 compressed mmCIF downloads
succeeded: **156,617,514 bytes**. Cached RCSB search JSON occupies 1,378,639
additional bytes. No structures or raw logs are committed.

Main runs: **300/300 Zn**, **300/300 water**, **16/20 RNA/Mg** complete.
Four large RNA entries hit the explicit size cap (250,000 retained heavy
atoms or 50,000 patches): 4V9P, 5FDU, 6XQE, 9DFD. No wall timeout (120 s),
enumeration cap (2,000,000 candidates), or solver failure occurred.

### Important sampling limitation

Pairwise-passing triples and quadruples were enumerated exhaustively in every
complete structure. Support solves use a uniform reservoir of at most 100
candidates **per structure/order/classic-ligand stratum**. Consequently:

- Main hard fractions below are inverse-inclusion-weighted estimates, not
  exhaustive support counts. Classic Zn fractions are exact: all 2,894
  classic triples and 537 classic quadruples were solved.
- Candidate sampling occurred in 261 Zn, all 300 water, and all 16 complete
  RNA/Mg structures. No whole water structures or regions were subsampled.
- Recall is the **observed sampled-candidate geometry coverage**, a lower
  bound on exhaustive coverage. Particularly low water coverage is not an
  estimate of exhaustive water recall. It cannot support comparison with
  trained site predictors. Clusters/site likewise describes observed sampled
  witnesses only; no extrapolation of cluster counts is attempted.

## Hard negatives among pairwise-passing candidates

Hard means tau > 0.3 despite every pair passing. The clash column recomputes
the fraction after rejecting witnesses within 1.8 Angstrom of any retained
non-ligand heavy atom. All atoms in selected ligand patches are exempted.

| Cohort/stratum | k | Pairwise-passing | Solved sample | Hard fraction | After clash |
|---|---:|---:|---:|---:|---:|
| Zn, all defined patches | 3 | 102,923 | 31,752 | 51.80% | 26.22% |
| Zn, all defined patches | 4 | 5,441 | 5,352 | 70.99% | 48.93% |
| Zn, classic side chains only | 3 | 2,894 | 2,894 | 8.60% | 7.01% |
| Zn, classic side chains only | 4 | 537 | 537 | 11.92% | 9.92% |
| Water | 3 | 4,604,222 | 30,000 | 7.65% | 6.47% |
| Water | 4 | 6,392,943 | 30,000 | 17.73% | 16.06% |
| RNA/Mg, complete structures | 3 | 517,712 | 1,600 | 17.52% | 7.46% |
| RNA/Mg, complete structures | 4 | 505,913 | 1,600 | 44.34% | 33.97% |

Classic means His/Cys/Asp/Glu **side-chain** patches, not backbone O patches
from those residues. Unions retain alternative atoms; a single atom is chosen
from each patch, never all alternatives simultaneously.

### Initial-release time split

`time_split_test=True` means initial PDB release **on/after 2024-05-01**.
Dates come from the first mmCIF audit revision date; all 620 dates were found.
Identifier CSVs contain both the date and boolean. No fitting uses this split.
The summary CSV also contains classic-stratum and clash-filtered split rows.

| Cohort | Test flag | Structures n | k=3 passing n | k=3 hard | k=4 passing n | k=4 hard |
|---|---|---:|---:|---:|---:|---:|
| Zn, all patches | False | 262 | 87,508 | 51.27% | 4,557 | 69.52% |
| Zn, all patches | True | 38 | 15,415 | 54.84% | 884 | 78.58% |
| Water | False | 240 | 3,701,515 | 7.55% | 5,151,325 | 17.56% |
| Water | True | 60 | 902,707 | 8.08% | 1,241,618 | 18.42% |

Classic Zn pre-cutoff: 242/2,604 = 9.29% for triples, 63/484 = 13.02%
for quadruples. Post-cutoff: 7/290 = 2.41%, 1/53 = 1.89%; these are small
candidate counts and do not justify a temporal generalization claim.

## Site coverage, localization and clutter

True sites follow occupancy and local-partner criteria; first model and
highest-occupancy atom altloc only. Crystal symmetry is not expanded. All
ions/waters are removed before constructing ligand inputs. Other retained
heavy atoms contribute to clash checks. Zn and water patches use protein;
Mg patches use RNA, not DNA. Site coverage pools sampled k=3 and k=4 witnesses.

| Metric | Zn | Water |
|---|---:|---:|
| True sites | 640 | 9,441 |
| Recovered within 1 Angstrom | 574/640 = 89.69% | 125/9,441 = 1.32% |
| Recovered within 2 Angstrom | 631/640 = 98.59% | 622/9,441 = 6.59% |
| Clash-filtered coverage, 1 / 2 Angstrom | 89.38% / 98.28% | 1.32% / 5.84% |
| Median nearest-witness error, all sites with predictions | 0.142 A | 4.473 A |
| Median error among 1-Angstrom recoveries | 0.115 A | 0.667 A |
| True-ligand-set eta > 1e-6 Angstrom | 627/640 = 97.97% | 7,178/9,441 = 76.03% |
| Observed witness clusters / true site | 23.25 | 4.94 |
| Same, after clash filter | 16.36 | 1.24 |

Clusters are connected components of the 1-Angstrom witness-neighbor graph.
Coverage is per-site existence, not one-to-one matched precision. Full
benchmark quality filters (symmetry-contact exclusion, B-factor validation,
sequence deduplication, curated ion validation) are not implemented here.

## Solver verification

The adapter calls `cspph.core.gsph1._active_set_support` and
`_support_certificate`, the repository's weighted 1-center reference backend.
It subtracts the smallest ligand radius before solving, then subtracts that
same constant from the returned tau. One shifted radius is zero, making the
backend's nonnegative-tau constraint inactive as a restriction. This preserves
the signed minimax optimum and its point, unlike simply clipping negative tau.
All microcontact selections in each sampled union are solved and certified.

**50 random 3/4-atom weighted cases vs independent SciPy SLSQP epigraph:**
max absolute tau difference **4.5365933232233147e-11 Angstrom**; certificate
acceptance **50/50**. Main complete structures: **169,619 weighted solves,
zero certificate failures (100%)**. No C++ or GPU acceleration was used.
This calibrates the portable reference pipeline, not an accelerated deployment.

Tests: pilot tests **10 passed in 0.32s**; the existing geometry tests were
also exercised with the earlier pilot tests (**13 passed in 1.02s**, including
all seven existing patch-support/property tests).

## Runtime and memory calibration

| Cohort | Structure wall median / p90 / max, seconds | Candidate mean, ms | Candidate median / p90 / max, ms | Peak RSS max, KiB |
|---|---|---:|---|---:|
| Zn, n=300 | 0.886 / 2.032 / 7.218 | 8.707 | 4.588 / 16.563 / 219.260 | 262,020 |
| Water, n=300 | 1.875 / 3.557 / 6.728 | 7.982 | 6.056 / 13.582 / 221.045 | 148,028 |
| RNA/Mg, n=20 including censored | 1.260 / 6.197 / 8.365 | 5.508 | 5.773 / 6.677 / 40.091 | 790,628 |

Candidate timings include alternative selections, certification and atom-set
cache lookup, but exclude clique enumeration, clash testing and truth metrics.
Individual weighted-solve means: Zn 5.049 ms, water 5.858 ms, Mg 5.040 ms.
The 16 complete Mg structures alone have wall median/p90/max
1.231 / 3.787 / 6.266 seconds. Whole-process wall medians, including startup,
were 1.195 / 2.167 / 1.586 seconds for Zn/water/Mg respectively (one initial
check per cohort lacks a parent-process timer).

Measured worker CPU totals: Zn **419.624 s**, water **752.277 s**, RNA/Mg
**53.863 s**; main runs total **0.3405 core-hours**, excluding downloads.
These are the *sampled* pilot costs, not full-exhaustive candidate costs.

Multiplying each structure/order/stratum's measured mean candidate time by
its exact passing count gives candidate-solving-only projections:

| Cohort | Complete pilot cohort, core-hours | Per 1,000 similar structures, core-hours |
|---|---:|---:|
| Zn, 300 structures | 0.213 | 0.710 |
| Water, 300 structures | 25.093 | 83.642 |
| RNA/Mg, 16 noncensored structures | 1.542 | 96.369 |

These are rough calibration numbers, not ETAs: exclude parsing/graph/truth
overhead, do not model changing cache hit rates or CPU contention, and omit
censored ribosomes. Do not extrapolate sampled wall medians directly to
exhaustive water extraction. All exact counts and raw timing sums are in CSV.

### Large-ribosome check (separate regional calibration)

5FDU is a 2.9-Angstrom Thermus thermophilus 70S ribosome. Its full input has
**286,517 retained atoms / 89,201 patches** and was size-censored in the main
cohort. An additional seeded, truth-independent 15-Angstrom ligand-centered
region retained 1,295 atoms / 175 patches, including a clash-check shell.

- 1,251 passing triples, 100 sampled: **24% hard**, **9.375% after clash**.
- 1,181 passing quadruples, 100 sampled: **51% hard**, **35.714% after clash**.
- End-to-end regional wall **5.376 s**, including full-structure parsing and
  cropping **4.239 s**; peak RSS **492,564 KiB**; 217 certified weighted solves.
- Not pooled with whole-structure fractions; it does **not** measure full
  ribosome candidate-enumeration time.

## Interpretation and artifacts

The broad Zn patch universe exceeds the design's 20% hard-negative threshold,
even after clash filtering. **Classic Zn alone does not**: 8.6% / 11.9%
before clash. Water also remains below 20% at k=3/4 here. Thus the broad Zn
result must not be advertised as proof of necessity for classic Zn chemistry,
and the sampled water recall cannot decide localization performance.

Artifacts: `exp09_metal_site_pilot_summary.csv` (including temporal strata),
`exp09_metal_site_pilot_per_structure.csv` (sizes, timing, RSS, censoring,
release date), `exp09_metal_site_pilot_candidate_strata.csv`,
`exp09_metal_site_pilot_sites.csv`, `exp09_metal_site_pilot_regional.csv`, and
`exp09_metal_site_pilot_measurements.json` under the standard results folders.
The experiment card, config, runner, tests and three identifier lists live in
`experiments/exp09_metal_site_pilot/`.
