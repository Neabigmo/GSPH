# Mathematical Specification (frozen V33)

## Objects

- **Microcontact** `c = (donor_atom, acceptor_atom, distance, tau_c)` —
  a chemically directed atom-pair interaction with endpoint-ball clearance.
- **Interaction patch** `P` — a set of alternative microcontacts sharing
  donor/acceptor site identity (union-valued: any realization may satisfy
  the patch).
- **Common support** of patch sets `P_1..P_k`:
  `tau(P_1..P_k) = min_x max_i min_{c in P_i} clearance(x, c)` solved
  exactly by enumerated affine quadratics (<=4 active balls in 3D) with a
  numerical certificate (`certificate_passed`).
- **Pairwise bound** `tau_pw(pi) = max_{i<j} tau(P_i, P_j)` — strictly a
  lower bound on `tau_GSPH(pi)`; the gap `eta_k = tau - tau_pw` is the
  higher-order support increment.
- **GLMY path complex** `Omega_0..Omega_3` over the directed patch graph;
  admissible chains only (coefficient field `F_1000003`, rank by modular
  prime consensus).
- **Birth rules** compared on identical complexes: `cs`, `pairwise`,
  `facetmax`, `edge` (bottleneck min-contact distance), `shuffled`.
- **Persistence** = birth/death intervals per homology dimension over the
  discrete scale grid `[0, 0.5, 2.0]` Å (NOT continuous persistence).
- **Shared Hodge** readout `G_k = B_k^T B_k + 1e-10 I`; PSD and
  harmonic-nullity=Betti audited separately.

## Key theorems (see manuscript V0.4)

1. Pairwise compatibility does not determine common support
   (explicit counterexample constructions, EXP-01).
2. `tau` is 1-Lipschitz in per-atom bounded coordinate perturbations
   (EXP-04, 1,600 tests / 0 violations).
3. Rigid-motion invariance to ~1e-13 Å.
4. Helly-type feasibility certificate for <=4 active constraints in 3D.
5. Selection–Helly formula, union counterexample, affine Jung bound and
   pairwise interleaving: see `docs/THEORY_UNION_SUPPORT.md`. "Determined by
   at most four patches" holds only for single-contact patches.

## Numerical conventions

- vdW radii table in `cspph.core.gsph1.VDW_RADII`.
- Intramolecular clash filter: exclude `d < 0.6*(r_a+r_b)` (RNA/protein
  intra only; never intermolecular docking).
- `valid=false` / `incomplete=true` records are stored explicitly,
  never zero-filled.
