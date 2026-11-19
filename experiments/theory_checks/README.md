# Union-support numerical checks

Run from the repository root with the project's NumPy/SciPy environment:

```sh
python experiments/theory_checks/check_union_support_bounds.py
```

The fixed-seed script checks 2,000 random families (3–5 patches, independently
1–3 contacts each, independent coordinates in a 10 angstrom cube), prints a
results table and the highest-ratio random witness, and exits nonzero on any
failure. All comparisons use absolute tolerance `1e-6`. Random radii come from
`VDW_RADII` in `src/cspph/core/gsph1.py`: O=1.52, N=1.55, C=1.70, S=1.80.
These are numerical checks, not proofs or exhaustive searches for sharpness.

For random atoms we call `cspph.geometry.patch_support._atom_support`, requiring
a finite certified answer. The synthetic equal-radius examples require 1.6,
which is not in the element table; these use a thin wrapper around the same
`_active_set_support` solver in `cspph/core/gsph1.py`, with explicit radii and
the shared `_support_certificate`. Contacts at a vertex use the same atom as
both endpoints, exactly representing two coincident equal-radius balls.

Union support is the minimum over selections of one contact per patch of
the support of their combined atom IDs. Enumeration prunes only when the
maximum two-atom separation bound is no better than an already solved
selection. This matches `compute_patch_support`'s objective: that function
uses branch-and-bound and certified atom solves instead of naive enumeration.
The script also compares both implementations on the first three random
families. The API collapses repeated **patch IDs**, returns zero for an empty
family, and reports no finite certified result for an empty patch; a finite
`max_combinations` can yield an incomplete answer. Here all patches have unique
IDs, are nonempty, and comparisons use the unlimited default search.

The spread uses donor-to-donor and acceptor-to-acceptor distances within each
patch, not all cross-endpoint distances. The affine bound is
`r_min + tau <= sqrt(3/2) * (r_max + tau_pw + delta)`.
The maximum-ratio random sample is printed in full for reproducibility; it
need not approach the sharp constant. The single-contact regular tetrahedron
does attain that constant: with side `2r`, its pair support is zero and its
full support is `(sqrt(3/2)-1)r`.

For the hollow triangle, any two patches share a vertex but all three do not.
An optimal selection uses two distinct vertices, so support is `D/2-r=2.4`,
**not** `D/sqrt(3)-r`. Similarly, for the hollow tetrahedron every triple
shares a vertex; the full family needs two vertices and again has support
`2.4`. The script checks all maximal proper subfamilies in both constructions.
