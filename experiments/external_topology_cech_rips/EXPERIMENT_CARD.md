# Čech, α and Rips on molecular controls

普通 Čech 是零半径点支撑下共同支撑的特例。在 patch 层，τ 是加厚支撑集族的 Čech 型神经出生值，pairwise 是它的 Rips 型对应。

**控制队列，不是原 16 个结构的复现。** Original decoy paths will be added only when matching exports become available. RCSB natives are not substituted for docking/prediction decoys.

## Data and graph

- Download the public first-model PDB files for RNA controls **1A9L** and **1A60**, outside the checkout. The report records their SHA256 hashes.
- Reuse the EXP-03 molecular-control settings from `configs/experiments/exp03_filtration_comparison.yaml`: upstream event fraction 0.25, per-nucleotide quota 8, intramolecular bond exclusion 4, clash floor 0.6 times summed vdW radii. This is a chemically selected RNA contact cloud, not a protein–DNA interface.
- Existing `extract_intra` builds microcontacts, alternative-contact patches, and the directed donor/acceptor relay graph. An edge exists when acceptor-to-donor covalent distance is 1–3 bonds; its weight is that minimum distance. There is no graph/path cap. Enumerate regular paths through order 3 for H0–H2. Symmetrize these same edges for the `und` path controls.
- For point-cloud controls, use all unique heavy-atom centers in `patch.atom_ids`, sorted by parser atom ID. Full α and Rips have no point cap. For explicit Čech and exhaustive simplex validation, use the first **24** of those atoms (or all if fewer); the report includes retained atom IDs. Apply the same subset to α/Rips in subset comparisons. This is an explicitly new computational subset, not a claim about EXP-03's original decoys.

## Point-cloud layer

- α: GUDHI `AlphaComplex(precision="exact")`, square-root filtration values.
- Rips: GUDHI, maximum edge length divided by two. Include simplices through dimension 3, sufficient to compute H2.
- Čech: every simplex through dimension 3 has its exact minimum-enclosing-ball radius, via Welzl miniball in float64. Singular cases use exhaustive affine active-sphere candidates. No approximate sphere or radius jitter is used.
- Zero-radius specialization: singleton atom supports are balls with radius zero. Call the existing `_active_set_support(centers, zeros)` geometric kernel and `_support_certificate`; a point patch is this singleton support, not a molecular contact lens. Compare against independent miniball on **every edge, triangle and tetrahedron of the 24-atom subset**, even if its birth exceeds the barcode cutoff. This is the solver used inside molecular patch support; no chemical radii, contacts, or production definitions are replaced. Abort on disagreement above 1e-8 Å or a missing certificate.
- Radius cutoff **2 Å**, matching EXP-03's maximum geometric parameter numerically. Zero-radius radii and patch inflation are different geometric objects despite sharing units. Point homology uses F₂. Čech–α agreement is a barcode statement, not equality of their simplex lists.

## Patch layer and archives

- Control τ and pairwise directly use existing patch support. For higher paths, pairwise takes the maximum support over **all** two-patch subsets, not only adjacent ones; singletons/pairs retain their own τ.
- Directed `edgew` is the maximum weight on adjacent edges, singleton zero. Its parameter is **bond steps**, not Å. Comparisons with τ are explicitly numerical, not unit-calibrated geometric distances.
- Use the existing exact event-value GLMY reducer over F₁₀₀₀₀₀₃, without rounding births or event subsampling. Each rule is restricted to parameter ≤2. Persistent classes surviving the cutoff are right-censored (+∞ in the restricted filtration).
- `und` remains the symmetrized **path complex**, not a simplicial complex. The full support-family nerve has Čech-type births; pairwise is its flag/Rips-type counterpart. Restricting those births to allowed paths and taking GLMY homology is a different construction. Nonconvex patch supports also do not satisfy a general good-cover nerve theorem. Do not assert that these barcodes describe the union of support sets.
- Reuse `results/tables/filtration_compare_summary.csv` only for available archived counts. Detailed intervals come from `filtration_compare.json`, overridden by `filtration_compare_rna_clash06.json`; the former CSV contains pre-clash RNA. Select the 16 IDs in `barcode_comparison_16structure.csv` and never mix pre-clash RNA with clash06.
- Original intervals are on the shared **[0, 0.5, 2] grid**, not full-event persistence. Convert birth/death indices to this grid; censored death index 3 becomes +∞, not a fictitious 3 Å. If detailed intervals are absent, leave bottleneck/equality missing. Count-only records cannot establish equality or a bottleneck distance. Original coefficient-field/numerical details are inherited from the archive, not relabeled as a new exact-field computation.

## Comparisons and outputs

Report interval counts, GUDHI bottleneck (`e=0`), and equality at tolerance **1e-8**, separately for each structure and H0/H1/H2. Consistently discard numerical intervals of lifetime ≤1e-8 and preserve multiplicities. Equality uses perfect bipartite interval matching without the diagonal; binary search on endpoint costs gives the minimum possible maximum endpoint difference, also recorded in the CSV. This is separate from the GUDHI bottleneck, which permits diagonal matches. An infinite bottleneck can arise from different numbers of right-censored bars; it is not an implementation error.

- Full point cloud: α vs Rips.
- Matched 24-atom subset: Čech vs α, Čech vs zero-radius τ, Čech vs Rips.
- Patch controls and available archives: undirected τ vs pairwise, directed τ vs pairwise, directed τ vs edgew.
- `results/tables/cech_rips_compare.csv`: one structure/order/pair per row, with source, scope, units and missing status.
- `results/reports/cech_rips_compare.md`: grouped summaries, truncations, zero-radius errors, versions, provenance and timing. No point/patch cross-layer bottleneck is presented as a direction-only ablation.

## Reproduce

Install the repository environment plus `gudhi==3.13.0` and `miniball==1.2.0`. Runtime versions are recorded in the report. Raw coordinates, extraction caches and timing logs stay outside the checkout; output tables/reports are written to their versioned locations.

```sh
mkdir -p "$HOME/cech-rips-data"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m experiments.external_topology_cech_rips.run \
  --data-dir "$HOME/cech-rips-data" --workers 8
```

Only load caches generated by this runner with unchanged inputs and extraction settings; remove a structure's `.patch_control.json` from the external data directory to re-extract it. Cached extraction times are distinguished from the wall time of the current command.
