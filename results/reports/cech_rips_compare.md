# Čech, α and Rips controls

普通 Čech 是零半径点支撑下共同支撑的特例。在 patch 层，τ 是加厚支撑集族的 Čech 型神经出生值，pairwise 是它的 Rips 型对应。

**控制队列，不是原 16 个结构的复现。** 1A9L、1A60 为 RCSB 第一模型；原队列仅复用归档，不以 native 替代 decoy。

## Summary

Counts pool H0–H2 per structure. Distances are GUDHI bottleneck distances on the stated parameter; infinity is retained.

| Cohort | Layer | Comparison | Identical / available | Missing | Median | Maximum |
|---|---|---|---:|---:|---:|---:|
| RNA_clash06_archive | patch_legacy_grid | dir_tau vs dir_edgew | 0/24 | 0 | inf | inf |
| RNA_clash06_archive | patch_legacy_grid | dir_tau vs dir_pairwise | 14/24 | 0 | 0 | inf |
| RNA_clash06_archive | patch_legacy_grid | und_tau vs und_pairwise | 12/24 | 0 | 0.375 | inf |
| RNA_control | patch_events | dir_tau vs dir_edgew | 0/6 | 0 | inf | inf |
| RNA_control | patch_events | dir_tau vs dir_pairwise | 2/6 | 0 | 0.025488504 | 0.112355116 |
| RNA_control | patch_events | und_tau vs und_pairwise | 2/6 | 0 | 0.0931584289 | inf |
| RNA_control | points_cech_subset | cech vs alpha | 6/6 | 0 | 3.33066907e-16 | 4.4408921e-16 |
| RNA_control | points_cech_subset | cech vs rips | 4/6 | 0 | 3.33066907e-16 | 0.0762931206 |
| RNA_control | points_cech_subset | cech vs zero_tau | 6/6 | 0 | 9.99200722e-16 | 1.33226763e-15 |
| RNA_control | points_full | alpha vs rips | 2/6 | 0 | inf | inf |
| protein_DNA_archive | patch_legacy_grid | dir_tau vs dir_edgew | 7/21 | 3 | inf | inf |
| protein_DNA_archive | patch_legacy_grid | dir_tau vs dir_pairwise | 20/21 | 3 | 0 | inf |
| protein_DNA_archive | patch_legacy_grid | und_tau vs und_pairwise | 19/21 | 3 | 0 | inf |

## Scope and interpretation

- Point-cloud scales are radii in Å, capped at 2 Å; H0–H2 uses simplices through dimension 3. Full α and Rips use all unique heavy-atom centers appearing in retained patches.
- Exact Čech and zero-radius checks use the first 24 atoms in ascending parser atom-ID order, or the entire cloud if smaller. This additional computational truncation is not an EXP-03 truncation. All edges, triangles and tetrahedra of this subset are checked, including births above 2 Å; only barcode construction is capped at 2 Å. Subset α/Rips use exactly these same atoms.
- α is GUDHI exact-precision Delaunay α with sqrt of squared filtration. Čech uses independent Welzl miniball (float64; exhaustive active-sphere fallback for singular simplices), not an approximate enclosing ball. Zero-radius τ calls the existing weighted active-set support solver with radii zero and checks its certificate; neither vdW radii nor patch definitions are edited.
- Point simplicial homology uses F_2; the existing exact GLMY reducer uses F_1000003. No cross-layer equality is inferred across these coefficient fields. Numerical intervals of lifetime ≤1e-8 are discarded consistently. Identical means a multiplicity-preserving perfect interval matching with endpoint error ≤1e-8, without diagonal matches. This is checked independently of the GUDHI bottleneck (which permits the diagonal).
- At the 2-unit cutoff, surviving classes have death +∞ in the restricted filtration. These are right-censored, not evidence of essential classes in the unbounded complex. Unequal surviving multiplicities can therefore give infinite bottleneck distance.
- The frozen grid is [0, 0.5, 2]. Archived indices are mapped back to this shared grid; death index 3 is right-censored, never interpreted as a measured death at 3 Å. No full-event distances are inferred from the grid.
- The summary CSV contains pre-clash RNA and omits arms; use its counts only when detailed intervals are absent. Matching detailed RNA clash06 JSON overrides pre-clash records. Missing detailed intervals stay missing; counts alone do not establish equal barcodes.
- `und_*` archives and controls use the symmetrized graph's regular GLMY path complex, not simplicial homology of the support-family nerve. Their birth rules are Čech-/Rips-type, but these persistence objects must not be identified with ordinary point-cloud Čech/Rips.
- `edgew` is max adjacent edge weight (Chowdhury–Mémoli-type path filtration). Here weights are minimum covalent bond-transfer distances in bond steps, not Å. τ/edgew distances and equality flags are numerical parameter controls, not geometrically calibrated Å distances.
- Point-cloud and patch vertices differ. No direct bottleneck between their diagrams is claimed to isolate direction alone. These controls establish specialization and filtration differences, not predictive benefit or prevalence in the original 16 decoys.

## Control audit

| Structure | Full atoms | Čech atoms | Patches | Directed paths | Symmetrized paths | Extraction s | Points s | Patch persistence s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1A9L | 144 | 24 | 104 | 1578 | 6308 | 97.508 | 14.454 | 0.290 |
| 1A60 | 157 | 24 | 125 | 2346 | 10951 | 151.273 | 13.621 | 1.189 |

- 1A9L PDB SHA256: `68c1576ad8162edf4d7f51400884baa7a2e78f274d6e2abc7e39db85e8d49ff4`.
- 1A9L retained Čech atom IDs: `[7, 14, 15, 17, 18, 37, 38, 40, 60, 61, 63, 76, 80, 81, 83, 103, 104, 106, 112, 117, 126, 127, 145, 146]`.
- 1A9L zero-radius checks by simplex dimension: `{"1": {"max_abs_error_A": 7.105427357601002e-15, "n_simplices": 276}, "2": {"max_abs_error_A": 8.881784197001252e-15, "n_simplices": 2024}, "3": {"max_abs_error_A": 8.881784197001252e-15, "n_simplices": 10626}}`.


- 1A60 PDB SHA256: `376f697769dd61ba07c734bdecc296f2ac5712047d80182a8ad00ea803780d02`.
- 1A60 retained Čech atom IDs: `[14, 15, 17, 18, 37, 38, 40, 60, 61, 63, 69, 80, 83, 84, 105, 106, 108, 125, 126, 128, 141, 145, 146, 157]`.
- 1A60 zero-radius checks by simplex dimension: `{"1": {"max_abs_error_A": 8.881784197001252e-15, "n_simplices": 276}, "2": {"max_abs_error_A": 1.0658141036401503e-14, "n_simplices": 2024}, "3": {"max_abs_error_A": 1.0658141036401503e-14, "n_simplices": 10626}}`.


Total command wall time: 278.545 s (cached extraction times are listed separately).

Software: `{"biopython": "1.85", "gudhi": "3.13.0", "miniball": "1.2.0", "numpy": "2.2.6", "pandas": "2.3.3", "scipy": "1.15.3"}`.

Sources: `experiments/exp03_filtration_comparison/EXPERIMENT_CARD.md`; `configs/experiments/exp03_filtration_comparison.yaml`; `results/tables/{filtration_compare_summary.csv,filtration_compare.json,filtration_compare_rna_clash06.json,barcode_comparison_16structure.csv}`.

Run instructions and mathematical qualifications: [experiment card](../../experiments/external_topology_cech_rips/EXPERIMENT_CARD.md).
