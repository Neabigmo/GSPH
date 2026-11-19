# Numerical audit of common support on archived molecular paths

**The checked paths satisfy the affine Jung bound.** Only 21 selected
witness paths from three structures have enough archived geometry to
compute the ratio. The 16-structure path cohort cannot exhibit a
four-patch gap: its paths contain at most four distinct patches.

## 1. Affine Jung bound and spread

For each selected path, recompute certified common support `τ` and the
maximum certified support `τ_pw` of **all** patch pairs, not only adjacent
pairs. Compute the van der Waals radius extrema on all contact endpoints
and use the maximum of the within-donor and within-acceptor diameters
across patches for `δ`. The recomputed supports agree with the archived
values within 1.1 × 10⁻¹⁴ Å.

| Structures | Paths | Minimum ratio | Q1 | Median | Q3 | Maximum ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All available witnesses | 21 | 0.3550 | 0.5136 | 0.6407 | 0.9920 | 1.1185 |
| 1FOK docking decoy | 7 | 0.4459 | 0.4949 | 0.6277 | 0.9924 | 0.9948 |
| 3dRNA-3IVN-3 | 7 | 0.5074 | 0.6102 | 0.9909 | 1.0003 | 1.0925 |
| SimRNA-3IVN-5 | 7 | 0.3550 | 0.5014 | 0.6007 | 0.9908 | 1.1185 |

No ratio exceeds the exact bound √(3/2). For `η = τ − τ_pw` versus
`δ`, pooled Pearson `r = 0.071` and Spearman `ρ = 0.254`.
These are descriptive only: the gallery selected paths partly on
`η` itself, and the two RNA decoys share a target. It does not
estimate the correlation or ratio distribution in the full cohort.
Differences within 4 × 10⁻⁶ Å count as zero for the correlations.

Five of ten paths with only single-contact patches have positive `η`.
The largest such gap is 0.4496 Å in SimRNA-3IVN-5. Alternatives are thus
not necessary for a pairwise gap; they are necessary for a gap to
four-patch support, as distinguished in [the theory](https://github.com/Neabigmo/GSPH/pull/4).

## 2. Strict four-patch excess

In the [path landscape](../tables/path_landscape.parquet), all 72,433
directed path records and all 400,579 symmetrized path records (16
structures) have order at most three. The entire patch family is
admissible in the maximum defining four-patch support, so every path has
`τ = τ^(4)` and the fraction with `τ > τ^(4)` is zero for each
graph. This is equality by definition, not an empirical Helly test.
Paths with at least five **distinct** patches are needed for that test.

## 3. Barcode differences and the RNA–protein–DNA comparison

On the three-scale grid (0, 0.5, 2 Å), the
[archived cohort summary](../tables/barcode_comparison_16structure.csv)
reports different common-support and pairwise barcodes in 7/8 RNA
structures versus 2/8 protein–DNA structures **for each graph mode**.
The audit verifies the interval multisets (including multiplicity) in
30 of the 32 structure–graph pairs. The two remaining entries are from
one 1HJC decoy and are retained as summary-only: its three-scale
intervals are absent from the committed detailed archives.

This shows more frequent differences on the coarse grid for RNA, not
larger differences in Å or an effect of `δ`. The Jung bound allows
larger differences as spread increases; it does not imply a positive
correlation. Legacy event indices do not give comparable Å endpoints
across birth rules; use the [exact-event audit](https://github.com/Neabigmo/GSPH/pull/3)
for a structure-level distance when its cohort inputs are available.

## Minimal inputs for the remaining checks

1. For the full ratio distribution and path-level correlation, export the
   certified `complex_paths` and `tau_table` records, plus **one row per
   patch** with the matching patch ID, donor/acceptor spread maximum
   `δ_P`, and contact-endpoint radius extrema. Alternatives with
   coordinates and the radii used by the solver are equivalent inputs;
   the compact patch statistics suffice. Use the same extraction for
   all three exports. Report path-level correlation by order for paths
   with at least three distinct patches; for one or two patches the
   pairwise filtration uses the full-family support.
2. For a nontrivial four-patch test, enumerate paths with at least five
   distinct patches and export certified support for the full family and
   all four-patch subsets. Do not report missing or uncertified subsets
   as equality.
3. For the structure-level correlation, join the mean of path-level
   `δ` on a fixed graph and path cohort to bottleneck distances in Å
   for the same graph, separately for H1 and H2. Compare the spread and
   distance distributions across tasks; do not treat two decoys of the
   same target or two graph modes as independent replicates.

## Reproduce

From the repository root with the package dependencies installed:

```sh
PYTHONPATH=src python -m experiments.union_support_audit
```

The command reads only the committed witness galleries, path summaries,
and discrete barcode archives. It prints per-path ratios and per-structure
barcode flags as JSON, with no data download.
