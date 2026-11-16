# GSPH JCIM Manuscript V0.4 — Required Corrections (clash06 audit)

All numbers below are measured from the clash06 rerun artifacts
(`results/deep_subset_rna_clash06/`, `deep_subset_rna_complex_clash06/`,
`deep_subset_pd_complex/`, `results/ablation/filtration_compare_rna_clash06.json`,
`_feat/rna_clash06/`). The V0.4 manuscript cites pre-clash V31 numbers in
three places that are now superseded.

## 1. Support / Barcode totals table (V31 deep-subset table)

| Scenario | Support records | H0–H2 intervals | V0.4 value | Corrected |
|---|---|---|---|---|
| Protein–DNA | 12,488 | 294 | same | unchanged |
| RNA | **59,945** | **1,873** | 59,496 / 1,870 | **update** |
| Total | **72,433** | **2,167** | 71,984 / 2,164 | **update** |

RNA changes are confined to two SimRNA decoys affected by the steric-clash
filter (SimRNA-3IVN-5: 6,914→7,267 paths; SimRNA-5SWD-7: 8,962→9,058).
3dRNA / FARFAR2 structures are bit-identical. 16 structures / 8 targets
unchanged.

## 2. P0-1 same-graph / same-path filtration comparison — CONCLUSION REVERSED

If V0.4 reports the old result ("86/90 barcode pairs identical, ~96%
agreement"), it must be rewritten. That number was a false negative caused
by steric-clash contacts (tau ≈ 0 always-active edges compressing birth
differences).

Corrected clash06 result (`filtration_compare_rna_clash06.json`, 8 RNA
structures × dir/und graphs):

- tau-filtration vs pairwise-filtration barcodes **identical in only
  2/16 structure×graph combinations (12.5%)**; **14/16 differ (87.5%)**.
- Birth-value level: tau > max(pairwise tau) on 27–41% of directed paths
  (per-structure range; e.g. 3dRNA-3IVN-3 ≈34% dir / 36% und positive
  deltas).
- Example: SimRNA-3IVN-5 und complex — tau filtration 136 intervals
  (H1=5, H2=6) vs pairwise 129 (H1=2, H2=2).
- True edge-weight bottleneck control (T6 fix): discordant fraction
  5.4% over 2,700 pair comparisons (was ~1% under the old diameter
  surrogate); shuffled-order control breaks ordering in 100% of trials.
- Chain-map commutation residuals ≤ ~1e-12 on every complex.

Suggested wording: "On clash-filtered structures, the common-support
filtration produces persistence barcodes that differ from the
pairwise-support filtration in 87.5% of structure–graph combinations,
demonstrating that higher-order common support carries information not
recoverable from pairwise compatibility alone."

Do NOT claim universality: 2/16 combinations remained identical, and the
comparison uses three discrete scales [0, 0.5, 2].

## 3. Certified-valid support statistics (fills JCIM_READINESS item)

- RNA clash06: 59,945 / 59,945 support records certified valid (100%),
  all orders k0–k3; zero invalid/incomplete after Hodge retry
  (H3_HODGE_DENSE_MAX=1024).
- Protein–DNA: 12,488 / 12,488 certified valid (100%), all orders k0–k3.
- Combined: 72,433 / 72,433 valid.

## 4. RNA predictive result (unchanged — robust negative)

Clash filtering did not alter RNA quality-assessment conclusions:
best arm |rho| 0.546→0.551; CleanFull |rho| 0.561→0.535; hard-pair
CleanFull−A1 +0.0246 (unchanged); B9−B2 −0.0094 (unchanged);
I_TH interaction non-significant. GSPH still does not exceed rsRNASP
(~0.58). Report as cross-object validation without predictive gain.

## 5. Methods addition

Add steric-clash prior to contact generation: intramolecular contacts
require d >= 0.6 * (r_vdW,donor + r_vdW,acceptor); applied to RNA and
protein intramolecular paths, NOT to protein–DNA intermolecular docking
paths; recorded as `clash_filtered_count` in audit metadata;
GSPH_H3_CLASH_RATIO=0 disables. 287/826 RNA structures had >=1 filtered
contact (SimRNA 151 > FARFAR2 46 ≈ RNAComposer 43 ≈ 3dRNA 43 >> native).

## 6. Full 16-structure filtration comparison — FINAL (job 49206184)

All 16 deep-subset structures (8 PD incl. previously missing
1HJC_ftd142 + 8 RNA clash06) compared on HPC; per-structure JSONs in
`results/cmp_out_16/`. Barcode identity = exact multiset equality of
(birth,death) intervals per dimension.

| Task | graph-combos | barcode-identical | barcode-different |
|---|---|---|---|
| RNA (8 structures) | 16 | 2 (12.5%) | **14 (87.5%)** |
| PD (8 structures) | 16 | 12 (75%) | 4 (25%) |
| **All** | **32** | **14 (43.75%)** | **18 (56.25%)** |

Honest reading: the barcode-level divergence is strong and consistent for
RNA (87.5%), present but weaker for PD (25%) — consistent with PD paths
being shorter/sparser so pairwise bounds coincide more often. Birth-value
analysis (tau > max pairwise tau on a large fraction of paths) applies to
both tasks. Do not state a single pooled percentage without the per-task
breakdown.

Chain-map validity: all 32 complexes `chain_map_valid=true`,
residuals ~0.

## 7. Common-witness visualization (P1) — DONE

- `ablation/witness_dump.py` + `ablation/render_witness.py`.
- Three certified examples rendered (PNG 300dpi + PDF + witness JSON):
  - `results/witness/fig_witness_1FOK_ftd158.*` (PD; 201 candidate paths,
    gaps 0.34–0.38 Å on 4-patch paths)
  - `results/witness/fig_witness_SimRNA3IVN5.*` (RNA clash06; 4,192
    candidates, gaps up to ~0.45 Å)
  - `results/witness/fig_witness_3dRNA3IVN3.*` (RNA clean; 3,818
    candidates)
- Every rendered witness carries `certificate_passed=true` — the marked
  point is a certified common-support center, not an energy minimum.
