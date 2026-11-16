# Paper integration notes

The JCIM manuscript (V0.4, achemso format, 27 pp) is maintained outside
this repository. This directory tracks only integration metadata:

- Manuscript package: `GSPH_JCIM_Paper_Package_V0.4` (LaTeX + PDF +
  figure sources + SI plan + SHA256 manifest).
- Required numeric corrections after the clash06 audit are frozen in
  `results/reports/manuscript_v04_corrections.md` — apply before final
  submission:
  1. Support totals: PD 12,488 + RNA 59,945 = 72,433 records;
     H0–H2 intervals 294 + 1,873 = 2,167.
  2. P0-1 filtration comparison must be reported per task:
     RNA barcode divergence 14/16 (87.5%), PD 4/16 (25%).
  3. Certified-valid supports: 72,433/72,433 (100%).
  4. Methods: intramolecular clash prior `d >= 0.6*(r_a+r_b)`.
- Figures: `results/figures/` holds the certified common-witness
  figures (PD 1FOK, RNA SimRNA-3IVN-5, 3dRNA-3IVN-3).

When the manuscript is added, place `main.tex`, `refs.bib`,
`supporting_information.tex`, and `figures/` here and add `build.sh`.
