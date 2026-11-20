# Main figure: common support versus pairwise support

Files: `fig_common_support_main.pdf` (vector), `.svg`, `.png` (600 dpi).
Regenerate with `python scripts/figure_common_support.py draw`.

**Title.** Common support sees what pairwise support cannot, and never by more than √(3/2).

**Caption.**
(a) Three supports whose pairs all intersect at scale τ_pw share no point until
the larger scale τ. Pairwise support declares the path present at τ_pw; common
support places its birth at τ, where a single witness point (star) lies in all
three. The difference η = τ − τ_pw is the gap.
(b) Support ratio (r_min + τ)/(r_max + τ_pw + δ) for all 473,012 certified paths
of 16 structures (RNA 340,874; protein–DNA 132,138; directed and symmetrized
graphs, orders 0–3). Paths right of 1 need common support. The affine Jung
bound √(3/2) ≈ 1.225 holds for every path: the largest ratio is 1.141 and no
path enters the hatched region.
(c) H2 barcodes in Å on identical path sets (symmetrized graph), common support
above the axis and pairwise support below; bars longer than 0.01 Å are drawn and
counted. Common support gives 111 bars against 54 in an RNA decoy (+3.57 Å total
finite persistence) and 77 against 22 in a protein–DNA decoy (+4.99 Å).
(d) Extra H2 bars (longer than 0.01 Å) and extra total finite persistence under
common support for each of the 16 structures (symmetrized graph). All 16
structures gain bars: 28–62 in RNA and 1–55 in protein–DNA. Circles mark the
structures in (c). On the directed graph the barcodes differ in 8 of 8 RNA
structures and 5 of 8 protein–DNA structures (H1 5 of 8, H2 3 of 8); the
supplementary figure shows the directed-graph panels.

**Bar-count convention.** Bar counts on both figures use bars longer than
0.01 Å; total finite persistence uses all bars. Counts at minimum lengths 0,
1e-6, 1e-3 and 1e-2 Å are in `union_barcode_count_by_min_length.csv`.

# Supplementary figure: directed graph

Files: `fig_common_support_directed.pdf`, `.svg`, `.png`, drawn by the same command.

**Title.** Directed graph: common support changes H1 in 8 of 8 RNA and 5 of 8
protein–DNA structures.

**Caption.**
(a) H1 barcodes in Å on the directed graph for the two structures of the main
figure, common support above the axis and pairwise support below; bars longer
than 0.01 Å. Bars that reach the right edge are infinite.
(b, c) H1 and H2 bar counts (longer than 0.01 Å) per structure under pairwise
support (grey) and common support (colour). RNA barcodes differ in 8 of 8
structures in both H1 and H2. Protein–DNA barcodes differ in 5 of 8 structures
in H1 and 3 of 8 in H2; the remaining directed protein–DNA barcodes are
identical under the two supports, and all 8 differ on the symmetrized graph.

**Evidence register entries.** (a) 1.6; (b) 1.5; (c, d) 1.3 and 1.4.

**Data.** Path totals, the largest ratio, the number of bound violations, the
"barcodes differ" flags and total finite persistence are read from the audited
tables (`union_rna_ratio_distribution.csv`, `union_audit_pd_ratio_distribution.csv`,
`union_rna_barcode_summary.csv`, `union_audit_pd_barcodes.csv`). The histogram
(`union_ratio_histogram.csv`), the drawn barcodes
(`union_barcode_intervals_examples.json`) and the bar counts above 0.01 Å
(`union_barcode_count_by_min_length.csv`) come from a re-export of the 16
structures from public coordinates; its path counts, ratio quantiles, barcode
difference flags and total persistence agree with the audited tables. Bars no
longer than 1e-6 Å are numerical.
