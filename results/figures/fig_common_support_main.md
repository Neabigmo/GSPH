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
above the axis and pairwise support below. Common support gives 173 bars against
55 in an RNA decoy (+3.57 Å total finite persistence) and 133 against 22 in a
protein–DNA decoy (+4.99 Å).
(d) Extra H2 bars and extra total finite persistence under common support for
each of the 16 structures (symmetrized graph). All 16 structures gain bars:
48–118 in RNA and 1–111 in protein–DNA. Circles mark the structures in (c).

**Evidence register entries.** (a) 1.6; (b) 1.5; (c, d) 1.3 and 1.4.

**Data.** Every number printed on the figure is read from the audited tables
(`union_rna_ratio_distribution.csv`, `union_audit_pd_ratio_distribution.csv`,
`union_rna_barcode_summary.csv`, `union_audit_pd_barcodes.csv`). The histogram
(`union_ratio_histogram.csv`) and the two drawn barcodes
(`union_barcode_intervals_examples.json`) come from a re-export of the 16
structures from public coordinates; its path counts, ratio quantiles, barcode
difference flags and total persistence agree with the audited tables. Bars no
longer than 1e-6 Å are numerical and are not drawn;
`union_barcode_count_by_min_length.csv` lists bar counts at minimum lengths
0, 1e-6, 1e-3 and 1e-2 Å.
