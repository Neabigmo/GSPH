# EXP-03: Same-Graph Filtration Comparison (tau vs pairwise)

- **Purpose**: Test whether common support and pairwise support give
  different barcodes on the same directed graph and regular path set.
- **Data**: The audit cohort has eight protein–DNA and eight RNA
  structures. Reuse the certified complex-path and support dumps; the
  archived barcode JSON alone does not contain the full scale vectors.

## Full birth-value comparison

```sh
python -m experiments.exp03_filtration_comparison.compare_exact \
  --dir dumps --sid STRUCTURE_ID --graph dir --out barcodes.json
```

The runner requires both `<sid>.complex_paths.parquet` and
`<sid>.tau_table.parquet` from the same graph extraction. Export every
path of orders 0–3 and every pair of patches in those paths, including
nonconsecutive pairs. The support table must contain only certified
values. A missing pair support or an uncertified path is an error, not
a reason to compare different path sets. The two rules
share the union of all supplied geometric birth values. GLMY linear
algebra is exact over `F_1000003`; the geometric values keep their
original numerical precision. The output lists all half-open H0–H2
intervals in Å, keeping duplicates and using JSON `null` for infinity.

The finite-field reducer obtains a filtration-adapted basis in each
degree as the kernel of `[∂ | -I]`, admitting each slack column when
its corresponding face becomes allowed. Projecting out the slack
coordinates identifies this kernel with the GLMY chain space on every
scale, and the projection commutes with inclusions. Ordinary filtered
boundary reduction then pairs the generators without rebuilding a
complex at every scale or converting a floating-point basis to integers.

## Cloud molecular controls

Use the first model of the public RCSB RNA entries 1A9L and 1A60. The
same directed graph is used for both rules per model. To keep extraction
small, target 25% of upstream contact events with a per-nucleotide
quota of eight. There is no path or scale cap on
the resulting graph. These are molecular controls, not members of the
16-structure audit cohort. Coordinate parsing needs Biopython.

| RNA model | Paths | Union of Å birth values | H1 bars, common / pairwise | H2 bars, common / pairwise | Export wall time / peak RSS | Reduction time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1A9L | 1,578 | 644 | 58 / 55 | 5 / 5 | 90 s / 214 MiB | 0.025 s |
| 1A60 | 2,346 | 763 | 98 / 96 | 1 / 1 | 114 s / 219 MiB | 0.044 s |

Both H0 diagrams agree. Both H1 and H2 diagrams differ in Å, even
though the H2 counts agree. Ignoring matching conflicts, take the
distance from each interval to its closest match in the other diagram
or the diagonal, then take the maximum in both directions. This gives
lower bounds on bottleneck distance of 0.0765 / 0.140 Å (H1 / H2) for
1A9L and 0.112 / 0.0062 Å for 1A60. Essential intervals cannot match
the diagonal. These two controls establish a difference on their graphs
but do not establish its prevalence or predictive value.

```sh
mkdir -p dumps
for id in 1A9L 1A60; do
  curl -fsSL "https://files.rcsb.org/download/$id.pdb" -o "dumps/$id.pdb"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python -m experiments.exp03_filtration_comparison.dump_rna_example \
    "dumps/$id.pdb" --out-dir dumps
  python -m experiments.exp03_filtration_comparison.compare_exact \
    --dir dumps --sid "$id" --out "dumps/$id.barcodes.json"
done
```

## Remaining cohort

The cloud repository lacks the path and support dumps for the original
cohort. Export the two Parquets per structure from the original jobs,
with no raw dataset transfer. Verify the dumps come from the same
extraction settings as the archived directed graphs; when complete
dumps exist, no geometric solve needs to be repeated.

| Largest pending cases | Archived directed paths | Historical extraction wall time |
| --- | ---: | ---: |
| Largest protein–DNA case (1HJC) | 10,256 | 28.5 min |
| Previously failed RNA case (SimRNA-3L0U-2) | 6,914 | 13.1 min |
| Largest RNA case (FARFAR2-5SWD) | 10,827 | 16.7 min |

Start the re-audit with the failed RNA case and the largest case in each
task. From complete dumps, provision one CPU core, 4 GiB RAM, and a
30-minute wall limit per structure as a first profiling job, not a
measured bound. If the dumps must be regenerated, allow at least one
hour per structure for extraction on comparable hardware and avoid
rebuilding the legacy event-scale snapshots. The original geometry
pipeline took about 2.35 hours in summed wall time over the cohort,
excluding subset export and re-audit. Sparse elimination fill-in
depends on graph connectivity: record wall time and peak RSS, and move
only outliers to a larger node. The exact reducer runs on CPU, not GPU.

## Legacy evidence

The original three-scale comparison remains a discrete result. In the
archived event-value comparison, six RNA structures used subsampling
under at least one rule, including runs capped at 1,500 values, and one
RNA structure failed. Of the nine structures with neither rule
subsampled, six already differ in the number of H1 or H2 intervals
(five protein–DNA, one RNA). This count is independent of the scale
index convention. The legacy mixed-tier equality flags are not a
full birth-value comparison in Å; compare against the
[archived summary](../../results/tables/exp03_eventvalue_summary.csv).
