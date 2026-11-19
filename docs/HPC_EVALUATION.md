# Offline CPU evaluation

The login node stages data and downloads; compute jobs never download. Unpack a
branch archive into a fresh code directory before submission. The scripts
in `scripts/slurm/` use `CODE_DIR`, `PYTHON` (default
`$HOME/gsph_paper/env/bin/python`) and `OUT_DIR`; select the partition at `sbatch`
time. Set `PYTHONPATH` to `src` and the repository root for direct module commands.
All output directories must be outside the repository. No container is needed.

## Frozen hard pairs from compact exports

```bash
python -m experiments.exp06_scoring_ablation.hardpair_frozen \
  --config configs/experiments/exp06_hardpair_frozen.yaml \
  --v2-dir "$V2_DIR" --out "$OUT_DIR"
```

Provide the physical_compact_v2, topology_features_v2 and hodge_features_v2
Parquets for each configured task, plus `oof_v2_<task>.parquet` (or
`oof_<task>.parquet`). **The compact feature exports do not contain labels.**
The original `run_ablation_v2` gets labels from unavailable raw task tables;
the supplied OOF tables preserve those same `label_cadscore` values in `y_true`
and `group` in `group_id`. Only those identifiers and labels are read; OOF scores,
models and splits are never used to select pairs or train these new models.
Labels repeated across old models must agree, and every feature candidate must
have a finite label. Conflicting labels between available OOF files are an error.

`build_v2_features` exports `structure_id`, `row`, `group`, `protein`, `kind`
when available. The old runner assumes identical row order; the new loader joins
one-to-one on `(structure_id, group)`, checks candidate-pool equality and verifies
all overlapping columns (including `row`) instead of silently aligning positions.
Shuffled exports are therefore safe. Only feature names registered for the task
in `results/tables/feature_manifest_v2.csv` enter tier selection. Each tier must
add nonempty columns. **No regex extensions were necessary:** the v2 manifest
already supplies nonempty matches for all three tiers using the frozen `patch_`,
`tau_2_`, `path_support_(mean|max)_k1`, `tau_[34]_`, `eta_`, `delta_tau_`,
`path_support_(mean|max)_k[23]`, `persistence_`, and `.*_glmy_betti_` families.
The `barcode_` rule remains available for future legacy barcode sidecars. Their original
anchoring and classifier exclusions are unchanged. No physical/Hodge columns
are promoted into a different tier merely because of their export file name.

The Slurm wrapper requests 32 CPUs and writes an effective configuration changing
only `rank_params.n_jobs` to the allocation. That effective file is hashed by the
runner; quantiles, Tier-1-only pair definitions, CV and all ten seeds are unchanged.

## Full union-support and Å barcode audit

Legacy complex dumps without patch geometry cannot recover endpoint spreads or
per-patch radius extremes: path/support/edge tables contain scalar births and
graph weights, and patch_audit contains structure-wide statistics, not atomic
coordinates and endpoint membership. Re-extract the original PDBs; do not infer
zero spread or substitute the archived witness subset. The fixed cohort is in
`manifests/union_audit_structures.tsv`, checked against the archived census.
`deep_complex_dump --fixed-ids <manifest>` bypasses all target/decoy selection and
requires unique PDBs and finite official labels. It records PDB/manifest hashes
and the RNA clash setting, and fails if an extraction fails. Stage only the 16
PDBs with the usual PD `Bound_docking/<target>/relaxed_top_complexes` and RNA
`PM_decoy_set/<target>` layouts, plus official per-target label files.
`scripts/slurm/union_redump.sh` takes `STRUCTURE_ROOT` and `OUT_DIR`, re-exports
both tasks with RNA clash ratio 0.6, then runs the full cohort audit in the same
offline job. It preserves the legacy dumps and writes new dumps below `OUT_DIR`.

```bash
python -m experiments.union_audit_batch --dumps "$PD_DUMPS" "$RNA_DUMPS" \
  --cohort results/tables/path_landscape.parquet --out "$OUT_DIR"
```

The driver calls `union_support_audit.audit_dumps` and the same `load_births` /
`compare_event_barcodes` functions as `compare_exact`, once for every structure
and both graph types. The optional cohort census checks the original 16 IDs and
all per-order path counts. Duplicate IDs across extraction variants are refused;
choose the clash-filtered RNA directory, not both RNA variants. Missing or corrupt
triplets cannot be replaced by barcode/landscape summary tables. Errors are recorded,
the remaining structures run, and incomplete audits return nonzero.

Outputs include every per-path ratio, ratio quantiles by graph/order, all bound
violations, every single-contact path with positive eta at zero delta, exact
Å interval JSONs and H0–H2 comparisons. `summary.json` aggregates the entire audit.
Barcode counts retain multiplicity; finite total persistence is summed in Å and
infinite bars counted separately (never silently assigned a finite lifetime).
`count_delta` and persistence deltas mean common minus pairwise. Exact-event
algebra can be expensive; a three-day/100 GiB union job is a cap, not a measured ETA.

## Baseline registry: definitions before comparisons

The conflicting claim is in `experiments/baselines/rna/eval_rna_baselines.py`:
its old prose said higher DI is better, but `evaluate` correlates predicted
quality `-score_kbt` with `-di`, treating lower DI as better. The prose now agrees
with the actual calculation. DI is the deformation-index error metric; verify
the distributed labels before reporting any sign-sensitive comparison:

```bash
python -m experiments.baseline_checks --pm-root "$PM_LABEL_ROOT" \
  --tables-root "$LAPTOP_TABLES" --out "$OUT_DIR"
```

Copy exactly **`PM_decoy_set/RMSD/*.rmsd` and `PM_decoy_set/DI/*.DI`**, preserving
the `RMSD/` and `DI/` directories under one parent. All 20 target label pairs are
small text; **do not copy any PDBs**. Rows are joined by target and decoy filename,
not position. Positive per-target RMSD–DI Spearman supports lower DI being better,
because lower RMSD is better; mixed or negative signs require investigation of
the label definition, not an automatic sign flip. Correlation is an empirical
orientation check, not a substitute for a metric definition. No real PM label
files are committed, so the observed direction on the owner's data remains pending.

The apparent RNA count conflict is explained by the committed
`results/tables/candidate_coverage.json`: **970 audit records**, of which **826
successful feature rows** survive audit, then **20 native structures excluded**
leaves **806 scored decoys over 20 targets**. These are recorded pipeline counts,
not three alternative filesystem counts; audit records need not be unique PDBs.
Read-only laptop counts (including native PDBs) can be obtained with:

```bash
PM=/path/to/PM_decoy_set
printf 'target directories: '; find "$PM" -mindepth 1 -maxdepth 1 -type d ! -name RMSD ! -name DI | wc -l
printf 'PDB files: '; find "$PM" -type f -iname '*.pdb' | wc -l
printf 'RMSD files: '; find "$PM/RMSD" -type f -name '*.rmsd' | wc -l
printf 'DI files: '; find "$PM/DI" -type f -name '*.DI' | wc -l
```

`baseline_checks` reports schema size, key columns and group counts separately
for each candidate/OOF table (and task where present). Named PD tasks must have
47 groups, not the historical 57-target rigid perturbation pool. Counts on the
owner's tables remain to be measured; the committed coverage records 47 for both.

## Ion calibration: select, download, compute, summarize

```bash
python -m experiments.ion_calibration --stage prepare --out "$CALIBRATION"
python -m experiments.ion_calibration --stage download --out "$CALIBRATION"
# Offline compute:
python -m experiments.ion_calibration --stage run --out "$CALIBRATION" --workers 8
python -m experiments.ion_calibration --stage summarize --out "$CALIBRATION"
```

The frozen manifest uses the committed per-structure atom counts, five rank-size
bins per cohort, all historically censored cases, and both size extrema. Seeded
sampling fills the remaining quota: 20 Zn + 20 water + 10 Mg. Every nonempty
size/censor stratum is represented. Extrema occupy separate certainty strata;
historically censored strata are censused, while remaining strata are sampled
uniformly without replacement. The manifest, source/config hashes, download
ledger, worker JSONs, `/usr/bin/time -v` files and summaries stay under the supplied
directory. A changed selection or config requires a new output directory.

Each selected structure runs through the unchanged pilot worker, with a solver
crosscheck before the batch. The download stage fetches only the selected unique
mmCIF IDs. The compute stage does not call the old internet-enabled `run` stage or
the old `summarize` stage that overwrites committed tables.

Projection uses inverse size/censor-stratum sampling fractions, not an unweighted
mean of the 50. Full cohort sizes are 300 Zn, 300 water and 20 Mg from the committed
cohort. Sampled costs include the whole worker CPU time, including censored runs.
Exhaustive projections cover candidate solving only in completed strata; outputs
explicitly report per-1000-complete costs and the complete-only part of the full
cohort. They exclude parsing, enumeration, and unknown censored costs, so they are
not exhaustive wall-time estimates. Solver elapsed seconds are a CPU-cost proxy
under single-threaded workers; only sampled worker CPU hours are directly measured.
RSS and both sampled/weighted censor fractions
are reported. Missing/error workers make the summary fail and suppress cost
projections for the affected cohort instead of counting them as free work.
