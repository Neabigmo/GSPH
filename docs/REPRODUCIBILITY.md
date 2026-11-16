# Reproducibility

## Environment

```bash
conda env create -f environment.yml
conda activate cspph
# or: pip install -e .
```

Python >= 3.10; core deps numpy/scipy/pandas/pyarrow/scikit-learn/pyyaml.
Optional C++ path/support backends are built from `src/cpp/` sources via
the existing backend loaders (graceful NumPy fallback if unavailable).

## Frozen V33 reproduction

```bash
python scripts/validate_manifest.py --manifest manifests/deep_subset_v33.csv
python scripts/run_experiment.py --config configs/experiments/exp03_filtration_comparison.yaml
python scripts/run_experiment.py --config configs/experiments/exp06_scoring_ablation.yaml
python scripts/reproduce_paper.py --version v33
python scripts/make_figures.py --figure witness
```

`reproduce_paper.py` only reads frozen tables under `results/tables/`
and `results/reports/`; it never re-downloads raw data or overwrites
results.

## Data requirements

Raw datasets are NOT in this repo. Required inputs (place under
`data/` — gitignored):

- rsRNASP PM decoy set (826 PDBs) — `tasks/rna/`
- FTDMP Bound/Unbound docking decoys — `tasks/protein_dna/`
- M223 membrane set — `tasks/membrane/` (paused)

Manifests record target lists and expected counts; validators check
structure hashes when provided.

## HPC

Large extractions ran on SLURM (see `results/reports/` job IDs
49178410/49196422/49198410/49198418/49206184/49206187). Experiment
scripts accept `--shard/--n-shards` for array execution; see the legacy
`*_array.sbatch` examples in the workspace history.
