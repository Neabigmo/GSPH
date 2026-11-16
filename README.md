# CS-PPH / GSPH — Common-Support Persistent Path Homology

Reproducible research repository for the JCIM manuscript on
**union-valued interaction patches, common spatial witnesses, and
higher-order persistent path structure** in molecular complexes.

Center claim under test:

> A chemically admissible molecular path should be born when all of its
> union-valued interaction patches acquire one common spatial witness,
> rather than when their pairwise relations are independently satisfied.

Pipeline:

```
Atomic microcontacts -> Interaction patches -> Union support (tau)
  -> GLMY Omega0-Omega3 -> Persistence barcodes -> Shared Hodge spectra
  -> P/T/H feature groups -> grouped-CV predictors
```

## Repository layout

| Path | Contents |
|---|---|
| `src/cspph/` | Canonical package: `geometry/`, `chemistry/`, `graph/`, `filtration/`, `topology/`, `spectral/`, `features/`, `evaluation/`, `io/`, `core/` |
| `src/gsph6/`, `src/runners/` | Backward-compatibility shims + legacy runners (all `gsph6.*` / `runners.*` imports keep working) |
| `tasks/` | Task adapters: `rna/`, `protein_dna/`, `membrane/` (membrane paused) |
| `experiments/` | EXP-01 … EXP-08 runners + published-baseline scripts |
| `configs/` | YAML configs for geometry, clash filter, filtration, tasks, experiments |
| `manifests/` | Dataset/target/deep-subset manifests + `legacy_file_map.csv` |
| `tests/` | unit / property / integration / regression tests |
| `scripts/` | `run_experiment.py`, `validate_manifest.py`, `reproduce_paper.py`, `make_figures.py` |
| `results/` | `tables/` + `reports/` + `figures/` tracked; `raw/`, `processed/` gitignored |
| `paper/` | Manuscript integration notes (paper sources maintained separately) |
| `docs/` | Mathematical spec, data dictionary, protocol, reproducibility |

## Quick start

```bash
conda env create -f environment.yml   # or: pip install -e .
conda activate cspph
python -c "import cspph; print(cspph.__version__)"
pytest tests/unit tests/property -q
```

Run a frozen experiment (requires data manifests — see `docs/REPRODUCIBILITY.md`):

```bash
python scripts/run_experiment.py --config configs/experiments/exp03_filtration_comparison.yaml
python scripts/reproduce_paper.py --version v33
```

## Frozen evidence base (V33)

- 72,433 certified-valid support records; 2,167 H0–H2 discrete intervals
- Same-graph filtration comparison: 16 structures × 2 graphs = 32 combos;
  barcode divergence RNA 14/16, PD 4/16
- Bounded perturbation 1,600 tests / 0 violations; solver audit 120/120
- See `results/reports/ablation_report_v3.md` and
  `results/reports/manuscript_v04_corrections.md`

## Data policy

Raw PDB/decoy files and large Parquet outputs are **not** redistributed.
`manifests/` records sources, target lists, and SHA256 checksums;
`results/raw/` and `results/processed/` are gitignored (use DVC/LFS for
large artifacts). The intramolecular clash rule
(`d < 0.6*(r_a+r_b)` excluded) applies to RNA/protein intramolecular
extraction only, never to protein–DNA docking interfaces.

## Citation

See `CITATION.cff`. Manuscript: *GSPH JCIM Manuscript V0.4* (in preparation).
