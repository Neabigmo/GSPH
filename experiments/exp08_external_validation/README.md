# EXP-08: Independent Protein–DNA external validation

## Status: data prepared, feature extraction pending HPC slot

## Dataset
- Source: Dockground non-redundant protein–DNA set (sequence-redundancy
  reduced at 30% protein identity), `dna-non-redundant-seq.csv.gz`,
  https://dockground.compbio.ku.edu/downloads/bound/bound_downloads/
- Excludes all 47 FTDMP/Prot-DNABenchmark development targets.
- Selected: 20 targets (`exp08_targets.csv`), spread across the list.
- Bound complexes downloaded from RCSB (`pdb/*.pdb`).

## Decoy protocol (documented limitation)
No public docking engine was run; decoys are **rigid-body perturbations**
of the DNA duplex, not FTDMP docking poses:
- Tier 1 (8/24): whole-duplex rotation (15–180°) + translation 2–15 Å.
- Tier 2 (16/24): per-chain independent perturbation 3–20 Å (partial
  unbinding analogue).
- Labels: ligand-RMSD of DNA heavy atoms after Kabsch superposition of
  protein chains; native = 0; `native_like` <4 Å, `medium` <10 Å,
  `decoy` ≥10 Å (`manifest.csv`).
- This validates that CS-PPH features rank the true interface above
  geometrically perturbed decoys on **targets never used in development**.
  It does NOT replicate FTDMP docking-pose discrimination difficulty.

## Files
- `scripts/make_decoys.py` — decoy generator (seed 20261117)
- `scripts/to_ftdmp_layout.py` — converts to FTDMP-compatible layout
- `exp08_targets.csv` — selected PDB codes + DNA chain ids
- Data (not committed, ~132 MB): `exp08_ftdmp.tar.gz` on HPC at
  `/work/share/acn8s56xc0/gsph_h3_deep/`, extracted to `exp08_ftdmp/`
- Feature extraction: `pd_ftdmp_driver.py --root exp08_ftdmp
  --out exp08_feat --shard 0..9 --nshards 10` (sbatch exp08_feat,
  pending QOS slot)

## Interpretation boundary
Perturbation decoys are easier than docking decoys; a positive result
supports transfer of the representation, a negative result is
inconclusive about docking-pose ranking. Report accordingly.
