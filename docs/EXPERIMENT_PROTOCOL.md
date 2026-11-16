# Experiment Protocol

Every experiment directory `experiments/expXX_*/` maps to
`configs/experiments/expXX_*.yaml` and must produce an
`experiment_card.md` + `experiment_metadata.json` on completion
(see plan §4.3 for required keys).

## Status at repository freeze (V33)

| EXP | Status | Evidence |
|---|---|---|
| 01 pairwise construction | done | `results/tables/` geometry cases; see ablation report |
| 02 molecular witnesses | partial | 3 structures x 6 certified examples (`results/figures/`) |
| 03 filtration comparison | done (16 structures) | `results/tables/barcode_comparison_16structure.csv`; fine-grid/event-value/bootstrap pending |
| 04 math audits | partial | perturbation 1,600/0 violations; solver 120/120; chain 60/60; 500-case stress pending |
| 05 molecular landscape | partial | 16-structure complex dumps; per-order aggregation pending |
| 06 scoring ablation | partial | 10-seed hard-pair PD bound; 20-seed grouped CV pending |
| 07 sensitivity | planned | clash/radius/patch-repr/path-order sweeps |
| 08 external validation | blocked on data | requires >=10 independent PD targets |

## Rules

1. Filtration comparators must pass face-monotonicity, `d^2=0`,
   inclusion chain-map and commutation tests (`tests/integration/`).
2. `valid=false`/`incomplete=true` are explicit, never zero-filled.
3. Prediction experiments share identical split manifests across the
   seven P/T/H arms; target is always the CV group.
4. Seed policy: repeated GroupKFold seeds are fold partitions, NOT
   independent biological replicates; inference uses target-cluster
   bootstrap (>=2000).
5. Paper numbers are generated only from `results/tables/` frozen files.
