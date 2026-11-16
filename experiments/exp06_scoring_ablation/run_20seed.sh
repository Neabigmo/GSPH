#!/bin/bash
# EXP-06: 20-seed grouped CV. Run on HPC or locally with feature parquets
# in the expected _feat locations. Produces oof_<task>_s<seed>.parquet.
set -euo pipefail
cd "$(dirname "$0")/../.."
AB=GSPH_H3/ablation/run_ablation.py
for task in pd_bound pd_unbound rna; do
  for seed in $(seq 0 19); do
    python "$AB" --task "$task" --seed "$seed" --tag "_s${seed}"
  done
done
python experiments/exp06_scoring/merge_oof.py
