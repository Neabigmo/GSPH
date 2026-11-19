#!/usr/bin/env bash
#SBATCH --job-name=union-redump
#SBATCH --cpus-per-task=24
#SBATCH --mem=90G
#SBATCH --time=3-00:00:00
set -uo pipefail
ROOT=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
PYTHON=${PYTHON:-$HOME/gsph_paper/env/bin/python}
: "${OUT_DIR:?Set OUT_DIR}" "${STRUCTURE_ROOT:?Set STRUCTURE_ROOT to the staged 16-structure tree}"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GSPH_H3_CLASH_RATIO=0.6
status=0
for task in pd rna; do
  "$PYTHON" -m experiments.exp05_molecular_landscape.deep_complex_dump \
    --task "$task" --fixed-ids "$ROOT/manifests/union_audit_structures.tsv" \
    --ftdmp-root "$STRUCTURE_ROOT/pd" --pm-root "$STRUCTURE_ROOT/rna/PM_decoy_set" \
    --out "$OUT_DIR/dumps/$task" || status=$?
done
"$PYTHON" -m experiments.union_audit_batch --dumps "$OUT_DIR/dumps/pd" "$OUT_DIR/dumps/rna" \
  --cohort "$ROOT/results/tables/path_landscape.parquet" --out "$OUT_DIR/audit" || status=$?
exit "$status"
