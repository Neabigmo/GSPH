#!/usr/bin/env bash
#SBATCH --job-name=union-audit
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
set -euo pipefail
ROOT=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
PYTHON=${PYTHON:-$HOME/gsph_paper/env/bin/python}
OUT_DIR=${OUT_DIR:?Set OUT_DIR}
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if (($# == 0)); then
  : "${DUMP_LIST:?Pass dump directories as arguments or a newline-delimited DUMP_LIST file}"
  mapfile -t directories < "$DUMP_LIST"
  set -- "${directories[@]}"
fi
exec "$PYTHON" -m experiments.union_audit_batch --out "$OUT_DIR" \
  --cohort "$ROOT/results/tables/path_landscape.parquet" --dumps "$@"
