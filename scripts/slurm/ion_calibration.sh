#!/usr/bin/env bash
#SBATCH --job-name=ion-calibration
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
set -uo pipefail
ROOT=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
PYTHON=${PYTHON:-$HOME/gsph_paper/env/bin/python}
OUT_DIR=${OUT_DIR:-${1:-}}
: "${OUT_DIR:?Set OUT_DIR or first argument; login-node prepare/download must already be complete}"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
status=0
"$PYTHON" -m experiments.ion_calibration --stage run --out "$OUT_DIR" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" || status=$?
"$PYTHON" -m experiments.ion_calibration --stage summarize --out "$OUT_DIR" || status=$?
exit "$status"
