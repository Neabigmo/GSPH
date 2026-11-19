#!/usr/bin/env bash
#SBATCH --job-name=ion-full
#SBATCH --cpus-per-task=32
#SBATCH --mem=90G
#SBATCH --time=3-00:00:00
set -uo pipefail
ROOT=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
PYTHON=${PYTHON:-$HOME/gsph_paper/env/bin/python}
DATA=${DATA:?Set DATA to the prepared full-cohort data folder (selection.json and downloaded mmCIF)}
CFG=${CFG:-$ROOT/configs/experiments/exp09_metal_site_full.yaml}
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
status=0
"$PYTHON" experiments/exp09_metal_site_pilot/run.py --stage crosscheck --config "$CFG" --data "$DATA" || status=$?
"$PYTHON" experiments/exp09_metal_site_pilot/run.py --stage run --config "$CFG" --data "$DATA" || status=$?
"$PYTHON" experiments/exp09_metal_site_pilot/run.py --stage summarize --config "$CFG" --data "$DATA" || status=$?
exit "$status"
