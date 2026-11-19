#!/usr/bin/env bash
#SBATCH --job-name=ext-metal
#SBATCH --array=0-31
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%A_%a.out
set -uo pipefail
ROOT=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
PYTHON=${PYTHON:?Set PYTHON to the environment made by setup_env.sh}
TOOLS=${TOOLS:?Set TOOLS to the folder made by setup_env.sh}
OUT_DIR=${OUT_DIR:?Set OUT_DIR to an output folder outside the repository}
METHOD=${METHOD:?Set METHOD to metal3d or allmetal3d}
BATCH_SIZE=${BATCH_SIZE:-8}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK
cd "$ROOT"
"$PYTHON" experiments/external_metal_site_run/run.py --method "$METHOD" --index "$SLURM_ARRAY_TASK_ID" \
  --tools "$TOOLS" --out "$OUT_DIR" --batch-size "$BATCH_SIZE"
