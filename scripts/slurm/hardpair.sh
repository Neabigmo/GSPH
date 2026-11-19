#!/usr/bin/env bash
#SBATCH --job-name=hardpair
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
set -euo pipefail
ROOT=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
PYTHON=${PYTHON:-$HOME/gsph_paper/env/bin/python}
V2_DIR=${V2_DIR:-${1:-}}
OUT_DIR=${OUT_DIR:-${2:-}}
: "${V2_DIR:?Set V2_DIR or first argument}" "${OUT_DIR:?Set OUT_DIR or second argument}"
mkdir -p "$OUT_DIR"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" - "$ROOT" "$OUT_DIR" "${SLURM_CPUS_PER_TASK:-32}" <<'PY'
from pathlib import Path
import sys
import yaml
root, out, cores = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
config = yaml.safe_load((root / 'configs/experiments/exp06_hardpair_frozen.yaml').read_text())
config['rank_params']['n_jobs'] = cores
(out / 'effective_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
print('XGBoost n_jobs =', cores, flush=True)
PY
exec "$PYTHON" -m experiments.exp06_scoring_ablation.hardpair_frozen \
  --config "$OUT_DIR/effective_config.yaml" --v2-dir "$V2_DIR" --out "$OUT_DIR"
