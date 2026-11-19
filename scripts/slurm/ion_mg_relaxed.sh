#!/usr/bin/env bash
#SBATCH --job-name=ion-mg-relaxed
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
# Rerun Mg structures censored in the full-cohort run with relaxed caps and a per-structure time limit.
set -uo pipefail
ROOT=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
PYTHON=${PYTHON:-$HOME/gsph_paper/env/bin/python}
DATA=${DATA:?Set DATA to the full-cohort data folder}
OUT_DIR=${OUT_DIR:?Set OUT_DIR}
CFG=${CFG:-$ROOT/configs/experiments/exp09_metal_site_mg_relaxed.yaml}
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
mkdir -p "$OUT_DIR/records"
limit=$(awk '/^timeout_seconds:/{print $2 + 60}' "$CFG")
"$PYTHON" - "$DATA/structures" > "$OUT_DIR/censored_mg.txt" <<'PY'
import json, sys
from pathlib import Path
for p in sorted(Path(sys.argv[1]).glob('mg_*.json')):
    r = json.loads(p.read_text())
    if r.get('status', '').endswith('_censored'):
        print(r['pdb_id'], r['status'])
PY
echo "censored Mg in full run: $(wc -l < "$OUT_DIR/censored_mg.txt")"
while read -r pdb status; do
  /usr/bin/time -v timeout "$limit" "$PYTHON" experiments/exp09_metal_site_pilot/run.py --stage worker --config "$CFG" \
    --data "$DATA" --cohort mg --pdb-id "$pdb" --output "$OUT_DIR/records/mg_$pdb.json" 2> "$OUT_DIR/records/mg_$pdb.time" \
    || echo "{\"cohort\": \"mg\", \"pdb_id\": \"$pdb\", \"status\": \"timeout_censored\"}" > "$OUT_DIR/records/mg_$pdb.json.fail"
done < "$OUT_DIR/censored_mg.txt"
"$PYTHON" - "$OUT_DIR" <<'PY'
import csv, json, sys
from pathlib import Path
out = Path(sys.argv[1]); rows = []
for line in (out / 'censored_mg.txt').read_text().split('\n'):
    if not line.strip():
        continue
    pdb, before = line.split()
    path = out / 'records' / f'mg_{pdb}.json'
    record = json.loads(path.read_text()) if path.exists() else json.loads((out / 'records' / f'mg_{pdb}.json.fail').read_text())
    rows.append(dict(pdb_id=pdb, status_full_caps=before, status_relaxed_caps=record.get('status'),
                     atoms=record.get('atoms'), patches=record.get('patches'), wall_seconds=record.get('wall_seconds'),
                     peak_rss_kib=record.get('peak_rss_kib')))
with open(out / 'mg_relaxed_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['pdb_id']); w.writeheader(); w.writerows(rows)
print(f"rerun {len(rows)}; still censored {sum(str(r['status_relaxed_caps']).endswith('_censored') for r in rows)}")
PY
