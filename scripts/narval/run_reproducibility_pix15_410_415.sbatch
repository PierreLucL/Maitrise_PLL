#!/bin/bash
#SBATCH --job-name=curbd_repro15
#SBATCH --account=def-mides323
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5%2
#SBATCH --no-requeue
#SBATCH --output=/scratch/pllar11/Maitrise_PLL/slurm/repro_pix15_%A_%a.out

set -euo pipefail
module purge
module load python/3.11.5
source "$HOME/curbd_env/bin/activate"

### sbatch est appele depuis une copie figee du code.
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MAITRISE_DATA_DIR=/scratch/pllar11/Datasets
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

CONFIG=scripts/narval/configs/reproducibility_pix15_410_415.json
OUTPUT_DIR="narval_reproducibility_pix15_410_415/${SLURM_ARRAY_JOB_ID}"

python scripts/narval/check_reproducibility_snapshot.py
printf 'Snapshot: %s\nJob: %s task %s\n' "$PWD" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
python scripts/curbd/loop.py --config "$CONFIG" \
  --job-index "$SLURM_ARRAY_TASK_ID" --data-dir "$MAITRISE_DATA_DIR" \
  --output-dir "$OUTPUT_DIR"

### loop.py capture les erreurs : on les fait aussi remonter dans le statut Slurm.
python - "$OUTPUT_DIR" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv
import sys
from pathlib import Path
path = Path('results') / sys.argv[1] / f'loop_summary_job_{int(sys.argv[2]):04d}.csv'
with path.open() as f:
    rows = list(csv.DictReader(f))
if len(rows) != 1 or rows[0]['status'] != 'done' or not Path(rows[0]['save_path']).is_file():
    raise SystemExit(f'Echec ou resultat manquant : {path}')
print('Resultat confirme:', rows[0]['save_path'])
PY
