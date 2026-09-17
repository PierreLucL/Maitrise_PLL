#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -s submission.jobid ]; then cat submission.jobid; exit 0; fi
mkdir .submission_lock
trap 'rmdir .submission_lock' EXIT
if [ -e submission.started ]; then echo 'Tentative sans reçu : vérifier Slurm avant de resoumettre.' >&2; exit 1; fi
module purge
module load python/3.11.5
source "$HOME/curbd_env/bin/activate"
export MAITRISE_DATA_DIR=/scratch/pllar11/Datasets MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for variant in corrected; do
  (cd "$variant"; PYTHONPATH="$PWD/src" python -m unittest discover -s tests -p test_reproducibility.py -v) > "preflight_${variant}.log" 2>&1 || { cat "preflight_${variant}.log"; exit 1; }
done
for task in 0 1 2 3; do
  python run_paired_controls.py --root "$PWD" --task "$task" --dry-run > "preflight_task${task}.log" 2>&1 || { cat "preflight_task${task}.log"; exit 1; }
done
sbatch --test-only run_paired_controls.sbatch
date -u > submission.started
sbatch --parsable run_paired_controls.sbatch > submission.jobid.tmp
mv submission.jobid.tmp submission.jobid
cat submission.jobid
squeue -r -j "$(cut -d';' -f1 submission.jobid)" -o '%.18i %.20j %.2t %.10M %.10l %.4C %.10m %R'
