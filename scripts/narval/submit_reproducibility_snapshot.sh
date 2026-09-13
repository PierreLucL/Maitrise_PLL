#!/bin/bash
### A executer depuis le snapshot sur Narval, dans un shell login.
set -euo pipefail
cd "$(dirname "$0")/../.."

### Une nouvelle tentative retourne le recu existant au lieu de soumettre un doublon.
if [ -s submission.jobid ]; then
  printf 'Soumission deja enregistree: '
  cat submission.jobid
  exit 0
fi
if ! mkdir .submission_lock; then
  echo 'Soumission precedente en cours ou interrompue; verifier Slurm avant de reessayer.' >&2
  exit 1
fi
trap 'rmdir .submission_lock 2>/dev/null || true' EXIT

module purge
module load python/3.11.5
source "$HOME/curbd_env/bin/activate"
export PYTHONPATH="$PWD/src"
export MAITRISE_DATA_DIR=/scratch/pllar11/Datasets
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

python scripts/narval/check_reproducibility_snapshot.py
python -m unittest discover -s tests -v > preflight_tests.log 2>&1 || {
  cat preflight_tests.log
  exit 1
}
mkdir -p /scratch/pllar11/Maitrise_PLL/slurm /scratch/pllar11/Maitrise_PLL/results
if [ ! -e results ]; then
  ln -s /scratch/pllar11/Maitrise_PLL/results results
fi
python scripts/curbd/loop.py \
  --config scripts/narval/configs/reproducibility_pix15_410_415.json \
  --output-dir reproducibility_preflight --dry-run
sbatch --test-only scripts/narval/run_reproducibility_pix15_410_415.sbatch

### Si la connexion tombe apres sbatch, le recu demeure sur Narval.
if [ -e submission.started ]; then
  echo 'Tentative anterieure sans recu : verifier sacct/squeue avant toute nouvelle soumission.' >&2
  exit 1
fi
date -u > submission.started
sbatch --parsable scripts/narval/run_reproducibility_pix15_410_415.sbatch > submission.jobid.tmp
mv submission.jobid.tmp submission.jobid
printf 'Soumission enregistree: '
cat submission.jobid
squeue -r -j "$(cut -d';' -f1 submission.jobid)" -o '%.18i %.20j %.2t %.10M %.10l %.4C %.10m %R'
