#!/bin/bash
#SBATCH --job-name=curbd_ttminus1
#SBATCH --account=def-mides323
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3%2
#SBATCH --no-requeue
#SBATCH --output=/scratch/pllar11/Maitrise_PLL/slurm/ttminus1_replicates_%A_%a.out
set -euo pipefail
module purge
module load python/3.11.5
source "$HOME/curbd_env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export MAITRISE_DATA_DIR=/scratch/pllar11/Datasets
export MPLBACKEND=Agg MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python run_paired_controls.py --root "$PWD" --task "$SLURM_ARRAY_TASK_ID"
