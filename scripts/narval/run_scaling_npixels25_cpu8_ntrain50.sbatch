#!/bin/bash
#SBATCH --job-name=curbd_scale_c8
#SBATCH --account=def-mides323
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/pllar11/Maitrise_PLL/slurm/scaling_cpu8_%A_%a.out

set -euo pipefail

### Benchmark CPU: meme config, juste le nombre de threads qui change.
module purge
module load python/3.11.5
source "$HOME/curbd_env/bin/activate"

PROJECT_DIR="/scratch/pllar11/Maitrise_PLL"
export MAITRISE_DATA_DIR="/scratch/pllar11/Datasets"
export PYTHONPATH="$PROJECT_DIR/src"
export MPLBACKEND="Agg"
export PYTHONUNBUFFERED="1"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/slurm"

python scripts/curbd/loop.py \
  --config scripts/narval/configs/scaling_npixels25_g08_sigma4_mouse410_ntrain50.json \
  --job-index "$SLURM_ARRAY_TASK_ID" \
  --data-dir "$MAITRISE_DATA_DIR" \
  --output-dir "narval_scaling_npixels25_cpu8_ntrain50/${SLURM_ARRAY_JOB_ID}"
