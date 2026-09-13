#!/bin/bash
#SBATCH --job-name=curbd_amp001
#SBATCH --account=def-mides323
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-3%2
#SBATCH --output=/scratch/pllar11/Maitrise_PLL/slurm/amp001_%A_%a.out

set -euo pipefail

### Scout article-like: meme resolution gagnante, mais bruit externe plus proche du papier.
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

CONFIG="scripts/narval/configs/scout_article_like_amp001_g08_g12_sigma4_410_415_ntrain1000.json"
OUTPUT_DIR="narval_scout_article_like_amp001_g08_g12_sigma4_410_415_ntrain1000/${SLURM_ARRAY_JOB_ID}"

echo "Job array: ${SLURM_ARRAY_JOB_ID}"
echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Noeud: $(hostname)"
echo "Debut: $(date)"

python scripts/curbd/loop.py \
  --config "$CONFIG" \
  --job-index "$SLURM_ARRAY_TASK_ID" \
  --data-dir "$MAITRISE_DATA_DIR" \
  --output-dir "$OUTPUT_DIR"

echo "Fin: $(date)"
