#!/bin/bash
#SBATCH --job-name=curbd_screen_m6
#SBATCH --account=def-mides323
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-259%20
#SBATCH --output=/scratch/pllar11/Maitrise_PLL/slurm/screen_m6_%A_%a.out

set -euo pipefail

### Narval aime qu'on declare clairement nos modules avant de partir le moteur.
module purge
module load python/3.11.5
source "$HOME/curbd_env/bin/activate"

### Tous les chemins cluster vivent ici; rien ne depend du disque externe du Mac.
PROJECT_DIR="/scratch/pllar11/Maitrise_PLL"
export MAITRISE_DATA_DIR="/scratch/pllar11/Datasets"
export PYTHONPATH="$PROJECT_DIR/src"
export MPLBACKEND="Agg"
export PYTHONUNBUFFERED="1"

### On donne les CPU demandes aux libs numeriques sans lancer quatre copies de Python.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/slurm"

CONFIG="scripts/narval/configs/screening_preprocessing_m6.json"
OUTPUT_DIR="narval_screening_preprocessing_m6/${SLURM_ARRAY_JOB_ID}"

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
