#!/bin/bash
#SBATCH --job-name=curbd_pix_tiny
#SBATCH --account=def-mides323
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=0-5%2
#SBATCH --output=/scratch/pllar11/Maitrise_PLL/slurm/pix_tiny_%A_%a.out

set -euo pipefail

### Scout tiny pixels: on teste plus de sous-regions, mais court, question de pas bruler Narval pour rien.
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

CONFIG="scripts/narval/configs/scout_tiny_npixels15_20_25_g08_sigma4_410_415_ntrain100.json"
OUTPUT_DIR="narval_scout_tiny_npixels15_20_25_g08_sigma4_410_415_ntrain100/${SLURM_ARRAY_JOB_ID}"

echo "Job array: ${SLURM_ARRAY_JOB_ID}"
echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Noeud: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Debut: $(date)"

python scripts/curbd/loop.py \
  --config "$CONFIG" \
  --job-index "$SLURM_ARRAY_TASK_ID" \
  --data-dir "$MAITRISE_DATA_DIR" \
  --output-dir "$OUTPUT_DIR"

echo "Fin: $(date)"
