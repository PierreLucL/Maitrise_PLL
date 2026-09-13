#!/bin/bash
#SBATCH --job-name=curbd_dff_off
#SBATCH --account=def-mides323
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/pllar11/Maitrise_PLL/slurm-%j.out

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

### On donne les quatre CPU aux libs numeriques sans lancer quatre copies du job.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$PROJECT_DIR"

echo "Job: $SLURM_JOB_ID"
echo "Noeud: $(hostname)"
echo "Debut: $(date)"

python scripts/curbd/loop.py \
  --data-dir "$MAITRISE_DATA_DIR"

echo "Fin: $(date)"
