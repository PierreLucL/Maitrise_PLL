#!/bin/bash
#SBATCH --job-name=plot_rnn_traces
#SBATCH --account=def-mides323
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=/scratch/pllar11/Maitrise_PLL/plot-timeseries-%j.out

set -euo pipefail

### Petit job de figures: pas besoin de repartir un entrainement de quatre heures.
module purge
module load python/3.11.5
source "$HOME/curbd_env/bin/activate"

PROJECT_DIR="/scratch/pllar11/Maitrise_PLL"
RUN_DIR="$PROJECT_DIR/results/CURBD_DFF_OFF_1000runs_415_410/run_du_2026-08-20_06h40"
export MAITRISE_DATA_DIR="/scratch/pllar11/Datasets"
export PYTHONPATH="$PROJECT_DIR/src"
export MPLBACKEND="Agg"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}"

cd "$PROJECT_DIR"

python scripts/visualization/figure_timeseries_pipeline.py \
  --pkl "$RUN_DIR/config000_C9_M6_mouse415_n_pixels100_lissage_sigma2_tauRNN0.33_dtFactor2_g0.8_ampInWN0.01_P01_nRunTrain1000_use_dffFalse_use_global_regressionFalse.pkl" \
  --data-root "$MAITRISE_DATA_DIR" \
  --duration-sec 30 \
  --n-regions 8

python scripts/visualization/figure_timeseries_pipeline.py \
  --pkl "$RUN_DIR/config001_C9_M6_mouse410_n_pixels100_lissage_sigma2_tauRNN0.33_dtFactor2_g0.8_ampInWN0.01_P01_nRunTrain1000_use_dffFalse_use_global_regressionFalse.pkl" \
  --data-root "$MAITRISE_DATA_DIR" \
  --duration-sec 30 \
  --n-regions 8

echo "Figures terminees dans $RUN_DIR/timeseries_pipeline_figures"
