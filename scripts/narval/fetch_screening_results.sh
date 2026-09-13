#!/bin/bash
### Rapatrie les resultats d'une job array Narval vers le repo local.
### Usage: bash scripts/narval/fetch_screening_results.sh 1506925

set -euo pipefail

JOB_ID="${1:-1506925}"
REMOTE="${REMOTE:-Narval}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/scratch/pllar11/Maitrise_PLL}"
REMOTE_RUN_DIR="results/narval_screening_preprocessing_m6/${JOB_ID}"
LOCAL_RUN_DIR="results/narval_screening_preprocessing_m6/${JOB_ID}"

echo "Job ID: ${JOB_ID}"
echo "Remote: ${REMOTE}:${REMOTE_PROJECT}/${REMOTE_RUN_DIR}"
echo "Local : ${LOCAL_RUN_DIR}"

echo
echo "1) Verification et aggregation sur Narval"
ssh "${REMOTE}" "
  set -euo pipefail
  cd '${REMOTE_PROJECT}'
  echo 'CSV jobs produits:'
  find '${REMOTE_RUN_DIR}' -maxdepth 1 -name 'loop_summary_job_*.csv' | wc -l
  python scripts/utils/aggregate_loop_summaries.py '${REMOTE_RUN_DIR}'
"

echo
echo "2) Telechargement des CSV, PKL et logs de resultats"
mkdir -p "${LOCAL_RUN_DIR}"
rsync -avz --partial \
  "${REMOTE}:${REMOTE_PROJECT}/${REMOTE_RUN_DIR}/" \
  "${LOCAL_RUN_DIR}/"

echo
echo "3) Resume local"
find "${LOCAL_RUN_DIR}" -maxdepth 1 -name "loop_summary*.csv" -print
echo "Import termine."
