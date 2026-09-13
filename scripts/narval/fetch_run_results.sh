#!/bin/bash
### Rapatrie une run Narval vers le repo local.
### Usage: bash scripts/narval/fetch_run_results.sh narval_visual_rnn_trace_410_415_ntrain1000 1234567

set -euo pipefail

RUN_NAME="${1:?Donne le nom du dossier de run, ex: narval_visual_rnn_trace_410_415_ntrain1000}"
JOB_ID="${2:?Donne le job id Narval, ex: 1506925}"
REMOTE="${REMOTE:-Narval}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/scratch/pllar11/Maitrise_PLL}"
REMOTE_RUN_DIR="results/${RUN_NAME}/${JOB_ID}"
LOCAL_RUN_DIR="results/${RUN_NAME}/${JOB_ID}"

echo "Run: ${RUN_NAME}"
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
