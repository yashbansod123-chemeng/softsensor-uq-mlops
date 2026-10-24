#!/usr/bin/env bash
# Train, evaluate and quality-gate an MVEM soft sensor.
# Usage: scripts/run_pipeline.sh [-d ccpp|synthetic] [-a ARCH] [-e EPOCHS] [-r MAX_RMSE] [-o OUT_DIR] [-- extra CLI args]
set -euo pipefail

DATASET="ccpp"
ARCH="mean_and_hidden"
EPOCHS=200
MAX_RMSE=""
OUT_ROOT="runs"

usage() { sed -n '2,3p' "$0"; exit "${1:-0}"; }

while getopts ":d:a:e:r:o:h" opt; do
  case "${opt}" in
    d) DATASET="${OPTARG}" ;;
    a) ARCH="${OPTARG}" ;;
    e) EPOCHS="${OPTARG}" ;;
    r) MAX_RMSE="${OPTARG}" ;;
    o) OUT_ROOT="${OPTARG}" ;;
    h) usage 0 ;;
    \?) echo "Unknown option: -${OPTARG}" >&2; usage 1 ;;
    :) echo "Option -${OPTARG} needs a value" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))

if ! [[ "${EPOCHS}" =~ ^[0-9]+$ ]]; then
  echo "EPOCHS must be a positive integer, got '${EPOCHS}'" >&2
  exit 1
fi

# Dataset-specific default gate (original units): CCPP output is in MW
if [[ -z "${MAX_RMSE}" ]]; then
  case "${DATASET}" in
    ccpp) MAX_RMSE=5.0 ;;
    synthetic) MAX_RMSE=1.0 ;;
    *) echo "Unknown dataset '${DATASET}'" >&2; exit 1 ;;
  esac
fi

GIT_SHA="${GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo local)}"
export GIT_SHA
RUN_DIR="${OUT_ROOT}/${DATASET}-${ARCH}-${GIT_SHA}"
mkdir -p "${RUN_DIR}"
LOG="${RUN_DIR}/pipeline.log"

echo "[$(date -u +%FT%TZ)] dataset=${DATASET} arch=${ARCH} epochs=${EPOCHS} max_rmse=${MAX_RMSE} sha=${GIT_SHA}" | tee "${LOG}"

set +e
python -m softsensor.cli \
  --dataset "${DATASET}" --arch "${ARCH}" --epochs "${EPOCHS}" \
  --max-rmse "${MAX_RMSE}" --out "${RUN_DIR}" "$@" 2>&1 | tee -a "${LOG}"
status=${PIPESTATUS[0]}
set -e

case "${status}" in
  0) echo "PASS: model accepted (${RUN_DIR})" | tee -a "${LOG}" ;;
  2) echo "FAIL: quality gate rejected the model (${RUN_DIR})" | tee -a "${LOG}" ;;
  *) echo "ERROR: pipeline crashed with exit code ${status}" | tee -a "${LOG}" ;;
esac
exit "${status}"
