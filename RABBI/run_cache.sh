#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <params-number>"
  exit 1
fi

NUM="$1"
if ! [[ "${NUM}" =~ ^[0-9]+$ ]]; then
  echo "Error: <params-number> must be a positive integer."
  exit 1
fi

PARAM_ID="params${NUM}"
PARAM_FILE="params/${PARAM_ID}.yml"
QY_PREFIX="data/QY/qy_${PARAM_ID}"
SHELVE_PREFIX="${PARAM_ID}_"
PLOT_PREFIX="${PARAM_ID}_"
LOG_FILE="tmp/cache_${PARAM_ID}_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Error: parameter file '${PARAM_FILE}' not found."
  exit 1
fi

mkdir -p tmp

/usr/bin/time -v python3 -m cli cache \
  --param "${PARAM_FILE}" \
  --qy-prefix "${QY_PREFIX}" \
  --solvers RABBI OFFline Robust \
  --shelve-dir data/shelve \
  --shelve-prefix "${SHELVE_PREFIX}" \
  --plots multi_k_results multi_k_ratio multi_k_regret \
  --plot-dir data/pics \
  --plot-prefix "${PLOT_PREFIX}" \
  2>&1 | tee "${LOG_FILE}"
