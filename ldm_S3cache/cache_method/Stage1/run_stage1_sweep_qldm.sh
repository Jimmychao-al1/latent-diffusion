#!/usr/bin/env bash
# Stage-1 (Q-LDM) sweep — same grid as FP run_stage1_sweep_ldm.sh
#
# Usage:
#   bash ldm_S3cache/cache_method/Stage1/run_stage1_sweep_qldm.sh
# Optional overrides:
#   STAGE0_DIR=... BASE_OUT=... BASE_FIG=... LAMBDA_LIST="0.25 0.5 1.0 2.0" bash ...
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

STAGE0_DIR="${STAGE0_DIR:-ldm_S3cache/cache_method/Stage0/stage0_output_qldm}"
BASE_OUT="${BASE_OUT:-ldm_S3cache/cache_method/Stage1/stage1_output_qldm}"
BASE_FIG="${BASE_FIG:-ldm_S3cache/cache_method/Stage1/stage1_figures_qldm}"

SCHEDULER="ldm_S3cache/cache_method/Stage1/stage1_scheduler_ldm.py"
VERIFY="ldm_S3cache/cache_method/Stage1/verify_scheduler_ldm.py"
VISUALIZE="ldm_S3cache/cache_method/Stage1/visualize_stage1_ldm.py"

# Match FP stage1_output_ldm retained configs only.
SWEEP_SPECS=(
  "15 3 0.5"
  "15 3 1.0"
  "20 3 1.0"
)

TOTAL=${#SWEEP_SPECS[@]}
CUR=0

echo "================================================================"
echo "Stage-1 Q-LDM sweep"
echo "STAGE0_DIR=${STAGE0_DIR}"
echo "BASE_OUT=${BASE_OUT}"
echo "BASE_FIG=${BASE_FIG}"
echo "SWEEP_SPECS=( ${SWEEP_SPECS[*]} )"
echo "TOTAL_RUNS=${TOTAL}"
echo "================================================================"

for SPEC in "${SWEEP_SPECS[@]}"; do
  read -r K SW LAMBDA <<< "${SPEC}"
  CUR=$((CUR + 1))
  TAG="K${K}_sw${SW}_lam${LAMBDA}"
      OUT_DIR="${BASE_OUT}/sweep_${TAG}"
      FIG_DIR="${BASE_FIG}/sweep_${TAG}"
      mkdir -p "${OUT_DIR}" "${FIG_DIR}"

      echo ""
      echo "[$CUR/$TOTAL] ▶ ${TAG}"
      echo "  out: ${OUT_DIR}"
      echo "  fig: ${FIG_DIR}"

      python3 "${SCHEDULER}" \
        --stage0_dir "${STAGE0_DIR}" \
        --output_dir "${OUT_DIR}" \
        --K "${K}" \
        --smooth_window "${SW}" \
        --lambda "${LAMBDA}"

      python3 "${VERIFY}" --config "${OUT_DIR}/scheduler_config.json"

      python3 "${VISUALIZE}" \
        --stage1_output_dir "${OUT_DIR}" \
        --output_dir "${FIG_DIR}"
done

EXPORT="ldm_S3cache/cache_method/Stage1/export_stage1_sweep_csv_ldm.py"
python3 "${EXPORT}" \
  --stage1_output_dir "${BASE_OUT}" \
  --tag_suffix "3runs"

echo ""
echo "================================================================"
echo "✅ Q-LDM sweep 完成"
echo "Results:"
echo "  ${BASE_OUT}/sweep_*"
echo "  ${BASE_FIG}/sweep_*"
echo "  ${BASE_OUT}/csv_exports/lambda/"
echo "================================================================"
