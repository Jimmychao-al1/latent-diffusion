#!/usr/bin/env bash
# Stage-1 (LDM) sweep
# - K: {10,15,20,25,30,35}
# - smooth_window: {3,5,9}
# - lambda: 1.0
#
# Usage:
#   bash ldm_S3cache/cache_method/Stage1/run_stage1_sweep_ldm.sh
# Optional overrides:
#   STAGE0_DIR=... BASE_OUT=... BASE_FIG=... LAMBDA=1.0 bash ...
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

STAGE0_DIR="${STAGE0_DIR:-ldm_S3cache/cache_method/Stage0/stage0_output_ldm}"
BASE_OUT="${BASE_OUT:-ldm_S3cache/cache_method/Stage1/stage1_output_ldm}"
BASE_FIG="${BASE_FIG:-ldm_S3cache/cache_method/Stage1/stage1_figures_ldm}"

SCHEDULER="ldm_S3cache/cache_method/Stage1/stage1_scheduler_ldm.py"
VERIFY="ldm_S3cache/cache_method/Stage1/verify_scheduler_ldm.py"
VISUALIZE="ldm_S3cache/cache_method/Stage1/visualize_stage1_ldm.py"

K_LIST=(10 15 20 25 30 35)
SW_LIST=(3 5 9)
LAMBDA="${LAMBDA:-1.0}"

TOTAL=$(( ${#K_LIST[@]} * ${#SW_LIST[@]} ))
CUR=0

echo "================================================================"
echo "Stage-1 LDM sweep"
echo "STAGE0_DIR=${STAGE0_DIR}"
echo "BASE_OUT=${BASE_OUT}"
echo "BASE_FIG=${BASE_FIG}"
echo "K_LIST=( ${K_LIST[*]} )"
echo "SW_LIST=( ${SW_LIST[*]} )"
echo "LAMBDA=${LAMBDA}"
echo "TOTAL_RUNS=${TOTAL}"
echo "================================================================"

for K in "${K_LIST[@]}"; do
  for SW in "${SW_LIST[@]}"; do
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
done

echo ""
echo "================================================================"
echo "✅ sweep 完成"
echo "Results:"
echo "  ${BASE_OUT}/sweep_*"
echo "  ${BASE_FIG}/sweep_*"
echo "================================================================"
