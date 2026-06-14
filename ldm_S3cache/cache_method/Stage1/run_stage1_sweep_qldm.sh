#!/usr/bin/env bash
# Stage-1 (Q-LDM) retained sweep — 3 configs for Stage 2 FID validation
#
# 1. K8_sw5_lam0.5   — sweep champion (lowest cost_J)
# 2. K12_sw5_lam0.5  — mid-K between 8 and 15
# 3. K15_sw3_lam0.5  — same K/sw as FP LDM best, lambda=0.5 for Q-LDM evidence
#
# Usage:
#   bash ldm_S3cache/cache_method/Stage1/run_stage1_sweep_qldm.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

STAGE0_DIR="${STAGE0_DIR:-ldm_S3cache/cache_method/Stage0/stage0_output_qldm}"
BASE_OUT="${BASE_OUT:-ldm_S3cache/cache_method/Stage1/stage1_output_qldm}"
BASE_FIG="${BASE_FIG:-ldm_S3cache/cache_method/Stage1/stage1_figures_qldm}"

SCHEDULER="ldm_S3cache/cache_method/Stage1/stage1_scheduler_ldm.py"
VERIFY="ldm_S3cache/cache_method/Stage1/verify_scheduler_ldm.py"
VISUALIZE="ldm_S3cache/cache_method/Stage1/visualize_stage1_ldm.py"
SUMMARIZE="ldm_S3cache/cache_method/Stage1/summarize_stage1_sweep_qldm.py"

# K SW lambda
SWEEP_SPECS=(
  "8 5 0.5"
  "12 5 0.5"
  "15 3 0.5"
)

TOTAL=${#SWEEP_SPECS[@]}
CUR=0
FAILS=0

echo "================================================================"
echo "Stage-1 Q-LDM retained sweep (3 configs)"
echo "STAGE0_DIR=${STAGE0_DIR}"
echo "BASE_OUT=${BASE_OUT}"
echo "SWEEP_SPECS=( ${SWEEP_SPECS[*]} )"
echo "================================================================"

# Step 0: clear old outputs (keep directory)
if [[ -d "${BASE_OUT}" ]]; then
  echo "Clearing old outputs under ${BASE_OUT} ..."
  find "${BASE_OUT}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi
if [[ -d "${BASE_FIG}" ]]; then
  echo "Clearing old figures under ${BASE_FIG} ..."
  find "${BASE_FIG}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi
mkdir -p "${BASE_OUT}" "${BASE_FIG}"

for SPEC in "${SWEEP_SPECS[@]}"; do
  read -r K SW LAMBDA <<< "${SPEC}"
  CUR=$((CUR + 1))
  TAG="K${K}_sw${SW}_lam${LAMBDA}"
  OUT_DIR="${BASE_OUT}/sweep_${TAG}"
  FIG_DIR="${BASE_FIG}/sweep_${TAG}"
  mkdir -p "${OUT_DIR}" "${FIG_DIR}"

  echo ""
  echo "[$CUR/$TOTAL] ▶ ${TAG}"

  if ! python3 "${SCHEDULER}" \
    --stage0_dir "${STAGE0_DIR}" \
    --output_dir "${OUT_DIR}" \
    --K "${K}" \
    --smooth_window "${SW}" \
    --lambda "${LAMBDA}"; then
    echo "  ✗ scheduler failed for ${TAG}"
    FAILS=$((FAILS + 1))
    continue
  fi

  if ! python3 "${VERIFY}" --config "${OUT_DIR}/scheduler_config.json"; then
    echo "  ✗ verify failed for ${TAG}"
    FAILS=$((FAILS + 1))
    continue
  fi

  python3 "${VISUALIZE}" \
    --stage1_output_dir "${OUT_DIR}" \
    --output_dir "${FIG_DIR}" || true
done

python3 "${SUMMARIZE}" --stage1_output_dir "${BASE_OUT}"

echo ""
echo "================================================================"
echo "✅ Q-LDM retained sweep finished (${CUR} attempted, ${FAILS} failed)"
echo "  ${BASE_OUT}/sweep_K8_sw5_lam0.5"
echo "  ${BASE_OUT}/sweep_K12_sw5_lam0.5"
echo "  ${BASE_OUT}/sweep_K15_sw3_lam0.5"
echo "  ${BASE_OUT}/sweep_summary.csv"
echo "================================================================"
