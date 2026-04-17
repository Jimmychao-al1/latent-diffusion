#!/usr/bin/env bash
# Run single block A->B->C for LDM b_SVD (in-memory).
# Usage: bash ldm_S3cache/cache_method/b_SVD/run_single_block_pipeline_ldm.sh "model.output_blocks.11" [N]

set -euo pipefail

if [[ -z "${1:-}" ]]; then
  echo "Usage: $0 <block_name> [target_N]"
  echo "Example: $0 model.output_blocks.11 16"
  exit 1
fi

BLOCK="$1"
TARGET_N="${2:-32}"
SAFE_NAME=$(echo "$BLOCK" | tr '.' '_')

DEFAULT_PYTHON="/home/jimmy/anaconda3/envs/ldm/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON}
else
  PYTHON_BIN=${PYTHON_BIN:-python3}
fi

NUM_STEPS=${NUM_STEPS:-200}
BATCH_SIZE=${BATCH_SIZE:-16}
RESUME=${RESUME:-models/ldm/ffhq256/model.ckpt}
SVD_ROOT=${SVD_ROOT:-ldm_S3cache/cache_method/b_SVD}
LOG_DIR="${SVD_ROOT}/logs"
mkdir -p "$LOG_DIR"

SIM_NPZ="ldm_S3cache/cache_method/a_L1_L2_cosine/T_${NUM_STEPS}/v2_latest/result_npz/${SAFE_NAME}.npz"

echo "=============================="
echo "LDM SVD single-block pipeline (A->B->C in-memory)"
echo "  Block: $BLOCK"
echo "  N: $TARGET_N"
echo "  T: $NUM_STEPS"
echo "  Similarity NPZ: $SIM_NPZ"
echo "=============================="

if [[ ! -f "$SIM_NPZ" ]]; then
  echo "Error: similarity npz not found: $SIM_NPZ"
  echo "Please run a_L1_L2_cosine first."
  exit 1
fi

"$PYTHON_BIN" ldm_S3cache/cache_method/b_SVD/collect_features_for_svd_ldm.py \
  --resume "$RESUME" \
  --num_steps "$NUM_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --svd_target_block "$BLOCK" \
  --svd_target_N "$TARGET_N" \
  --svd_output_root "$SVD_ROOT" \
  --in_memory_pipeline \
  --representative-t -1 \
  --energy-threshold 0.98 \
  --similarity_npz "$SIM_NPZ" \
  --log_file "$LOG_DIR/svd_feature_${SAFE_NAME}.log"

echo ""
echo "=============================="
echo "Done"
echo "  SVD JSON:    ${SVD_ROOT}/T_${NUM_STEPS}/svd_metrics/${SAFE_NAME}.json"
echo "  Correlation: ${SVD_ROOT}/T_${NUM_STEPS}/correlation/${SAFE_NAME}.json"
echo "  Figures:     ${SVD_ROOT}/T_${NUM_STEPS}/correlation/figures/${SAFE_NAME}_*.png"
echo "=============================="
