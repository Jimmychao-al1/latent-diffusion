#!/usr/bin/env bash
# Q-LDM Stage 0b: SVD evidence for all 25 UNet blocks (A->B->C in-memory).
# Usage:
#   bash ldm_S3cache/cache_method/b_SVD/run_svd_all_blocks_qldm.sh [target_N] [start_from_block]

set -euo pipefail

cd "$(git -C /home/jimmy/latent-diffusion rev-parse --show-toplevel)"

export PYTHONPATH="/home/jimmy/TFMQ-DM:/home/jimmy/TFMQ-DM/stable-diffusion:/home/jimmy/latent-diffusion:/home/jimmy/latent-diffusion/src/taming-transformers:/home/jimmy/latent-diffusion/ldm_S3cache/cache_method:${PYTHONPATH:-}"

TARGET_N="${1:-32}"
START_FROM_BLOCK="${2:-}"
RUN_SINGLE_SCRIPT="ldm_S3cache/cache_method/b_SVD/run_single_block_pipeline_qldm.sh"
LOG_ROOT="ldm_S3cache/cache_method/stage0_output_qldm"
MAIN_LOG="${LOG_ROOT}/stage0b_qldm.log"

mkdir -p "${LOG_ROOT}"

declare -A OVERRIDE_N=(
  # ["model.output_blocks.11"]=16
)
declare -A OVERRIDE_BATCH_SIZE=(
  ["model.output_blocks.8"]="8"
)

BLOCKS=(
  "model.input_blocks.0"
  "model.input_blocks.1"
  "model.input_blocks.2"
  "model.input_blocks.3"
  "model.input_blocks.4"
  "model.input_blocks.5"
  "model.input_blocks.6"
  "model.input_blocks.7"
  "model.input_blocks.8"
  "model.input_blocks.9"
  "model.input_blocks.10"
  "model.input_blocks.11"
  "model.middle_block"
  "model.output_blocks.0"
  "model.output_blocks.1"
  "model.output_blocks.2"
  "model.output_blocks.3"
  "model.output_blocks.4"
  "model.output_blocks.5"
  "model.output_blocks.6"
  "model.output_blocks.7"
  "model.output_blocks.8"
  "model.output_blocks.9"
  "model.output_blocks.10"
  "model.output_blocks.11"
)

if [[ ! -f "$RUN_SINGLE_SCRIPT" ]]; then
  echo "Error: cannot find $RUN_SINGLE_SCRIPT"
  exit 1
fi

if [[ -n "$START_FROM_BLOCK" ]]; then
  FOUND_START=0
  for BLOCK in "${BLOCKS[@]}"; do
    if [[ "$BLOCK" == "$START_FROM_BLOCK" ]]; then
      FOUND_START=1
      break
    fi
  done
  if [[ "$FOUND_START" != "1" ]]; then
    echo "Error: start_from_block not in block list: $START_FROM_BLOCK"
    exit 1
  fi
fi

{
  echo "=== Q-LDM Stage 0b (SVD) start: $(date) ==="
  echo "TARGET_N=${TARGET_N}"
  echo "CALI_CKPT=${CALI_CKPT:-/home/jimmy/TFMQ-DM/cali_ckpt/ffhq256_w8a8.pth}"
  echo "SVD_ROOT=${SVD_ROOT:-ldm_S3cache/cache_method/stage0_output_qldm/b_SVD}"
  if [[ -n "$START_FROM_BLOCK" ]]; then
    echo "Start from: $START_FROM_BLOCK"
  fi
  echo "=============================="
  echo "Q-LDM SVD all-block single pipeline"
  echo "  Total blocks: ${#BLOCKS[@]}"
  echo "  Default target_N: $TARGET_N"
  echo "=============================="

  STARTED=0
  DONE_COUNT=0

  for BLOCK in "${BLOCKS[@]}"; do
    if [[ -n "$START_FROM_BLOCK" && "$STARTED" != "1" ]]; then
      if [[ "$BLOCK" == "$START_FROM_BLOCK" ]]; then
        STARTED=1
      else
        continue
      fi
    fi

    N="${OVERRIDE_N[$BLOCK]:-$TARGET_N}"
    CURRENT_BATCH_SIZE="${OVERRIDE_BATCH_SIZE[$BLOCK]:-16}"

    echo ""
    echo "--------------------------------------------------"
    echo "[$(date +%H:%M:%S)] Running block: $BLOCK  (N=$N, bs=$CURRENT_BATCH_SIZE)"
    echo "--------------------------------------------------"

    BATCH_SIZE="$CURRENT_BATCH_SIZE" bash "$RUN_SINGLE_SCRIPT" "$BLOCK" "$N"
    DONE_COUNT=$((DONE_COUNT + 1))
  done

  echo ""
  echo "=============================="
  echo "All done"
  echo "  Executed blocks: $DONE_COUNT"
  echo "=== Q-LDM Stage 0b done: $(date) ==="
} 2>&1 | tee -a "$MAIN_LOG"
