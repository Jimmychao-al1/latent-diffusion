#!/usr/bin/env bash
# Run all LDM blocks with single-block in-memory pipeline (A->B->C).
# Usage:
#   bash ldm_S3cache/cache_method/b_SVD/run_svd_all_blocks_single_pipeline_ldm.sh [target_N] [start_from_block]

set -euo pipefail

TARGET_N="${1:-32}"
START_FROM_BLOCK="${2:-}"
RUN_SINGLE_SCRIPT="ldm_S3cache/cache_method/b_SVD/run_single_block_pipeline_ldm.sh"

# Optional per-block N override for memory-heavy blocks.
# Keep the associative array declared even when empty.
declare -A OVERRIDE_N=(
  # ["model.output_blocks.11"]=16
)

# LDM FFHQ256 block list (12 + 1 + 12 = 25)
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

echo "=============================="
echo "LDM SVD all-block single pipeline"
echo "  Total blocks: ${#BLOCKS[@]}"
echo "  Default target_N: $TARGET_N"
if [[ -n "$START_FROM_BLOCK" ]]; then
  echo "  Start from: $START_FROM_BLOCK"
fi
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
  echo ""
  echo "--------------------------------------------------"
  echo "Running block: $BLOCK  (N=$N)"
  echo "--------------------------------------------------"

  bash "$RUN_SINGLE_SCRIPT" "$BLOCK" "$N"
  DONE_COUNT=$((DONE_COUNT + 1))
done

echo ""
echo "=============================="
echo "All done"
echo "  Executed blocks: $DONE_COUNT"
echo "=============================="
