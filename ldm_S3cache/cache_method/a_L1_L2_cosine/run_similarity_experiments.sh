#!/usr/bin/env bash
# LDM similarity experiments runner (all 25 blocks)
#
# Usage:
#   bash ldm_S3cache/cache_method/a_L1_L2_cosine/run_similarity_experiments.sh
#   bash ldm_S3cache/cache_method/a_L1_L2_cosine/run_similarity_experiments.sh model.output_blocks.0
#
# Optional env overrides:
#   RESUME=models/ldm/ffhq256/model.ckpt
#   NUM_STEPS=100
#   N_SAMPLES=128
#   BATCH_SIZE=32
#   ETA=1.0
#   COLLECT_PER_BATCH=20
#   SAMPLE_STRATEGY=random
#   SIMILARITY_DTYPE=float32
#   OUTPUT_ROOT=ldm_S3cache/cache_method/a_L1_L2_cosine
#   BASE_COLLECTOR_PY=/home/jimmy/diffae/QATcode/cache_method/a_L1_L2_cosine/similarity_calculation.py
#   SAVE_GENERATED_PNGS=1
#   GENERATED_ROOT=ldm_S3cache/cache_method/generated_images
#   CLEAR_GENERATED_DIR=1
#   PYTHON=python

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-python}"
SCRIPT="ldm_S3cache/cache_method/a_L1_L2_cosine/similarity_calculation_ldm.py"

RESUME="${RESUME:-models/ldm/ffhq256/model.ckpt}"
NUM_STEPS="${NUM_STEPS:-200}"
N_SAMPLES="${N_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ETA="${ETA:-0.0}"
COLLECT_PER_BATCH="${COLLECT_PER_BATCH:-20}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-random}"
SIMILARITY_DTYPE="${SIMILARITY_DTYPE:-float32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ldm_S3cache/cache_method/a_L1_L2_cosine}"
BASE_COLLECTOR_PY="${BASE_COLLECTOR_PY:-/home/jimmy/diffae/QATcode/cache_method/a_L1_L2_cosine/similarity_calculation.py}"
SAVE_GENERATED_PNGS="${SAVE_GENERATED_PNGS:-0}"
GENERATED_ROOT="${GENERATED_ROOT:-ldm_S3cache/cache_method/generated_images}"
CLEAR_GENERATED_DIR="${CLEAR_GENERATED_DIR:-1}"

START_FROM_BLOCK="${1:-}"

LOG_ROOT="${OUTPUT_ROOT}/logs"
LOG_DIR="${LOG_ROOT}/v2"
mkdir -p "${LOG_DIR}"

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

if [[ -n "${START_FROM_BLOCK}" ]]; then
  FOUND_START=0
  for BLOCK in "${BLOCKS[@]}"; do
    if [[ "${BLOCK}" == "${START_FROM_BLOCK}" ]]; then
      FOUND_START=1
      break
    fi
  done
  if [[ "${FOUND_START}" != "1" ]]; then
    echo "ERROR: start block not in block list: ${START_FROM_BLOCK}" >&2
    exit 1
  fi
fi

echo "================================================================"
echo "LDM similarity experiments (all blocks)"
echo "  SCRIPT=${SCRIPT}"
echo "  RESUME=${RESUME}"
echo "  NUM_STEPS=${NUM_STEPS}"
echo "  N_SAMPLES=${N_SAMPLES}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  ETA=${ETA}"
echo "  COLLECT_PER_BATCH=${COLLECT_PER_BATCH}"
echo "  SAMPLE_STRATEGY=${SAMPLE_STRATEGY}"
echo "  SIMILARITY_DTYPE=${SIMILARITY_DTYPE}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  TOTAL_BLOCKS=${#BLOCKS[@]}"
if [[ -n "${START_FROM_BLOCK}" ]]; then
  echo "  START_FROM_BLOCK=${START_FROM_BLOCK}"
fi
echo "================================================================"

STARTED=0
DONE=0

for BLOCK in "${BLOCKS[@]}"; do
  if [[ -n "${START_FROM_BLOCK}" && "${STARTED}" != "1" ]]; then
    if [[ "${BLOCK}" == "${START_FROM_BLOCK}" ]]; then
      STARTED=1
    else
      continue
    fi
  fi

  SAFE_NAME="$(echo "${BLOCK}" | tr '.' '_')"
  LOG_FILE="${LOG_DIR}/similarity_${SAFE_NAME}.log"
  CURRENT_BATCH_SIZE="${BATCH_SIZE}"
  if [[ "${BLOCK}" == "model.output_blocks.8" ]]; then
    CURRENT_BATCH_SIZE="16"
  fi

  echo ""
  echo "--------------------------------------------------"
  echo "Running block: ${BLOCK}"
  echo "Batch size: ${CURRENT_BATCH_SIZE}"
  echo "Log: ${LOG_FILE}"
  echo "--------------------------------------------------"

  EXTRA_ARGS=()
  if [[ "${SAVE_GENERATED_PNGS}" == "1" ]]; then
    EXTRA_ARGS+=(
      --save_generated_pngs
      --generated_root "${GENERATED_ROOT}"
      --generated_subdir "similarity_${SAFE_NAME}_T${NUM_STEPS}"
    )
    if [[ "${CLEAR_GENERATED_DIR}" == "1" ]]; then
      EXTRA_ARGS+=(--clear_generated_dir)
    fi
  fi

  "${PYTHON}" "${SCRIPT}" \
    --resume "${RESUME}" \
    --num_steps "${NUM_STEPS}" \
    --n_samples "${N_SAMPLES}" \
    --batch_size "${CURRENT_BATCH_SIZE}" \
    --eta "${ETA}" \
    --target_block "${BLOCK}" \
    --collect_per_batch "${COLLECT_PER_BATCH}" \
    --sample_strategy "${SAMPLE_STRATEGY}" \
    --similarity_dtype "${SIMILARITY_DTYPE}" \
    --output_root "${OUTPUT_ROOT}" \
    --base_collector_py "${BASE_COLLECTOR_PY}" \
    --log_file "${LOG_FILE}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"

  DONE=$((DONE + 1))
done

echo ""
echo "================================================================"
echo "Done. Completed blocks: ${DONE}"
echo "Logs: ${LOG_DIR}"
echo "================================================================"
