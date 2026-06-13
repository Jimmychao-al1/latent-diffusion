#!/usr/bin/env bash
# Q-LDM Stage 0a: L1/Cosine similarity for all 25 blocks
set -euo pipefail

cd "$(git -C /home/jimmy/latent-diffusion rev-parse --show-toplevel)"

export PYTHONPATH="/home/jimmy/TFMQ-DM:/home/jimmy/TFMQ-DM/stable-diffusion:/home/jimmy/latent-diffusion:/home/jimmy/latent-diffusion/src/taming-transformers:/home/jimmy/latent-diffusion/ldm_S3cache/cache_method:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/jimmy/anaconda3/envs/ldm/bin/python}"
SCRIPT="ldm_S3cache/cache_method/a_L1_L2_cosine/similarity_calculation_ldm.py"
CALI_CKPT="${CALI_CKPT:-/home/jimmy/TFMQ-DM/cali_ckpt/ffhq256_w8a8.pth}"
RESUME="${RESUME:-models/ldm/ffhq256/model.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ldm_S3cache/cache_method/stage0_output_qldm/a_L1_L2_cosine}"
NUM_STEPS="${NUM_STEPS:-200}"
N_SAMPLES="${N_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"

LOG_ROOT="${OUTPUT_ROOT}/logs/v2"
mkdir -p "${LOG_ROOT}"

BLOCKS=(
  "model.input_blocks.0" "model.input_blocks.1" "model.input_blocks.2"
  "model.input_blocks.3" "model.input_blocks.4" "model.input_blocks.5"
  "model.input_blocks.6" "model.input_blocks.7" "model.input_blocks.8"
  "model.input_blocks.9" "model.input_blocks.10" "model.input_blocks.11"
  "model.middle_block"
  "model.output_blocks.0" "model.output_blocks.1" "model.output_blocks.2"
  "model.output_blocks.3" "model.output_blocks.4" "model.output_blocks.5"
  "model.output_blocks.6" "model.output_blocks.7" "model.output_blocks.8"
  "model.output_blocks.9" "model.output_blocks.10" "model.output_blocks.11"
)

echo "=== Q-LDM Stage 0a (L1/Cosine) start: $(date) ==="
echo "CALI_CKPT=${CALI_CKPT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

for BLOCK in "${BLOCKS[@]}"; do
  SAFE_NAME="$(echo "${BLOCK}" | tr '.' '_')"
  LOG_FILE="${LOG_ROOT}/similarity_${SAFE_NAME}.log"
  CURRENT_BATCH_SIZE="${BATCH_SIZE}"
  if [[ "${BLOCK}" == "model.output_blocks.8" ]]; then
    CURRENT_BATCH_SIZE="8"
  fi

  echo "[$(date +%H:%M:%S)] Running ${BLOCK} (bs=${CURRENT_BATCH_SIZE})"
  "${PYTHON}" "${SCRIPT}" \
    --resume "${RESUME}" \
    --num_steps "${NUM_STEPS}" \
    --n_samples "${N_SAMPLES}" \
    --batch_size "${CURRENT_BATCH_SIZE}" \
    --eta 0.0 \
    --target_block "${BLOCK}" \
    --collect_per_batch 20 \
    --sample_strategy random \
    --similarity_dtype float32 \
    --output_root "${OUTPUT_ROOT}" \
    --cali_ckpt "${CALI_CKPT}" \
    --log_file "${LOG_FILE}" \
    2>&1 | tee -a "${LOG_FILE}"
done

echo "=== Q-LDM Stage 0a done: $(date) ==="
