#!/usr/bin/env bash
# Q-LDM unified evidence: L1/cosine + SVD in one pass (replaces separate 0a + 0b).
set -euo pipefail

cd "$(git -C /home/jimmy/latent-diffusion rev-parse --show-toplevel)"

export PYTHONPATH="/home/jimmy/TFMQ-DM:/home/jimmy/TFMQ-DM/stable-diffusion:/home/jimmy/latent-diffusion:/home/jimmy/latent-diffusion/src/taming-transformers:/home/jimmy/latent-diffusion/ldm_S3cache/cache_method:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/jimmy/anaconda3/envs/ldm/bin/python}"
SCRIPT="ldm_S3cache/cache_method/evidence/collect_evidence_qldm.py"

RESUME="${RESUME:-models/ldm/ffhq256/model.ckpt}"
CALI_CKPT="${CALI_CKPT:-/home/jimmy/TFMQ-DM/cali_ckpt/ffhq256_w8a8.pth}"
NUM_STEPS="${NUM_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_BATCHES="${N_BATCHES:-8}"
TARGET_N="${TARGET_N:-32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ldm_S3cache/cache_method/stage0_output_qldm/evidence_unified}"
LOG_FILE="${LOG_FILE:-ldm_S3cache/cache_method/stage0_output_qldm/evidence_unified.log}"

echo "=== Q-LDM Unified Evidence start: $(date) ===" | tee -a "$LOG_FILE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT" | tee -a "$LOG_FILE"

"$PYTHON" "$SCRIPT" \
  --resume "$RESUME" \
  --num_steps "$NUM_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --n_batches "$N_BATCHES" \
  --svd_target_N "$TARGET_N" \
  --output_root "$OUTPUT_ROOT" \
  --cali_ckpt "$CALI_CKPT" \
  --log_file "$LOG_FILE" \
  2>&1 | tee -a "$LOG_FILE"

echo "=== Q-LDM Unified Evidence done: $(date) ===" | tee -a "$LOG_FILE"
