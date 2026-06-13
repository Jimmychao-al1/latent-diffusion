#!/usr/bin/env bash
# Q-LDM Stage 0b unified: collect all supported blocks in one run.

set -euo pipefail

cd "$(git -C /home/jimmy/latent-diffusion rev-parse --show-toplevel)"

export PYTHONPATH="/home/jimmy/TFMQ-DM:/home/jimmy/TFMQ-DM/stable-diffusion:/home/jimmy/latent-diffusion:/home/jimmy/latent-diffusion/src/taming-transformers:/home/jimmy/latent-diffusion/ldm_S3cache/cache_method:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/jimmy/anaconda3/envs/ldm/bin/python}"
SCRIPT="ldm_S3cache/cache_method/b_SVD/collect_features_for_svd_ldm_unified.py"

RESUME="${RESUME:-models/ldm/ffhq256/model.ckpt}"
CALI_CKPT="${CALI_CKPT:-/home/jimmy/TFMQ-DM/cali_ckpt/ffhq256_w8a8.pth}"
NUM_STEPS="${NUM_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TARGET_N="${TARGET_N:-32}"
MAX_BATCHES="${MAX_BATCHES:-64}"

SVD_ROOT="${SVD_ROOT:-ldm_S3cache/cache_method/stage0_output_qldm/b_SVD_unified}"
SIM_NPZ_ROOT="${SIM_NPZ_ROOT:-ldm_S3cache/cache_method/stage0_output_qldm/a_L1_L2_cosine/T_200/v2_latest/result_npz}"
LOG_FILE="${LOG_FILE:-ldm_S3cache/cache_method/stage0_output_qldm/stage0b_qldm_unified.log}"

echo "=== Q-LDM Stage 0b Unified start: $(date) ===" | tee -a "$LOG_FILE"
echo "SVD_ROOT=$SVD_ROOT" | tee -a "$LOG_FILE"
echo "SIM_NPZ_ROOT=$SIM_NPZ_ROOT" | tee -a "$LOG_FILE"

"$PYTHON_BIN" "$SCRIPT" \
  --resume "$RESUME" \
  --num_steps "$NUM_STEPS" \
  --svd_target_N "$TARGET_N" \
  --batch_size "$BATCH_SIZE" \
  --max_batches "$MAX_BATCHES" \
  --svd_output_root "$SVD_ROOT" \
  --similarity_npz_root "$SIM_NPZ_ROOT" \
  --cali_ckpt "$CALI_CKPT" \
  --log_file "$LOG_FILE"

echo "=== Q-LDM Stage 0b Unified done: $(date) ===" | tee -a "$LOG_FILE"
