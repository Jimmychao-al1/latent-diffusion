#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# User Config
# ------------------------------------------------------------
CKPT="${CKPT:-models/ldm/ffhq256/model.ckpt}"
CONFIG="${CONFIG:-configs/latent-diffusion/ffhq-ldm-vq-4.yaml}"

# Set exactly one:
REAL_IMAGE_DIR="${REAL_IMAGE_DIR:-}"
REAL_LMDB="${REAL_LMDB:-}"

OUT_ROOT="${OUT_ROOT:-outputs}"
RESULTS_JSON="${RESULTS_JSON:-results/fid_results_ldm.json}"

# Stage2 refined scheduler (cache run)
SCHEDULER_JSON="${SCHEDULER_JSON:-ldm_S3cache/cache_method/Stage2/stage2_output_ldm/src_K15_sw3_lam1.0/02_refined_blockwise/stage2_refined_scheduler_config.json}"
SCHEDULER_NAME="${SCHEDULER_NAME:-K15_sw3_lam1.0}"

# ------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "/home/jimmy/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source /home/jimmy/anaconda3/etc/profile.d/conda.sh
  conda activate ldm
fi

if [[ -n "${REAL_IMAGE_DIR}" && -n "${REAL_LMDB}" ]]; then
  echo "[ERROR] Set only one of REAL_IMAGE_DIR or REAL_LMDB."
  exit 1
fi
if [[ -z "${REAL_IMAGE_DIR}" && -z "${REAL_LMDB}" ]]; then
  echo "[ERROR] REAL_IMAGE_DIR / REAL_LMDB must set one."
  exit 1
fi

COMMON_ARGS=(
  --ckpt "${CKPT}"
  --config "${CONFIG}"
  --out_root "${OUT_ROOT}"
  --results_json "${RESULTS_JSON}"
)
if [[ -n "${REAL_IMAGE_DIR}" ]]; then
  COMMON_ARGS+=( --real_image_dir "${REAL_IMAGE_DIR}" )
else
  COMMON_ARGS+=( --real_lmdb "${REAL_LMDB}" )
fi

PY_SCRIPT="ldm_S3cache/cache_method/start_run/sample_stage2_cache_scheduler_ldm.py"

echo "============================================================"
echo "[1/2] Baseline (LDM, no cache, --no_npz)"
echo "============================================================"
python "${PY_SCRIPT}" \
  --mode baseline \
  --scheduler_name baseline_no_npz \
  --no_npz \
  "${COMMON_ARGS[@]}"

echo "============================================================"
echo "[2/2] Cache (Stage2 refined scheduler)"
echo "============================================================"
python "${PY_SCRIPT}" \
  --mode cache \
  --scheduler_json "${SCHEDULER_JSON}" \
  --scheduler_name "${SCHEDULER_NAME}" \
  "${COMMON_ARGS[@]}"

# ------------------------------------------------------------
# Add more cache runs by duplicating this block:
#
# python "${PY_SCRIPT}" \
#   --mode cache \
#   --scheduler_json "ldm_S3cache/cache_method/Stage2/stage2_output_ldm/src_.../02_refined_blockwise/stage2_refined_scheduler_config.json" \
#   --scheduler_name "K20_sw3_lam1.0" \
#   "${COMMON_ARGS[@]}"
# ------------------------------------------------------------

echo "[DONE] shared results json: ${RESULTS_JSON}"
