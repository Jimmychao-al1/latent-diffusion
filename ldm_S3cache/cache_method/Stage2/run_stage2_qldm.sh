#!/usr/bin/env bash
# Stage2-LDM (Q-LDM): 3 retained Stage1 sources
# - each source: Pass1(global) -> blockwise threshold -> Pass2(blockwise)
# - verifies threshold JSON and final refined scheduler JSON

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="/home/jimmy/TFMQ-DM:/home/jimmy/TFMQ-DM/stable-diffusion:/home/jimmy/latent-diffusion:/home/jimmy/latent-diffusion/src/taming-transformers:/home/jimmy/latent-diffusion/ldm_S3cache/cache_method:${PYTHONPATH:-}"

PY_REFINE="python ldm_S3cache/cache_method/Stage2/stage2_runtime_refine_ldm.py"
PY_THRESH="python ldm_S3cache/cache_method/Stage2/build_blockwise_thresholds_ldm.py"
PY_VERIFY="python ldm_S3cache/cache_method/Stage2/verify_stage2_ldm.py"

OUT_BASE="${OUT_BASE:-ldm_S3cache/cache_method/Stage2/stage2_output_qldm}"
STAGE1_DIR="${STAGE1_DIR:-ldm_S3cache/cache_method/Stage1/stage1_output_qldm}"
SEED="${SEED:-0}"
CALI_CKPT="${CALI_CKPT:-/home/jimmy/TFMQ-DM/cali_ckpt/ffhq256_w8a8.pth}"

# Pass1 global thresholds
ZONE="${ZONE:-0.02}"
PEAK="${PEAK:-0.08}"

# baseline_908030 defaults
Q_ZONE="${Q_ZONE:-0.90}"
Q_PEAK="${Q_PEAK:-0.80}"
PEAK_OVER_ZONE_MIN="${PEAK_OVER_ZONE_MIN:-1.3}"
THRESH_SOURCE="${THRESH_SOURCE:-ported_from_baseline_908030}"

# diagnostics settings (fixed by request)
EVAL_NUM_IMAGES="${EVAL_NUM_IMAGES:-8}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-1}"

RESUME="${RESUME:-models/ldm/ffhq256/model.ckpt}"
BLOCK_MAP="${BLOCK_MAP:-ldm_block_map_ffhq256.json}"

SRC_BASELINE="${SRC_BASELINE:-${STAGE1_DIR}/sweep_K8_sw5_lam0.5/scheduler_config.json}"
SRC_CAND1="${SRC_CAND1:-${STAGE1_DIR}/sweep_K12_sw5_lam0.5/scheduler_config.json}"
SRC_CAND2="${SRC_CAND2:-${STAGE1_DIR}/sweep_K15_sw3_lam0.5/scheduler_config.json}"

run_one() {
  local src_name="$1"
  local sched="$2"

  if [[ ! -f "${sched}" ]]; then
    echo "ERROR: missing scheduler_config: ${sched}" >&2
    exit 1
  fi

  local base_dir="${OUT_BASE}/${src_name}"
  local pass1_dir="${base_dir}/00_global_refine"
  local th_dir="${base_dir}/01_blockwise_threshold"
  local pass2_dir="${base_dir}/02_refined_blockwise"
  local th_json="${th_dir}/stage2_thresholds_blockwise.json"

  mkdir -p "${pass1_dir}" "${th_dir}" "${pass2_dir}"

  echo ""
  echo "================================================================"
  echo "▶ Stage2-LDM source: ${src_name}"
  echo "  scheduler: ${sched}"
  echo "  output:    ${base_dir}"
  echo "================================================================"

  echo "[1/5] Pass1 global refine"
  ${PY_REFINE} \
    --scheduler_config "${sched}" \
    --output_dir "${pass1_dir}" \
    --seed "${SEED}" \
    --zone_l1_threshold "${ZONE}" \
    --peak_l1_threshold "${PEAK}" \
    --resume "${RESUME}" \
    --block_map "${BLOCK_MAP}" \
    --eval-num-images "${EVAL_NUM_IMAGES}" \
    --eval-chunk-size "${EVAL_CHUNK_SIZE}" \
    --cali_ckpt "${CALI_CKPT}"

  echo "[2/5] Build blockwise threshold (baseline_908030)"
  ${PY_THRESH} \
    --diagnostics "${pass1_dir}/stage2_runtime_diagnostics.json" \
    --output "${th_json}" \
    --q_zone "${Q_ZONE}" \
    --q_peak "${Q_PEAK}" \
    --peak_over_zone_ratio_min "${PEAK_OVER_ZONE_MIN}" \
    --source "${THRESH_SOURCE}"

  echo "[3/5] Verify threshold config"
  ${PY_VERIFY} --threshold-config "${th_json}"

  echo "[4/5] Pass2 blockwise refine"
  ${PY_REFINE} \
    --scheduler_config "${sched}" \
    --output_dir "${pass2_dir}" \
    --seed "${SEED}" \
    --zone_l1_threshold "${ZONE}" \
    --peak_l1_threshold "${PEAK}" \
    --threshold-config "${th_json}" \
    --resume "${RESUME}" \
    --block_map "${BLOCK_MAP}" \
    --eval-num-images "${EVAL_NUM_IMAGES}" \
    --eval-chunk-size "${EVAL_CHUNK_SIZE}" \
    --cali_ckpt "${CALI_CKPT}"

  echo "[5/5] Verify refined scheduler config"
  ${PY_VERIFY} "${pass2_dir}/stage2_refined_scheduler_config.json"

  echo "✅ Completed source: ${src_name}"
}

echo "================================================================"
echo "Stage2-LDM (Q-LDM) — 3 retained Stage1 configs"
echo "OUT_BASE=${OUT_BASE}"
echo "STAGE1_DIR=${STAGE1_DIR}"
echo "SEED=${SEED}"
echo "CALI_CKPT=${CALI_CKPT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "================================================================"

# Step 0: clear old outputs (pre-EMA-fix artifacts from 06-13)
if [[ -d "${OUT_BASE}" ]]; then
  echo "Clearing old outputs under ${OUT_BASE} ..."
  find "${OUT_BASE}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi
mkdir -p "${OUT_BASE}"
echo "Cleared. Ready for fresh Stage2 run."

run_one "sweep_K8_sw5_lam0.5" "${SRC_BASELINE}"
run_one "sweep_K12_sw5_lam0.5" "${SRC_CAND1}"
run_one "sweep_K15_sw3_lam0.5" "${SRC_CAND2}"

echo ""
echo "================================================================"
echo "✅ Stage2-LDM all sources complete"
echo "Results root: ${OUT_BASE}"
echo "================================================================"
