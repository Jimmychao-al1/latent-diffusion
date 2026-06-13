#!/usr/bin/env bash
set -euo pipefail

# FP LDM + S3-Cache: run cache FID for all three Stage2 refined schedulers.
# Order: best combo first (K15_sw3_lam1.0), then the other two candidates.

CKPT="${CKPT:-models/ldm/ffhq256/model.ckpt}"
CONFIG="${CONFIG:-models/ldm/ffhq256/config.yaml}"
REAL_IMAGE_DIR="${REAL_IMAGE_DIR:-ffhq-dataset/images1024x1024}"
REAL_LMDB="${REAL_LMDB:-}"
OUT_ROOT="${OUT_ROOT:-outputs}"
RESULTS_JSON="${RESULTS_JSON:-results/fid_results_ldm.json}"
N_SAMPLES="${N_SAMPLES:-5000}"

if [[ "${N_SAMPLES}" == "50000" ]]; then
  FID_TAG="50k"
else
  FID_TAG="5k"
fi

RESULTS_ROOT="${RESULTS_ROOT:-ldm_S3cache/cache_method/start_run/results/fid_${FID_TAG}}"
RUNS_INDEX="${RUNS_INDEX:-${RESULTS_ROOT}/runs_index.jsonl}"
STAGE2_ROOT="${STAGE2_ROOT:-ldm_S3cache/cache_method/Stage2/stage2_output_ldm}"

FORCE_FULL_PREFIX_STEPS="${FORCE_FULL_PREFIX_STEPS:-0}"
FORCE_FULL_RUNTIME_BLOCKS="${FORCE_FULL_RUNTIME_BLOCKS:-}"
SAFETY_FIRST_INPUT_BLOCK="${SAFETY_FIRST_INPUT_BLOCK:-0}"

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

mkdir -p "${RESULTS_ROOT}"

COMMON_ARGS=(
  --ckpt "${CKPT}"
  --config "${CONFIG}"
  --out_root "${OUT_ROOT}"
  --results_json "${RESULTS_JSON}"
  --n_samples "${N_SAMPLES}"
)
if [[ -n "${REAL_IMAGE_DIR}" ]]; then
  COMMON_ARGS+=( --real_image_dir "${REAL_IMAGE_DIR}" )
else
  COMMON_ARGS+=( --real_lmdb "${REAL_LMDB}" )
fi

if [[ "${FORCE_FULL_PREFIX_STEPS}" != "0" ]]; then
  COMMON_ARGS+=( --force-full-prefix-steps "${FORCE_FULL_PREFIX_STEPS}" )
fi
if [[ -n "${FORCE_FULL_RUNTIME_BLOCKS}" ]]; then
  COMMON_ARGS+=( --force-full-runtime-blocks "${FORCE_FULL_RUNTIME_BLOCKS}" )
fi
if [[ "${SAFETY_FIRST_INPUT_BLOCK}" == "1" ]]; then
  COMMON_ARGS+=( --safety-first-input-block )
fi

PY_SCRIPT="ldm_S3cache/cache_method/start_run/sample_stage2_cache_scheduler_ldm.py"
LOG_ROOT="${RESULTS_ROOT}/batch_logs"
mkdir -p "${LOG_ROOT}"
BATCH_LOG="${LOG_ROOT}/fp_stage2_cache_all_$(date +%Y%m%d_%H%M%S).log"

make_run_dir() {
  local name="$1"
  local d t
  d="$(date +%Y%m%d)"
  t="$(date +%m%d_%H)"
  echo "${RESULTS_ROOT}/${d}/${name}/${t}_${name}"
}

run_cache() {
  local src_name="$1"
  local scheduler_name="$2"
  local scheduler_json="${STAGE2_ROOT}/${src_name}/02_refined_blockwise/stage2_refined_scheduler_config.json"
  local run_dir

  if [[ ! -f "${scheduler_json}" ]]; then
    echo "[ERROR] missing scheduler_json: ${scheduler_json}" >&2
    exit 1
  fi

  run_dir="$(make_run_dir "${scheduler_name}")"
  mkdir -p "${run_dir}"

  echo ""
  echo "============================================================"
  echo "Cache run: ${scheduler_name}"
  echo "  scheduler_json: ${scheduler_json}"
  echo "  run_output_dir: ${run_dir}"
  echo "============================================================"

  python "${PY_SCRIPT}" \
    --mode cache \
    --scheduler_json "${scheduler_json}" \
    --scheduler_name "${scheduler_name}" \
    --run-output-dir "${run_dir}" \
    --runs-index-path "${RUNS_INDEX}" \
    --log_file "${run_dir}/run.log" \
    "${COMMON_ARGS[@]}"

  echo "  -> summary: ${run_dir}/summary.json"
}

{
  echo "============================================================"
  echo "FP LDM + S3-Cache: all Stage2 cache schedulers"
  echo "N_SAMPLES=${N_SAMPLES} SEED=0 (hardcoded in python)"
  echo "RESULTS_ROOT=${RESULTS_ROOT}"
  echo "RUNS_INDEX=${RUNS_INDEX}"
  echo "============================================================"

  run_cache "src_K15_sw3_lam1.0" "K15_sw3_lam1.0"
  run_cache "src_K15_sw3_lam0.5" "K15_sw3_lam0.5"
  run_cache "src_K20_sw3_lam1.0" "K20_sw3_lam1.0"

  echo ""
  echo "[DONE] all three cache runs complete"
  echo "[DONE] shared results json: ${RESULTS_JSON}"
  echo "[DONE] runs index       : ${RUNS_INDEX}"
} 2>&1 | tee -a "${BATCH_LOG}"
