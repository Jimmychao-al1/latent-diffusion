#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# User Config
# ------------------------------------------------------------
CKPT="${CKPT:-models/ldm/ffhq256/model.ckpt}"
CONFIG="${CONFIG:-models/ldm/ffhq256/config.yaml}"

# Set exactly one.
REAL_IMAGE_DIR="${REAL_IMAGE_DIR:-ffhq-dataset/images1024x1024}"
REAL_LMDB="${REAL_LMDB:-}"

OUT_ROOT="${OUT_ROOT:-outputs}"
RESULTS_JSON="${RESULTS_JSON:-results/fid_results_ldm.json}"
N_SAMPLES="${N_SAMPLES:-5000}" # 5000 or 50000
if [[ "${N_SAMPLES}" == "50000" ]]; then
  FID_TAG="50k"
else
  FID_TAG="5k"
fi
RESULTS_ROOT="${RESULTS_ROOT:-ldm_S3cache/cache_method/start_run/results/fid_${FID_TAG}}"
RUNS_INDEX="${RUNS_INDEX:-${RESULTS_ROOT}/runs_index.jsonl}"

# Stage2 refined scheduler (cache run)
SCHEDULER_JSON="${SCHEDULER_JSON:-ldm_S3cache/cache_method/Stage2/stage2_output_ldm/src_K15_sw3_lam1.0/02_refined_blockwise/stage2_refined_scheduler_config.json}"
SCHEDULER_NAME="${SCHEDULER_NAME:-K15_sw3_lam1.0}"

# Optional runtime overrides (union with Stage2 mask)
FORCE_FULL_PREFIX_STEPS="${FORCE_FULL_PREFIX_STEPS:-0}"
FORCE_FULL_RUNTIME_BLOCKS="${FORCE_FULL_RUNTIME_BLOCKS:-}"
SAFETY_FIRST_INPUT_BLOCK="${SAFETY_FIRST_INPUT_BLOCK:-0}" # 1=true, 0=false

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

make_run_dir() {
  local name="$1"
  local d t
  d="$(date +%Y%m%d)"
  t="$(date +%m%d_%H)"
  echo "${RESULTS_ROOT}/${d}/${name}/${t}_${name}"
}

run_one() {
  local mode="$1"
  local name="$2"
  local run_dir
  run_dir="$(make_run_dir "${name}")"
  mkdir -p "${run_dir}"

  if [[ "${mode}" == "baseline" ]]; then
    python "${PY_SCRIPT}" \
      --mode baseline \
      --scheduler_name "${name}" \
      --no_npz \
      --run-output-dir "${run_dir}" \
      --runs-index-path "${RUNS_INDEX}" \
      --log_file "${run_dir}/run.log" \
      "${COMMON_ARGS[@]}"
  else
    python "${PY_SCRIPT}" \
      --mode cache \
      --scheduler_json "${SCHEDULER_JSON}" \
      --scheduler_name "${name}" \
      --run-output-dir "${run_dir}" \
      --runs-index-path "${RUNS_INDEX}" \
      --log_file "${run_dir}/run.log" \
      "${COMMON_ARGS[@]}"
  fi

  echo "  -> summary: ${run_dir}/summary.json"
}

echo "============================================================"
echo "[1/2] Baseline (LDM, no cache, --no_npz)"
echo "============================================================"
run_one baseline baseline_no_npz

echo "============================================================"
echo "[2/2] Cache (Stage2 refined scheduler)"
echo "============================================================"
run_one cache "${SCHEDULER_NAME}"

# ------------------------------------------------------------
# Add more cache runs by duplicating block below:
#
# SCHEDULER_JSON="ldm_S3cache/cache_method/Stage2/stage2_output_ldm/src_K20_sw3_lam1.0/02_refined_blockwise/stage2_refined_scheduler_config.json"
# run_one cache "K20_sw3_lam1.0"
# ------------------------------------------------------------

echo ""
echo "[DONE] shared results json: ${RESULTS_JSON}"
echo "[DONE] runs index       : ${RUNS_INDEX}"
