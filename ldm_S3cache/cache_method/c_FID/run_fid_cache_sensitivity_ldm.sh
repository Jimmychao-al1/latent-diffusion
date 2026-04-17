#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PYTHON="/home/jimmy/anaconda3/envs/ldm/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON}
else
  PYTHON_BIN=${PYTHON_BIN:-python3}
fi
SCRIPT="ldm_S3cache/cache_method/c_FID/fid_cache_sensitivity_ldm.py"

RESUME=${RESUME:-models/ldm/ffhq256/model.ckpt}
DATASET_TAG=${DATASET_TAG:-ffhq256}
REAL_IMAGE_DIR=${REAL_IMAGE_DIR:-ffhq-dataset/images1024x1024}
BLOCK_MAP=${BLOCK_MAP:-ldm_block_map_ffhq256.json}

NUM_STEPS=${NUM_STEPS:-200}
NUM_IMAGES=${NUM_IMAGES:-1000}
BATCH_SIZE=${BATCH_SIZE:-16}
FID_BATCH_SIZE=${FID_BATCH_SIZE:-32}
FID_DIMS=${FID_DIMS:-192}
IMG_SIZE=${IMG_SIZE:-256}
SEED=${SEED:-0}
ETA=${ETA:-0}
K=${K:-3}

RUN_BASELINE=${RUN_BASELINE:-1}
RUN_ALL_LAYERS=${RUN_ALL_LAYERS:-1}
TARGET_LAYER=${TARGET_LAYER:-}
CLEANUP_AT_END=${CLEANUP_AT_END:-1}

OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/c_fid}
RESULTS_JSON=${RESULTS_JSON:-ldm_S3cache/cache_method/c_FID/fid_sensitivity_results_ldm.json}
LOG_FILE=${LOG_FILE:-ldm_S3cache/cache_method/c_FID/fid_cache_sensitivity_ldm.log}

COMMON_ARGS=(
  --resume "$RESUME"
  --dataset_tag "$DATASET_TAG"
  --output_root "$OUTPUT_ROOT"
  --results_json "$RESULTS_JSON"
  --log_file "$LOG_FILE"
  --real_image_dir "$REAL_IMAGE_DIR"
  --block_map "$BLOCK_MAP"
  --num_steps "$NUM_STEPS"
  --num_images "$NUM_IMAGES"
  --batch_size "$BATCH_SIZE"
  --fid_batch_size "$FID_BATCH_SIZE"
  --fid_dims "$FID_DIMS"
  --img_size "$IMG_SIZE"
  --seed "$SEED"
  --eta "$ETA"
)

DATASET_OUT_DIR="${OUTPUT_ROOT}/${DATASET_TAG}"
REAL_READY=0

run_one() {
  local extra_args=("$@")
  local call_args=("${COMMON_ARGS[@]}")

  if [[ "$REAL_READY" == "1" ]]; then
    call_args+=(--skip_real_prepare)
  fi

  "$PYTHON_BIN" "$SCRIPT" "${call_args[@]}" "${extra_args[@]}"
  REAL_READY=1
}

if [[ "$RUN_BASELINE" == "1" ]]; then
  echo "[c_FID] Running baseline..."
  run_one --baseline
fi

if [[ "$RUN_ALL_LAYERS" != "1" ]]; then
  if [[ -z "$TARGET_LAYER" ]]; then
    echo "TARGET_LAYER is required when RUN_ALL_LAYERS=0"
    exit 1
  fi
  if [[ "$K" != "3" && "$K" != "4" && "$K" != "5" ]]; then
    echo "K must be one of: 3,4,5 (got: $K)"
    exit 1
  fi
  echo "[c_FID] Running single layer: ${TARGET_LAYER} (k=${K})"
  run_one --k "$K" --layer "$TARGET_LAYER"

  if [[ "$CLEANUP_AT_END" == "1" && -d "$DATASET_OUT_DIR" ]]; then
    echo "[c_FID] Cleanup dataset output dir: ${DATASET_OUT_DIR}"
    rm -rf "$DATASET_OUT_DIR"
  fi
  exit 0
fi

for k in 3 4 5; do
  echo "[c_FID] Running all layers with k=${k}"
  run_one --k "$k"
done

if [[ "$CLEANUP_AT_END" == "1" && -d "$DATASET_OUT_DIR" ]]; then
  echo "[c_FID] Cleanup dataset output dir: ${DATASET_OUT_DIR}"
  rm -rf "$DATASET_OUT_DIR"
fi

echo "[c_FID] Completed. results=${RESULTS_JSON}"
