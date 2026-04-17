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
SEED=${SEED:-0}
ETA=${ETA:-0}
K=${K:-3}

RUN_BASELINE=${RUN_BASELINE:-1}
RUN_ALL_LAYERS=${RUN_ALL_LAYERS:-1}
TARGET_LAYER=${TARGET_LAYER:-}

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
  --seed "$SEED"
  --eta "$ETA"
)

if [[ "$RUN_BASELINE" == "1" ]]; then
  echo "[c_FID] Running baseline..."
  "$PYTHON_BIN" "$SCRIPT" "${COMMON_ARGS[@]}" --baseline
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
  "$PYTHON_BIN" "$SCRIPT" "${COMMON_ARGS[@]}" --k "$K" --layer "$TARGET_LAYER"
  exit 0
fi

layers=()
for i in $(seq 0 11); do
  layers+=("encoder_layer_${i}")
done
layers+=("middle_layer")
for i in $(seq 0 11); do
  layers+=("decoder_layer_${i}")
done

for k in 3 4 5; do
  echo "[c_FID] Running all ${#layers[@]} layers with k=${k}"
  for idx in "${!layers[@]}"; do
    layer="${layers[$idx]}"
    echo "[k=${k}] [$((idx+1))/${#layers[@]}] layer=${layer}"
    "$PYTHON_BIN" "$SCRIPT" "${COMMON_ARGS[@]}" --k "$k" --layer "$layer"
  done
done

echo "[c_FID] Completed. results=${RESULTS_JSON}"
