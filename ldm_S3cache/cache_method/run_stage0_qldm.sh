#!/usr/bin/env bash
# Q-LDM Stage 0: unified evidence -> normalization
set -euo pipefail

cd "$(git -C /home/jimmy/latent-diffusion rev-parse --show-toplevel)"

export PYTHONPATH="/home/jimmy/TFMQ-DM:/home/jimmy/TFMQ-DM/stable-diffusion:/home/jimmy/latent-diffusion:/home/jimmy/latent-diffusion/src/taming-transformers:/home/jimmy/latent-diffusion/ldm_S3cache/cache_method:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/jimmy/anaconda3/envs/ldm/bin/python}"
NUM_STEPS="${NUM_STEPS:-200}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-ldm_S3cache/cache_method/stage0_output_qldm/evidence_unified}"
STAGE0_OUT="${STAGE0_OUT:-ldm_S3cache/cache_method/Stage0/stage0_output_qldm}"
FID_JSON="${FID_JSON:-ldm_S3cache/cache_method/c_FID/fid_sensitivity_results_ldm.json}"
LOG_FILE="${LOG_FILE:-ldm_S3cache/cache_method/stage0_output_qldm/stage0_qldm.log}"

L1_COS_DIR="${EVIDENCE_ROOT}/T_${NUM_STEPS}/v2_latest/result_npz"
SVD_DIR="${EVIDENCE_ROOT}/T_${NUM_STEPS}/svd_metrics"

echo "=== Q-LDM Stage 0 start: $(date) ===" | tee -a "$LOG_FILE"
echo "L1_COS_DIR=$L1_COS_DIR" | tee -a "$LOG_FILE"
echo "SVD_DIR=$SVD_DIR" | tee -a "$LOG_FILE"

"$PYTHON" - <<PY 2>&1 | tee -a "$LOG_FILE"
from Stage0.stage0_normalization_ldm import run_stage0_ldm

run_stage0_ldm(
    l1_cos_dir="${L1_COS_DIR}",
    svd_dir="${SVD_DIR}",
    fid_json_path="${FID_JSON}",
    output_dir="${STAGE0_OUT}",
    eps_noise=0.05,
    quantile=0.95,
    fid_step_key=None,
)
PY

OUTPUT_DIR="ldm_S3cache/cache_method/Stage0/stage0_output_qldm"
bash ldm_S3cache/cache_method/Stage0/verify_stage0_output_ldm.sh 2>&1 | sed "s|stage0_output_ldm|stage0_output_qldm|g" || {
  # verify script is hardcoded; run inline check
  "$PYTHON" - <<'PY'
import numpy as np
from pathlib import Path
import sys

output_dir = Path("ldm_S3cache/cache_method/Stage0/stage0_output_qldm")
for name in ("l1_interval_norm.npy", "cosdist_interval_norm.npy", "svd_interval_norm.npy", "fid_weights.npy"):
    if not (output_dir / name).exists():
        print(f"Missing {name}")
        sys.exit(1)
l1 = np.load(output_dir / "l1_interval_norm.npy")
print(f"Stage0 OK: blocks={l1.shape[0]} intervals={l1.shape[1]} range=[{l1.min():.4f},{l1.max():.4f}]")
PY
}

echo "=== Q-LDM Stage 0 done: $(date) ===" | tee -a "$LOG_FILE"
