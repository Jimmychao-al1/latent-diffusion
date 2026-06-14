#!/usr/bin/env bash
set -uo pipefail

# Q-LDM FID@5K batch: baseline + 3 Stage2 refined cache schedulers.
# Order: baseline first (EMA gate), then K8 champion, K12 mid, K15 FP-comparable.

CKPT="${CKPT:-models/ldm/ffhq256/model.ckpt}"
CONFIG="${CONFIG:-models/ldm/ffhq256/config.yaml}"
CALI_CKPT="${CALI_CKPT:-/home/jimmy/TFMQ-DM/cali_ckpt/ffhq256_w8a8.pth}"
REAL_IMAGE_DIR="${REAL_IMAGE_DIR:-ffhq-dataset/images1024x1024}"
REAL_LMDB="${REAL_LMDB:-}"
OUT_ROOT="${OUT_ROOT:-outputs}"
RESULTS_JSON="${RESULTS_JSON:-results/fid_results_qldm.json}"
N_SAMPLES="${N_SAMPLES:-5000}"

if [[ "${N_SAMPLES}" == "50000" ]]; then
  FID_TAG="50k"
else
  FID_TAG="5k"
fi

RESULTS_ROOT="${RESULTS_ROOT:-ldm_S3cache/cache_method/start_run/results/fid_${FID_TAG}_qldm}"
RUNS_INDEX="${RUNS_INDEX:-${RESULTS_ROOT}/runs_index.jsonl}"
STAGE2_ROOT="${STAGE2_ROOT:-ldm_S3cache/cache_method/Stage2/stage2_output_qldm}"

FORCE_FULL_PREFIX_STEPS="${FORCE_FULL_PREFIX_STEPS:-0}"
FORCE_FULL_RUNTIME_BLOCKS="${FORCE_FULL_RUNTIME_BLOCKS:-}"
SAFETY_FIRST_INPUT_BLOCK="${SAFETY_FIRST_INPUT_BLOCK:-0}"

# Baseline FID gate: stop cache runs if outside ~11.8x (EMA sanity check).
BASELINE_FID_LO="${BASELINE_FID_LO:-11.5}"
BASELINE_FID_HI="${BASELINE_FID_HI:-12.2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "/home/jimmy/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source /home/jimmy/anaconda3/etc/profile.d/conda.sh
  conda activate ldm
fi

export PYTHONPATH="${REPO_ROOT}/TFMQ-DM:${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -n "${REAL_IMAGE_DIR}" && -n "${REAL_LMDB}" ]]; then
  echo "[ERROR] Set only one of REAL_IMAGE_DIR or REAL_LMDB."
  exit 1
fi
if [[ -z "${REAL_IMAGE_DIR}" && -z "${REAL_LMDB}" ]]; then
  echo "[ERROR] REAL_IMAGE_DIR / REAL_LMDB must set one."
  exit 1
fi

if [[ ! -f "${CALI_CKPT}" ]]; then
  echo "[ERROR] missing cali_ckpt: ${CALI_CKPT}" >&2
  exit 1
fi

mkdir -p "${RESULTS_ROOT}"

COMMON_ARGS=(
  --ckpt "${CKPT}"
  --config "${CONFIG}"
  --cali_ckpt "${CALI_CKPT}"
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
BATCH_LOG="${LOG_ROOT}/qldm_fid_batch_$(date +%Y%m%d_%H%M%S).log"

declare -a RUN_LABELS=()
declare -a RUN_FIDS=()
declare -a RUN_STATUSES=()
declare -a RUN_DIRS=()
declare -a RUN_F_RATIOS=(
  "100%"
  "26.02%"
  "26.52%"
  "27.02%"
)

make_run_dir() {
  local name="$1"
  local d t
  d="$(date +%Y%m%d)"
  t="$(date +%m%d_%H)"
  echo "${RESULTS_ROOT}/${d}/${name}/${t}_${name}"
}

read_fid_from_summary() {
  local summary_json="$1"
  python - <<'PY' "${summary_json}"
import json, sys
p = sys.argv[1]
with open(p, encoding="utf-8") as f:
    d = json.load(f)
fid = d.get("fid_5k") or d.get("fid_score") or d.get("fid")
print("" if fid is None else f"{float(fid):.3f}")
PY
}

print_run_config() {
  echo ""
  echo "=== Run 1: baseline ==="
  echo "mode: baseline (no cache)"
  echo "cali_ckpt: ${CALI_CKPT}"
  echo "batch_size: 16, seed: 0, eta: 0.0, steps: 200, n_samples: ${N_SAMPLES}"
  echo ""
  echo "=== Run 2: K8_sw5_lam0.5 ==="
  echo "mode: cache"
  echo "scheduler: ${STAGE2_ROOT}/sweep_K8_sw5_lam0.5/02_refined_blockwise/stage2_refined_scheduler_config.json"
  echo "cali_ckpt: ${CALI_CKPT}"
  echo "batch_size: 16, seed: 0, eta: 0.0, steps: 200, n_samples: ${N_SAMPLES}"
  echo ""
  echo "=== Run 3: K12_sw5_lam0.5 ==="
  echo "mode: cache"
  echo "scheduler: ${STAGE2_ROOT}/sweep_K12_sw5_lam0.5/02_refined_blockwise/stage2_refined_scheduler_config.json"
  echo "cali_ckpt: ${CALI_CKPT}"
  echo "batch_size: 16, seed: 0, eta: 0.0, steps: 200, n_samples: ${N_SAMPLES}"
  echo ""
  echo "=== Run 4: K15_sw3_lam0.5 ==="
  echo "mode: cache"
  echo "scheduler: ${STAGE2_ROOT}/sweep_K15_sw3_lam0.5/02_refined_blockwise/stage2_refined_scheduler_config.json"
  echo "cali_ckpt: ${CALI_CKPT}"
  echo "batch_size: 16, seed: 0, eta: 0.0, steps: 200, n_samples: ${N_SAMPLES}"
}

verify_paths() {
  local ok=0
  local sched
  for sched in \
    "${STAGE2_ROOT}/sweep_K8_sw5_lam0.5/02_refined_blockwise/stage2_refined_scheduler_config.json" \
    "${STAGE2_ROOT}/sweep_K12_sw5_lam0.5/02_refined_blockwise/stage2_refined_scheduler_config.json" \
    "${STAGE2_ROOT}/sweep_K15_sw3_lam0.5/02_refined_blockwise/stage2_refined_scheduler_config.json"
  do
    if [[ ! -f "${sched}" ]]; then
      echo "[ERROR] missing scheduler_json: ${sched}" >&2
      ok=1
    elif ! python -c "import json; json.load(open('${sched}'))" 2>/dev/null; then
      echo "[ERROR] scheduler_json not readable/valid JSON: ${sched}" >&2
      ok=1
    else
      echo "[OK] ${sched}"
    fi
  done
  return "${ok}"
}

run_baseline() {
  local scheduler_name="baseline_no_npz_qldm"
  local run_dir
  run_dir="$(make_run_dir "${scheduler_name}")"
  mkdir -p "${run_dir}"

  echo ""
  echo "============================================================"
  echo "Baseline run (no cache)"
  echo "  run_output_dir: ${run_dir}"
  echo "============================================================"

  if python "${PY_SCRIPT}" \
    --mode baseline \
    --no_npz \
    --scheduler_name "${scheduler_name}" \
    --run-output-dir "${run_dir}" \
    --runs-index-path "${RUNS_INDEX}" \
    --log_file "${run_dir}/run.log" \
    "${COMMON_ARGS[@]}"; then
    RUN_STATUSES+=("success")
  else
    echo "[FAILED] baseline run exited non-zero" >&2
    RUN_STATUSES+=("failed")
    RUN_FIDS+=("N/A")
    RUN_DIRS+=("${run_dir}")
    RUN_LABELS+=("Baseline (no cache)")
    return 1
  fi

  local fid
  fid="$(read_fid_from_summary "${run_dir}/summary.json")"
  RUN_LABELS+=("Baseline (no cache)")
  RUN_FIDS+=("${fid:-N/A}")
  RUN_DIRS+=("${run_dir}")
  echo "  -> FID@5K: ${fid:-N/A}"
  echo "  -> summary: ${run_dir}/summary.json"
  return 0
}

run_cache() {
  local src_name="$1"
  local scheduler_name="$2"
  local scheduler_json="${STAGE2_ROOT}/${src_name}/02_refined_blockwise/stage2_refined_scheduler_config.json"
  local run_dir

  if [[ ! -f "${scheduler_json}" ]]; then
    echo "[ERROR] missing scheduler_json: ${scheduler_json}" >&2
    RUN_LABELS+=("${scheduler_name}")
    RUN_FIDS+=("N/A")
    RUN_STATUSES+=("skipped")
    RUN_DIRS+=("")
    return 1
  fi

  run_dir="$(make_run_dir "${scheduler_name}")"
  mkdir -p "${run_dir}"

  echo ""
  echo "============================================================"
  echo "Cache run: ${scheduler_name}"
  echo "  scheduler_json: ${scheduler_json}"
  echo "  run_output_dir: ${run_dir}"
  echo "============================================================"

  if python "${PY_SCRIPT}" \
    --mode cache \
    --scheduler_json "${scheduler_json}" \
    --scheduler_name "${scheduler_name}" \
    --run-output-dir "${run_dir}" \
    --runs-index-path "${RUNS_INDEX}" \
    --log_file "${run_dir}/run.log" \
    "${COMMON_ARGS[@]}"; then
    RUN_STATUSES+=("success")
  else
    echo "[FAILED] cache run ${scheduler_name} exited non-zero" >&2
    RUN_STATUSES+=("failed")
    RUN_FIDS+=("N/A")
    RUN_DIRS+=("${run_dir}")
    RUN_LABELS+=("${scheduler_name}")
    return 1
  fi

  local fid
  fid="$(read_fid_from_summary "${run_dir}/summary.json")"
  RUN_LABELS+=("${scheduler_name}")
  RUN_FIDS+=("${fid:-N/A}")
  RUN_DIRS+=("${run_dir}")
  echo "  -> FID@5K: ${fid:-N/A}"
  echo "  -> summary: ${run_dir}/summary.json"
  return 0
}

check_baseline_gate() {
  local fid="$1"
  if [[ -z "${fid}" || "${fid}" == "N/A" ]]; then
    echo ""
    echo "[ABORT] Baseline FID unavailable — skipping cache runs."
    return 1
  fi
  python - <<PY "${fid}" "${BASELINE_FID_LO}" "${BASELINE_FID_HI}"
import sys
fid = float(sys.argv[1])
lo, hi = float(sys.argv[2]), float(sys.argv[3])
if lo <= fid <= hi:
    sys.exit(0)
print(f"[ABORT] Baseline FID={fid:.3f} outside [{lo}, {hi}] — EMA fix may be broken; skipping cache runs.")
sys.exit(1)
PY
}

print_summary_table() {
  local baseline_fid="${RUN_FIDS[0]:-N/A}"
  echo ""
  echo "============================================================"
  echo "FID@5K Summary (Q-LDM)"
  echo "============================================================"
  echo "| Config              | FID@5K | vs Baseline | F_ratio | Status  |"
  echo "|---------------------|--------|-------------|---------|---------|"

  local i label fid status f_ratio vs
  for i in 0 1 2 3; do
    label="${RUN_LABELS[$i]:-?}"
    fid="${RUN_FIDS[$i]:-N/A}"
    status="${RUN_STATUSES[$i]:-?}"
    f_ratio="${RUN_F_RATIOS[$i]:-?}"
    if [[ "${i}" -eq 0 ]]; then
      vs="—"
    elif [[ "${fid}" == "N/A" || "${baseline_fid}" == "N/A" ]]; then
      vs="N/A"
    else
      vs="$(python - <<PY "${fid}" "${baseline_fid}"
import sys
fid, base = float(sys.argv[1]), float(sys.argv[2])
pct = (fid - base) / base * 100.0
sign = "+" if pct >= 0 else ""
print(f"{sign}{pct:.1f}%")
PY
)"
    fi
    printf "| %-19s | %-6s | %-11s | %-7s | %-7s |\n" "${label}" "${fid}" "${vs}" "${f_ratio}" "${status}"
  done

  echo ""
  echo "FP LDM reference:"
  echo "  FP LDM baseline FID@5K = 11.84"
  echo "  FP LDM + S3-Cache (K15sw3lam1.0) FID@5K = 11.71 (ΔFID = -1.1%)"
  echo ""
  echo "[DONE] batch log: ${BATCH_LOG}"
  echo "[DONE] runs index (append): ${RUNS_INDEX}"
}

{
  echo "============================================================"
  echo "Q-LDM FID@5K: baseline + 3 Stage2 cache schedulers"
  echo "N_SAMPLES=${N_SAMPLES} batch_size=16 seed=0 eta=0.0 steps=200"
  echo "CALI_CKPT=${CALI_CKPT}"
  echo "RESULTS_ROOT=${RESULTS_ROOT}"
  echo "RUNS_INDEX=${RUNS_INDEX}"
  echo "STAGE2_ROOT=${STAGE2_ROOT}"
  echo "============================================================"

  echo ""
  echo "=== Parameter confirmation ==="
  print_run_config

  echo ""
  echo "=== Path verification ==="
  if ! verify_paths; then
    echo "[ERROR] path verification failed — aborting batch."
    exit 1
  fi

  RUNS_INDEX_LINES_BEFORE=0
  if [[ -f "${RUNS_INDEX}" ]]; then
    RUNS_INDEX_LINES_BEFORE="$(wc -l < "${RUNS_INDEX}")"
  fi

  run_baseline || true
  baseline_fid="${RUN_FIDS[0]:-N/A}"

  if ! check_baseline_gate "${baseline_fid}"; then
    print_summary_table
    exit 2
  fi

  run_cache "sweep_K8_sw5_lam0.5" "K8_sw5_lam0.5" || true
  run_cache "sweep_K12_sw5_lam0.5" "K12_sw5_lam0.5" || true
  run_cache "sweep_K15_sw3_lam0.5" "K15_sw3_lam0.5" || true

  RUNS_INDEX_LINES_AFTER=0
  if [[ -f "${RUNS_INDEX}" ]]; then
    RUNS_INDEX_LINES_AFTER="$(wc -l < "${RUNS_INDEX}")"
  fi
  echo ""
  echo "runs_index.jsonl: ${RUNS_INDEX_LINES_BEFORE} -> ${RUNS_INDEX_LINES_AFTER} lines (append-only)"

  print_summary_table
} 2>&1 | tee -a "${BATCH_LOG}"
