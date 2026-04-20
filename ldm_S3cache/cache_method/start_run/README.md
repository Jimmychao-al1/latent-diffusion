# start_run (LDM)

Stage2 之後的 LDM 正式 FID 採樣入口，支援 baseline 與 cache scheduler，並輸出 DiffAE 風格 run artifacts。

## 固定規格

| item | value |
|---|---|
| `ddim_steps` | `200` |
| `eta` | `0` |
| `seed` | `0` |
| `batch_size` | `32` |
| `n_samples` | `5000` |
| `fid_dims` | `2048` |
| `dataset_tag` | `ffhq256` |

## 檔案

- `sample_stage2_cache_scheduler_ldm.py`
  - `--mode baseline`：原生 LDM（需 `--no_npz`）
  - `--mode cache`：載入 Stage2 refined scheduler + runtime cache hook
- `runFidWithStage2SchedulerLdm.sh`
  - 固定執行：baseline -> cache
  - 自動產出 per-run artifacts 與 `runs_index.jsonl`

## 重要：時間軸對齊

cache scheduler 在 LDM 端採用 **loop index** 判定：

- loop index `0..199`
- `0` 為第一個 denoise step
- `199` 為最後一個 denoise step

程式會先建立 `raw_ddim_timestep -> loop_index` 映射，再以 loop index 判定 full/reuse。

## Stage2 JSON 路徑慣例

```
ldm_S3cache/cache_method/Stage2/stage2_output_ldm/<src>/02_refined_blockwise/stage2_refined_scheduler_config.json
```

## 單次執行（Python）

### baseline

```bash
python ldm_S3cache/cache_method/start_run/sample_stage2_cache_scheduler_ldm.py \
  --mode baseline \
  --no_npz \
  --ckpt models/ldm/ffhq256/model.ckpt \
  --config models/ldm/ffhq256/config.yaml \
  --real_image_dir ffhq-dataset/images1024x1024 \
  --scheduler_name baseline_no_npz \
  --run-output-dir ldm_S3cache/cache_method/start_run/results/fid_5k/$(date +%Y%m%d)/baseline_no_npz/$(date +%m%d_%H)_baseline_no_npz \
  --runs-index-path ldm_S3cache/cache_method/start_run/results/fid_5k/runs_index.jsonl \
  --results_json results/fid_results_ldm.json
```

### cache

```bash
python ldm_S3cache/cache_method/start_run/sample_stage2_cache_scheduler_ldm.py \
  --mode cache \
  --ckpt models/ldm/ffhq256/model.ckpt \
  --config models/ldm/ffhq256/config.yaml \
  --real_image_dir ffhq-dataset/images1024x1024 \
  --scheduler_json ldm_S3cache/cache_method/Stage2/stage2_output_ldm/src_K15_sw3_lam1.0/02_refined_blockwise/stage2_refined_scheduler_config.json \
  --scheduler_name K15_sw3_lam1.0 \
  --run-output-dir ldm_S3cache/cache_method/start_run/results/fid_5k/$(date +%Y%m%d)/K15_sw3_lam1.0/$(date +%m%d_%H)_K15_sw3_lam1.0 \
  --runs-index-path ldm_S3cache/cache_method/start_run/results/fid_5k/runs_index.jsonl \
  --results_json results/fid_results_ldm.json
```

## 一鍵執行（Shell）

```bash
bash ldm_S3cache/cache_method/start_run/runFidWithStage2SchedulerLdm.sh
```

可用環境變數：

- `CKPT`
- `CONFIG`
- `REAL_IMAGE_DIR` / `REAL_LMDB`（二擇一）
- `SCHEDULER_JSON`
- `SCHEDULER_NAME`
- `RESULTS_JSON`
- `RESULTS_ROOT`
- `RUNS_INDEX`
- `FORCE_FULL_PREFIX_STEPS`
- `FORCE_FULL_RUNTIME_BLOCKS`
- `SAFETY_FIRST_INPUT_BLOCK=1`

## 輸出（對齊 DiffAE 風格）

每個 run 目錄（`--run-output-dir`）至少包含：

- `run_manifest.json`
- `summary.json`
- `detail_stats.json`
- `run.log`

cache 模式額外輸出：

- `scheduler_config.snapshot.json`
- `cache_runtime_overrides_run.json`

全域輸出：

- `results/fid_results_ldm.json`（共享 list，append）
- `runs_index.jsonl`（每行一筆精簡結果）

## 如何判斷 cache 是否真的套用

看 `summary.json` 與 `detail_stats.json`：

- `cache_hook_cache_hits`
- `cache_hook_recompute_hits`
- `cache_hook_cache_ratio`
- `hook_stats.per_block_cache_hits`

若 `cache_hook_cache_hits > 0`，代表 scheduler 已實際參與 runtime 決策。  
目前實作為 `cache_execution_mode=true_skip`：reuse 時會在 block 前直接回傳 cached tensor，真正跳過該 block 計算。
