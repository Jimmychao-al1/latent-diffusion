# LDM `start_run` (Stage2 Scheduler FID Entry)

本目錄提供 S3-Cache 在 LDM 端的正式 FID 入口，統一 baseline 與 Stage2 scheduler 實驗紀錄。

## 固定規格（不可動）

| item | value |
|---|---|
| `ddim_steps` | `200` |
| `eta` | `0` |
| `seed` | `0` |
| `batch_size` | `32` |
| `n_samples` | `5000` (`FID@5K`) |
| `fid_dims` | `2048` |
| `dataset_tag` | `ffhq256` |

## 檔案

- `sample_stage2_cache_scheduler_ldm.py`
  - `--mode baseline`：原生 LDM，不套 scheduler，且需 `--no_npz`
  - `--mode cache`：讀 Stage2 refined JSON，掛 runtime cache hook 後採樣
- `runFidWithStage2SchedulerLdm.sh`
  - 固定流程：baseline -> cache，並 append 到同一份結果 JSON

## Loop Index 注意事項（重要）

cache 模式的 scheduler 決策使用 **loop index**（`0..199`）：

- `loop_idx=0` = 第一個 denoise step
- `loop_idx=199` = 最後一個 denoise step

程式內會從 DDIM raw timestep 建立 `raw_t -> loop_idx` 映射，hook 只用 `loop_idx` 比對 scheduler，避免把 raw timestep 當成 scheduler index。

## Stage2 Scheduler JSON 路徑慣例

```
ldm_S3cache/cache_method/Stage2/stage2_output_ldm/<src>/02_refined_blockwise/stage2_refined_scheduler_config.json
```

## 單獨跑 Python

### 1) baseline（必須 `--no_npz`）

```bash
python ldm_S3cache/cache_method/start_run/sample_stage2_cache_scheduler_ldm.py \
  --mode baseline \
  --no_npz \
  --ckpt models/ldm/ffhq256/model.ckpt \
  --config configs/latent-diffusion/ffhq-ldm-vq-4.yaml \
  --real_image_dir /path/to/ffhq/images \
  --scheduler_name baseline_no_npz \
  --out_root outputs \
  --results_json results/fid_results_ldm.json
```

### 2) cache（Stage2 refined scheduler）

```bash
python ldm_S3cache/cache_method/start_run/sample_stage2_cache_scheduler_ldm.py \
  --mode cache \
  --ckpt models/ldm/ffhq256/model.ckpt \
  --config configs/latent-diffusion/ffhq-ldm-vq-4.yaml \
  --real_image_dir /path/to/ffhq/images \
  --scheduler_json ldm_S3cache/cache_method/Stage2/stage2_output_ldm/src_K15_sw3_lam1.0/02_refined_blockwise/stage2_refined_scheduler_config.json \
  --scheduler_name K15_sw3_lam1.0 \
  --out_root outputs \
  --results_json results/fid_results_ldm.json
```

`--real_image_dir` 與 `--real_lmdb` 二擇一，不能同時提供。

## 一鍵跑 shell

```bash
bash ldm_S3cache/cache_method/start_run/runFidWithStage2SchedulerLdm.sh
```

腳本流程固定：
1. baseline (`--mode baseline --no_npz`)
2. cache (`--mode cache`)

後續要跑多個 scheduler，可直接複製 shell 中 cache block。

## 輸出與共享結果 JSON

- 生成圖：`outputs/start_run/fid_5k/<timestamp>_<run_label>/gen_images`
- 真實圖 cache：`outputs/start_run/real_eval_cache_ffhq256_5k`
- 共享結果：`results/fid_results_ldm.json`（list append）

每筆記錄欄位：

```json
{
  "run_type": "baseline_no_npz | stage2_scheduler",
  "scheduler_name": "...",
  "scheduler_json": "abs path or null",
  "fid": 0.0,
  "fid_at": "5k",
  "n_samples": 5000,
  "dataset_tag": "ffhq256",
  "ddim_steps": 200,
  "eta": 0,
  "seed": 0,
  "batch_size": 32,
  "fid_dims": 2048,
  "no_npz": true,
  "timestamp": "YYYYmmdd_HHMMSS",
  "sample_time_min": 0.0,
  "fid_time_min": 0.0,
  "gen_dir": "abs path",
  "eval_dir": "abs path",
  "ckpt": "abs path"
}
```
