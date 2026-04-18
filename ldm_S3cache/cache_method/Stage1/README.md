# Stage 1 (LDM)

本目錄是 LDM 版本的 Stage 1 scheduler，從 Stage 0 輸出合成：
- `shared_zones`
- 每個 block 在每個 zone 的 `k_per_zone`
- `expanded_mask`（步序 i：`i=0 -> t=T-1`）

## 檔案

- `stage1_scheduler_ldm.py`：主程式（產生 3 個 JSON）
- `verify_scheduler_ldm.py`：驗證 `scheduler_config.json`
- `visualize_stage1_ldm.py`：輸出 Stage 1 圖表

## 時間軸約定

- `time_order = "ddim_descending_Tminus1_to_0"`
- `expanded_mask[b, i]`：
  - `i=0` 對應 `t=T-1`
  - `i=T-1` 對應 `t=0`
- `True = full compute (F)`、`False = reuse (R)`
- 強制 `expanded_mask[:, 0] = True`

## Stage 0 依賴（固定）

預設從：
- `ldm_S3cache/cache_method/Stage0/stage0_output_ldm`

必要檔案：
- `block_names.npy`
- `l1_interval_norm.npy`
- `cosdist_interval_norm.npy`
- `svd_interval_norm.npy`
- `fid_w_ldm_clip.npy`（僅此檔名，不做 fallback）
- `axis_interval_def.npy`
- `t_curr_interval.npy`

## 執行

在 repo root (`/home/jimmy/latent-diffusion`)：

```bash
python3 ldm_S3cache/cache_method/Stage1/stage1_scheduler_ldm.py
python3 ldm_S3cache/cache_method/Stage1/verify_scheduler_ldm.py \
  --config ldm_S3cache/cache_method/Stage1/stage1_output_ldm/scheduler_config.json
python3 ldm_S3cache/cache_method/Stage1/visualize_stage1_ldm.py \
  --stage1_output_dir ldm_S3cache/cache_method/Stage1/stage1_output_ldm \
  --output_dir ldm_S3cache/cache_method/Stage1/stage1_figures_ldm
```

## 預設參數（目前）

- `K=10`
- `smooth_window=5`
- `lambda=1.0`
- `k_min=1`
- `k_max=4`
- `min_zone_len=2`

你後續可以用 sweep 再挑正式 baseline。
