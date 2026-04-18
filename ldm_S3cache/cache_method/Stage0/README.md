# Stage-0（LDM，T=200）

與 DiffAE Stage-0 參考實作 **計算與圖表樣式對齊**；本目錄腳本與輸出路徑皆為 LDM 專用檔名（`*_ldm`）。

## 腳本

| 檔案 | 說明 |
|------|------|
| `stage0_normalization_ldm.py` | 載入 a/b/c 實驗結果、正規化、寫入 `stage0_output_ldm/` |
| `stage0_visualization_ldm.py` | 讀取輸出並繪圖至 `stage0_figures_ldm/` |
| `verify_stage0_output_ldm.sh` | 檢查 `stage0_output_ldm/` 內陣列與權重 |

## 前置資料

1. **L1 / Cosine**：`a_L1_L2_cosine/T_200/v2_latest/result_npz/*.npz`
2. **SVD**：`b_SVD/T_200/svd_metrics/*.json`
3. **FID**：`c_FID/fid_sensitivity_results_ldm.json`（`results["T200"]`）

## 執行（自 repo 根目錄）

```bash
python3 ldm_S3cache/cache_method/Stage0/stage0_normalization_ldm.py
python3 ldm_S3cache/cache_method/Stage0/stage0_visualization_ldm.py
bash ldm_S3cache/cache_method/Stage0/verify_stage0_output_ldm.sh
```

- 數值輸出目錄：`Stage0/stage0_output_ldm/`（含 `stage0_metadata_ldm.json`、`fid_w_ldm_clip.npy` 等）
- 圖表目錄：`Stage0/stage0_figures_ldm/`

## API

`run_stage0_ldm(..., fid_step_key=None)`：`None` 時依序嘗試 `T200` → `T100` → `100steps`。

採樣推理請用 `scripts/sample_diffusion.py`（Stage-0 本身不跑採樣）。
