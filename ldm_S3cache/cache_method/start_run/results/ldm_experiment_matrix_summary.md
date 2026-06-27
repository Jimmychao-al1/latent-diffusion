# LDM / Q-LDM S3-Cache 實驗矩陣摘要

**建立日期：** 2026-06-27  
**最後更新：** 2026-06-27  
**資料來源：** `start_run/results/**/summary.json` 逐檔驗證 + `results/fid_results_*.json`  
**機器設定：** LDM FFHQ256, DDIM steps=200, eta=0, seed=0  
**Checkpoint：** `/home/jimmy/latent-diffusion/models/ldm/ffhq256/model.ckpt`

---

## 量化方法

本矩陣中 **Q-LDM** 實驗採用下列論文的 post-training quantization（PTQ）方法；**FP LDM** 為全精度 baseline，未做量化。

| 模型標籤 | 量化方法 | 論文 | 頂會 | 本實驗設定 |
|----------|----------|------|------|------------|
| Q-LDM | **TFMQ-DM** | Huang et al., *TFMQ-DM: Temporal Feature Maintenance Quantization for Diffusion Models* | **CVPR 2024**（Highlight Poster (top 2.8%)） | W8A8；cali ckpt：`/home/jimmy/TFMQ-DM/cali_ckpt/ffhq256_w8a8.pth`；base ckpt：FFHQ256 LDM |

**參考連結**

- TFMQ-DM 論文：[arXiv:2311.16503](https://arxiv.org/abs/2311.16503) · [CVPR 2024 Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_TFMQ-DM_Temporal_Feature_Maintenance_Quantization_for_Diffusion_Models_CVPR_2024_paper.html) · [GitHub](https://github.com/ModelTC/TFMQ-DM)
- 期刊延伸版：*Temporal Feature Matters for Diffusion Model Quantization* — **TPAMI 2025**（[arXiv:2407.19547](https://arxiv.org/abs/2407.19547)）

---

## 主結果（`is_primary: true`）

| ID | Model | Cache | Steps | FID@5K | FID@50K | ρ(M) | ΔFID@5K | ΔFID@50K |
|----|-------|-------|-------|--------|---------|------|---------|----------|
| FP-B-DDIM200 | FP LDM | — | 200 | 11.840 | 5.828 | 1.000 | — | — |
| FP-S3-K15sw3lam1 | FP LDM | S3-Cache | 200 | 11.710 | 5.761 | 0.282 | -1.1% | -1.2% |
| QL-B-DDIM200 | Q-LDM | — | 200 | 11.777 | 5.968 | 1.000 | — | — |
| QL-S3-K8sw5lam0.5 | Q-LDM | S3-Cache | 200 | 11.598 | 5.788 | 0.265 | -1.5% | -3.0% |

---

## 全部結果

### A. FP LDM Baseline

| ID | FID@5K | FID@50K | ρ | ΔFID@5K | Source |
|----|--------|---------|---|---------|--------|
| FP-B-DDIM200 ★ | 11.840 | 5.828 | 1.000 | — | `ldm_S3cache/cache_method/start_run/results/fid_5k/20260420/baseline_no_npz/0420_08_baseline_no_npz/summary.json` + `ldm_S3cache/cache_method/start_run/results/fid_50k/20260420/baseline_no_npz/0420_17_baseline_no_npz/summary.json` |

### B. FP LDM + S3-Cache

| ID | FID@5K | FID@50K | ρ | ΔFID@5K | Source |
|----|--------|---------|---|---------|--------|
| FP-S3-K15sw3lam1 ★ | 11.710 | 5.761 | 0.282 | -1.1% | `ldm_S3cache/cache_method/start_run/results/fid_5k/20260613/K15_sw3_lam1.0/0613_20_K15_sw3_lam1.0/summary.json` + `ldm_S3cache/cache_method/start_run/results/fid_50k/20260420/K15_sw3_lam1.0/0420_12_K15_sw3_lam1.0/summary.json` |
| FP-S3-K15sw3lam0.5 | 11.710 | — | 0.282 | -1.1% | `ldm_S3cache/cache_method/start_run/results/fid_5k/20260613/K15_sw3_lam0.5/0613_20_K15_sw3_lam0.5/summary.json` |
| FP-S3-K20sw3lam1 | 11.721 | — | 0.290 | -1.0% | `ldm_S3cache/cache_method/start_run/results/fid_5k/20260613/K20_sw3_lam1.0/0613_21_K20_sw3_lam1.0/summary.json` |

### C. Q-LDM Baseline

| ID | FID@5K | FID@50K | ρ | ΔFID@5K | Source |
|----|--------|---------|---|---------|--------|
| QL-B-DDIM200 ★ | 11.777 | 5.968 | 1.000 | — | `ldm_S3cache/cache_method/start_run/results/fid_5k_qldm/20260614/baseline_no_npz_qldm/0614_17_baseline_no_npz_qldm/summary.json` + `ldm_S3cache/cache_method/start_run/results/fid_50k_qldm/20260615/baseline_no_npz_qldm/0615_09_baseline_no_npz_qldm/summary.json` |

### D. Q-LDM + S3-Cache

| ID | FID@5K | FID@50K | ρ | ΔFID@5K | Source |
|----|--------|---------|---|---------|--------|
| QL-S3-K8sw5lam0.5 ★ | 11.598 | 5.788 | 0.265 | -1.5% | `ldm_S3cache/cache_method/start_run/results/fid_5k_qldm/20260614/K8_sw5_lam0.5/0614_19_K8_sw5_lam0.5/summary.json` + `ldm_S3cache/cache_method/start_run/results/fid_50k_qldm/20260616/K8_sw5_lam0.5/0616_10_K8_sw5_lam0.5/summary.json` |
| QL-S3-K12sw5lam0.5 | 11.677 | — | 0.270 | -0.8% | `ldm_S3cache/cache_method/start_run/results/fid_5k_qldm/20260614/K12_sw5_lam0.5/0614_20_K12_sw5_lam0.5/summary.json` |
| QL-S3-K15sw3lam0.5 | 12.080 | — | 0.313 | +2.6% | `ldm_S3cache/cache_method/start_run/results/fid_5k_qldm/20260614/K15_sw3_lam0.5/0614_21_K15_sw3_lam0.5/summary.json` |

### E. DeepCache（對照）

| ID | FID@5K | FID@50K | ρ | ΔFID@5K | Source |
|----|--------|---------|---|---------|--------|
| FP-DC-int10 | — | 8.131 | — | +39.5% (@50K) | `./deepcache_results/deepcache_r5_50k/interval10_samples50000_steps200_eta0.0/summary.json` |
| QL-DC-int10 | 14.315 | — | — | +21.6% | `outputs/deepcache_quant_5k/interval10_samples5000_steps200_eta0.0/summary.json` |

---

## 驗證過程中發現的不一致

### 1. Q-LDM baseline 5K 有多筆 run（已選 canonical）

- **0614_12**（batch_size=32）：FID@5K ≈ 13.88 — 排除
- **0614_17**（batch_size=16）：FID@5K = 11.777 — **canonical**

### 2. 早期 Q-LDM S3 sweep（0613，batch32）

`fid_results_qldm.json` 中 0613 的 K15/K20 sweep（FID@5K ≈ 15.9）與 0614 sweep 設定不同，
本矩陣以 `start_run/results/fid_5k_qldm/20260614/` 的 0614 sweep 為準。

### 3. FP LDM K15 5K 重複 run

0420_09 / 0420_10 / 0613_20 結果相同（FID=11.710），canonical 取最新 **0613_20**。

### 4. Q-LDM S3 50K FID 略優於 baseline

QL-S3-K8sw5λ0.5 FID@50K=5.788 vs QL-B FID@50K=5.968（ΔFID=-3.0%）。
建議論文引用時註明此為單次 run 結果。

---

## 待完成實驗

- FP LDM S3 50K for K20/K15_lam0.5 configs
- Q-LDM S3 50K for K12/K15 configs
- Q-LDM DeepCache 50K

---

## 檔案位置

- **結構化 JSON：** `/home/jimmy/latent-diffusion/ldm_S3cache/cache_method/start_run/results/ldm_experiment_matrix.json`
- **本摘要：** `/home/jimmy/latent-diffusion/ldm_S3cache/cache_method/start_run/results/ldm_experiment_matrix_summary.md`
- **重建腳本：** `ldm_S3cache/cache_method/start_run/build_ldm_experiment_matrix.py`
