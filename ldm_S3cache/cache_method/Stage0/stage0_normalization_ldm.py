"""
Stage-0：Loader + Normalization（LDM，`stage0_normalization_ldm.py`，T=200 baseline）

此模組讀取 Stage-0 的原始實驗資料並產生正規化的 interval-wise 指標。
計算順序、資料語意、方程式與 DiffAE `stage0e_normalization.py` 對齊；預設路徑與
FID step key 適配本 repo（`T_200`、`results["T200"]`）。

資料來源（預設）：
1. L1 / Cosine: ldm_S3cache/cache_method/a_L1_L2_cosine/T_200/v2_latest/result_npz/*.npz
2. SVD drift: ldm_S3cache/cache_method/b_SVD/T_200/svd_metrics/*.json
3. FID sensitivity: ldm_S3cache/cache_method/c_FID/fid_sensitivity_results_ldm.json（`results["T200"]`）

輸出：
- 正規化的 interval-wise 指標 (B, T-1)，其中 interval i 代表 step i → i+1
- FID-based block weights w_b (B,)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("Stage0_LDM")


# =============================================================================
# 一、載入 interval-wise 指標
# =============================================================================


def load_interval_metrics(
    l1_cos_dir: str,
    svd_dir: str,
    strict: bool = False,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, Dict[str, str], List[str]]:
    """
    掃描兩個資料夾，讀取所有 block 的 L1 step mean / cosine distance / SVD interval distance。

    Args:
        l1_cos_dir: L1/Cosine npz 檔案目錄
        svd_dir: SVD metrics JSON 目錄
        strict: 若為 True，任一 block 載入失敗時最終會 raise RuntimeError

    Returns:
        block_names, L1_interval, CosDist_interval, SVD_interval, source_keys, failed_blocks

    Interval mapping（與 similarity_calculation / a_L1_L2_cosine .npz 一致）：
    - 欄位索引 j ∈ [0..T-2]：**interval j** = 沿 **analysis axis** 在點 j 與 j+1 之間的變化
    - L1: 正式優先使用 l1_step_mean[j]；若缺失則 fallback 至 l1_rate_step_mean[j]（legacy）
    - Cos: 1.0 - cos_step_mean[j]
    - SVD: subspace_dist[j+1]（見 DiffAE 原始對應）
    """
    l1_cos_path = Path(l1_cos_dir)
    svd_path = Path(svd_dir)

    if not l1_cos_path.exists():
        raise FileNotFoundError(f"L1/Cosine 目錄不存在: {l1_cos_path}")
    if not svd_path.exists():
        raise FileNotFoundError(f"SVD 目錄不存在: {svd_path}")

    npz_files = sorted(l1_cos_path.glob("*.npz"))
    if len(npz_files) == 0:
        raise ValueError(f"在 {l1_cos_path} 中找不到任何 .npz 檔案")

    block_names: List[str] = []
    L1_list: List[np.ndarray] = []
    CosDist_list: List[np.ndarray] = []
    SVD_list: List[np.ndarray] = []
    failed_blocks: List[str] = []
    l1_source_used: Optional[str] = None

    LOGGER.info("載入 %d 個 block 的 interval-wise 指標...", len(npz_files))

    for npz_file in npz_files:
        block_slug = npz_file.stem

        if block_slug.startswith("model_"):
            rest = block_slug[6:]
            parts = rest.split("_")
            if len(parts) >= 3 and parts[-1].isdigit():
                block_name = "model." + "_".join(parts[:-1]) + "." + parts[-1]
            else:
                block_name = "model." + rest
        else:
            block_name = block_slug

        try:
            data = np.load(npz_file)
            if "l1_step_mean" in data:
                l1_step_mean = data["l1_step_mean"]
                l1_source_key = "l1_step_mean"
            elif "l1_rate_step_mean" in data:
                l1_step_mean = data["l1_rate_step_mean"]
                l1_source_key = "l1_rate_step_mean (fallback)"
                LOGGER.warning(
                    "[legacy fallback] block=%s 缺少 l1_step_mean，改用 l1_rate_step_mean",
                    block_name,
                )
            else:
                raise ValueError(
                    f"block={block_name}: similarity npz 缺少 L1 key，"
                    f"找不到 'l1_step_mean' 或 'l1_rate_step_mean'"
                )
            if "cos_step_mean" not in data:
                raise ValueError(f"block={block_name}: similarity npz 缺少必要 key 'cos_step_mean'")
            cos_step_mean = data["cos_step_mean"]

            if l1_step_mean.ndim != 1:
                raise ValueError(
                    f"block={block_name}, key={l1_source_key}: shape={l1_step_mean.shape}, expected 1D (T-1,)"
                )
            if cos_step_mean.ndim != 1:
                raise ValueError(
                    f"block={block_name}, key=cos_step_mean: shape={cos_step_mean.shape}, expected 1D (T-1,)"
                )
            T_minus_1 = int(l1_step_mean.shape[0])
            if int(cos_step_mean.shape[0]) != T_minus_1:
                raise ValueError(
                    f"block={block_name}: 長度不一致，"
                    f"{l1_source_key}={l1_step_mean.shape}, cos_step_mean={cos_step_mean.shape}"
                )

            svd_json_path = svd_path / f"{block_slug}.json"
            if not svd_json_path.exists():
                LOGGER.warning("SVD JSON 不存在，跳過 block: %s", block_name)
                continue

            with open(svd_json_path, "r", encoding="utf-8") as f:
                svd_data = json.load(f)
            if "subspace_dist" not in svd_data:
                raise ValueError(f"block={block_name}: SVD JSON 缺少必要 key 'subspace_dist'")
            subspace_dist = np.array(svd_data["subspace_dist"])
            if subspace_dist.ndim != 1:
                raise ValueError(
                    f"block={block_name}, key=subspace_dist: shape={subspace_dist.shape}, expected 1D (T,)"
                )

            L1_interval = l1_step_mean
            CosDist_interval = 1.0 - cos_step_mean
            SVD_interval = subspace_dist[1:]

            if len(L1_interval) != T_minus_1:
                raise ValueError(
                    f"block={block_name}, key={l1_source_key}: len={len(L1_interval)}, expected={T_minus_1}"
                )
            if len(CosDist_interval) != T_minus_1:
                raise ValueError(
                    f"block={block_name}, key=cos_step_mean: len={len(CosDist_interval)}, expected={T_minus_1}"
                )
            if len(SVD_interval) != T_minus_1:
                raise ValueError(
                    f"block={block_name}, key=subspace_dist[1:]: len={len(SVD_interval)}, expected={T_minus_1}, "
                    f"raw_subspace_dist_shape={subspace_dist.shape}"
                )

            if l1_source_used is None:
                l1_source_used = l1_source_key
            elif l1_source_used != l1_source_key:
                l1_source_used = "mixed(l1_step_mean + l1_rate_step_mean fallback)"

            block_names.append(block_name)
            L1_list.append(L1_interval)
            CosDist_list.append(CosDist_interval)
            SVD_list.append(SVD_interval)

        except Exception as e:
            LOGGER.error("載入 %s 時出錯: %s", block_name, e)
            raise ValueError(f"載入 {block_name} 時出錯: {e}") from e

    if len(block_names) == 0:
        raise ValueError("沒有成功載入任何 block 的資料")

    if failed_blocks:
        LOGGER.warning("共有 %d 個 block 載入失敗/跳過：%s", len(failed_blocks), failed_blocks)
        if strict:
            raise RuntimeError(f"strict=True，且有 block 載入失敗：{failed_blocks}")

    L1_interval_all = np.stack(L1_list, axis=0)
    CosDist_interval_all = np.stack(CosDist_list, axis=0)
    SVD_interval_all = np.stack(SVD_list, axis=0)

    LOGGER.info("成功載入 %d 個 block", len(block_names))
    LOGGER.info("   L1_interval shape: %s", L1_interval_all.shape)
    LOGGER.info("   CosDist_interval shape: %s", CosDist_interval_all.shape)
    LOGGER.info("   SVD_interval shape: %s", SVD_interval_all.shape)

    source_keys = {
        "l1_source_key": l1_source_used or "unknown",
        "cos_source_key": "cos_step_mean",
        "svd_source_key": "subspace_dist[1:]",
    }

    return block_names, L1_interval_all, CosDist_interval_all, SVD_interval_all, source_keys, failed_blocks


# =============================================================================
# 二、Min-max 正規化
# =============================================================================


def normalize_minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """對 x 全體元素做 min-max 正規化到 [0, 1]。"""
    valid_mask = np.isfinite(x)

    if not np.any(valid_mask):
        LOGGER.warning("輸入陣列全為 NaN/Inf，回傳全零")
        return np.zeros_like(x, dtype=np.float32)

    x_valid = x[valid_mask]
    x_min = x_valid.min()
    x_max = x_valid.max()

    if x_max - x_min <= eps:
        LOGGER.warning("數值範圍過小 (max - min = %.2e <= eps)，回傳全零", x_max - x_min)
        return np.zeros_like(x, dtype=np.float32)

    x_norm = (x - x_min) / (x_max - x_min)
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=1.0, neginf=0.0)
    x_norm = np.clip(x_norm, 0.0, 1.0)

    return x_norm.astype(np.float32)


# =============================================================================
# 三、載入 FID sensitivity 資料
# =============================================================================

_DEFAULT_FID_STEP_KEY_ORDER = ("T200", "T100", "100steps")


def load_delta_fid_from_json(
    fid_json_path: str,
    step_key: Optional[str] = None,
) -> Tuple[str, List[str], Dict[str, Dict[int, float]]]:
    """
    從 JSON 讀取 Delta-FID（與 DiffAE 相同結構；LDM 預設使用 results[\"T200\"]）。

    Args:
        fid_json_path: FID sensitivity 結果 JSON 路徑
        step_key: 若指定，使用 results[step_key]；若為 None，依序嘗試
            T200、T100、100steps（與 DiffAE 相容）

    Returns:
        resolved_step_key: 實際使用的 results 子鍵
        block_names: block 名稱列表（FID JSON 內 layer 名稱）
        delta_fid: delta_fid[block_name][k] = delta_FID at hop k, k in {3,4,5}
    """
    fid_path = Path(fid_json_path)
    if not fid_path.exists():
        raise FileNotFoundError(f"FID JSON 不存在: {fid_path}")

    with open(fid_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    results_dict = results.get("results", {})

    if step_key is not None:
        if step_key not in results_dict:
            available_keys = list(results_dict.keys())
            raise ValueError(
                f"FID JSON 中沒有指定的 step_key={step_key!r}。可用的 keys: {available_keys}"
            )
        resolved = step_key
    else:
        resolved = None
        for k in _DEFAULT_FID_STEP_KEY_ORDER:
            if k in results_dict:
                resolved = k
                break
        if resolved is None:
            available_keys = list(results_dict.keys())
            raise ValueError(
                f"FID JSON 中找不到可用的 step key（嘗試: {_DEFAULT_FID_STEP_KEY_ORDER}）。"
                f"可用的 keys: {available_keys}"
            )

    step_results = results_dict[resolved]
    baseline_fid = step_results.get("baseline_fid")

    if baseline_fid is None:
        raise ValueError(f"找不到 baseline_fid ({resolved})")

    LOGGER.info("Baseline FID (%s): %.4f", resolved, float(baseline_fid))

    delta_fid: Dict[str, Dict[int, float]] = {}
    k_values = [3, 4, 5]

    for k in k_values:
        k_key = f"k{k}"
        if k_key not in step_results:
            LOGGER.warning("找不到 k=%s 的資料，跳過", k)
            continue

        for layer_name, layer_data in step_results[k_key].items():
            if layer_name not in delta_fid:
                delta_fid[layer_name] = {}

            delta = layer_data.get("delta")
            if delta is not None:
                delta_fid[layer_name][k] = float(delta)

    block_names = sorted(delta_fid.keys())

    LOGGER.info("成功載入 %d 個 block 的 Delta-FID (%s)", len(block_names), resolved)
    LOGGER.info("   k values: %s", k_values)

    return resolved, block_names, delta_fid


# =============================================================================
# 四、FID-based block weight 計算
# =============================================================================


def compute_fid_weights(
    block_names: List[str],
    delta_fid: Dict[str, Dict[int, float]],
    eps_noise: float = 0.05,
    quantile: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """計算 FID-based block weights w_b（與 DiffAE 相同邏輯）。"""
    B = len(block_names)
    S = np.zeros(B, dtype=np.float32)

    for i, block_name in enumerate(block_names):
        if block_name not in delta_fid:
            LOGGER.warning("Block %s 沒有 FID 資料，使用 S_b = 0", block_name)
            continue

        delta_dict = delta_fid[block_name]
        delta_pos_values = []

        for k in [3, 4, 5]:
            if k in delta_dict:
                delta = delta_dict[k]
                delta_pos = max(0.0, delta - eps_noise)
                delta_pos_values.append(delta_pos)

        if len(delta_pos_values) > 0:
            S[i] = max(delta_pos_values)

    LOGGER.info("原始 S 統計：min=%.4f, max=%.4f, mean=%.4f", S.min(), S.max(), S.mean())

    if np.all(S == 0):
        LOGGER.warning("所有 S_b = 0，回傳全零權重")
        w_clip = np.zeros(B, dtype=np.float32)
        w_rank = np.zeros(B, dtype=np.float32)
        return w_clip, w_rank

    hi = np.quantile(S, quantile)
    S_clip = np.minimum(S, hi)

    LOGGER.info("Quantile clipping (q=%s): hi=%.4f", quantile, hi)
    LOGGER.info("S_clip 統計：min=%.4f, max=%.4f, mean=%.4f", S_clip.min(), S_clip.max(), S_clip.mean())

    h = S_clip.max()
    if h <= 0:
        LOGGER.warning("S_clip.max() <= 0，回傳全零權重")
        w_clip = np.zeros(B, dtype=np.float32)
    else:
        w_clip = S_clip / h

    LOGGER.info(
        "w_clip 統計：min=%.4f, max=%.4f, mean=%.4f",
        w_clip.min(),
        w_clip.max(),
        w_clip.mean(),
    )

    w_rank = rank_based_weights(S)
    LOGGER.info(
        "w_rank 統計：min=%.4f, max=%.4f, mean=%.4f",
        w_rank.min(),
        w_rank.max(),
        w_rank.mean(),
    )

    return w_clip, w_rank


def rank_based_weights(S: np.ndarray) -> np.ndarray:
    """對 S (B,) 做排序，回傳 [0, 1] 的 rank 權重。"""
    B = S.shape[0]
    order = np.argsort(S)
    w_rank = np.zeros_like(S, dtype=np.float32)

    if B > 1:
        for i, idx in enumerate(order):
            w_rank[idx] = i / (B - 1)
    else:
        w_rank[0] = 1.0

    return w_rank


# =============================================================================
# 五、主入口函式
# =============================================================================


def run_stage0_ldm(
    l1_cos_dir: str,
    svd_dir: str,
    fid_json_path: str,
    output_dir: str,
    eps_noise: float = 0.05,
    quantile: float = 0.95,
    strict: bool = False,
    fid_step_key: Optional[str] = None,
):
    """
    Stage-0（LDM）主流程：讀取 + 正規化 + 輸出。

    Args:
        fid_step_key: 傳入 `load_delta_fid_from_json`；None 時自動在 T200 / T100 / 100steps 中擇一。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 80)
    LOGGER.info("Stage-0 (LDM): Loader + Normalization")
    LOGGER.info("=" * 80)

    LOGGER.info("\n[步驟 1] 載入 L1 / Cosine / SVD interval-wise 指標...")
    (
        block_names_metric,
        L1_interval_all,
        CosDist_interval_all,
        SVD_interval_all,
        source_keys,
        failed_blocks,
    ) = load_interval_metrics(
        l1_cos_dir=l1_cos_dir,
        svd_dir=svd_dir,
        strict=strict,
    )

    LOGGER.info("\n[步驟 2] Min-max 正規化（三種指標獨立處理）...")

    LOGGER.info("  正規化 L1_interval...")
    L1_interval_norm = normalize_minmax(L1_interval_all)
    LOGGER.info(
        "    L1_interval_norm: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
        L1_interval_norm.min(),
        L1_interval_norm.max(),
        L1_interval_norm.mean(),
        L1_interval_norm.std(),
    )

    LOGGER.info("  正規化 CosDist_interval...")
    CosDist_interval_norm = normalize_minmax(CosDist_interval_all)
    LOGGER.info(
        "    CosDist_interval_norm: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
        CosDist_interval_norm.min(),
        CosDist_interval_norm.max(),
        CosDist_interval_norm.mean(),
        CosDist_interval_norm.std(),
    )

    LOGGER.info("  正規化 SVD_interval...")
    SVD_interval_norm = normalize_minmax(SVD_interval_all)
    LOGGER.info(
        "    SVD_interval_norm: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
        SVD_interval_norm.min(),
        SVD_interval_norm.max(),
        SVD_interval_norm.mean(),
        SVD_interval_norm.std(),
    )

    LOGGER.info("\n[步驟 3] 載入 FID sensitivity 並計算 block weights...")
    fid_resolved_key, _, delta_fid = load_delta_fid_from_json(
        fid_json_path=fid_json_path,
        step_key=fid_step_key,
    )

    def metric_to_fid_name(block_name: str) -> str:
        if "input_blocks" in block_name:
            idx = block_name.split(".")[-1]
            return f"encoder_layer_{idx}"
        if "middle_block" in block_name:
            return "middle_layer"
        if "output_blocks" in block_name:
            idx = block_name.split(".")[-1]
            return f"decoder_layer_{idx}"
        return block_name

    delta_fid_aligned = {}
    for block_name in block_names_metric:
        fid_layer_name = metric_to_fid_name(block_name)
        if fid_layer_name in delta_fid:
            delta_fid_aligned[block_name] = delta_fid[fid_layer_name]
        else:
            LOGGER.warning(
                "Block %s (FID: %s) 在 FID 資料中找不到，使用空字典",
                block_name,
                fid_layer_name,
            )
            delta_fid_aligned[block_name] = {}

    all_delta_values = [
        float(delta)
        for per_block in delta_fid_aligned.values()
        for delta in per_block.values()
        if np.isfinite(delta)
    ]
    if all_delta_values:
        delta_min = min(all_delta_values)
        delta_max = max(all_delta_values)
        LOGGER.info("FID delta 統計：min=%.4f, max=%.4f", delta_min, delta_max)
        if delta_max <= eps_noise:
            LOGGER.warning(
                "目前 eps_noise=%.4f >= max(delta)=%.4f，FID 權重將全部變成 0。"
                "建議降低 eps_noise（例如 0.05 或 0.0）。",
                eps_noise,
                delta_max,
            )

    w_clip, w_rank = compute_fid_weights(
        block_names=block_names_metric,
        delta_fid=delta_fid_aligned,
        eps_noise=eps_noise,
        quantile=quantile,
    )

    LOGGER.info("\n[步驟 4] 檢查數值有效性...")

    def check_array(arr, name):
        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()
        in_range = (arr >= 0).all() and (arr <= 1).all()
        LOGGER.info("  %s: NaN=%s, Inf=%s, in [0,1]=%s", name, has_nan, has_inf, in_range)
        if has_nan or has_inf or not in_range:
            LOGGER.error("    %s 包含異常值！", name)
        else:
            LOGGER.info("    %s 正常", name)

    check_array(L1_interval_norm, "L1_interval_norm")
    check_array(CosDist_interval_norm, "CosDist_interval_norm")
    check_array(SVD_interval_norm, "SVD_interval_norm")
    check_array(w_clip, "fid_w_clip")
    check_array(w_rank, "fid_w_rank")

    LOGGER.info("\n[步驟 5] 存檔...")

    block_names_array = np.array(block_names_metric, dtype=object)

    np.save(output_path / "block_names.npy", block_names_array)
    np.save(output_path / "block_names_metric.npy", block_names_array)
    np.save(output_path / "l1_interval_norm.npy", L1_interval_norm)
    np.save(output_path / "cosdist_interval_norm.npy", CosDist_interval_norm)
    np.save(output_path / "svd_interval_norm.npy", SVD_interval_norm)
    np.save(output_path / "fid_w_ldm_clip.npy", w_clip)
    np.save(output_path / "fid_weights.npy", w_clip)
    np.save(output_path / "fid_w_ldm_rank.npy", w_rank)
    np.save(output_path / "delta_fid.npy", np.array(delta_fid_aligned, dtype=object), allow_pickle=True)

    interval_len = int(L1_interval_norm.shape[1])
    t_curr_interval = (interval_len - 1) - np.arange(interval_len, dtype=np.int32)
    np.save(output_path / "t_curr_interval.npy", t_curr_interval)
    np.save(
        output_path / "axis_interval_def.npy",
        np.array(
            "interval-wise: analysis interval index j (0..T-2) keeps internal order; display label is t_curr=(T-2)-j",
            dtype=object,
        ),
    )

    metadata = {
        "l1_source_key": source_keys["l1_source_key"],
        "cos_source_key": source_keys["cos_source_key"],
        "svd_source_key": source_keys["svd_source_key"],
        "fid_results_step_key": fid_resolved_key,
        "t_axis_definition": "interval-wise: analysis interval index j (0..T-2) keeps internal order; display label is t_curr=(T-2)-j",
        "similarity_root": str(Path(l1_cos_dir).resolve()),
        "svd_root": str(Path(svd_dir).resolve()),
        "fid_json_path": str(Path(fid_json_path).resolve()),
        "fid_eps_noise": float(eps_noise),
        "fid_quantile": float(quantile),
        "failed_blocks_count": len(failed_blocks),
        "failed_blocks": failed_blocks,
        "strict_mode": bool(strict),
    }
    with open(output_path / "stage0_metadata_ldm.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    LOGGER.info("所有檔案已儲存至: %s", output_path)
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Stage-0 (LDM) 完成！")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    l1_cos_dir = repo_root / "ldm_S3cache/cache_method/a_L1_L2_cosine/T_200/v2_latest/result_npz"
    svd_dir = repo_root / "ldm_S3cache/cache_method/b_SVD/T_200/svd_metrics"
    fid_json_path = repo_root / "ldm_S3cache/cache_method/c_FID/fid_sensitivity_results_ldm.json"
    output_dir = repo_root / "ldm_S3cache/cache_method/Stage0/stage0_output_ldm"

    required_inputs = [
        ("l1_cos_dir", l1_cos_dir, True),
        ("svd_dir", svd_dir, True),
        ("fid_json_path", fid_json_path, False),
    ]
    for label, p, expect_dir in required_inputs:
        if expect_dir and not p.is_dir():
            raise FileNotFoundError(f"[Stage0_LDM] default {label} directory not found: {p}")
        if (not expect_dir) and not p.is_file():
            raise FileNotFoundError(f"[Stage0_LDM] default {label} file not found: {p}")

    run_stage0_ldm(
        l1_cos_dir=str(l1_cos_dir),
        svd_dir=str(svd_dir),
        fid_json_path=str(fid_json_path),
        output_dir=str(output_dir),
        eps_noise=0.05,
        quantile=0.95,
        fid_step_key=None,
    )
