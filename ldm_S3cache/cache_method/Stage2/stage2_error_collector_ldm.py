"""
Stage2 (LDM): baseline/cache 兩趟特徵收集與誤差統計。

透過 callback 收集每個 runtime block、每個 DDIM timestep 的輸出 tensor，
再計算 baseline vs cache 的 L1 / L2 / cosine，並做 block-step / block-zone 聚合。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ldm_S3cache.cache_method.Stage2.stage2_scheduler_adapter_ldm import RUNTIME_LAYER_NAMES

LOGGER = logging.getLogger("Stage2ErrorCollectorLDM")
_LOG_T = logging.getLogger("Stage2RuntimeRefineLDM")


def _flatten_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a2 = a.flatten(1)
    b2 = b.flatten(1)
    return F.cosine_similarity(a2, b2, dim=1, eps=1e-8)


def _l1_scalar(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().mean()


def _l2_rms(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).mean().sqrt()


def _zone_ddim_timesteps(shared_zones: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for z in shared_zones:
        zid = int(z["id"])
        ts = int(z["t_start"])
        te = int(z["t_end"])
        out[zid] = list(range(te, ts + 1))
    return out


class Stage2ErrorCollectorLDM:
    """
    蒐集 baseline / cache 兩趟每層每步特徵。

    features 結構：
      feats[run_name][runtime_name][step_idx] = tensor (cpu float)
    其中 step_idx=0 對應第一步 DDIM t=T-1。
    """

    def __init__(self, T: int, device: Optional[torch.device] = None):
        self.T = int(T)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._run_name = "baseline"
        self._debug_write_count = 0
        self._feats: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {
            "baseline": {},
            "cache": {},
        }

    def set_run(self, name: str) -> None:
        """切換收集桶：baseline 或 cache。"""
        if name not in ("baseline", "cache"):
            raise ValueError("run name must be 'baseline' or 'cache'")
        self._run_name = name

    def clear_storage(self) -> None:
        """釋放收集到的 tensor。"""
        self._feats = {"baseline": {}, "cache": {}}
        self._debug_write_count = 0

    def make_cache_debug_callback(self) -> Callable[..., None]:
        """
        產生 callback，供 runtime cache wrapper 在每層輸出時呼叫。

        callback 簽名：
          cb(runtime_name: str, output: Tensor, *, recompute: bool, ddim_timestep: int)
        """
        return self.on_layer_output

    def on_layer_output(
        self,
        runtime_name: str,
        output: torch.Tensor,
        *,
        recompute: bool,
        ddim_timestep: int,
    ) -> None:
        """寫入單一 layer-step tensor。"""
        del recompute
        t = int(ddim_timestep)
        if not (0 <= t < self.T):
            LOGGER.warning(
                "[Stage2-LDM] on_layer_output: expect ddim_timestep in [0, %s), got %s (layer=%s)",
                self.T,
                t,
                runtime_name,
            )
            return

        if runtime_name not in RUNTIME_LAYER_NAMES:
            LOGGER.warning("[Stage2-LDM] unknown runtime_name=%s; skip", runtime_name)
            return

        step_idx = (self.T - 1) - t
        x = output.detach().float().cpu()
        bucket = self._feats[self._run_name].setdefault(runtime_name, {})
        bucket[step_idx] = x
        self._debug_write_count += 1

    def debug_snapshot_line(self, phase: str) -> str:
        """單行摘要，方便 log 追蹤是否真的有在收資料。"""
        b = self._feats["baseline"]
        c = self._feats["cache"]
        n0b = len(b.get("encoder_layer_0", {}))
        n0c = len(c.get("encoder_layer_0", {}))
        return (
            f"[collector debug] phase={phase} id={id(self)} writes={self._debug_write_count} "
            f"baseline_layers={len(b)} enc0_steps={n0b} cache_layers={len(c)} enc0_steps={n0c}"
        )

    def compute_diagnostics(
        self,
        shared_zones: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """將 baseline/cache 收集結果轉為 Stage2 diagnostics dict。"""
        per_block_step_error: Dict[str, Dict[str, Dict[str, float]]] = {}
        per_block_zone_error: Dict[str, Dict[str, Dict[str, Any]]] = {}
        zone_ts = _zone_ddim_timesteps(shared_zones)

        all_l1: List[float] = []
        all_l2: List[float] = []
        all_cos: List[float] = []

        for rt in RUNTIME_LAYER_NAMES:
            per_block_step_error[rt] = {}
            per_block_zone_error[rt] = {}
            bmap = self._feats["baseline"].get(rt, {})
            cmap = self._feats["cache"].get(rt, {})

            for step_idx in range(self.T):
                ddim_t = (self.T - 1) - step_idx
                key = str(ddim_t)
                if step_idx not in bmap or step_idx not in cmap:
                    raise RuntimeError(
                        f"[Stage2-LDM] missing feature for {rt} step_idx={step_idx} (ddim_t={ddim_t}); "
                        f"baseline keys={sorted(bmap.keys())}, cache keys={sorted(cmap.keys())}"
                    )
                a = bmap[step_idx]
                b = cmap[step_idx]
                if a.shape != b.shape:
                    raise RuntimeError(
                        f"{rt} step_idx={step_idx}: shape mismatch {a.shape} vs {b.shape}"
                    )

                l1 = float(_l1_scalar(a, b))
                l2 = float(_l2_rms(a, b))
                cos = float(_flatten_cosine(a, b).mean())
                per_block_step_error[rt][key] = {"l1": l1, "l2": l2, "cosine": cos}
                all_l1.append(l1)
                all_l2.append(l2)
                all_cos.append(cos)

            for zid, ts in zone_ts.items():
                zs_l1: List[float] = []
                zs_l2: List[float] = []
                zs_cos: List[float] = []
                for ddim_t in ts:
                    st = per_block_step_error[rt].get(str(ddim_t))
                    if st is None:
                        continue
                    zs_l1.append(float(st["l1"]))
                    zs_l2.append(float(st["l2"]))
                    zs_cos.append(float(st["cosine"]))
                if not zs_l1:
                    per_block_zone_error[rt][str(zid)] = {
                        "mean_l1": float("nan"),
                        "mean_l2": float("nan"),
                        "mean_cosine": float("nan"),
                        "num_steps": len(ts),
                        "num_compared_in_zone": 0,
                    }
                else:
                    per_block_zone_error[rt][str(zid)] = {
                        "mean_l1": float(np.mean(zs_l1)),
                        "mean_l2": float(np.mean(zs_l2)),
                        "mean_cosine": float(np.mean(zs_cos)),
                        "num_steps": len(ts),
                        "num_compared_in_zone": len(zs_l1),
                    }

        global_summary = {
            "mean_l1": float(np.mean(all_l1)) if all_l1 else None,
            "mean_l2": float(np.mean(all_l2)) if all_l2 else None,
            "mean_cosine": float(np.mean(all_cos)) if all_cos else None,
            "num_entries": len(all_l1),
            "note": "包含 reuse 步：cache 側可為 cached tensor 與 baseline 全算比較。",
        }

        return {
            "per_block_step_error": per_block_step_error,
            "per_block_zone_error": per_block_zone_error,
            "global_summary": global_summary,
            "T": self.T,
            "time_axis_note": (
                "step_idx 0..T-1：0=第一步(DDIM t=T-1)，T-1=最後一步(DDIM t=0)；"
                "per_block_step_error key 為字串化 DDIM timestep t"
            ),
        }


def aggregate_per_timestep(
    per_block_step_error: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """對所有 block 聚合每個 DDIM timestep 的 mean/max L1。"""
    by_t: Dict[str, List[float]] = {}
    for _rt, steps in per_block_step_error.items():
        for t_str, m in steps.items():
            by_t.setdefault(t_str, []).append(float(m["l1"]))

    out: Dict[str, Dict[str, float]] = {}
    for t_str, vals in by_t.items():
        out[t_str] = {
            "mean_l1": float(np.mean(vals)),
            "max_l1": float(np.max(vals)),
        }
    return out
