"""
Stage2 (LDM) 主流程：
- 讀入 Stage1 scheduler_config.json
- baseline / cache 兩趟採樣並收集 runtime block 特徵誤差
- 單輪 refinement（zone 降 k + peak 補 mask）
- 輸出 refined scheduler 與診斷 JSON
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf


def _bootstrap_local_paths() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    candidate_paths = [
        repo_root,
        os.path.join(repo_root, "src", "taming-transformers"),
        os.path.join(repo_root, "src", "clip"),
    ]
    for path in candidate_paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_local_paths()

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential
from ldm.util import instantiate_from_config

from ldm_S3cache.cache_method.Stage2.stage2_error_collector_ldm import (
    Stage2ErrorCollectorLDM,
    aggregate_per_timestep,
)
from ldm_S3cache.cache_method.Stage2.stage2_scheduler_adapter_ldm import (
    EXPECTED_NUM_BLOCKS,
    FIRST_INPUT_RUNTIME_BLOCK_NAME,
    RUNTIME_LAYER_NAMES,
    apply_cache_scheduler_runtime_overrides,
    cache_runtime_override_variant_label,
    cache_scheduler_to_jsonable,
    ddim_timestep_to_step_index,
    load_stage1_scheduler_config,
    rebuild_expanded_mask_from_shared_zones_and_k_per_zone,
    runtime_name_to_block_id,
    stage1_block_to_runtime_block,
    stage1_mask_to_runtime_cache_scheduler,
    validate_stage1_scheduler_config,
)

LOGGER = logging.getLogger("Stage2RuntimeRefineLDM")
_STAGE2_DIR = Path(__file__).resolve().parent
_STAGE2_LOG_FILE = _STAGE2_DIR / "stage2_runtime_refine_ldm.log"
_LOG_FMT = logging.Formatter("%(asctime)s [%(levelname)s] [Stage2-LDM] %(message)s")


class MultiBlockRuntimeCacheLDM:
    """
    在 LDM UNet 上做 runtime block cache 模擬（hook-based）。

    注意：reuse 時會以 cached tensor 取代該層輸出，
    但 layer 計算仍先發生（hook 無法跳過 forward 計算）。
    """

    def __init__(
        self,
        *,
        unet: torch.nn.Module,
        runtime_to_unet_name: Dict[str, str],
        total_steps: int,
        recompute_timesteps_by_runtime: Dict[str, Set[int]],
        raw_t_to_step_idx: Optional[Dict[int, int]] = None,
        callback: Optional[Any] = None,
        cache_enabled: bool = True,
    ) -> None:
        self.unet = unet
        self.runtime_to_unet_name = dict(runtime_to_unet_name)
        self.total_steps = int(total_steps)
        self.recompute_timesteps_by_runtime = {
            k: set(int(x) for x in v) for k, v in recompute_timesteps_by_runtime.items()
        }
        self.raw_t_to_step_idx = (
            {int(k): int(v) for k, v in dict(raw_t_to_step_idx or {}).items()}
            if raw_t_to_step_idx is not None
            else None
        )
        self.callback = callback
        self.cache_enabled = bool(cache_enabled)

        self._hooks: List[Any] = []
        self._target_modules: Dict[str, torch.nn.Module] = {}
        self._cache: Dict[str, Optional[torch.Tensor]] = {
            rt: None for rt in self.runtime_to_unet_name
        }

        self._forward_counter = -1
        self._current_loop_step: Optional[int] = None
        self._current_ddim_t: Optional[int] = None
        self._last_loop_step: Optional[int] = None

        self.recompute_hits = 0
        self.cache_hits = 0

    def _resolve_module_from_unet_path(self, path: str) -> torch.nn.Module:
        cur: Any = self.unet
        for tok in path.split("."):
            if tok.isdigit():
                cur = cur[int(tok)]
            else:
                if not hasattr(cur, tok):
                    raise ValueError(f"unet path not found: {path!r} (missing token {tok!r})")
                cur = getattr(cur, tok)
        if not isinstance(cur, torch.nn.Module):
            raise TypeError(f"resolved object is not torch.nn.Module: path={path!r}")
        return cur

    def _model_pre_hook(
        self,
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        del module

        raw_t: Optional[int] = None
        t_obj: Any = None
        if "timesteps" in kwargs:
            t_obj = kwargs.get("timesteps")
        elif len(args) >= 2:
            t_obj = args[1]

        if torch.is_tensor(t_obj) and t_obj.numel() >= 1:
            raw_t = int(t_obj.flatten()[0].item())

        loop_step: Optional[int] = None
        ddim_t: Optional[int] = None
        if raw_t is not None and self.raw_t_to_step_idx is not None and raw_t in self.raw_t_to_step_idx:
            loop_step = int(self.raw_t_to_step_idx[raw_t])
            if 0 <= loop_step < self.total_steps:
                ddim_t = (self.total_steps - 1) - loop_step
            else:
                loop_step = None
                ddim_t = None

        self._forward_counter += 1
        if loop_step is None:
            loop_step = self._forward_counter % self.total_steps
            ddim_t = (self.total_steps - 1) - int(loop_step)

        self._current_loop_step = int(loop_step)
        self._current_ddim_t = ddim_t

        # New sample starts from step_idx=0 (DDIM t=T-1). Clear stale cache once at sequence start.
        if self._current_loop_step == 0 and self._last_loop_step != 0:
            for rt in self._cache:
                self._cache[rt] = None

        self._last_loop_step = int(self._current_loop_step)

    def _make_block_hook(self, runtime_name: str):
        def _hook(module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            if not torch.is_tensor(output):
                return output

            step = self._current_loop_step
            if step is None:
                return output

            if self._current_ddim_t is not None:
                ddim_t = int(self._current_ddim_t)
            else:
                ddim_t = (self.total_steps - 1) - int(step)

            if not self.cache_enabled:
                y = output
                recompute = True
                self.recompute_hits += 1
            else:
                cached = self._cache[runtime_name]
                should_recompute = (
                    (ddim_t in self.recompute_timesteps_by_runtime[runtime_name])
                    or (cached is None)
                )
                if should_recompute:
                    y = output
                    self._cache[runtime_name] = output.detach()
                    recompute = True
                    self.recompute_hits += 1
                else:
                    y = cached.clone()
                    recompute = False
                    self.cache_hits += 1

            if self.callback is not None:
                self.callback(
                    runtime_name,
                    y,
                    recompute=recompute,
                    ddim_timestep=int(ddim_t),
                )
            return y

        return _hook

    def __enter__(self) -> "MultiBlockRuntimeCacheLDM":
        if set(self.runtime_to_unet_name.keys()) != set(RUNTIME_LAYER_NAMES):
            raise ValueError(
                "runtime_to_unet_name keys mismatch runtime inventory; "
                f"got {sorted(self.runtime_to_unet_name.keys())}"
            )

        self._hooks.append(
            self.unet.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        )

        for rt in RUNTIME_LAYER_NAMES:
            mod = self._resolve_module_from_unet_path(self.runtime_to_unet_name[rt])
            if not isinstance(mod, TimestepEmbedSequential):
                raise TypeError(
                    f"runtime block {rt} -> {self.runtime_to_unet_name[rt]!r} is not TimestepEmbedSequential"
                )
            self._target_modules[rt] = mod
            self._hooks.append(mod.register_forward_hook(self._make_block_hook(rt)))

        LOGGER.info(
            "Runtime cache hooks enabled | cache_enabled=%s | blocks=%d",
            self.cache_enabled,
            len(self._target_modules),
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._target_modules.clear()
        for rt in self._cache:
            self._cache[rt] = None

    def stats(self) -> Dict[str, int]:
        return {
            "recompute_hits": int(self.recompute_hits),
            "cache_hits": int(self.cache_hits),
            "total_hook_calls": int(self.recompute_hits + self.cache_hits),
        }


def _configure_stage2_logging() -> None:
    if LOGGER.handlers:
        return
    LOGGER.setLevel(logging.INFO)

    h_err = logging.StreamHandler(sys.stderr)
    h_err.setFormatter(_LOG_FMT)
    LOGGER.addHandler(h_err)

    h_file = logging.FileHandler(_STAGE2_LOG_FILE, mode="a", encoding="utf-8")
    h_file.setFormatter(_LOG_FMT)
    LOGGER.addHandler(h_file)

    LOGGER.propagate = False


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / p).resolve()


def _resolve_resume_to_logdir_and_ckpt(resume: str) -> Tuple[Path, Path]:
    resume_p = _resolve_repo_path(resume)
    if not resume_p.exists():
        raise FileNotFoundError(f"Cannot find resume path: {resume_p}")

    if resume_p.is_file():
        logdir = resume_p.parent
        ckpt = resume_p
    else:
        logdir = resume_p
        ckpt = logdir / "model.ckpt"

    if not ckpt.is_file():
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt}")
    return logdir, ckpt


def _load_config(logdir: Path) -> Any:
    cfg_path = logdir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Cannot find config.yaml in: {logdir}")
    return OmegaConf.load(str(cfg_path))


def _load_model(config: Any, ckpt: Path, device: torch.device) -> torch.nn.Module:
    LOGGER.info("Loading model from: %s", ckpt)
    pl_sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state_dict = pl_sd.get("state_dict", pl_sd)

    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    LOGGER.info("load_state_dict: missing=%d unexpected=%d", len(missing), len(unexpected))

    model = model.to(device)
    model.eval()
    return model


def _canonical_to_unet_name(canonical: str) -> str:
    if canonical.startswith("model."):
        return canonical[len("model.") :]
    return canonical


def _load_block_map_runtime_to_unet(block_map_path: Path) -> Dict[str, str]:
    if not block_map_path.is_file():
        raise FileNotFoundError(f"block map not found: {block_map_path}")
    payload = json.loads(block_map_path.read_text(encoding="utf-8"))
    blocks = payload.get("blocks", [])
    if not blocks:
        raise ValueError(f"No blocks found in block map: {block_map_path}")

    runtime_to_unet: Dict[str, str] = {}
    for b in blocks:
        runtime = str(b["runtime_name"])
        canonical = str(b["canonical_name"])
        runtime_to_unet[runtime] = _canonical_to_unet_name(canonical)

    if set(runtime_to_unet.keys()) != set(RUNTIME_LAYER_NAMES):
        missing = sorted(set(RUNTIME_LAYER_NAMES) - set(runtime_to_unet.keys()))
        extra = sorted(set(runtime_to_unet.keys()) - set(RUNTIME_LAYER_NAMES))
        raise ValueError(
            "block map runtime names mismatch Stage1-LDM runtime inventory: "
            f"missing={missing}, extra={extra}"
        )
    return runtime_to_unet


def _run_single_sampling_pass(
    *,
    model: torch.nn.Module,
    sampler: DDIMSampler,
    x_T: torch.Tensor,
    num_steps: int,
    eta: float,
    runtime_ctx: Any,
) -> None:
    unet = model.model.diffusion_model
    shape = [unet.in_channels, unet.image_size, unet.image_size]
    bsz = int(x_T.shape[0])

    with model.ema_scope("stage2_ldm_refine"):
        with runtime_ctx:
            _samples, _ = sampler.sample(
                S=int(num_steps),
                batch_size=bsz,
                shape=shape,
                conditioning=None,
                eta=float(eta),
                verbose=False,
                x_T=x_T,
            )


def _aggregate_step_metrics_inplace(
    agg: Dict[str, Dict[str, Dict[str, float]]],
    per_block_step_error: Dict[str, Dict[str, Dict[str, float]]],
    *,
    batch_size: int,
) -> None:
    w = float(batch_size)
    for rt, steps in per_block_step_error.items():
        rt_acc = agg.setdefault(rt, {})
        for t_str, m in steps.items():
            slot = rt_acc.setdefault(
                t_str,
                {"sum_l1": 0.0, "sum_l2_sq": 0.0, "sum_cos": 0.0, "weight": 0.0},
            )
            l1 = float(m["l1"])
            l2 = float(m["l2"])
            cos = float(m["cosine"])
            slot["sum_l1"] += l1 * w
            slot["sum_l2_sq"] += (l2 * l2) * w
            slot["sum_cos"] += cos * w
            slot["weight"] += w


def _finalize_per_block_step_error(
    agg: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rt, steps in agg.items():
        row: Dict[str, Dict[str, float]] = {}
        for t_str, slot in steps.items():
            w = float(slot["weight"])
            if w <= 0.0:
                continue
            row[t_str] = {
                "l1": float(slot["sum_l1"] / w),
                "l2": float(math.sqrt(max(slot["sum_l2_sq"] / w, 0.0))),
                "cosine": float(slot["sum_cos"] / w),
            }
        out[rt] = row
    return out


def _compute_per_block_zone_error(
    per_block_step_error: Dict[str, Dict[str, Dict[str, float]]],
    shared_zones: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    per_block_zone_error: Dict[str, Dict[str, Dict[str, Any]]] = {}
    zone_ts: Dict[int, List[int]] = {}
    for z in shared_zones:
        zid = int(z["id"])
        ts = int(z["t_start"])
        te = int(z["t_end"])
        zone_ts[zid] = list(range(te, ts + 1))

    for rt, steps in per_block_step_error.items():
        per_block_zone_error[rt] = {}
        for zid, ts in zone_ts.items():
            zs_l1: List[float] = []
            zs_l2: List[float] = []
            zs_cos: List[float] = []
            for ddim_t in ts:
                st = steps.get(str(ddim_t))
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
    return per_block_zone_error


def _build_diagnostics_from_aggregated_steps(
    *,
    per_block_step_error: Dict[str, Dict[str, Dict[str, float]]],
    shared_zones: List[Dict[str, Any]],
    T: int,
) -> Dict[str, Any]:
    per_block_zone_error = _compute_per_block_zone_error(per_block_step_error, shared_zones)

    all_l1: List[float] = []
    all_l2: List[float] = []
    all_cos: List[float] = []
    for steps in per_block_step_error.values():
        for m in steps.values():
            all_l1.append(float(m["l1"]))
            all_l2.append(float(m["l2"]))
            all_cos.append(float(m["cosine"]))

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
        "T": int(T),
        "time_axis_note": (
            "step_idx 0..T-1：0=第一步(DDIM t=T-1)，T-1=最後一步(DDIM t=0)；"
            "per_block_step_error key 為字串化 DDIM timestep t"
        ),
    }


def _load_blockwise_threshold_config(
    path: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """讀取 blockwise threshold JSON；回傳 (runtime_name -> entry, block_id -> entry, root)。"""
    from ldm_S3cache.cache_method.Stage2.verify_stage2_ldm import (
        verify_blockwise_threshold_config_dict,
    )

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"threshold config not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    verify_blockwise_threshold_config_dict(data)

    by_runtime: Dict[str, Dict[str, Any]] = {}
    by_id: Dict[int, Dict[str, Any]] = {}
    for entry in data["per_block"]:
        bid = int(entry["block_id"])
        rt = str(entry["runtime_name"])
        if rt in by_runtime:
            raise ValueError(f"duplicate runtime_name in threshold config: {rt}")
        by_runtime[rt] = entry
        by_id[bid] = entry

    if len(by_runtime) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"threshold config must contain exactly {EXPECTED_NUM_BLOCKS} runtime entries, got {len(by_runtime)}"
        )
    if set(by_id.keys()) != set(range(EXPECTED_NUM_BLOCKS)):
        raise ValueError(
            f"threshold config must contain block_id 0..{EXPECTED_NUM_BLOCKS - 1}, got {sorted(by_id.keys())}"
        )

    return by_runtime, by_id, data


def _parse_force_full_runtime_blocks(s: Optional[str]) -> List[str]:
    if not s or not str(s).strip():
        return []

    out: List[str] = []
    for raw in str(s).split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok in RUNTIME_LAYER_NAMES:
            rt = tok
        else:
            if tok.startswith("input_blocks.") or tok == "middle_block" or tok.startswith("output_blocks."):
                tok = f"model.{tok}"
            rt = stage1_block_to_runtime_block(tok)
        if rt not in out:
            out.append(rt)
    return out


def _nonnegative_int(s: str) -> int:
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


def _cache_runtime_override_contract(*, override_active: bool) -> Dict[str, str]:
    return {
        "must_reapply_same_runtime_overrides_for_sampling": "true" if override_active else "false",
        "refined_json_expanded_mask_is_algorithm_only": "true",
        "diagnostics_cache_pass_scheduler_field": "cache_scheduler_effective_for_cache_pass",
        "interpretation": (
            "Safety overrides apply only to cache diagnostic pass (runtime union). "
            "stage2_refined_scheduler_config.json masks are Stage2 refine results unless merged elsewhere."
        ),
    }


def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(x) for x in obj)
    if isinstance(obj, np.floating):
        return _json_safe(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def run_stage2_refine_ldm(
    *,
    scheduler_config_path: str,
    output_dir: str,
    seed: int = 0,
    zone_l1_threshold: float = 0.02,
    peak_l1_threshold: float = 0.08,
    resume: str = "models/ldm/ffhq256/model.ckpt",
    block_map: str = "ldm_block_map_ffhq256.json",
    device: Optional[torch.device] = None,
    threshold_config_path: Optional[str] = None,
    force_full_prefix_steps: int = 0,
    force_full_runtime_blocks: Optional[List[str]] = None,
    safety_first_input_block: bool = False,
    eval_num_images: int = 8,
    eval_chunk_size: Optional[int] = 1,
    eta: float = 0.0,
) -> Dict[str, Any]:
    """Public function run_stage2_refine_ldm."""
    _configure_stage2_logging()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage2-LDM runtime refine.")

    device = device or torch.device("cuda")
    _seed_all(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_stage1_scheduler_config(scheduler_config_path)
    validate_stage1_scheduler_config(cfg)
    T = int(cfg["T"])
    shared_zones: List[Dict[str, Any]] = cfg["shared_zones"]

    if float(eta) != 0.0:
        LOGGER.warning("Stage2-LDM enforces eta=0.0 for deterministic compare. Using eta=0.0.")
    eta = 0.0

    cache_sched_stage1 = stage1_mask_to_runtime_cache_scheduler(cfg)

    blocks_eff = list(force_full_runtime_blocks or [])
    if safety_first_input_block and FIRST_INPUT_RUNTIME_BLOCK_NAME not in blocks_eff:
        blocks_eff.append(FIRST_INPUT_RUNTIME_BLOCK_NAME)

    cache_sched_effective, override_meta = apply_cache_scheduler_runtime_overrides(
        cache_sched_stage1,
        T,
        force_full_prefix_steps=int(force_full_prefix_steps),
        force_full_runtime_blocks=blocks_eff,
    )
    override_meta["variant_label"] = cache_runtime_override_variant_label(
        force_full_prefix_steps=int(force_full_prefix_steps),
        force_full_runtime_blocks=list(force_full_runtime_blocks or []),
        safety_first_input_block=bool(safety_first_input_block),
    )
    override_meta["force_full_runtime_blocks_effective"] = list(blocks_eff)
    override_meta["safety_first_input_block"] = bool(safety_first_input_block)

    _ov_active = bool(int(force_full_prefix_steps) > 0 or blocks_eff)
    _contract = _cache_runtime_override_contract(override_active=_ov_active)

    runtime_to_unet_name = _load_block_map_runtime_to_unet(_resolve_repo_path(block_map))

    logdir, ckpt = _resolve_resume_to_logdir_and_ckpt(resume)
    config = _load_config(logdir)
    model = _load_model(config, ckpt, device=device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=T, ddim_eta=eta, verbose=False)
    ddim_desc = [int(x) for x in np.flip(sampler.ddim_timesteps)]
    if len(ddim_desc) != T:
        raise RuntimeError(
            f"Unexpected DDIM schedule length: got {len(ddim_desc)}, expected T={T}"
        )
    raw_t_to_step_idx = {int(raw_t): int(step_idx) for step_idx, raw_t in enumerate(ddim_desc)}

    total_eval_images = int(eval_num_images)
    if total_eval_images < 1:
        raise ValueError(f"eval_num_images must be >= 1, got {total_eval_images}")
    chunk_size = (
        int(eval_chunk_size) if eval_chunk_size is not None else min(4, total_eval_images)
    )
    if chunk_size < 1:
        raise ValueError(f"eval_chunk_size must be >= 1, got {chunk_size}")
    chunk_size = min(chunk_size, total_eval_images)

    unet = model.model.diffusion_model
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    agg_step_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    done = 0
    chunk_idx = 0

    while done < total_eval_images:
        bsz = min(chunk_size, total_eval_images - done)
        x_T = torch.randn(
            (bsz, unet.in_channels, unet.image_size, unet.image_size),
            generator=rng,
            device=device,
        )

        collector = Stage2ErrorCollectorLDM(T=T, device=device)
        cb = collector.make_cache_debug_callback()

        try:
            collector.set_run("baseline")
            _seed_all(seed + chunk_idx)
            with MultiBlockRuntimeCacheLDM(
                unet=unet,
                runtime_to_unet_name=runtime_to_unet_name,
                total_steps=T,
                recompute_timesteps_by_runtime=cache_sched_effective,
                raw_t_to_step_idx=raw_t_to_step_idx,
                callback=cb,
                cache_enabled=False,
            ) as baseline_ctx:
                _run_single_sampling_pass(
                    model=model,
                    sampler=sampler,
                    x_T=x_T,
                    num_steps=T,
                    eta=eta,
                    runtime_ctx=baseline_ctx,
                )
            LOGGER.info(
                "%s | baseline_hook_stats=%s",
                collector.debug_snapshot_line(f"after_baseline_chunk_{chunk_idx}"),
                baseline_ctx.stats(),
            )

            collector.set_run("cache")
            _seed_all(seed + chunk_idx)
            with MultiBlockRuntimeCacheLDM(
                unet=unet,
                runtime_to_unet_name=runtime_to_unet_name,
                total_steps=T,
                recompute_timesteps_by_runtime=cache_sched_effective,
                raw_t_to_step_idx=raw_t_to_step_idx,
                callback=cb,
                cache_enabled=True,
            ) as cache_ctx:
                _run_single_sampling_pass(
                    model=model,
                    sampler=sampler,
                    x_T=x_T,
                    num_steps=T,
                    eta=eta,
                    runtime_ctx=cache_ctx,
                )
            LOGGER.info(
                "%s | cache_hook_stats=%s",
                collector.debug_snapshot_line(f"after_cache_chunk_{chunk_idx}"),
                cache_ctx.stats(),
            )

            chunk_diag = collector.compute_diagnostics(shared_zones)
            _aggregate_step_metrics_inplace(
                agg_step_metrics,
                chunk_diag["per_block_step_error"],
                batch_size=bsz,
            )
        finally:
            collector.clear_storage()
            del collector

        done += bsz
        chunk_idx += 1

    per_block_step_agg = _finalize_per_block_step_error(agg_step_metrics)
    diagnostics = _build_diagnostics_from_aggregated_steps(
        per_block_step_error=per_block_step_agg,
        shared_zones=shared_zones,
        T=T,
    )
    diagnostics["cache_scheduler_input"] = cache_scheduler_to_jsonable(cache_sched_stage1)
    diagnostics["cache_scheduler_effective_for_cache_pass"] = cache_scheduler_to_jsonable(
        cache_sched_effective
    )
    diagnostics["cache_scheduler_runtime_overrides"] = dict(override_meta)
    diagnostics["cache_runtime_override_contract"] = dict(_contract)
    diagnostics["scheduler_config_path"] = str(Path(scheduler_config_path).resolve())

    per_block_step = diagnostics["per_block_step_error"]
    per_block_zone = diagnostics["per_block_zone_error"]
    per_t = aggregate_per_timestep(per_block_step)

    blockwise_by_runtime: Optional[Dict[str, Dict[str, Any]]] = None
    blockwise_by_id: Optional[Dict[int, Dict[str, Any]]] = None
    threshold_mode = "global"
    threshold_meta_diag: Dict[str, Any] = {
        "threshold_mode": "global",
        "global_zone_l1": float(zone_l1_threshold),
        "global_peak_l1": float(peak_l1_threshold),
        "note": (
            "Single global thresholds from CLI. Use build_blockwise_thresholds_ldm.py + "
            "--threshold-config for per-block thresholds."
        ),
    }

    if threshold_config_path:
        blockwise_by_runtime, blockwise_by_id, tc_doc = _load_blockwise_threshold_config(
            threshold_config_path
        )
        threshold_mode = "blockwise_quantile"
        threshold_meta_diag = {
            "threshold_mode": threshold_mode,
            "threshold_config_path": str(Path(threshold_config_path).resolve()),
            "method": tc_doc.get("method"),
            "source": tc_doc.get("source"),
            "source_diagnostics_path": tc_doc.get("source_diagnostics_path"),
            "q_zone": tc_doc.get("q_zone"),
            "q_peak": tc_doc.get("q_peak"),
            "peak_over_zone_ratio_min": tc_doc.get("peak_over_zone_ratio_min"),
        }

    diagnostics["stage2_threshold_meta"] = threshold_meta_diag
    diagnostics["block_identity_semantics"] = {
        "scheduler_local_block_id": "scheduler-local id from scheduler_config blocks[].id",
        "canonical_runtime_block_id": "canonical runtime index from runtime_name",
    }

    refined = copy.deepcopy(cfg)
    refined["version"] = "stage2_refined_v1_ldm"
    refined["stage2_meta"] = {
        "zone_l1_threshold": float(zone_l1_threshold),
        "peak_l1_threshold": float(peak_l1_threshold),
        "seed": int(seed),
        "threshold_mode": threshold_mode,
        "threshold_config_path": str(Path(threshold_config_path).resolve())
        if threshold_config_path
        else None,
        "cache_runtime_overrides": {
            "variant_label": override_meta.get("variant_label"),
            "force_full_prefix_steps": int(force_full_prefix_steps),
            "force_full_runtime_blocks_effective": list(blocks_eff),
            "safety_first_input_block": bool(safety_first_input_block),
            "first_input_runtime_block_name": FIRST_INPUT_RUNTIME_BLOCK_NAME,
            **_contract,
        },
    }

    k_touch: List[Dict[str, Any]] = []
    blocks = sorted(refined["blocks"], key=lambda b: int(b["id"]))

    for b in blocks:
        rt = stage1_block_to_runtime_block(str(b["name"]))
        runtime_bid = runtime_name_to_block_id(rt)
        if blockwise_by_runtime is not None and rt not in blockwise_by_runtime:
            raise RuntimeError(f"threshold config missing runtime_name {rt} (block id={b['id']})")

        zone_thr_used = (
            float(blockwise_by_runtime[rt]["zone_l1_threshold"])
            if blockwise_by_runtime is not None
            else float(zone_l1_threshold)
        )

        kz = [int(x) for x in b["k_per_zone"]]
        for z in shared_zones:
            zid = int(z["id"])
            st = per_block_zone.get(rt, {}).get(str(zid), {})
            ml1 = float(st.get("mean_l1", 0.0))
            if not math.isnan(ml1) and ml1 > zone_thr_used:
                if zid < 0 or zid >= len(kz):
                    raise RuntimeError(f"block {b['id']}: bad zone id {zid}")
                old = kz[zid]
                kz[zid] = max(1, old - 1)
                k_touch.append(
                    {
                        "block_id": b["id"],
                        "scheduler_local_block_id": int(b["id"]),
                        "runtime_name": rt,
                        "canonical_runtime_block_id": int(runtime_bid),
                        "zone_id": zid,
                        "k_before": old,
                        "k_after": kz[zid],
                        "threshold_mode": threshold_mode,
                        "zone_l1_threshold_used": zone_thr_used,
                    }
                )
        b["k_per_zone"] = kz

    for b in blocks:
        bid = int(b["id"])
        b["expanded_mask"] = rebuild_expanded_mask_from_shared_zones_and_k_per_zone(
            shared_zones,
            [int(x) for x in b["k_per_zone"]],
            T,
            block_id=bid,
        )

    mask_touch: List[Dict[str, Any]] = []
    for b in blocks:
        rt = stage1_block_to_runtime_block(str(b["name"]))
        runtime_bid = runtime_name_to_block_id(rt)
        peak_thr_used = (
            float(blockwise_by_runtime[rt]["peak_l1_threshold"])
            if blockwise_by_runtime is not None
            else float(peak_l1_threshold)
        )
        row = list(b["expanded_mask"])

        for t_str, m in per_block_step.get(rt, {}).items():
            ddim_t = int(t_str)
            if float(m["l1"]) <= peak_thr_used:
                continue
            si = ddim_timestep_to_step_index(ddim_t, T)
            was_reuse = not bool(row[si])
            row[si] = True
            mask_touch.append(
                {
                    "block_id": b["id"],
                    "scheduler_local_block_id": int(b["id"]),
                    "runtime_name": rt,
                    "canonical_runtime_block_id": int(runtime_bid),
                    "ddim_timestep": ddim_t,
                    "step_index": si,
                    "was_reuse_before_peak_repair": was_reuse,
                    "expanded_mask_after": True,
                    "threshold_mode": threshold_mode,
                    "peak_l1_threshold_used": peak_thr_used,
                }
            )
        b["expanded_mask"] = row

    for b in blocks:
        rt = stage1_block_to_runtime_block(str(b["name"]))
        runtime_bid = runtime_name_to_block_id(rt)
        if not bool(b["expanded_mask"][0]):
            b["expanded_mask"][0] = True
            peak_thr_used = (
                float(blockwise_by_runtime[rt]["peak_l1_threshold"])
                if blockwise_by_runtime is not None
                else float(peak_l1_threshold)
            )
            mask_touch.append(
                {
                    "block_id": b["id"],
                    "scheduler_local_block_id": int(b["id"]),
                    "runtime_name": rt,
                    "canonical_runtime_block_id": int(runtime_bid),
                    "note": "enforce first step full compute (step_idx=0 -> DDIM t=T-1)",
                    "expanded_mask_after": True,
                    "threshold_mode": threshold_mode,
                    "peak_l1_threshold_used": peak_thr_used,
                }
            )

    refined_cache_sched = stage1_mask_to_runtime_cache_scheduler(refined)
    diagnostics["refined_cache_scheduler"] = cache_scheduler_to_jsonable(refined_cache_sched)

    per_block_thr_summary: Optional[List[Dict[str, Any]]] = None
    if blockwise_by_id is not None:
        per_block_thr_summary = [
            {
                "block_id": int(blockwise_by_id[i]["block_id"]),
                "canonical_runtime_block_id": int(
                    blockwise_by_id[i].get(
                        "canonical_runtime_block_id",
                        blockwise_by_id[i]["block_id"],
                    )
                ),
                "canonical_name": blockwise_by_id[i]["canonical_name"],
                "runtime_name": blockwise_by_id[i]["runtime_name"],
                "zone_l1_threshold": float(blockwise_by_id[i]["zone_l1_threshold"]),
                "peak_l1_threshold": float(blockwise_by_id[i]["peak_l1_threshold"]),
            }
            for i in range(EXPECTED_NUM_BLOCKS)
        ]

    summary = {
        "cache_runtime_overrides": {
            "variant_label": override_meta.get("variant_label"),
            "force_full_prefix_steps": int(force_full_prefix_steps),
            "force_full_runtime_blocks_effective": list(blocks_eff),
            "safety_first_input_block": bool(safety_first_input_block),
            "first_input_runtime_block_name": FIRST_INPUT_RUNTIME_BLOCK_NAME,
            **_contract,
            "note": (
                "Diagnostics/refinement used cache pass scheduler = "
                "cache_scheduler_effective_for_cache_pass. Refined JSON masks are Stage2 results only."
            ),
        },
        "zone_k_adjustments": k_touch,
        "peak_mask_adjustments": mask_touch,
        "block_identity_semantics": {
            "block_id": "backward-compatible alias of scheduler_local_block_id",
            "scheduler_local_block_id": "id in scheduler JSON; do not interpret as canonical runtime index",
            "canonical_runtime_block_id": "canonical runtime index matching runtime_name",
        },
        "threshold_mode": threshold_mode,
        "global_thresholds": {
            "zone_l1": float(zone_l1_threshold),
            "peak_l1": float(peak_l1_threshold),
        },
        "per_block_thresholds": per_block_thr_summary,
        "thresholds": {
            "zone_l1": float(zone_l1_threshold),
            "peak_l1": float(peak_l1_threshold),
        },
        "aggregate_per_ddim_timestep_l1": per_t,
    }

    with open(out / "stage2_runtime_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(diagnostics), f, indent=2, ensure_ascii=False)
    with open(out / "stage2_refined_scheduler_config.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(refined), f, indent=2, ensure_ascii=False)
    with open(out / "stage2_refinement_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)
    with open(out / "cache_runtime_overrides_run.json", "w", encoding="utf-8") as f:
        json.dump(
            _json_safe(
                {
                    "stage": "stage2_runtime_refine_ldm",
                    "scheduler_config_path": str(Path(scheduler_config_path).resolve()),
                    "T": int(T),
                    "seed": int(seed),
                    **dict(override_meta),
                    "cache_runtime_override_contract": _contract,
                }
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    LOGGER.info("寫入 %s", out)

    return {
        "output_dir": str(out),
        "diagnostics": diagnostics,
        "summary": summary,
        "refined_config": refined,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage2 runtime refine for LDM (single-pass). Run twice for pass1/pass2.",
    )

    g_in = p.add_argument_group("Stage1 input / output")
    g_in.add_argument("--scheduler_config", type=str, required=True)
    g_in.add_argument("--output_dir", type=str, required=True)
    g_in.add_argument("--seed", type=int, default=0)

    g_thr = p.add_argument_group("Threshold (global unless --threshold-config)")
    g_thr.add_argument("--zone_l1_threshold", type=float, default=0.02)
    g_thr.add_argument("--peak_l1_threshold", type=float, default=0.08)
    g_thr.add_argument("--threshold-config", type=str, default=None)

    g_model = p.add_argument_group("Model")
    g_model.add_argument("--resume", type=str, default="models/ldm/ffhq256/model.ckpt")
    g_model.add_argument("--block_map", type=str, default="ldm_block_map_ffhq256.json")
    g_model.add_argument("--eta", type=float, default=0.0)

    g_eval = p.add_argument_group("Diagnostics eval")
    g_eval.add_argument("--eval-num-images", type=_nonnegative_int, default=8)
    g_eval.add_argument("--eval-chunk-size", type=_nonnegative_int, default=1)

    g_safe = p.add_argument_group("Safety overrides (cache pass only)")
    g_safe.add_argument("--force-full-prefix-steps", type=_nonnegative_int, default=0)
    g_safe.add_argument(
        "--force-full-runtime-blocks",
        type=str,
        default="",
        help="Comma separated runtime names (or canonical names) to force full compute.",
    )
    g_safe.add_argument("--safety-first-input-block", action="store_true")

    args = p.parse_args()

    _configure_stage2_logging()
    LOGGER.info(
        "----- Stage2-LDM run start | scheduler_config=%s | output_dir=%s | seed=%s -----",
        args.scheduler_config,
        args.output_dir,
        args.seed,
    )

    run_stage2_refine_ldm(
        scheduler_config_path=args.scheduler_config,
        output_dir=args.output_dir,
        seed=int(args.seed),
        zone_l1_threshold=float(args.zone_l1_threshold),
        peak_l1_threshold=float(args.peak_l1_threshold),
        resume=args.resume,
        block_map=args.block_map,
        threshold_config_path=args.threshold_config,
        force_full_prefix_steps=int(args.force_full_prefix_steps),
        force_full_runtime_blocks=_parse_force_full_runtime_blocks(args.force_full_runtime_blocks),
        safety_first_input_block=bool(args.safety_first_input_block),
        eval_num_images=int(args.eval_num_images),
        eval_chunk_size=int(args.eval_chunk_size),
        eta=float(args.eta),
    )


if __name__ == "__main__":
    main()
