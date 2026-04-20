#!/usr/bin/env python3
"""
LDM start_run entrypoint for formal FID experiments with optional Stage2 scheduler.

Goals:
- Baseline and cache runs share the same output/reporting contract.
- Cache scheduler decision is strictly based on loop index (0..T-1).
- Output artifacts align with DiffAE start_run style.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import glob
import json
import logging
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
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
from ldm_S3cache.cache_method.Stage2.stage2_scheduler_adapter_ldm import (
    FIRST_INPUT_RUNTIME_BLOCK_NAME,
    RUNTIME_LAYER_NAMES,
    apply_cache_scheduler_runtime_overrides,
    cache_runtime_override_variant_label,
    cache_scheduler_to_jsonable,
    load_stage1_scheduler_config,
    stage1_block_to_runtime_block,
    stage1_mask_to_runtime_cache_scheduler,
    validate_stage1_scheduler_config,
)
from scripts.sample_diffusion import (
    compute_fid,
    export_real_from_image_dir,
    export_real_from_lmdb,
    run,
)


# Fixed formal experiment spec
DDIM_STEPS = 200
ETA = 0.0
SEED = 0
BATCH_SIZE = 32
N_SAMPLES = 5000
FID_DIMS = 2048
DATASET_TAG = "ffhq256"
FID_AT = "5k"

REPO_ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger("LDMStartRun")


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def _setup_logging(log_file: Optional[str]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        p = _resolve_repo_path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(p), encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [LDM-start_run] %(message)s",
        handlers=handlers,
        force=True,
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _count_pngs(dir_path: Path) -> int:
    return len(glob.glob(str(dir_path / "*.png"))) if dir_path.exists() else 0


def _append_json_list(path: Path, record: Dict[str, Any]) -> None:
    history: List[Any] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                history = existing
            elif isinstance(existing, dict):
                history = [existing]
        except Exception:
            history = []
    history.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")


def _append_runs_index(index_path: Path, entry: Dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _fid_index_key(num_images: int) -> str:
    if num_images == 5000:
        return "FID@5K"
    if num_images == 50000:
        return "FID@50K"
    return f"FID@{num_images}"


def _round_fid_index(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return round(float(x), 3)


def _sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\\-]", "_", s)


def _compact_run_index_id(start_dt: dt.datetime, scheduler_name: str) -> str:
    return f"{start_dt.strftime('%m%d%H%M%S')}__{_sanitize_name(scheduler_name)}"


def _repo_rel_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p.resolve())


def _get_git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _nonnegative_int(s: str) -> int:
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


def _parse_force_full_runtime_blocks(s: str) -> List[str]:
    if not s.strip():
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_model(config_path: Path, ckpt_path: Path) -> torch.nn.Module:
    config = OmegaConf.load(str(config_path))
    pl_sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = pl_sd.get("state_dict", pl_sd)
    model = instantiate_from_config(config.model)
    model.load_state_dict(state_dict, strict=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    model = model.cuda()
    model.eval()
    return model


def _build_raw_t_to_loop_idx(model: torch.nn.Module, ddim_steps: int, eta: float) -> Dict[int, int]:
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)
    # DDIM execution order is descending raw timesteps (e.g. 999 -> 0).
    raw_desc = [int(x) for x in np.flip(sampler.ddim_timesteps)]
    return {raw_t: step_idx for step_idx, raw_t in enumerate(raw_desc)}


def _ddim_to_loop_scheduler(cache_scheduler_ddim: Dict[str, Set[int]], T: int) -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = {}
    for rt, rec_ddim in cache_scheduler_ddim.items():
        out[rt] = {int((T - 1) - i) for i in rec_ddim}
    return out


def _cfg_snapshot_with_effective_expanded_masks(
    base_cfg: Dict[str, Any],
    effective_ddim: Dict[str, Set[int]],
    *,
    T: int,
) -> Dict[str, Any]:
    out = copy.deepcopy(base_cfg)
    for b in out.get("blocks", []):
        rt = stage1_block_to_runtime_block(str(b["name"]))
        rec = effective_ddim[rt]
        row: List[bool] = []
        for step_idx in range(T):
            ddim_i = (T - 1) - step_idx
            row.append(bool(ddim_i in rec))
        b["expanded_mask"] = row
    out["snapshot_meta"] = {
        "source": "effective_runtime_cache_scheduler_after_cli_overrides",
        "expanded_mask_semantics": "step_idx=0 is first denoise step; True=full compute",
        "note": (
            "If runtime overrides are applied (e.g. force-full-prefix), "
            "k_per_zone may not reconstruct expanded_mask exactly."
        ),
    }
    return out


def _compute_schedule_stats_from_effective_runtime_scheduler(
    cache_scheduler_ddim: Dict[str, Set[int]],
    *,
    T: int,
    shared_zones: List[Dict[str, Any]],
) -> Dict[str, Any]:
    keys = set(cache_scheduler_ddim.keys())
    if keys != set(RUNTIME_LAYER_NAMES):
        raise ValueError(
            "effective cache scheduler keys mismatch runtime layers; "
            f"missing={sorted(set(RUNTIME_LAYER_NAMES)-keys)} extra={sorted(keys-set(RUNTIME_LAYER_NAMES))}"
        )

    zone_ts: Dict[int, List[int]] = {}
    for z in shared_zones:
        zid = int(z["id"])
        t_s, t_e = int(z["t_start"]), int(z["t_end"])
        zone_ts[zid] = list(range(t_e, t_s + 1))

    total_full = 0
    total_reuse = 0
    full_blocks = 0
    per_block_full: Dict[str, int] = {}
    per_block_reuse: Dict[str, int] = {}
    per_zone: Dict[str, Dict[str, Any]] = {
        str(zid): {"full_count": 0, "reuse_count": 0, "num_timesteps_in_zone": len(ts)}
        for zid, ts in zone_ts.items()
    }

    for rt in RUNTIME_LAYER_NAMES:
        rec = cache_scheduler_ddim[rt]
        n_full = len(rec)
        n_reuse = T - n_full
        total_full += n_full
        total_reuse += n_reuse
        per_block_full[rt] = n_full
        per_block_reuse[rt] = n_reuse
        if n_full == T:
            full_blocks += 1
        for zid, ts in zone_ts.items():
            zkey = str(zid)
            for i in ts:
                if i in rec:
                    per_zone[zkey]["full_count"] += 1
                else:
                    per_zone[zkey]["reuse_count"] += 1

    num_blocks = len(RUNTIME_LAYER_NAMES)
    num_cells = T * num_blocks
    return {
        "T": T,
        "num_blocks": num_blocks,
        "total_full_compute_count": total_full,
        "total_cache_reuse_count": total_reuse,
        "full_compute_ratio": round(total_full / num_cells, 6),
        "full_compute_blocks_count": full_blocks,
        "per_block_recompute_count": per_block_full,
        "per_block_reuse_count": per_block_reuse,
        "per_block_recompute_ratio": {
            rt: round(per_block_full[rt] / T, 6) for rt in RUNTIME_LAYER_NAMES
        },
        "per_zone_recompute_stats": per_zone,
    }


def _format_effective_scheduler_fr_grid(cache_scheduler_ddim: Dict[str, Set[int]], *, T: int) -> Dict[str, Any]:
    ddim_order = list(range(T - 1, -1, -1))
    name_w = max(len(x) for x in RUNTIME_LAYER_NAMES)
    by_layer: Dict[str, str] = {}
    aligned_lines: List[str] = []
    for rt in RUNTIME_LAYER_NAMES:
        rec = cache_scheduler_ddim[rt]
        cells = [f"{('F' if i in rec else 'R'):^2}" for i in ddim_order]
        row = " ".join(cells)
        by_layer[rt] = row
        aligned_lines.append(f"{rt.ljust(name_w)} : {row}")
    return {
        "meta": {
            "source": "runtime_conf_cache_scheduler_after_cli_overrides",
            "timestep_order": "A_ddim_execution",
            "columns_left_to_right": "DDIM timestep i from T-1 down to 0",
            "ddim_timestep_i_left_to_right": ddim_order,
            "T": T,
            "num_blocks": len(RUNTIME_LAYER_NAMES),
            "cell_token": "F=full_compute R=cache_reuse",
        },
        "by_layer": by_layer,
        "aligned_lines": aligned_lines,
        "aligned_text_block": "\n".join(aligned_lines),
    }


def _load_stage2_summary_stats(scheduler_json_path: Path) -> Dict[str, Any]:
    summary_p = scheduler_json_path.parent / "stage2_refinement_summary.json"
    if not summary_p.is_file():
        return {}
    try:
        data = json.loads(summary_p.read_text(encoding="utf-8"))
        zone_adj: List[Dict[str, Any]] = data.get("zone_k_adjustments", [])
        peak_adj: List[Dict[str, Any]] = data.get("peak_mask_adjustments", [])
        per_zone_adj: Dict[str, int] = {}
        for e in zone_adj:
            zid = str(e.get("zone_id", "?"))
            per_zone_adj[zid] = per_zone_adj.get(zid, 0) + 1
        return {
            "zone_adjustments_count": len(zone_adj),
            "peak_adjustments_count": len(peak_adj),
            "per_zone_adjustment_stats": per_zone_adj,
        }
    except Exception as e:
        LOGGER.warning("Could not load stage2_refinement_summary.json: %s", e)
        return {}


def _load_effective_cache_scheduler(
    *,
    scheduler_json: str,
    num_steps: int,
    force_full_prefix_steps: int = 0,
    force_full_runtime_blocks: Optional[List[str]] = None,
    safety_first_input_block: bool = False,
    allow_missing_k_per_zone: bool = False,
) -> Tuple[Dict[str, Set[int]], Dict[str, Set[int]], Dict[str, Any], Dict[str, Any], Path]:
    cfg_path = _resolve_repo_path(scheduler_json)
    cfg = load_stage1_scheduler_config(cfg_path)
    validate_stage1_scheduler_config(cfg, require_k_per_zone=not allow_missing_k_per_zone)
    T_cfg = int(cfg["T"])
    if T_cfg != int(num_steps):
        raise ValueError(
            f"Scheduler T mismatch: scheduler T={T_cfg}, sampling num_steps={num_steps}"
        )

    sched_ddim = stage1_mask_to_runtime_cache_scheduler(
        cfg,
        require_k_per_zone=not allow_missing_k_per_zone,
    )
    blocks_eff = list(force_full_runtime_blocks or [])
    if safety_first_input_block and FIRST_INPUT_RUNTIME_BLOCK_NAME not in blocks_eff:
        blocks_eff.append(FIRST_INPUT_RUNTIME_BLOCK_NAME)
    effective_ddim, override_meta = apply_cache_scheduler_runtime_overrides(
        sched_ddim,
        T_cfg,
        force_full_prefix_steps=int(force_full_prefix_steps),
        force_full_runtime_blocks=blocks_eff,
    )
    override_meta["variant_label"] = cache_runtime_override_variant_label(
        force_full_prefix_steps=int(force_full_prefix_steps),
        force_full_runtime_blocks=list(force_full_runtime_blocks or []),
        safety_first_input_block=bool(safety_first_input_block),
    )
    override_meta["force_full_runtime_blocks_effective"] = list(blocks_eff)
    effective_loop = _ddim_to_loop_scheduler(effective_ddim, T_cfg)

    run_record: Dict[str, Any] = {
        "stage": "fid_sampling",
        "scheduler_config_path": str(cfg_path.resolve()),
        "T": T_cfg,
        "allow_missing_k_per_zone": bool(allow_missing_k_per_zone),
        **override_meta,
        "cache_scheduler_effective_for_sampling_ddim_i": cache_scheduler_to_jsonable(
            effective_ddim
        ),
    }
    return effective_ddim, effective_loop, run_record, cfg, cfg_path


def _runtime_to_unet_path(runtime_name: str) -> str:
    if runtime_name.startswith("encoder_layer_"):
        idx = int(runtime_name.split("_")[-1])
        return f"input_blocks.{idx}"
    if runtime_name == "middle_layer":
        return "middle_block"
    if runtime_name.startswith("decoder_layer_"):
        idx = int(runtime_name.split("_")[-1])
        return f"output_blocks.{idx}"
    raise ValueError(f"Unknown runtime_name: {runtime_name}")


class Stage2LoopIndexCacheHook:
    """
    Runtime cache hook using loop-index decisions (true skip mode).

    Note:
    - Reuse step returns cached tensor before calling target block forward.
    - Therefore underlying block compute is truly skipped on bypass path.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        recompute_loop_steps_by_runtime: Dict[str, Set[int]],
        raw_t_to_loop_idx: Dict[int, int],
        total_steps: int,
    ) -> None:
        self.model = model
        self.unet = model.model.diffusion_model
        self.recompute_loop_steps_by_runtime = {
            k: set(int(v) for v in vals) for k, vals in recompute_loop_steps_by_runtime.items()
        }
        self.raw_t_to_loop_idx = {int(k): int(v) for k, v in raw_t_to_loop_idx.items()}
        self.total_steps = int(total_steps)

        self._hooks: List[Any] = []
        self._cached: Dict[str, Optional[torch.Tensor]] = {
            rt: None for rt in RUNTIME_LAYER_NAMES
        }
        self._target_modules: Dict[str, torch.nn.Module] = {}
        self._orig_forward_by_runtime: Dict[str, Any] = {}
        self._had_instance_forward_attr: Dict[str, bool] = {}
        self._current_loop_idx: Optional[int] = None
        self._current_raw_t: Optional[int] = None
        self._last_loop_idx: Optional[int] = None
        self._forward_counter = -1
        self._warned_raw_t: Set[int] = set()
        self._fallback_count = 0

        self.cache_hits = 0  # alias of bypass_hits for backward compatibility
        self.recompute_hits = 0
        self.bypass_hits = 0
        self.per_block_cache_hits: Dict[str, int] = {rt: 0 for rt in RUNTIME_LAYER_NAMES}  # alias
        self.per_block_bypass_hits: Dict[str, int] = {rt: 0 for rt in RUNTIME_LAYER_NAMES}
        self.per_block_recompute_hits: Dict[str, int] = {rt: 0 for rt in RUNTIME_LAYER_NAMES}
        self.forced_recompute_due_to_empty_cache = 0

    def _resolve_module(self, path: str) -> torch.nn.Module:
        cur: Any = self.unet
        for tok in path.split("."):
            if tok.isdigit():
                cur = cur[int(tok)]
            else:
                cur = getattr(cur, tok)
        if not isinstance(cur, torch.nn.Module):
            raise TypeError(f"Resolved path is not module: {path}")
        return cur

    def _model_pre_hook(
        self,
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        del module
        self._forward_counter += 1

        t_obj = kwargs.get("timesteps") if "timesteps" in kwargs else (args[1] if len(args) >= 2 else None)
        raw_t: Optional[int] = None
        if torch.is_tensor(t_obj) and t_obj.numel() >= 1:
            raw_t = int(t_obj.flatten()[0].item())
        self._current_raw_t = raw_t

        if raw_t is not None and raw_t in self.raw_t_to_loop_idx:
            loop_idx = int(self.raw_t_to_loop_idx[raw_t])
        else:
            loop_idx = int(self._forward_counter % self.total_steps)
            self._fallback_count += 1
            if raw_t is not None and raw_t not in self._warned_raw_t:
                self._warned_raw_t.add(raw_t)
                LOGGER.warning(
                    "raw timestep %s not found in raw_t_to_loop_idx; fallback to counter loop_idx=%s",
                    raw_t,
                    loop_idx,
                )

        self._current_loop_idx = loop_idx
        if self._current_loop_idx == 0 and self._last_loop_idx != 0:
            for rt in self._cached:
                self._cached[rt] = None
        self._last_loop_idx = self._current_loop_idx

    def _make_true_skip_forward(self, runtime_name: str, orig_forward: Any):
        def _forward(*args: Any, **kwargs: Any) -> Any:
            loop_idx = self._current_loop_idx
            cached = self._cached[runtime_name]
            if loop_idx is None:
                # Outside expected denoise path: fallback to original compute.
                y = orig_forward(*args, **kwargs)
                if torch.is_tensor(y):
                    self._cached[runtime_name] = y.detach()
                self.recompute_hits += 1
                self.per_block_recompute_hits[runtime_name] += 1
                return y

            scheduled_recompute = loop_idx in self.recompute_loop_steps_by_runtime[runtime_name]
            should_recompute = scheduled_recompute or (cached is None)
            if should_recompute:
                if (not scheduled_recompute) and (cached is None):
                    self.forced_recompute_due_to_empty_cache += 1
                y = orig_forward(*args, **kwargs)
                if torch.is_tensor(y):
                    self._cached[runtime_name] = y.detach()
                self.recompute_hits += 1
                self.per_block_recompute_hits[runtime_name] += 1
                return y

            self.bypass_hits += 1
            self.cache_hits += 1
            self.per_block_bypass_hits[runtime_name] += 1
            self.per_block_cache_hits[runtime_name] += 1
            return cached.clone()

        return _forward

    def __enter__(self) -> "Stage2LoopIndexCacheHook":
        if set(self.recompute_loop_steps_by_runtime.keys()) != set(RUNTIME_LAYER_NAMES):
            raise ValueError("scheduler runtime keys mismatch RUNTIME_LAYER_NAMES")
        self._hooks.append(
            self.unet.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        )
        for rt in RUNTIME_LAYER_NAMES:
            mod = self._resolve_module(_runtime_to_unet_path(rt))
            if not isinstance(mod, TimestepEmbedSequential):
                raise TypeError(f"{rt} resolved module is not TimestepEmbedSequential")
            self._target_modules[rt] = mod
            self._had_instance_forward_attr[rt] = "forward" in mod.__dict__
            orig_forward = mod.forward
            self._orig_forward_by_runtime[rt] = orig_forward
            mod.forward = self._make_true_skip_forward(rt, orig_forward)  # type: ignore[method-assign]
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for rt, mod in self._target_modules.items():
            if self._had_instance_forward_attr.get(rt, False):
                mod.forward = self._orig_forward_by_runtime[rt]  # type: ignore[method-assign]
            else:
                # restore class-defined forward descriptor
                if "forward" in mod.__dict__:
                    delattr(mod, "forward")
        self._target_modules.clear()
        self._orig_forward_by_runtime.clear()
        self._had_instance_forward_attr.clear()
        for rt in self._cached:
            self._cached[rt] = None

    def stats(self) -> Dict[str, Any]:
        total = int(self.bypass_hits + self.recompute_hits)
        return {
            "cache_execution_mode": "true_skip",
            "cache_hits": int(self.cache_hits),
            "bypass_hits": int(self.bypass_hits),
            "recompute_hits": int(self.recompute_hits),
            "total_block_calls": total,
            "total_hook_calls": total,  # backward-compatible alias
            "cache_ratio": float(self.bypass_hits / total) if total > 0 else 0.0,
            "bypass_ratio": float(self.bypass_hits / total) if total > 0 else 0.0,
            "recompute_ratio": float(self.recompute_hits / total) if total > 0 else 0.0,
            "per_block_cache_hits": dict(self.per_block_cache_hits),
            "per_block_bypass_hits": dict(self.per_block_bypass_hits),
            "per_block_recompute_hits": dict(self.per_block_recompute_hits),
            "forced_recompute_due_to_empty_cache": int(self.forced_recompute_due_to_empty_cache),
            "raw_t_fallback_count": int(self._fallback_count),
        }


def _prepare_real_images(
    *,
    eval_dir: Path,
    n_samples: int,
    real_image_dir: Optional[str],
    real_lmdb: Optional[str],
) -> None:
    if bool(real_image_dir) == bool(real_lmdb):
        raise ValueError("Provide exactly one of --real_image_dir or --real_lmdb")

    if real_image_dir:
        export_real_from_image_dir(
            real_image_dir=str(_resolve_repo_path(real_image_dir)),
            eval_dir=str(eval_dir),
            num_images=n_samples,
            img_size=256,
            real_image_list=None,
        )
    else:
        export_real_from_lmdb(
            lmdb_path=str(_resolve_repo_path(str(real_lmdb))),
            eval_dir=str(eval_dir),
            num_images=n_samples,
            img_size=256,
            lmdb_resolution=256,
            lmdb_zfill=5,
        )

    real_count = _count_pngs(eval_dir)
    if real_count < n_samples:
        raise RuntimeError(f"real image count {real_count} < n_samples {n_samples}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LDM Stage2 scheduler FID entrypoint")
    p.add_argument("--mode", type=str, required=True, choices=["baseline", "cache"])
    p.add_argument("--no_npz", action="store_true", help="Baseline only: do not save npz")
    p.add_argument("--ckpt", type=str, default="models/ldm/ffhq256/model.ckpt")
    p.add_argument("--config", type=str, default="models/ldm/ffhq256/config.yaml")
    p.add_argument("--real_image_dir", type=str, default=None)
    p.add_argument("--real_lmdb", type=str, default=None)
    p.add_argument("--scheduler_json", type=str, default=None)
    p.add_argument("--scheduler_name", type=str, default="unknown")
    p.add_argument("--out_root", type=str, default="outputs")
    p.add_argument("--results_json", type=str, default="results/fid_results_ldm.json")

    p.add_argument("--force-full-prefix-steps", type=_nonnegative_int, default=0)
    p.add_argument("--force-full-runtime-blocks", type=str, default="")
    p.add_argument("--safety-first-input-block", action="store_true")
    p.add_argument("--allow-missing-k-per-zone", action="store_true")

    p.add_argument("--run-output-dir", type=str, default=None)
    p.add_argument("--runs-index-path", type=str, default=None)
    p.add_argument("--log_file", type=str, default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    _setup_logging(args.log_file)

    if args.mode == "baseline" and not args.no_npz:
        raise ValueError("Baseline mode must set --no_npz for formal start_run.")
    if args.mode == "cache" and args.no_npz:
        raise ValueError("--no_npz is baseline-only; do not set it in cache mode.")
    if args.mode == "cache" and not args.scheduler_json:
        raise ValueError("Cache mode requires --scheduler_json")

    _seed_all(SEED)

    ckpt_path = _resolve_repo_path(args.ckpt)
    config_path = _resolve_repo_path(args.config)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")

    start_dt = dt.datetime.now()
    timestamp = start_dt.strftime("%Y%m%d_%H%M%S")
    default_scheduler_name = "baseline_no_npz" if args.mode == "baseline" else str(args.scheduler_name)
    scheduler_name = str(args.scheduler_name) if args.scheduler_name else default_scheduler_name
    run_label = "baseline_no_npz" if args.mode == "baseline" else f"stage2_{scheduler_name}"

    out_root = _resolve_repo_path(args.out_root)
    if args.run_output_dir:
        run_output_dir = _resolve_repo_path(args.run_output_dir)
    else:
        run_output_dir = out_root / "start_run" / "results" / start_dt.strftime("%Y%m%d") / scheduler_name / f"{start_dt.strftime('%m%d_%H')}_{scheduler_name}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    gen_dir = out_root / "start_run" / f"fid_{FID_AT}" / f"{timestamp}_{run_label}" / "gen_images"
    eval_dir = out_root / "start_run" / f"real_eval_cache_{DATASET_TAG}_{FID_AT}"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    scheduler_json_abs: Optional[Path] = None
    effective_ddim: Optional[Dict[str, Set[int]]] = None
    effective_loop: Optional[Dict[str, Set[int]]] = None
    scheduler_cfg: Optional[Dict[str, Any]] = None
    override_run_record: Optional[Dict[str, Any]] = None
    hook_stats: Dict[str, Any] = {}
    stage2_sidecar_stats: Dict[str, Any] = {}

    runs_index_path = _resolve_repo_path(args.runs_index_path) if args.runs_index_path else None
    run_manifest = {
        "run_id": f"{start_dt.strftime('%Y%m%dT%H%M%S')}__{_sanitize_name(scheduler_name)}",
        "status": "running",
        "start_time": start_dt.isoformat(timespec="seconds"),
        "end_time": None,
        "duration_sec": None,
        "mode": args.mode,
        "scheduler_name": scheduler_name,
        "scheduler_config_path": None,
        "num_images": N_SAMPLES,
        "seed": SEED,
        "script_path": str(Path(__file__).resolve()),
        "command_argv": sys.argv[:],
        "git_commit": _get_git_commit(),
        "hostname": socket.gethostname(),
        "output_dir": str(run_output_dir.resolve()),
    }
    _write_json(run_output_dir / "run_manifest.json", run_manifest)

    score: Optional[float] = None
    sample_time_min = 0.0
    fid_time_min = 0.0
    try:
        LOGGER.info("mode=%s dataset=%s n_samples=%s", args.mode, DATASET_TAG, N_SAMPLES)
        LOGGER.info("gen_dir=%s", gen_dir)
        LOGGER.info("eval_dir=%s", eval_dir)

        model = _load_model(config_path, ckpt_path)
        raw_t_to_loop_idx = _build_raw_t_to_loop_idx(model, DDIM_STEPS, ETA)

        if args.mode == "cache":
            effective_ddim, effective_loop, override_run_record, scheduler_cfg, scheduler_json_abs = _load_effective_cache_scheduler(
                scheduler_json=str(args.scheduler_json),
                num_steps=DDIM_STEPS,
                force_full_prefix_steps=int(args.force_full_prefix_steps),
                force_full_runtime_blocks=_parse_force_full_runtime_blocks(str(args.force_full_runtime_blocks)),
                safety_first_input_block=bool(args.safety_first_input_block),
                allow_missing_k_per_zone=bool(args.allow_missing_k_per_zone),
            )
            run_manifest["scheduler_config_path"] = str(scheduler_json_abs.resolve())
            stage2_sidecar_stats = _load_stage2_summary_stats(scheduler_json_abs)
            LOGGER.info("scheduler_json=%s", scheduler_json_abs)
            LOGGER.info("scheduler semantics: loop index 0..%d", DDIM_STEPS - 1)

        t0 = time.time()
        if args.mode == "cache":
            assert effective_loop is not None
            with Stage2LoopIndexCacheHook(
                model=model,
                recompute_loop_steps_by_runtime=effective_loop,
                raw_t_to_loop_idx=raw_t_to_loop_idx,
                total_steps=DDIM_STEPS,
            ) as hook:
                run(
                    model,
                    str(gen_dir),
                    batch_size=BATCH_SIZE,
                    vanilla=False,
                    custom_steps=DDIM_STEPS,
                    eta=ETA,
                    n_samples=N_SAMPLES,
                    nplog=None,
                )
                hook_stats = hook.stats()
        else:
            run(
                model,
                str(gen_dir),
                batch_size=BATCH_SIZE,
                vanilla=False,
                custom_steps=DDIM_STEPS,
                eta=ETA,
                n_samples=N_SAMPLES,
                nplog=None,
            )
        sample_time_min = (time.time() - t0) / 60.0

        t1 = time.time()
        _prepare_real_images(
            eval_dir=eval_dir,
            n_samples=N_SAMPLES,
            real_image_dir=args.real_image_dir,
            real_lmdb=args.real_lmdb,
        )
        gen_count = _count_pngs(gen_dir)
        if gen_count != N_SAMPLES:
            raise RuntimeError(
                f"generated image count mismatch: expected {N_SAMPLES}, got {gen_count}"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        score = float(compute_fid(str(eval_dir), str(gen_dir), BATCH_SIZE, device, FID_DIMS))
        fid_time_min = (time.time() - t1) / 60.0

    except Exception as e:
        end_dt = dt.datetime.now()
        run_manifest.update(
            {
                "status": "failed",
                "end_time": end_dt.isoformat(timespec="seconds"),
                "duration_sec": round((end_dt - start_dt).total_seconds(), 2),
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(run_output_dir / "run_manifest.json", run_manifest)
        if runs_index_path is not None:
            fk = _fid_index_key(N_SAMPLES)
            _append_runs_index(
                runs_index_path,
                {
                    "rid": _compact_run_index_id(start_dt, scheduler_name),
                    fk: _round_fid_index(None),
                    "d": start_dt.strftime("%Y%m%d"),
                    "sch": scheduler_name,
                    "r": None,
                    "seed": SEED,
                    "out": _repo_rel_path(run_output_dir),
                    "sum": None,
                    "st": "failed",
                },
            )
        raise

    # Success artifacts
    end_dt = dt.datetime.now()
    duration_sec = round((end_dt - start_dt).total_seconds(), 2)
    run_manifest.update(
        {
            "status": "success",
            "end_time": end_dt.isoformat(timespec="seconds"),
            "duration_sec": duration_sec,
            "scheduler_config_path": str(scheduler_json_abs.resolve()) if scheduler_json_abs else None,
        }
    )
    _write_json(run_output_dir / "run_manifest.json", run_manifest)

    sched_stats: Dict[str, Any] = {}
    if args.mode == "cache":
        assert effective_ddim is not None and scheduler_cfg is not None
        sched_stats = _compute_schedule_stats_from_effective_runtime_scheduler(
            effective_ddim,
            T=int(scheduler_cfg["T"]),
            shared_zones=list(scheduler_cfg.get("shared_zones", [])),
        )
        snapshot = _cfg_snapshot_with_effective_expanded_masks(
            scheduler_cfg,
            effective_ddim,
            T=int(scheduler_cfg["T"]),
        )
        _write_json(run_output_dir / "scheduler_config.snapshot.json", snapshot)
        if override_run_record is not None:
            override_run_record["fid_score"] = score
            override_run_record["hook_stats"] = hook_stats
            _write_json(run_output_dir / "cache_runtime_overrides_run.json", override_run_record)

    summary_obj: Dict[str, Any] = {
        "run_id": run_manifest["run_id"],
        "status": "success",
        "mode": args.mode,
        "cache_execution_mode": hook_stats.get("cache_execution_mode"),
        "scheduler_name": scheduler_name,
        "scheduler_config_path": run_manifest.get("scheduler_config_path"),
        "force-prefix": "T" if int(args.force_full_prefix_steps) > 0 else "F",
        "force-full-prefix-steps": int(args.force_full_prefix_steps),
        "num_images": N_SAMPLES,
        "seed": SEED,
        "fid_5k": score,
        "sample_time_min": float(sample_time_min),
        "fid_time_min": float(fid_time_min),
        "full_compute_ratio": sched_stats.get("full_compute_ratio"),
        "total_full_compute_count": sched_stats.get("total_full_compute_count"),
        "total_cache_reuse_count": sched_stats.get("total_cache_reuse_count"),
        "full_compute_blocks_count": sched_stats.get("full_compute_blocks_count"),
        "zone_adjustments_count": stage2_sidecar_stats.get("zone_adjustments_count"),
        "peak_adjustments_count": stage2_sidecar_stats.get("peak_adjustments_count"),
        "cache_hook_cache_hits": hook_stats.get("cache_hits"),
        "cache_bypass_hits": hook_stats.get("bypass_hits"),
        "cache_hook_recompute_hits": hook_stats.get("recompute_hits"),
        "cache_hook_cache_ratio": hook_stats.get("cache_ratio"),
        "cache_bypass_ratio": hook_stats.get("bypass_ratio"),
        "forced_recompute_due_to_empty_cache": hook_stats.get("forced_recompute_due_to_empty_cache"),
        "start_time": run_manifest["start_time"],
        "end_time": run_manifest["end_time"],
        "duration_sec": run_manifest["duration_sec"],
        "detail_stats_path": str((run_output_dir / "detail_stats.json").resolve()),
    }
    _write_json(run_output_dir / "summary.json", summary_obj)

    detail_stats_obj: Dict[str, Any] = {
        "run_id": run_manifest["run_id"],
        "cache_execution_mode": hook_stats.get("cache_execution_mode"),
        "hook_stats": hook_stats,
        "per_block_bypass_hits": hook_stats.get("per_block_bypass_hits", {}),
        "per_block_recompute_count": sched_stats.get("per_block_recompute_count", {}),
        "per_block_reuse_count": sched_stats.get("per_block_reuse_count", {}),
        "per_zone_recompute_stats": sched_stats.get("per_zone_recompute_stats", {}),
        "per_zone_adjustment_stats": stage2_sidecar_stats.get("per_zone_adjustment_stats", {}),
        "raw_estimation_stats": {
            "full_compute_ratio": sched_stats.get("full_compute_ratio"),
            "total_full_compute_count": sched_stats.get("total_full_compute_count"),
            "total_cache_reuse_count": sched_stats.get("total_cache_reuse_count"),
            "full_compute_blocks_count": sched_stats.get("full_compute_blocks_count"),
            "T": sched_stats.get("T"),
            "num_blocks": sched_stats.get("num_blocks"),
            "per_block_recompute_ratio": sched_stats.get("per_block_recompute_ratio", {}),
        },
    }
    if args.mode == "cache" and effective_ddim is not None:
        detail_stats_obj["effective_scheduler_fr_grid"] = _format_effective_scheduler_fr_grid(
            effective_ddim,
            T=DDIM_STEPS,
        )
    _write_json(run_output_dir / "detail_stats.json", detail_stats_obj)

    if runs_index_path is not None:
        fk = _fid_index_key(N_SAMPLES)
        _append_runs_index(
            runs_index_path,
            {
                "rid": _compact_run_index_id(start_dt, scheduler_name),
                fk: _round_fid_index(score),
                "d": start_dt.strftime("%Y%m%d"),
                "sch": scheduler_name,
                "r": sched_stats.get("full_compute_ratio"),
                "seed": SEED,
                "out": _repo_rel_path(run_output_dir),
                "sum": _repo_rel_path(run_output_dir / "summary.json"),
                "st": "success",
            },
        )

    results_json = _resolve_repo_path(args.results_json)
    record = {
        "run_type": "baseline_no_npz" if args.mode == "baseline" else "stage2_scheduler",
        "scheduler_name": scheduler_name,
        "scheduler_json": str(scheduler_json_abs) if scheduler_json_abs is not None else None,
        "fid": score,
        "fid_at": FID_AT,
        "n_samples": N_SAMPLES,
        "dataset_tag": DATASET_TAG,
        "ddim_steps": DDIM_STEPS,
        "eta": ETA,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "fid_dims": FID_DIMS,
        "no_npz": bool(args.no_npz),
        "timestamp": timestamp,
        "sample_time_min": float(sample_time_min),
        "fid_time_min": float(fid_time_min),
        "gen_dir": str(gen_dir.resolve()),
        "eval_dir": str(eval_dir.resolve()),
        "ckpt": str(ckpt_path.resolve()),
        "run_output_dir": str(run_output_dir.resolve()),
        "cache_hook_cache_hits": hook_stats.get("cache_hits"),
        "cache_bypass_hits": hook_stats.get("bypass_hits"),
        "cache_hook_recompute_hits": hook_stats.get("recompute_hits"),
        "cache_bypass_ratio": hook_stats.get("bypass_ratio"),
        "forced_recompute_due_to_empty_cache": hook_stats.get("forced_recompute_due_to_empty_cache"),
    }
    _append_json_list(results_json, record)

    LOGGER.info("FID=%.6f", float(score))
    LOGGER.info("results_json=%s", results_json)
    LOGGER.info("run_output_dir=%s", run_output_dir)


if __name__ == "__main__":
    main()
