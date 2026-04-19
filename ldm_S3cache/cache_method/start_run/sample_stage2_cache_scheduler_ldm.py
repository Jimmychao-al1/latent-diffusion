#!/usr/bin/env python3
"""
LDM start_run entrypoint for formal FID@5K experiments.

Modes:
- baseline: native LDM DDIM sampling, no scheduler, must use --no_npz
- cache: Stage2 refined scheduler + runtime cache hook

Important:
- Scheduler decisions are made on loop index (0..199), not raw DDIM timestep.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import random
import shutil
import sys
import time
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
    RUNTIME_LAYER_NAMES,
    TIME_ORDER_EXPECTED,
    stage1_block_to_runtime_block,
)
from scripts.sample_diffusion import (
    compute_fid,
    export_real_from_image_dir,
    export_real_from_lmdb,
    run,
)


# Fixed formal experiment spec (do not change)
DDIM_STEPS = 200
ETA = 0.0
SEED = 0
BATCH_SIZE = 32
N_SAMPLES = 5000
FID_DIMS = 2048
DATASET_TAG = "ffhq256"
FID_AT = "5k"


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


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


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


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
    # DDIM execution order is descending raw timesteps (e.g. 999 -> 0)
    raw_desc = [int(x) for x in np.flip(sampler.ddim_timesteps)]
    return {raw_t: step_idx for step_idx, raw_t in enumerate(raw_desc)}


def _load_stage2_scheduler_loop_idx(path: Path, expected_steps: int) -> Dict[str, Set[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    T = int(payload.get("T", -1))
    if T != expected_steps:
        raise ValueError(f"Scheduler T={T} mismatch expected steps={expected_steps}")
    if payload.get("time_order") != TIME_ORDER_EXPECTED:
        raise ValueError(
            f"Scheduler time_order must be {TIME_ORDER_EXPECTED!r}, got {payload.get('time_order')!r}"
        )
    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        raise TypeError("scheduler_json['blocks'] must be a list")

    sched: Dict[str, Set[int]] = {}
    for b in blocks:
        name = str(b.get("name", ""))
        runtime_name = str(b.get("runtime_name") or stage1_block_to_runtime_block(name))
        expanded_mask = b.get("expanded_mask")
        if not isinstance(expanded_mask, list) or len(expanded_mask) != T:
            raise ValueError(
                f"Block {name!r}: expanded_mask length must be T={T}, "
                f"got {len(expanded_mask) if isinstance(expanded_mask, list) else type(expanded_mask)}"
            )
        # Critical: use loop index (0..T-1) directly.
        recompute_loop_steps = {int(i) for i, flag in enumerate(expanded_mask) if bool(flag)}
        if any(si < 0 or si >= T for si in recompute_loop_steps):
            raise ValueError(f"Block {runtime_name!r} has invalid loop step index in scheduler")
        sched[runtime_name] = recompute_loop_steps

    if set(sched.keys()) != set(RUNTIME_LAYER_NAMES):
        missing = sorted(set(RUNTIME_LAYER_NAMES) - set(sched.keys()))
        extra = sorted(set(sched.keys()) - set(RUNTIME_LAYER_NAMES))
        raise ValueError(
            "Runtime block mismatch in scheduler. "
            f"missing={missing}, extra={extra}"
        )
    return sched


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


class _Stage2LoopIndexCacheHook:
    """
    Runtime cache hook that always compares scheduler against loop index 0..T-1.
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
        self._current_loop_idx: Optional[int] = None
        self._last_loop_idx: Optional[int] = None
        self._forward_counter = -1
        self._warned_raw_t: Set[int] = set()

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

        if raw_t is not None and raw_t in self.raw_t_to_loop_idx:
            loop_idx = int(self.raw_t_to_loop_idx[raw_t])
        else:
            loop_idx = int(self._forward_counter % self.total_steps)
            if raw_t is not None and raw_t not in self._warned_raw_t:
                self._warned_raw_t.add(raw_t)
                print(
                    f"[WARN] raw timestep {raw_t} not found in raw_t_to_loop_idx; "
                    f"fallback to counter loop_idx={loop_idx}"
                )

        self._current_loop_idx = loop_idx
        if self._current_loop_idx == 0 and self._last_loop_idx != 0:
            for rt in self._cached:
                self._cached[rt] = None
        self._last_loop_idx = self._current_loop_idx

    def _make_block_hook(self, runtime_name: str):
        def _hook(module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            if not torch.is_tensor(output):
                return output
            loop_idx = self._current_loop_idx
            if loop_idx is None:
                return output
            cached = self._cached[runtime_name]
            should_recompute = (
                (loop_idx in self.recompute_loop_steps_by_runtime[runtime_name]) or (cached is None)
            )
            if should_recompute:
                self._cached[runtime_name] = output.detach()
                return output
            return cached.clone()

        return _hook

    def __enter__(self) -> "_Stage2LoopIndexCacheHook":
        if set(self.recompute_loop_steps_by_runtime.keys()) != set(RUNTIME_LAYER_NAMES):
            raise ValueError("scheduler runtime keys mismatch RUNTIME_LAYER_NAMES")

        self._hooks.append(
            self.unet.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        )
        for rt in RUNTIME_LAYER_NAMES:
            mod = self._resolve_module(_runtime_to_unet_path(rt))
            if not isinstance(mod, TimestepEmbedSequential):
                raise TypeError(f"{rt} resolved module is not TimestepEmbedSequential")
            self._hooks.append(mod.register_forward_hook(self._make_block_hook(rt)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for rt in self._cached:
            self._cached[rt] = None


def apply_s3cache_scheduler(
    model: torch.nn.Module,
    scheduler_cfg: Dict[str, Set[int]],
    *,
    raw_t_to_loop_idx: Dict[int, int],
    total_steps: int,
) -> _Stage2LoopIndexCacheHook:
    return _Stage2LoopIndexCacheHook(
        model=model,
        recompute_loop_steps_by_runtime=scheduler_cfg,
        raw_t_to_loop_idx=raw_t_to_loop_idx,
        total_steps=total_steps,
    )


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
    p = argparse.ArgumentParser(description="LDM Stage2 scheduler FID@5K entrypoint")
    p.add_argument("--mode", type=str, required=True, choices=["baseline", "cache"])
    p.add_argument("--no_npz", action="store_true", help="Baseline only: do not save npz")
    p.add_argument("--ckpt", type=str, default="models/ldm/ffhq256/model.ckpt")
    p.add_argument("--config", type=str, default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml")
    p.add_argument("--real_image_dir", type=str, default=None)
    p.add_argument("--real_lmdb", type=str, default=None)
    p.add_argument("--scheduler_json", type=str, default=None)
    p.add_argument("--scheduler_name", type=str, default="unknown_scheduler")
    p.add_argument("--out_root", type=str, default="outputs")
    p.add_argument("--results_json", type=str, default="results/fid_results_ldm.json")
    return p


def main() -> None:
    args = _build_parser().parse_args()
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

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = "baseline_no_npz" if args.mode == "baseline" else f"stage2_{args.scheduler_name}"

    out_root = _resolve_repo_path(args.out_root)
    run_dir = out_root / "start_run" / f"fid_{FID_AT}" / f"{timestamp}_{run_label}"
    gen_dir = run_dir / "gen_images"
    eval_dir = out_root / "start_run" / f"real_eval_cache_{DATASET_TAG}_{FID_AT}"
    _reset_dir(gen_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] mode={args.mode} dataset={DATASET_TAG} n_samples={N_SAMPLES}")
    print(f"[INFO] gen_dir={gen_dir}")
    print(f"[INFO] eval_dir={eval_dir}")

    model = _load_model(config_path, ckpt_path)
    raw_t_to_loop_idx = _build_raw_t_to_loop_idx(model, DDIM_STEPS, ETA)

    scheduler_cfg_loop_idx: Optional[Dict[str, Set[int]]] = None
    scheduler_json_abs: Optional[Path] = None
    if args.mode == "cache":
        scheduler_json_abs = _resolve_repo_path(str(args.scheduler_json))
        if not scheduler_json_abs.is_file():
            raise FileNotFoundError(f"scheduler_json not found: {scheduler_json_abs}")
        scheduler_cfg_loop_idx = _load_stage2_scheduler_loop_idx(scheduler_json_abs, DDIM_STEPS)
        print(f"[INFO] scheduler_json={scheduler_json_abs}")
        print("[INFO] scheduler semantics: loop index 0..199 (not raw DDIM timestep)")

    t0 = time.time()
    if args.mode == "cache":
        assert scheduler_cfg_loop_idx is not None
        with apply_s3cache_scheduler(
            model,
            scheduler_cfg_loop_idx,
            raw_t_to_loop_idx=raw_t_to_loop_idx,
            total_steps=DDIM_STEPS,
        ):
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
        raise RuntimeError(f"generated image count mismatch: expected {N_SAMPLES}, got {gen_count}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = float(compute_fid(str(eval_dir), str(gen_dir), BATCH_SIZE, device, FID_DIMS))
    fid_time_min = (time.time() - t1) / 60.0

    results_json = _resolve_repo_path(args.results_json)
    record = {
        "run_type": "baseline_no_npz" if args.mode == "baseline" else "stage2_scheduler",
        "scheduler_name": args.scheduler_name,
        "scheduler_json": str(scheduler_json_abs) if scheduler_json_abs is not None else None,
        "fid": fid,
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
    }
    _append_json_list(results_json, record)

    print(f"[DONE] FID={fid:.6f}")
    print(f"[DONE] results_json={results_json}")


if __name__ == "__main__":
    main()
