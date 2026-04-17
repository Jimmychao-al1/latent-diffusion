#!/usr/bin/env python3
"""
LDM b_SVD - Stage A feature collection (with optional in-memory A->B->C pipeline).

Collects per-step block features for one target block over DDIM sampling steps.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm.auto import tqdm


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

LOGGER = logging.getLogger("LDM_SVD_FeatureCollector")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _setup_logger(log_file: Optional[str]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = _resolve_repo_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [b_SVD-LDM] %(message)s",
        handlers=handlers,
        force=True,
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    cfg_files = sorted(glob.glob(str(logdir / "config.yaml")))
    if not cfg_files:
        raise FileNotFoundError(f"Cannot find config.yaml in: {logdir}")
    configs = [OmegaConf.load(cfg) for cfg in cfg_files]
    return OmegaConf.merge(*configs)


def _load_model(config: Any, ckpt: Path) -> torch.nn.Module:
    LOGGER.info("Loading model from: %s", ckpt)
    pl_sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state_dict = pl_sd.get("state_dict", pl_sd)

    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    LOGGER.info("load_state_dict: missing=%d unexpected=%d", len(missing), len(unexpected))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LDM SVD feature collection.")

    model = model.cuda()
    model.eval()
    return model


def _canonical_block_name(name: str) -> str:
    n = str(name).strip()
    for prefix in (
        "model.model.diffusion_model.",
        "model.diffusion_model.",
    ):
        if n.startswith(prefix):
            n = n[len(prefix) :]

    if n.startswith("model."):
        short = n[len("model.") :]
        if short.startswith("input_blocks.") or short == "middle_block" or short.startswith("output_blocks."):
            return "model." + short
        return n

    if n.startswith("input_blocks.") or n == "middle_block" or n.startswith("output_blocks."):
        return "model." + n
    return n


def _is_supported_unet_block_name(name: str) -> bool:
    return name.startswith("input_blocks.") or name == "middle_block" or name.startswith("output_blocks.")


class SvdFeatureCollector:
    def __init__(
        self,
        save_root: Path,
        max_timesteps: int,
        target_n: int,
        target_block: str,
    ):
        self.save_root = save_root
        self.max_timesteps = int(max_timesteps)
        self.target_n = int(target_n)
        self.target_block = _canonical_block_name(target_block)

        self.block_slug = self.target_block.replace(".", "_")
        self.hooks: List[Any] = []

        self._step_counter = -1
        self.current_step_idx = None

        self.feature_buffers: Dict[int, List[torch.Tensor]] = {t: [] for t in range(self.max_timesteps)}
        self.feature_counts: Dict[int, int] = {t: 0 for t in range(self.max_timesteps)}

        self.c = None
        self.h = None
        self.w = None

    def _create_step_pre_hook(self):
        def pre_hook(module, args, kwargs):
            del module, args, kwargs
            self._step_counter = (self._step_counter + 1) % self.max_timesteps
            self.current_step_idx = self.max_timesteps - 1 - self._step_counter

        return pre_hook

    def _create_block_hook(self, block_name: str):
        def hook_fn(module, inputs, output):
            del module, inputs
            if self._step_counter is None or self._step_counter < 0:
                return

            step_idx = int(self._step_counter)
            if not (0 <= step_idx < self.max_timesteps):
                return

            if self.feature_counts[step_idx] >= self.target_n:
                return

            if not torch.is_tensor(output):
                return

            out = output.detach().cpu()
            remain = self.target_n - self.feature_counts[step_idx]
            if out.shape[0] > remain:
                out = out[:remain]

            if self.c is None:
                _, self.c, self.h, self.w = out.shape
                LOGGER.info("Feature shape: C=%s H=%s W=%s", self.c, self.h, self.w)

            self.feature_buffers[step_idx].append(out)
            self.feature_counts[step_idx] += int(out.shape[0])

        return hook_fn

    def register_hooks(self, unet: nn.Module) -> None:
        registered = []
        for name, module in unet.named_modules():
            if not isinstance(module, TimestepEmbedSequential):
                continue
            if not _is_supported_unet_block_name(name):
                continue

            canonical = f"model.{name}"
            if canonical != self.target_block:
                continue

            self.hooks.append(module.register_forward_hook(self._create_block_hook(canonical)))
            registered.append(canonical)
            LOGGER.info("Register block hook: %s", canonical)

        if not registered:
            raise ValueError(
                f"Target block not found: {self.target_block}. "
                "Expected model.input_blocks.* / model.middle_block / model.output_blocks.*"
            )

        self.hooks.append(unet.register_forward_pre_hook(self._create_step_pre_hook(), with_kwargs=True))

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def min_collected(self) -> int:
        return int(min(self.feature_counts.values()))

    def has_enough(self) -> bool:
        return self.min_collected() >= self.target_n

    def build_feature_tensors(self) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        features: List[torch.Tensor] = []

        per_t_counts = []
        for t in range(self.max_timesteps):
            chunks = self.feature_buffers[t]
            if len(chunks) == 0:
                raise RuntimeError(f"No feature data at t={t}")
            tensor_t = torch.cat(chunks, dim=0)
            per_t_counts.append(int(tensor_t.shape[0]))
            features.append(tensor_t)

        actual_n = min(min(per_t_counts), self.target_n)
        trimmed = [x[:actual_n] for x in features]

        meta = {
            "block": self.block_slug,
            "target_block_name": self.target_block,
            "N": int(actual_n),
            "T": int(self.max_timesteps),
            "C": int(self.c),
            "H": int(self.h),
            "W": int(self.w),
            "index_convention": "analysis_index: 0..T-1",
            "display_t_mapping": "display_ddim_t=(T-1)-analysis_index",
        }
        return trimmed, meta

    def finalize(self) -> None:
        features, meta = self.build_feature_tensors()

        output_dir = self.save_root / self.block_slug
        output_dir.mkdir(parents=True, exist_ok=True)

        for t, tensor_t in enumerate(tqdm(features, desc="Writing features")):
            torch.save(tensor_t, output_dir / f"t_{t}.pt")

        with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        LOGGER.info("Finalize done: %s", output_dir)


@torch.no_grad()
def run_sampling_collection(
    *,
    model: torch.nn.Module,
    collector: SvdFeatureCollector,
    num_steps: int,
    batch_size: int,
    max_batches: int,
) -> None:
    unet = model.model.diffusion_model
    sampler = DDIMSampler(model)
    shape = [unet.in_channels, unet.image_size, unet.image_size]

    collector.register_hooks(unet)
    try:
        with model.ema_scope("SVD Feature Collection"):
            for bi in range(max_batches):
                if collector.has_enough():
                    LOGGER.info("Reached target_N for all timesteps at batch=%d", bi)
                    break

                samples, _ = sampler.sample(
                    S=num_steps,
                    batch_size=batch_size,
                    shape=shape,
                    eta=0.0,
                    verbose=False,
                )
                del samples
                torch.cuda.empty_cache()

            if not collector.has_enough():
                LOGGER.warning(
                    "Sampling ended without full target_N. min_collected=%d target_N=%d",
                    collector.min_collected(),
                    collector.target_n,
                )
    finally:
        collector.remove_hooks()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LDM SVD Feature Collection")
    p.add_argument("--resume", type=str, required=True, help="LDM checkpoint path or logdir")
    p.add_argument("--num_steps", "--n", type=int, default=200, help="DDIM steps T")
    p.add_argument("--svd_target_block", type=str, required=True, help="target block, e.g. model.output_blocks.0")
    p.add_argument("--svd_target_N", type=int, default=32, help="samples collected per timestep")
    p.add_argument(
        "--svd_output_root",
        type=str,
        default="ldm_S3cache/cache_method/b_SVD",
        help="output root",
    )
    p.add_argument("--log_file", "--lf", type=str, default=None, help="optional log file")

    p.add_argument("--batch_size", type=int, default=16, help="sampling batch size")
    p.add_argument("--max_batches", type=int, default=64, help="hard cap for sampling loops")

    p.add_argument(
        "--in_memory_pipeline",
        action="store_true",
        help="run Stage A->B->C in-memory (no t_{t}.pt writes)",
    )
    p.add_argument("--representative-t", type=int, default=-1)
    p.add_argument("--energy-threshold", type=float, default=0.98)
    p.add_argument("--no-compute-energy", action="store_true")
    p.add_argument("--similarity_npz", type=str, default=None)
    p.add_argument("--skip_correlation", action="store_true")

    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logger(args.log_file)
    _seed_all(args.seed)

    LOGGER.info("=" * 80)
    LOGGER.info("LDM b_SVD Stage A")
    LOGGER.info("target_block=%s target_N=%d T=%d", args.svd_target_block, args.svd_target_N, args.num_steps)
    LOGGER.info("in_memory_pipeline=%s", bool(args.in_memory_pipeline))
    LOGGER.info("=" * 80)

    logdir, ckpt = _resolve_resume_to_logdir_and_ckpt(args.resume)
    config = _load_config(logdir)
    model = _load_model(config, ckpt)

    output_root = _resolve_repo_path(args.svd_output_root)
    t_root = output_root / f"T_{args.num_steps}"
    svd_features_root = t_root / "svd_features"

    collector = SvdFeatureCollector(
        save_root=svd_features_root,
        max_timesteps=args.num_steps,
        target_n=args.svd_target_N,
        target_block=args.svd_target_block,
    )

    run_sampling_collection(
        model=model,
        collector=collector,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    if args.in_memory_pipeline:
        LOGGER.info("Run in-memory Stage B/C...")
        meta = {
            "block": collector.block_slug,
            "target_block_name": collector.target_block,
            "T": int(args.num_steps),
            "C": int(collector.c),
            "H": int(collector.h),
            "W": int(collector.w),
        }

        from svd_metrics_ldm import process_feature_buffers_in_memory

        svd_metrics_dir = t_root / "svd_metrics"
        svd_result = process_feature_buffers_in_memory(
            feature_buffers=collector.feature_buffers,
            meta=meta,
            target_n=args.svd_target_N,
            output_dir=svd_metrics_dir,
            representative_t=args.representative_t,
            energy_threshold=args.energy_threshold,
            compute_energy=not args.no_compute_energy,
        )
        LOGGER.info("Stage B done")

        if not args.skip_correlation:
            if args.similarity_npz is None:
                raise ValueError("--similarity_npz is required when running Stage C")
            from correlate_svd_similarity_ldm import process_single_correlation

            svd_json_path = svd_metrics_dir / f"{svd_result['block']}.json"
            corr_output_dir = t_root / "correlation"
            process_single_correlation(
                svd_json_path=svd_json_path,
                similarity_npz_path=_resolve_repo_path(args.similarity_npz),
                output_dir=corr_output_dir,
                plot_figures=True,
            )
            LOGGER.info("Stage C done")
    else:
        collector.finalize()
        LOGGER.info("Stage A finalize done (features written)")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
