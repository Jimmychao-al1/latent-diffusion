#!/usr/bin/env python3
"""
LDM b_SVD - Unified Stage A->B->C pipeline.

Collect SVD evidence for all supported UNet blocks in a single DDIM run:
- Stage A: gather per-timestep second-moment statistics for each block
- Stage B: compute SVD/subspace drift metrics per block
- Stage C: correlate with Stage 0a similarity NPZ per block
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from collect_features_for_svd_ldm import (
    _load_config,
    _load_model,
    _resolve_repo_path,
    _resolve_resume_to_logdir_and_ckpt,
    _seed_all,
    _setup_logger,
)
from correlate_svd_similarity_ldm import process_single_correlation
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential
from ldm_S3cache.cache_method.Stage2.stage2_scheduler_adapter_ldm import get_unet_for_hook
from svd_metrics_ldm import (
    compute_energy_ratios,
    compute_rank_r,
    compute_subspace_distance,
)

import logging


LOGGER = logging.getLogger("LDM_SVD_Unified")


def _is_supported_unet_block_name(name: str) -> bool:
    return name.startswith("input_blocks.") or name == "middle_block" or name.startswith("output_blocks.")


class UnifiedSvdCollector:
    def __init__(self, max_timesteps: int, target_n: int, include_blocks: Optional[List[str]] = None):
        self.max_timesteps = int(max_timesteps)
        self.target_n = int(target_n)
        self.include_blocks = set(include_blocks) if include_blocks else None

        self.hooks: List[Any] = []
        self._step_counter = -1

        # block_name -> state
        self.states: Dict[str, Dict[str, Any]] = {}

    def _create_step_pre_hook(self):
        def pre_hook(module, args, kwargs):
            del module, args, kwargs
            self._step_counter = (self._step_counter + 1) % self.max_timesteps

        return pre_hook

    def _create_block_hook(self, block_name: str):
        def hook_fn(module, inputs, output):
            del module, inputs
            step_idx = int(self._step_counter)
            if not (0 <= step_idx < self.max_timesteps):
                return
            if not torch.is_tensor(output) or output.ndim != 4:
                return

            st = self.states[block_name]
            remain = self.target_n - int(st["sample_counts"][step_idx])
            if remain <= 0:
                return

            out = output.detach().to(device="cpu", dtype=torch.float32)
            if out.shape[0] > remain:
                out = out[:remain]
            if out.shape[0] <= 0:
                return

            n, c, h, w = out.shape
            if st["C"] is None:
                st["C"], st["H"], st["W"] = int(c), int(h), int(w)
                LOGGER.info("Block %s feature shape: C=%d H=%d W=%d", block_name, c, h, w)
            else:
                if (st["C"], st["H"], st["W"]) != (int(c), int(h), int(w)):
                    raise RuntimeError(
                        f"Block {block_name} feature shape changed: "
                        f"{(st['C'], st['H'], st['W'])} -> {(int(c), int(h), int(w))}"
                    )

            flat = out.permute(1, 0, 2, 3).reshape(c, n * h * w)
            cov_chunk = flat @ flat.T  # (C, C), float32

            cov_sums = st["cov_sums"]
            if cov_sums[step_idx] is None:
                cov_sums[step_idx] = cov_chunk
            else:
                cov_sums[step_idx].add_(cov_chunk)

            st["sample_counts"][step_idx] += int(n)
            st["token_counts"][step_idx] += int(n * h * w)

        return hook_fn

    def register_hooks(self, unet: nn.Module) -> None:
        registered: List[str] = []
        for name, module in unet.named_modules():
            if not isinstance(module, TimestepEmbedSequential):
                continue
            if not _is_supported_unet_block_name(name):
                continue

            canonical = f"model.{name}"
            if self.include_blocks is not None and canonical not in self.include_blocks:
                continue

            self.states[canonical] = {
                "C": None,
                "H": None,
                "W": None,
                "cov_sums": [None] * self.max_timesteps,  # list[Tensor|None], each (C,C)
                "sample_counts": np.zeros((self.max_timesteps,), dtype=np.int32),
                "token_counts": np.zeros((self.max_timesteps,), dtype=np.int64),
            }
            self.hooks.append(module.register_forward_hook(self._create_block_hook(canonical)))
            registered.append(canonical)
            LOGGER.info("Register block hook: %s", canonical)

        if not registered:
            raise ValueError("No supported blocks registered.")

        self.hooks.append(unet.register_forward_pre_hook(self._create_step_pre_hook(), with_kwargs=True))
        LOGGER.info("Registered %d block hooks", len(registered))

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def min_collected(self) -> int:
        mins = [int(st["sample_counts"].min()) for st in self.states.values()]
        return int(min(mins)) if mins else 0

    def has_enough(self) -> bool:
        return self.min_collected() >= self.target_n

    def finalize_results(
        self,
        representative_t: int,
        energy_threshold: float,
        compute_energy: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}

        for block_name, st in tqdm(self.states.items(), desc="Finalize blocks"):
            c = st["C"]
            h = st["H"]
            w = st["W"]
            if c is None or h is None or w is None:
                raise RuntimeError(f"No features collected for block {block_name}")

            sample_counts = st["sample_counts"]
            token_counts = st["token_counts"]
            cov_sums = st["cov_sums"]

            eigenvalues_list: List[torch.Tensor] = []
            eigenvectors_list: List[torch.Tensor] = []

            for t in range(self.max_timesteps):
                cov = cov_sums[t]
                tokens = int(token_counts[t])
                if cov is None or tokens <= 0:
                    raise RuntimeError(f"Missing covariance for block={block_name} t={t}")
                sigma = (cov / float(tokens)).double()

                eigvals, eigvecs = torch.linalg.eigh(sigma)
                eigvals = torch.flip(eigvals, dims=[0])
                eigvecs = torch.flip(eigvecs, dims=[1])
                eigenvalues_list.append(eigvals)
                eigenvectors_list.append(eigvecs)

                cov_sums[t] = None

            rep_t = representative_t
            if rep_t < 0 or rep_t >= self.max_timesteps:
                rep_t = self.max_timesteps - 1

            eigenvalues_ref = eigenvalues_list[rep_t]
            rank_r = compute_rank_r(eigenvalues_ref, energy_threshold)
            cumulative_ref = torch.cumsum(eigenvalues_ref, dim=0) / eigenvalues_ref.sum()
            actual_energy = float(cumulative_ref[rank_r - 1].item())

            subspace_dist: List[float] = [0.0]
            for t in range(1, self.max_timesteps):
                dist = compute_subspace_distance(eigenvectors_list[t], eigenvectors_list[t - 1], rank_r)
                subspace_dist.append(float(dist))

            energy_ratio = compute_energy_ratios(eigenvalues_list) if compute_energy else None
            block_slug = block_name.replace(".", "_")
            result: Dict[str, Any] = {
                "block": block_slug,
                "target_block_name": block_name,
                "T": int(self.max_timesteps),
                "C": int(c),
                "N": int(sample_counts.min()),
                "H": int(h),
                "W": int(w),
                "rank_r": int(rank_r),
                "representative_t": int(rep_t),
                "energy_threshold": float(energy_threshold),
                "actual_energy_at_r": float(actual_energy),
                "timesteps": list(range(self.max_timesteps)),
                "subspace_dist": subspace_dist,
            }
            if energy_ratio is not None:
                result["energy_ratio"] = energy_ratio

            results[block_slug] = result

        return results


@torch.no_grad()
def run_sampling_collection(
    *,
    model: torch.nn.Module,
    collector: UnifiedSvdCollector,
    num_steps: int,
    batch_size: int,
    max_batches: int,
) -> None:
    unet = get_unet_for_hook(model)
    sampler = DDIMSampler(model)
    shape = [unet.in_channels, unet.image_size, unet.image_size]

    collector.register_hooks(unet)
    try:
        with model.ema_scope("SVD Unified Collection"):
            for bi in range(max_batches):
                if collector.has_enough():
                    LOGGER.info("Reached target_N for all blocks/timesteps at batch=%d", bi)
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


def _parse_block_list(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    items = [x.strip() for x in arg.split(",") if x.strip()]
    return items if items else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LDM Unified SVD Feature Collection")
    p.add_argument("--resume", type=str, required=True, help="LDM checkpoint path or logdir")
    p.add_argument("--num_steps", "--n", type=int, default=200, help="DDIM steps T")
    p.add_argument("--svd_target_N", type=int, default=32, help="samples collected per timestep")
    p.add_argument(
        "--svd_output_root",
        type=str,
        default="ldm_S3cache/cache_method/stage0_output_qldm/b_SVD",
        help="output root",
    )
    p.add_argument("--log_file", "--lf", type=str, default=None, help="optional log file")
    p.add_argument("--batch_size", type=int, default=16, help="sampling batch size")
    p.add_argument("--max_batches", type=int, default=64, help="hard cap for sampling loops")
    p.add_argument("--representative-t", type=int, default=-1)
    p.add_argument("--energy-threshold", type=float, default=0.98)
    p.add_argument("--no-compute-energy", action="store_true")
    p.add_argument(
        "--similarity_npz_root",
        type=str,
        default="ldm_S3cache/cache_method/stage0_output_qldm/a_L1_L2_cosine/T_200/v2_latest/result_npz",
        help="root directory containing per-block similarity npz files",
    )
    p.add_argument("--skip_correlation", action="store_true")
    p.add_argument(
        "--include_blocks",
        type=str,
        default=None,
        help="comma-separated canonical block names; default: all supported blocks",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--cali_ckpt",
        type=str,
        default=None,
        help="TFMQ-DM calibration checkpoint for Q-LDM mode",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logger(args.log_file)
    _seed_all(args.seed)

    include_blocks = _parse_block_list(args.include_blocks)

    LOGGER.info("=" * 80)
    LOGGER.info("LDM b_SVD Unified Stage A->B->C")
    LOGGER.info("target_N=%d T=%d", args.svd_target_N, args.num_steps)
    LOGGER.info("batch_size=%d max_batches=%d", args.batch_size, args.max_batches)
    LOGGER.info("include_blocks=%s", include_blocks if include_blocks else "ALL")
    LOGGER.info("=" * 80)

    logdir, ckpt = _resolve_resume_to_logdir_and_ckpt(args.resume)
    config = _load_config(logdir)
    model = _load_model(config, ckpt)
    if args.cali_ckpt:
        from Stage2.stage2_scheduler_adapter_ldm import setup_quantized_ldm

        model = setup_quantized_ldm(model, args.cali_ckpt)

    output_root = _resolve_repo_path(args.svd_output_root)
    t_root = output_root / f"T_{args.num_steps}"
    svd_metrics_dir = t_root / "svd_metrics"
    corr_output_dir = t_root / "correlation"
    similarity_npz_root = _resolve_repo_path(args.similarity_npz_root)

    collector = UnifiedSvdCollector(
        max_timesteps=args.num_steps,
        target_n=args.svd_target_N,
        include_blocks=include_blocks,
    )
    run_sampling_collection(
        model=model,
        collector=collector,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    LOGGER.info("Finalize Stage B for all blocks...")
    results = collector.finalize_results(
        representative_t=args.representative_t,
        energy_threshold=args.energy_threshold,
        compute_energy=not args.no_compute_energy,
    )

    svd_metrics_dir.mkdir(parents=True, exist_ok=True)
    for block_slug, result in results.items():
        out_path = svd_metrics_dir / f"{block_slug}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    LOGGER.info("Stage B done. Wrote %d json files", len(results))

    if not args.skip_correlation:
        corr_ok = 0
        for block_slug in tqdm(sorted(results.keys()), desc="Correlation"):
            svd_json_path = svd_metrics_dir / f"{block_slug}.json"
            similarity_npz_path = similarity_npz_root / f"{block_slug}.npz"
            if not similarity_npz_path.exists():
                LOGGER.warning("Missing similarity npz for %s: %s", block_slug, similarity_npz_path)
                continue
            process_single_correlation(
                svd_json_path=svd_json_path,
                similarity_npz_path=similarity_npz_path,
                output_dir=corr_output_dir,
                plot_figures=True,
            )
            corr_ok += 1
        LOGGER.info("Stage C done. Correlated %d blocks", corr_ok)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
