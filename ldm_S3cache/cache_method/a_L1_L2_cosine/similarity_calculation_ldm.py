#!/usr/bin/env python3
"""
Phase 1: Port a_L1_L2_cosine similarity collector to LDM UNet.

Key points:
- Uses full unconditional DDIM inference (no FID) to collect features.
- Reuses Diff-AE SimilarityCollector as base class.
- Only overrides register_hooks (LDM UNet block filtering and naming).
"""

from __future__ import annotations

import argparse
import ast
import glob
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf
from tqdm.auto import trange


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


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_COLLECTOR = Path(
    "/home/jimmy/diffae/QATcode/cache_method/a_L1_L2_cosine/similarity_calculation.py"
)

LOGGER = logging.getLogger("LDMSimilarity")


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
    return (REPO_ROOT / p).resolve()


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [LDM-Sim] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8"),
        ],
        force=True,
    )


def _custom_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    img = Image.fromarray(x)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


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
    base_configs = sorted(glob.glob(str(logdir / "config.yaml")))
    if not base_configs:
        raise FileNotFoundError(f"Cannot find config.yaml in: {logdir}")
    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    return OmegaConf.merge(*configs)


def _load_model(config: Any, ckpt: Path) -> torch.nn.Module:
    LOGGER.info("Loading model from: %s", ckpt)
    pl_sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state_dict = pl_sd.get("state_dict", pl_sd)

    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    LOGGER.info("load_state_dict: missing=%d unexpected=%d", len(missing), len(unexpected))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LDM DDIM sampling in this script.")

    model = model.cuda()
    model.eval()
    return model


def _extract_node_source(code: str, node: ast.AST) -> str:
    seg = ast.get_source_segment(code, node)
    if seg is not None:
        return seg
    lines = code.splitlines()
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        raise RuntimeError("AST node source extraction failed")
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def _load_similarity_collector_base(base_collector_py: Path) -> Type[Any]:
    if not base_collector_py.is_file():
        raise FileNotFoundError(f"Base collector file not found: {base_collector_py}")

    code = base_collector_py.read_text(encoding="utf-8")
    tree = ast.parse(code)
    class_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SimilarityCollector":
            class_node = node
            break
    if class_node is None:
        raise RuntimeError("Cannot find class SimilarityCollector in base collector file")

    class_src = _extract_node_source(code, class_node)
    ns: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "np": np,
        "torch": torch,
        "nn": nn,
        "Path": Path,
        "plt": plt,
        "pd": pd,
        "sns": sns,
        "LOGGER": LOGGER,
        "Tuple": Tuple,
        "List": List,
        "Dict": Dict,
        "Any": Any,
        "Optional": Optional,
    }
    exec(class_src, ns)  # noqa: S102
    return ns["SimilarityCollector"]


def _build_ldm_collector_class(base_cls: Type[Any]) -> Type[Any]:
    class LDMSimilarityCollector(base_cls):
        """
        LDM port of SimilarityCollector.

        Differences from base class:
        - TimestepEmbedSequential import from ldm.modules.diffusionmodules.openaimodel.
        - Hook filter only tracks UNet input_blocks / middle_block / output_blocks.
        - Default base_image_size is 64 (latent spatial size).
        - mapped_t_list is set to None because LDM DDIMSampler has no timestep_map.
          finalize() therefore stores mapped_t as -1 placeholders.
        """

        def __init__(
            self,
            save_root: str,
            max_timesteps: int,
            num_samples: int,
            target_block: Optional[str] = None,
            dtype: str = "float16",
            save_cosine_step_plot: bool = False,
            base_image_size: int = 64,
            device: Optional[torch.device] = None,
            sample_strategy: str = "first",
        ):
            super().__init__(
                save_root=save_root,
                max_timesteps=max_timesteps,
                num_samples=num_samples,
                target_block=target_block,
                dtype=dtype,
                save_cosine_step_plot=save_cosine_step_plot,
                base_image_size=base_image_size,
                device=device,
                sample_strategy=sample_strategy,
            )

        @staticmethod
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
                if (
                    short.startswith("input_blocks.")
                    or short == "middle_block"
                    or short.startswith("output_blocks.")
                ):
                    return "model." + short
                return n

            if (
                n.startswith("input_blocks.")
                or n == "middle_block"
                or n.startswith("output_blocks.")
            ):
                return "model." + n
            return n

        @staticmethod
        def _is_supported_unet_block_name(name: str) -> bool:
            return (
                name.startswith("input_blocks.")
                or name == "middle_block"
                or name.startswith("output_blocks.")
            )

        def register_hooks(self, unet: nn.Module, sampler) -> None:
            """Register hooks on LDM UNet (model.model.diffusion_model)."""
            self._step_counter = -1
            self.current_step_idx = None
            self.current_mapped_t = None

            # LDM DDIMSampler has no timestep_map. Keep placeholder behavior (-1 in finalize).
            self.mapped_t_list = None

            canonical_target = (
                self._canonical_block_name(self.target_block)
                if self.target_block is not None
                else None
            )

            registered: List[str] = []
            for name, module in unet.named_modules():
                if not isinstance(module, TimestepEmbedSequential):
                    continue
                if not self._is_supported_unet_block_name(name):
                    continue

                canonical_name = f"model.{name}"
                if canonical_target and canonical_name != canonical_target:
                    continue

                self.hooks.append(module.register_forward_hook(self._create_block_hook(canonical_name)))
                registered.append(canonical_name)
                LOGGER.info("[LDM Similarity] register block hook: %s", canonical_name)

            if not registered:
                raise ValueError(
                    "No LDM block hooks registered. target_block=%r. "
                    "Expected names like model.input_blocks.0 / model.middle_block / model.output_blocks.0"
                    % (self.target_block,)
                )

            self.hooks.append(
                unet.register_forward_pre_hook(self._create_step_pre_hook(), with_kwargs=True)
            )
            self.hooks.append(unet.register_forward_hook(self._create_model_post_hook()))
            LOGGER.info(
                "[LDM Similarity] registered %d block hooks + model step hooks",
                len(registered),
            )

        def _calc_metrics_batch(
            self, t1: torch.Tensor, t2: torch.Tensor, eps: float = 1e-6
        ):
            """
            Override base metrics to prevent float16 overflow on LDM blocks.

            LDM feature magnitudes can be much larger than Diff-AE, and float16
            dot/norm in cosine may produce inf/inf -> NaN for step-wise cosine.
            We upcast to float32 for stable reduction ops, while keeping output
            tensors compatible with base-class accumulators.
            """
            t1f = t1.to(torch.float32)
            t2f = t2.to(torch.float32)

            diff = torch.abs(t1f - t2f)
            l1_diff = diff.mean(dim=(1, 2, 3))
            l1_ref = (torch.abs(t1f).mean(dim=(1, 2, 3)) + torch.abs(t2f).mean(dim=(1, 2, 3))) / 2.0 + eps
            l1_vals = l1_diff / l1_ref

            l1_rate_diff = diff.sum(dim=(1, 2, 3))
            l1_rate_ref = torch.abs(t1f).sum(dim=(1, 2, 3)) + eps
            l1_rate_vals = l1_rate_diff / l1_rate_ref

            diff_sq = (t1f - t2f).pow(2).mean(dim=(1, 2, 3)).sqrt()
            l2_ref = (t1f.pow(2).mean(dim=(1, 2, 3)).sqrt() + t2f.pow(2).mean(dim=(1, 2, 3)).sqrt()) / 2.0 + eps
            l2_vals = diff_sq / l2_ref

            t1_flat = t1f.reshape(t1f.size(0), -1)
            t2_flat = t2f.reshape(t2f.size(0), -1)
            dot = (t1_flat * t2_flat).sum(dim=1)
            denom = torch.clamp(t1_flat.norm(dim=1) * t2_flat.norm(dim=1), min=eps)
            cos_vals = dot / denom
            cos_vals = torch.nan_to_num(cos_vals, nan=0.0, posinf=1.0, neginf=-1.0)
            cos_vals = torch.clamp(cos_vals, min=-1.0, max=1.0)

            return l1_vals, l1_rate_vals, l2_vals, cos_vals

    return LDMSimilarityCollector


def _run_unconditional_sampling(
    *,
    model: torch.nn.Module,
    collector: Any,
    num_steps: int,
    n_samples: int,
    batch_size: int,
    eta: float,
    save_generated_pngs: bool = False,
    generated_dir: Optional[Path] = None,
) -> int:
    unet = model.model.diffusion_model
    sampler = DDIMSampler(model)
    shape = [unet.in_channels, unet.image_size, unet.image_size]

    if save_generated_pngs:
        if generated_dir is None:
            raise ValueError("generated_dir is required when save_generated_pngs=True")
        generated_dir.mkdir(parents=True, exist_ok=True)

    collector.register_hooks(unet, sampler)
    generated = 0
    total_batches = math.ceil(max(n_samples, 1) / max(batch_size, 1))

    try:
        # Align with scripts/sample_diffusion.py: sampling uses EMA weights.
        with model.ema_scope("Similarity Collection"):
            for _ in trange(total_batches, desc="Sampling batches"):
                current_bs = min(batch_size, n_samples - generated)
                if current_bs <= 0:
                    break
                # Analysis protocol: always use deterministic DDIM (eta=0).
                # Keep `eta` in signature for compatibility, but ignore runtime input.
                samples, _ = sampler.sample(
                    S=num_steps,
                    batch_size=current_bs,
                    shape=shape,
                    eta=0.0,
                    verbose=False,
                )
                if save_generated_pngs:
                    x_sample = model.decode_first_stage(samples)
                    start_idx = generated
                    for bi in range(current_bs):
                        img = _custom_to_pil(x_sample[bi])
                        img.save(str(generated_dir / f"sample_{start_idx + bi:06d}.png"))
                    del x_sample
                generated += current_bs
                del samples
                torch.cuda.empty_cache()
    finally:
        collector.remove_hooks()

    collector.finalize()
    return generated


def _validate_gate(npz_path: Path, expected_steps: int) -> Dict[str, Any]:
    if not npz_path.is_file():
        raise FileNotFoundError(f"Expected npz output not found: {npz_path}")

    arr = np.load(npz_path)
    l1_step_mean = arr["l1_step_mean"]
    cos_step_mean = arr["cos_step_mean"]
    cosine = arr["cosine"]

    if expected_steps < 2:
        raise ValueError(f"expected_steps must be >=2, got {expected_steps}")

    expected_l1_shape = (expected_steps - 1,)
    expected_cos_step_shape = (expected_steps - 1,)
    expected_cos_shape = (expected_steps, expected_steps)

    if l1_step_mean.shape != expected_l1_shape:
        raise ValueError(
            f"Gate failed: l1_step_mean.shape={l1_step_mean.shape}, expected {expected_l1_shape}"
        )
    if cos_step_mean.shape != expected_cos_step_shape:
        raise ValueError(
            f"Gate failed: cos_step_mean.shape={cos_step_mean.shape}, expected {expected_cos_step_shape}"
        )
    if cosine.shape != expected_cos_shape:
        raise ValueError(
            f"Gate failed: cosine.shape={cosine.shape}, expected {expected_cos_shape}"
        )

    if not np.isfinite(l1_step_mean).all():
        raise ValueError("Gate failed: l1_step_mean contains NaN/Inf")
    if not np.isfinite(cos_step_mean).all():
        raise ValueError("Gate failed: cos_step_mean contains NaN/Inf")
    if not np.isfinite(cosine).all():
        raise ValueError("Gate failed: cosine contains NaN/Inf")

    diag = np.diag(cosine)
    diag_mean = float(diag.mean())
    diag_min = float(diag.min())
    if diag_mean < 0.99 or diag_min < 0.95:
        raise ValueError(
            "Gate failed: cosine diagonal not close to 1.0 "
            f"(mean={diag_mean:.6f}, min={diag_min:.6f})"
        )

    return {
        "expected_steps": int(expected_steps),
        "l1_step_mean_shape": list(l1_step_mean.shape),
        "cos_step_mean_shape": list(cos_step_mean.shape),
        "cosine_shape": list(cosine.shape),
        "l1_nan_count": int(np.isnan(l1_step_mean).sum()),
        "cos_step_nan_count": int(np.isnan(cos_step_mean).sum()),
        "cosine_nan_count": int(np.isnan(cosine).sum()),
        "cosine_diag_mean": diag_mean,
        "cosine_diag_min": diag_min,
        "cosine_diag_max": float(diag.max()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LDM a_L1_L2_cosine similarity collection (Phase 1)")
    p.add_argument("--resume", type=str, required=True, help="LDM checkpoint path or logdir")
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Kept for CLI compatibility. Analysis run always forces eta=0.",
    )
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--target_block", type=str, default="model.output_blocks.0")
    p.add_argument("--collect_per_batch", type=int, default=20)
    p.add_argument(
        "--output_root",
        type=str,
        default="ldm_S3cache/cache_method/a_L1_L2_cosine",
        help="Root output path (repo-relative or absolute)",
    )
    p.add_argument("--similarity_dtype", type=str, default="float32", choices=["float16", "float32"])
    p.add_argument(
        "--sample_strategy",
        type=str,
        default="first",
        choices=["first", "random", "uniform"],
    )
    p.add_argument(
        "--base_collector_py",
        type=str,
        default=str(DEFAULT_BASE_COLLECTOR),
        help="Path to Diff-AE similarity_calculation.py containing class SimilarityCollector",
    )
    p.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional log file path. Default is <output_root>/log/similarity_calculation_ldm.log",
    )
    p.add_argument(
        "--save_generated_pngs",
        action="store_true",
        help="Decode and save generated images to a dedicated cache-method directory (separate from sample_diffusion outputs).",
    )
    p.add_argument(
        "--generated_root",
        type=str,
        default="ldm_S3cache/cache_method/generated_images",
        help="Root directory for generated images when --save_generated_pngs is enabled.",
    )
    p.add_argument(
        "--generated_subdir",
        type=str,
        default=None,
        help="Optional subdir name under generated_root. Default auto: T{steps}_{target_block_slug}_seed{seed}",
    )
    p.add_argument(
        "--clear_generated_dir",
        action="store_true",
        help="If set, delete generated output directory before writing new images.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _seed_all(args.seed)

    if float(args.eta) != 0.0:
        LOGGER.warning(
            "Received --eta=%s, but analysis protocol forces eta=0.0. Using eta=0.0.",
            args.eta,
        )

    output_root = _resolve_repo_path(args.output_root)
    similarity_root = output_root / f"T_{args.num_steps}" / "v2_latest"
    log_file = _resolve_repo_path(args.log_file) if args.log_file else (output_root / "log" / "similarity_calculation_ldm.log")
    _setup_logging(log_file)

    LOGGER.info("=" * 80)
    LOGGER.info("Phase 1 | LDM a_L1_L2_cosine similarity collection")
    LOGGER.info("resume=%s", args.resume)
    LOGGER.info("target_block=%s", args.target_block)
    LOGGER.info("output_root=%s", output_root)
    LOGGER.info(
        "num_steps=%d n_samples=%d batch_size=%d eta=%s(forced)",
        args.num_steps,
        args.n_samples,
        args.batch_size,
        0.0,
    )
    LOGGER.info("=" * 80)

    logdir, ckpt = _resolve_resume_to_logdir_and_ckpt(args.resume)
    config = _load_config(logdir)
    model = _load_model(config, ckpt)

    base_cls = _load_similarity_collector_base(_resolve_repo_path(args.base_collector_py))
    LDMSimilarityCollector = _build_ldm_collector_class(base_cls)

    total_collect = args.collect_per_batch * math.ceil(max(args.n_samples, 1) / max(args.batch_size, 1))
    collector = LDMSimilarityCollector(
        save_root=str(similarity_root),
        max_timesteps=args.num_steps,
        num_samples=total_collect,
        target_block=args.target_block,
        dtype=args.similarity_dtype,
        save_cosine_step_plot=False,
        base_image_size=64,
        device=torch.device("cuda"),
        sample_strategy=args.sample_strategy,
    )
    collector._batch_collect_limit = int(args.collect_per_batch)

    generated_dir: Optional[Path] = None
    if args.save_generated_pngs:
        generated_root = _resolve_repo_path(args.generated_root)
        block_slug = collector._canonical_block_name(args.target_block).replace(".", "_")
        subdir = args.generated_subdir or f"T{args.num_steps}_{block_slug}_seed{args.seed}"
        generated_dir = generated_root / subdir
        if args.clear_generated_dir and generated_dir.exists():
            shutil.rmtree(generated_dir)
        generated_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Generated images will be saved to: %s", generated_dir)

    generated = _run_unconditional_sampling(
        model=model,
        collector=collector,
        num_steps=args.num_steps,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        eta=args.eta,
        save_generated_pngs=bool(args.save_generated_pngs),
        generated_dir=generated_dir,
    )
    LOGGER.info("Sampling done. generated=%d", generated)

    canonical_target = collector._canonical_block_name(args.target_block)
    slug = canonical_target.replace(".", "_")
    npz_path = similarity_root / "result_npz" / f"{slug}.npz"

    gate = _validate_gate(npz_path, expected_steps=int(args.num_steps))
    LOGGER.info("Phase 1 gate passed: %s", gate)


if __name__ == "__main__":
    main()
