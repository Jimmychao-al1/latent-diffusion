#!/usr/bin/env python3
"""
Phase 3 (c_FID): LDM cache sensitivity by single-block runtime caching.

This script ports Diff-AE c_FID workflow to LDM UNet:
- baseline FID (no cache)
- single-layer cache sensitivity (k-step recompute schedule)

Default experiment profile follows requested settings:
- seed=0, steps=200, num_images=1000, fid_dims=192, eta=0
- generated images under outputs/c_fid/
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import logging
import math
import os
import random
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image
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
from scripts.sample_diffusion import (
    compute_fid,
    export_real_from_lmdb,
)


LOGGER = logging.getLogger("LDMFIDSensitivity")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [c_FID-LDM] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8"),
        ],
        force=True,
    )


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
        raise RuntimeError("CUDA is required for c_FID script.")

    model = model.cuda()
    model.eval()
    return model


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


def _count_pngs(dir_path: Path) -> int:
    return len(list(dir_path.glob("*.png"))) if dir_path.exists() else 0


def _collect_images_recursive(root_dir: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths: List[str] = []
    root = Path(root_dir)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(str(p))
    paths.sort()
    return paths


def _resolve_listed_images(
    real_image_dir: str,
    real_image_list: str,
    all_paths: Sequence[str],
) -> List[str]:
    path_set = set(all_paths)
    by_name: Dict[str, str] = {}
    for p in all_paths:
        name = os.path.basename(p)
        if name in by_name and by_name[name] != p:
            continue
        by_name[name] = p

    resolved: List[str] = []
    missing: List[str] = []
    with open(real_image_list, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        candidate_rel = os.path.join(real_image_dir, line)
        if candidate_rel in path_set:
            resolved.append(candidate_rel)
            continue
        name_only = os.path.basename(line)
        if name_only in by_name:
            resolved.append(by_name[name_only])
            continue
        missing.append(line)

    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} entries from --real_image_list could not be resolved. "
            f"First unresolved entries: {preview}"
        )

    return resolved


def _export_real_from_image_dir_fixed(
    *,
    real_image_dir: str,
    eval_dir: Path,
    num_images: int,
    img_size: int,
    real_image_list: Optional[str] = None,
) -> None:
    all_paths = _collect_images_recursive(real_image_dir)
    if not all_paths:
        raise ValueError(f"No images found under --real_image_dir: {real_image_dir}")

    if real_image_list:
        selected_paths = _resolve_listed_images(real_image_dir, real_image_list, all_paths)
    else:
        selected_paths = list(all_paths)

    if num_images > len(selected_paths):
        raise ValueError(
            f"Requested num_images={num_images}, but only {len(selected_paths)} source images are available"
        )

    selected_paths = selected_paths[:num_images]

    preprocessor = A.Compose(
        [
            A.SmallestMaxSize(max_size=img_size),
            A.CenterCrop(height=img_size, width=img_size),
        ]
    )

    eval_dir.mkdir(parents=True, exist_ok=True)
    for index in trange(len(selected_paths), desc="Exporting real images"):
        resolved_path = selected_paths[index]
        img = Image.open(resolved_path).convert("RGB")
        arr = np.array(img).astype(np.uint8)
        arr = preprocessor(image=arr)["image"]
        arr = (arr / 127.5 - 1.0).astype(np.float32)
        arr = (arr * 127.5 + 127.5).astype(np.uint8)
        Image.fromarray(arr).save(str(eval_dir / f"{index}.png"))


def _load_block_map(block_map_path: Path) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    if not block_map_path.is_file():
        raise FileNotFoundError(f"block map not found: {block_map_path}")

    payload = json.loads(block_map_path.read_text(encoding="utf-8"))
    blocks = payload.get("blocks", [])
    if not blocks:
        raise ValueError(f"No blocks found in block map: {block_map_path}")

    blocks_sorted = sorted(blocks, key=lambda b: int(b.get("canonical_runtime_block_id", 10**9)))
    runtime_order: List[str] = []
    runtime_to_canonical: Dict[str, str] = {}
    canonical_to_runtime: Dict[str, str] = {}

    for b in blocks_sorted:
        runtime = str(b["runtime_name"])
        canonical = str(b["canonical_name"])
        runtime_order.append(runtime)
        runtime_to_canonical[runtime] = canonical
        canonical_to_runtime[canonical] = runtime

    return runtime_order, runtime_to_canonical, canonical_to_runtime


def _canonical_to_unet_name(canonical: str) -> str:
    if canonical.startswith("model."):
        return canonical[len("model.") :]
    return canonical


def _resolve_layer_arg(
    layer: str,
    runtime_to_canonical: Dict[str, str],
    canonical_to_runtime: Dict[str, str],
) -> Tuple[str, str]:
    val = str(layer).strip()

    if val in runtime_to_canonical:
        return val, runtime_to_canonical[val]

    if val in canonical_to_runtime:
        return canonical_to_runtime[val], val

    if val.startswith("input_blocks.") or val == "middle_block" or val.startswith("output_blocks."):
        canonical = f"model.{val}"
        if canonical in canonical_to_runtime:
            return canonical_to_runtime[canonical], canonical

    if val.startswith("model."):
        short = val[len("model.") :]
        if short.startswith("input_blocks.") or short == "middle_block" or short.startswith("output_blocks."):
            if val in canonical_to_runtime:
                return canonical_to_runtime[val], val

    raise ValueError(
        "Unknown --layer value. Use runtime name (encoder_layer_0) or canonical name "
        "(model.output_blocks.0). Got: %s" % layer
    )


def _create_recompute_steps(k: int, total_steps: int) -> set[int]:
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    steps = set(range(0, total_steps, k))
    # Align with Diff-AE script convention: force last loop step to recompute too.
    steps.add(total_steps - 1)
    return steps


class SingleBlockRuntimeCache:
    """Runtime cache hook for one TimestepEmbedSequential block in LDM UNet."""

    def __init__(
        self,
        *,
        unet: torch.nn.Module,
        target_unet_name: str,
        total_steps: int,
        recompute_steps: set[int],
    ) -> None:
        self.unet = unet
        self.target_unet_name = target_unet_name
        self.total_steps = int(total_steps)
        self.recompute_steps = set(int(s) for s in recompute_steps)

        self._hooks: List[Any] = []
        self._target_module: Optional[torch.nn.Module] = None
        self._cached_output: Optional[torch.Tensor] = None
        self._forward_counter = -1
        self._current_loop_step: Optional[int] = None

        self.cache_hits = 0
        self.recompute_hits = 0

    def _find_target_module(self) -> torch.nn.Module:
        for name, module in self.unet.named_modules():
            if name == self.target_unet_name:
                if not isinstance(module, TimestepEmbedSequential):
                    raise TypeError(
                        f"Target module exists but is not TimestepEmbedSequential: {self.target_unet_name}"
                    )
                return module
        raise ValueError(f"Target module not found in UNet: {self.target_unet_name}")

    def _model_pre_hook(self, module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        del module, args, kwargs
        self._forward_counter += 1
        self._current_loop_step = self._forward_counter % self.total_steps
        if self._current_loop_step == 0:
            # New sampling batch starts: cached feature should not leak across batches.
            self._cached_output = None

    def _target_forward_hook(
        self,
        module: torch.nn.Module,
        inputs: Tuple[Any, ...],
        output: Any,
    ) -> Any:
        del module, inputs

        if not torch.is_tensor(output):
            return output

        step = self._current_loop_step
        if step is None:
            return output

        should_recompute = (step in self.recompute_steps) or (self._cached_output is None)
        if should_recompute:
            self.recompute_hits += 1
            self._cached_output = output.detach()
            return output

        self.cache_hits += 1
        return self._cached_output.clone()

    def __enter__(self) -> "SingleBlockRuntimeCache":
        self._target_module = self._find_target_module()
        self._hooks.append(
            self.unet.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        )
        self._hooks.append(self._target_module.register_forward_hook(self._target_forward_hook))
        LOGGER.info(
            "Runtime cache enabled: target=%s total_steps=%d recompute_count=%d",
            self.target_unet_name,
            self.total_steps,
            len(self.recompute_steps),
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._target_module = None
        self._cached_output = None

    def stats(self) -> Dict[str, int]:
        return {
            "cache_hits": int(self.cache_hits),
            "recompute_hits": int(self.recompute_hits),
            "total_hook_calls": int(self.cache_hits + self.recompute_hits),
        }


def _ensure_real_images_exact_count(
    *,
    real_dir: Path,
    num_images: int,
    img_size: int,
    real_image_dir: Optional[str],
    real_image_list: Optional[str],
    real_lmdb: Optional[str],
    lmdb_resolution: int,
    lmdb_zfill: int,
    force_rebuild: bool,
) -> None:
    if not real_image_dir and not real_lmdb:
        raise ValueError("Provide either --real_image_dir or --real_lmdb")

    if force_rebuild and real_dir.exists():
        shutil.rmtree(real_dir)

    existing = _count_pngs(real_dir)
    if real_dir.exists() and existing != num_images:
        LOGGER.info(
            "Rebuilding real images because count mismatch: existing=%d expected=%d",
            existing,
            num_images,
        )
        shutil.rmtree(real_dir)

    if real_image_dir:
        _export_real_from_image_dir_fixed(
            real_image_dir=str(_resolve_repo_path(real_image_dir)),
            eval_dir=real_dir,
            num_images=num_images,
            img_size=img_size,
            real_image_list=str(_resolve_repo_path(real_image_list)) if real_image_list else None,
        )
    else:
        export_real_from_lmdb(
            lmdb_path=str(_resolve_repo_path(real_lmdb)),
            eval_dir=str(real_dir),
            num_images=num_images,
            img_size=img_size,
            lmdb_resolution=lmdb_resolution,
            lmdb_zfill=lmdb_zfill,
        )

    final_count = _count_pngs(real_dir)
    if final_count != num_images:
        raise RuntimeError(
            f"Real image count mismatch after export: expected {num_images}, got {final_count}"
        )


def _generate_images(
    *,
    model: torch.nn.Module,
    gen_dir: Path,
    n_samples: int,
    batch_size: int,
    num_steps: int,
    eta: float,
    cache_ctx: Optional[SingleBlockRuntimeCache],
) -> int:
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)

    unet = model.model.diffusion_model
    sampler = DDIMSampler(model)
    shape = [unet.in_channels, unet.image_size, unet.image_size]

    n_saved = 0
    total_batches = math.ceil(max(n_samples, 1) / max(batch_size, 1))
    runtime_ctx = cache_ctx if cache_ctx is not None else nullcontext()

    with model.ema_scope("c_FID generation"):
        with runtime_ctx:
            for _ in trange(total_batches, desc="Sampling batches"):
                current_bs = min(batch_size, n_samples - n_saved)
                if current_bs <= 0:
                    break

                samples, _ = sampler.sample(
                    S=num_steps,
                    batch_size=current_bs,
                    shape=shape,
                    eta=eta,
                    verbose=False,
                )
                x_sample = model.decode_first_stage(samples)

                for i in range(current_bs):
                    img = _custom_to_pil(x_sample[i])
                    img.save(str(gen_dir / f"sample_{n_saved + i:06d}.png"))

                n_saved += current_bs
                del samples
                del x_sample
                torch.cuda.empty_cache()

    return n_saved


def _load_results(results_path: Path, default_config: Dict[str, Any]) -> Dict[str, Any]:
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Results JSON must be dict: {results_path}")
        payload.setdefault("config", default_config)
        payload.setdefault("results", {})
        return payload

    return {
        "config": default_config,
        "results": {},
    }


def _save_results(results: Dict[str, Any], results_path: Path) -> None:
    results["last_updated"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def _should_skip(results: Dict[str, Any], step_key: str, k: int, runtime_name: str) -> bool:
    if step_key not in results.get("results", {}):
        return False
    k_key = f"k{k}"
    if k_key not in results["results"][step_key]:
        return False
    return runtime_name in results["results"][step_key][k_key]


def _baseline_matches_current_config(
    baseline_meta: Dict[str, Any],
    *,
    expected_num_images: int,
    expected_fid_dims: int,
    expected_num_steps: int,
) -> bool:
    if not isinstance(baseline_meta, dict):
        return False
    if int(baseline_meta.get("real_count", -1)) != int(expected_num_images):
        return False
    if int(baseline_meta.get("gen_count", -1)) != int(expected_num_images):
        return False
    if int(baseline_meta.get("fid_dims", -1)) != int(expected_fid_dims):
        return False
    if int(baseline_meta.get("num_steps", -1)) != int(expected_num_steps):
        return False
    return True


def _evaluate_fid_once(
    *,
    model: torch.nn.Module,
    real_dir: Path,
    gen_dir: Path,
    n_samples: int,
    batch_size: int,
    num_steps: int,
    eta: float,
    fid_batch_size: int,
    fid_dims: int,
    target_canonical: Optional[str],
    k: Optional[int],
) -> Dict[str, Any]:
    cache_stats: Dict[str, int] = {}
    cache_ctx: Optional[SingleBlockRuntimeCache] = None

    if target_canonical is not None:
        if k is None:
            raise ValueError("k is required when target_canonical is set")
        unet = model.model.diffusion_model
        target_unet_name = _canonical_to_unet_name(target_canonical)
        cache_ctx = SingleBlockRuntimeCache(
            unet=unet,
            target_unet_name=target_unet_name,
            total_steps=num_steps,
            recompute_steps=_create_recompute_steps(k, num_steps),
        )

    generated = _generate_images(
        model=model,
        gen_dir=gen_dir,
        n_samples=n_samples,
        batch_size=batch_size,
        num_steps=num_steps,
        eta=eta,
        cache_ctx=cache_ctx,
    )

    if generated != n_samples:
        raise RuntimeError(
            f"Generated image count mismatch: expected {n_samples}, loop generated {generated}"
        )

    gen_count = _count_pngs(gen_dir)
    if gen_count != n_samples:
        raise RuntimeError(
            f"Generated PNG count mismatch: expected {n_samples}, found {gen_count} in {gen_dir}"
        )

    real_count = _count_pngs(real_dir)
    if real_count != n_samples:
        raise RuntimeError(
            f"Real PNG count mismatch before FID: expected {n_samples}, found {real_count} in {real_dir}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_value = float(compute_fid(str(real_dir), str(gen_dir), fid_batch_size, device, dims=fid_dims))

    if cache_ctx is not None:
        cache_stats = cache_ctx.stats()

    return {
        "fid": fid_value,
        "real_count": int(real_count),
        "gen_count": int(gen_count),
        "fid_dims": int(fid_dims),
        "num_steps": int(num_steps),
        "real_dir": str(real_dir),
        "gen_dir": str(gen_dir),
        "cache_stats": cache_stats,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LDM c_FID cache sensitivity")
    p.add_argument("--resume", type=str, required=True, help="LDM checkpoint path or logdir")

    p.add_argument("--dataset_tag", type=str, default="ffhq256")
    p.add_argument("--output_root", type=str, default="outputs/c_fid")
    p.add_argument(
        "--results_json",
        type=str,
        default="ldm_S3cache/cache_method/c_FID/fid_sensitivity_results_ldm.json",
    )
    p.add_argument(
        "--log_file",
        type=str,
        default="ldm_S3cache/cache_method/c_FID/fid_cache_sensitivity_ldm.log",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--num_images", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eta", type=float, default=0.0)

    p.add_argument("--fid_batch_size", type=int, default=32)
    p.add_argument("--fid_dims", type=int, default=192)

    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--real_image_dir", type=str, default="ffhq-dataset/images1024x1024")
    p.add_argument("--real_image_list", type=str, default=None)
    p.add_argument("--real_lmdb", type=str, default=None)
    p.add_argument("--lmdb_resolution", type=int, default=256)
    p.add_argument("--lmdb_zfill", type=int, default=5)
    p.add_argument("--force_rebuild_real", action="store_true")

    p.add_argument(
        "--block_map",
        type=str,
        default="ldm_block_map_ffhq256.json",
        help="Phase0 block map JSON used for runtime/canonical name mapping.",
    )

    p.add_argument("--baseline", action="store_true", help="Run only baseline (no cache).")
    p.add_argument("--layer", type=str, default=None, help="Runtime or canonical block name.")
    p.add_argument(
        "--k",
        type=int,
        default=None,
        choices=[3, 4, 5],
        help="Cache recompute period for target layer(s). Must be one of: 3, 4, 5.",
    )
    p.add_argument(
        "--cleanup_generated",
        action="store_true",
        help="Delete generated images after each experiment (default keeps them under outputs/c_fid).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log_file = _resolve_repo_path(args.log_file)
    _setup_logging(log_file)

    _seed_all(args.seed)

    if float(args.eta) != 0.0:
        LOGGER.warning(
            "Received --eta=%s, but c_FID analysis protocol enforces eta=0.0. Using eta=0.0.",
            args.eta,
        )
    eta = 0.0

    if not args.baseline and args.k is None:
        raise ValueError("Provide --k for cache sensitivity runs, or use --baseline.")

    output_root = _resolve_repo_path(args.output_root)
    dataset_root = output_root / args.dataset_tag
    real_dir = dataset_root / f"real_images_n{args.num_images}_s{args.img_size}"

    results_path = _resolve_repo_path(args.results_json)
    block_map_path = _resolve_repo_path(args.block_map)

    runtime_order, runtime_to_canonical, canonical_to_runtime = _load_block_map(block_map_path)

    LOGGER.info("=" * 80)
    LOGGER.info("LDM c_FID cache sensitivity")
    LOGGER.info("resume=%s", args.resume)
    LOGGER.info("num_steps=%d num_images=%d batch_size=%d", args.num_steps, args.num_images, args.batch_size)
    LOGGER.info("eta=%.1f (forced) fid_dims=%d fid_batch_size=%d", eta, args.fid_dims, args.fid_batch_size)
    LOGGER.info("output_root=%s", output_root)
    LOGGER.info("results_json=%s", results_path)
    LOGGER.info("block_map=%s (blocks=%d)", block_map_path, len(runtime_order))
    LOGGER.info("=" * 80)

    _ensure_real_images_exact_count(
        real_dir=real_dir,
        num_images=args.num_images,
        img_size=args.img_size,
        real_image_dir=args.real_image_dir,
        real_image_list=args.real_image_list,
        real_lmdb=args.real_lmdb,
        lmdb_resolution=args.lmdb_resolution,
        lmdb_zfill=args.lmdb_zfill,
        force_rebuild=bool(args.force_rebuild_real),
    )
    LOGGER.info("Real image cache ready: %s (count=%d)", real_dir, _count_pngs(real_dir))

    logdir, ckpt = _resolve_resume_to_logdir_and_ckpt(args.resume)
    config = _load_config(logdir)
    model = _load_model(config, ckpt)

    default_config = {
        "seed": int(args.seed),
        "num_steps": int(args.num_steps),
        "num_images": int(args.num_images),
        "eta": float(eta),
        "fid_dims": int(args.fid_dims),
        "fid_batch_size": int(args.fid_batch_size),
        "dataset_tag": str(args.dataset_tag),
        "real_dir": str(real_dir),
        "block_map": str(block_map_path),
    }
    results = _load_results(results_path, default_config)

    step_key = f"T{args.num_steps}"
    results.setdefault("results", {})
    results["results"].setdefault(step_key, {})

    baseline_exists = "baseline_fid" in results["results"][step_key]
    baseline_meta = results["results"][step_key].get("baseline_meta", {})
    baseline_is_compatible = _baseline_matches_current_config(
        baseline_meta,
        expected_num_images=args.num_images,
        expected_fid_dims=args.fid_dims,
        expected_num_steps=args.num_steps,
    )

    if (not baseline_exists) or (not baseline_is_compatible):
        if baseline_exists and (not baseline_is_compatible):
            LOGGER.warning(
                "Existing baseline is incompatible with current config. Recomputing baseline "
                "(need num_images=%d, fid_dims=%d, num_steps=%d).",
                args.num_images,
                args.fid_dims,
                args.num_steps,
            )
        LOGGER.info("Computing baseline FID for %s...", step_key)
        baseline_exp = dataset_root / f"{step_key}_baseline_seed{args.seed}" / "gen_images"
        baseline_result = _evaluate_fid_once(
            model=model,
            real_dir=real_dir,
            gen_dir=baseline_exp,
            n_samples=args.num_images,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            eta=eta,
            fid_batch_size=args.fid_batch_size,
            fid_dims=args.fid_dims,
            target_canonical=None,
            k=None,
        )
        results["results"][step_key]["baseline_fid"] = baseline_result["fid"]
        results["results"][step_key]["baseline_meta"] = baseline_result
        _save_results(results, results_path)
        LOGGER.info("Baseline FID: %.6f", baseline_result["fid"])

        if args.cleanup_generated and baseline_exp.parent.exists():
            shutil.rmtree(baseline_exp.parent)
    else:
        LOGGER.info(
            "Baseline exists for %s: %.6f",
            step_key,
            float(results["results"][step_key]["baseline_fid"]),
        )

    baseline_fid = float(results["results"][step_key]["baseline_fid"])

    if args.baseline:
        LOGGER.info("Baseline-only mode complete.")
        return

    assert args.k is not None
    k_key = f"k{args.k}"
    results["results"][step_key].setdefault(k_key, {})

    if args.layer is not None:
        layer_targets = [_resolve_layer_arg(args.layer, runtime_to_canonical, canonical_to_runtime)]
    else:
        layer_targets = [(runtime, runtime_to_canonical[runtime]) for runtime in runtime_order]

    LOGGER.info("Running %d layer experiment(s) for %s %s", len(layer_targets), step_key, k_key)

    for idx, (runtime_name, canonical_name) in enumerate(layer_targets, 1):
        if _should_skip(results, step_key, args.k, runtime_name):
            LOGGER.info("[%d/%d] Skip existing: %s", idx, len(layer_targets), runtime_name)
            continue

        LOGGER.info("[%d/%d] Evaluate layer=%s canonical=%s", idx, len(layer_targets), runtime_name, canonical_name)
        exp_name = f"{step_key}_k{args.k}_{runtime_name}_seed{args.seed}"
        gen_dir = dataset_root / exp_name / "gen_images"

        run_result = _evaluate_fid_once(
            model=model,
            real_dir=real_dir,
            gen_dir=gen_dir,
            n_samples=args.num_images,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            eta=eta,
            fid_batch_size=args.fid_batch_size,
            fid_dims=args.fid_dims,
            target_canonical=canonical_name,
            k=args.k,
        )

        fid_value = float(run_result["fid"])
        delta = fid_value - baseline_fid

        results["results"][step_key][k_key][runtime_name] = {
            "fid": fid_value,
            "delta": delta,
            "canonical_name": canonical_name,
            "experiment_dir": str(gen_dir.parent),
            "real_count": int(run_result["real_count"]),
            "gen_count": int(run_result["gen_count"]),
            "cache_stats": run_result.get("cache_stats", {}),
        }
        _save_results(results, results_path)

        LOGGER.info(
            "Layer %s done: FID=%.6f delta=%+.6f real=%d gen=%d",
            runtime_name,
            fid_value,
            delta,
            run_result["real_count"],
            run_result["gen_count"],
        )

        if args.cleanup_generated and gen_dir.parent.exists():
            shutil.rmtree(gen_dir.parent)

    LOGGER.info("All requested experiments complete. Results: %s", results_path)


if __name__ == "__main__":
    main()
