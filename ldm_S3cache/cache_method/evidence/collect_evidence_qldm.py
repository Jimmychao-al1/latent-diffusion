#!/usr/bin/env python3
"""One-shot Stage-0 evidence collection for Q-LDM (L1/cosine + SVD in a single pass).

Mirrors ``dit_s3cache/evidence`` design: one model load, one DDIM sampling loop,
all 25 UNet blocks hooked simultaneously, legacy per-block exports for Stage 0.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import trange

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_METHOD = Path(__file__).resolve().parents[1]


def _bootstrap_paths() -> None:
    for path in (
        REPO_ROOT,
        REPO_ROOT / "src" / "taming-transformers",
        CACHE_METHOD,
        "/home/jimmy/TFMQ-DM",
        "/home/jimmy/TFMQ-DM/stable-diffusion",
    ):
        p = str(path)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    tfmq_sd = "/home/jimmy/TFMQ-DM/stable-diffusion/ldm"
    ldm_ld = str(REPO_ROOT / "ldm")
    import ldm as ldm_pkg

    ldm_pkg.__path__ = [p for p in (tfmq_sd, ldm_ld) if os.path.isdir(p)]
    for mod in list(sys.modules):
        if mod.startswith("ldm.modules.diffusionmodules.openaimodel"):
            del sys.modules[mod]


_bootstrap_paths()

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from ldm_S3cache.cache_method.Stage2.stage2_scheduler_adapter_ldm import (
    EXPECTED_NUM_BLOCKS,
    get_unet_for_hook,
    setup_quantized_ldm,
)
from evidence.collector import UnifiedEvidenceCollector
from evidence.utils import (
    export_legacy_stage0_inputs,
    save_unified_evidence_npz,
    verify_evidence_outputs,
)

LOGGER = logging.getLogger("LDM_Evidence_Collect")

DEFAULT_BLOCKS: Tuple[str, ...] = (
    "model.input_blocks.0",
    "model.input_blocks.1",
    "model.input_blocks.2",
    "model.input_blocks.3",
    "model.input_blocks.4",
    "model.input_blocks.5",
    "model.input_blocks.6",
    "model.input_blocks.7",
    "model.input_blocks.8",
    "model.input_blocks.9",
    "model.input_blocks.10",
    "model.input_blocks.11",
    "model.middle_block",
    "model.output_blocks.0",
    "model.output_blocks.1",
    "model.output_blocks.2",
    "model.output_blocks.3",
    "model.output_blocks.4",
    "model.output_blocks.5",
    "model.output_blocks.6",
    "model.output_blocks.7",
    "model.output_blocks.8",
    "model.output_blocks.9",
    "model.output_blocks.10",
    "model.output_blocks.11",
)
assert len(DEFAULT_BLOCKS) == EXPECTED_NUM_BLOCKS


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _resolve_resume_to_logdir_and_ckpt(resume: str) -> Tuple[Path, Path]:
    resume_p = _resolve_repo_path(resume)
    if not resume_p.exists():
        raise FileNotFoundError(f"Cannot find resume path: {resume_p}")
    if resume_p.is_file():
        return resume_p.parent, resume_p
    ckpt = resume_p / "model.ckpt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt}")
    return resume_p, ckpt


def _load_config(logdir: Path) -> Any:
    configs = sorted(glob.glob(str(logdir / "config.yaml")))
    if not configs:
        raise FileNotFoundError(f"Cannot find config.yaml in: {logdir}")
    return OmegaConf.merge(*[OmegaConf.load(cfg) for cfg in configs])


def _load_model(config: Any, ckpt: Path) -> torch.nn.Module:
    LOGGER.info("Loading model from: %s", ckpt)
    pl_sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state_dict = pl_sd.get("state_dict", pl_sd)
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    LOGGER.info("load_state_dict: missing=%d unexpected=%d", len(missing), len(unexpected))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evidence collection.")
    return model.cuda().eval()


def _setup_logger(log_file: Optional[str]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = _resolve_repo_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [LDM-Evidence] %(message)s",
        handlers=handlers,
        force=True,
    )


@torch.no_grad()
def run_collection(
    *,
    model: torch.nn.Module,
    collector: UnifiedEvidenceCollector,
    num_steps: int,
    batch_size: int,
    n_batches: int,
    use_ema_scope: bool,
) -> None:
    from contextlib import nullcontext

    unet = get_unet_for_hook(model)
    sampler = DDIMSampler(model)
    shape = [unet.in_channels, unet.image_size, unet.image_size]

    collector.register_hooks(unet)
    try:
        scope = model.ema_scope("Evidence Collection") if use_ema_scope else nullcontext()
        with scope:
            for bi in trange(n_batches, desc="Sampling batches"):
                if collector.has_enough_svd():
                    LOGGER.info("Reached target_N for all blocks/timesteps at batch=%d", bi)
                collector.reset_batch_state()
                samples, _ = sampler.sample(
                    S=num_steps,
                    batch_size=batch_size,
                    shape=shape,
                    eta=0.0,
                    verbose=False,
                )
                del samples
                torch.cuda.empty_cache()
    finally:
        collector.remove_hooks()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=str, default="models/ldm/ffhq256/model.ckpt")
    p.add_argument("--num_steps", "--n", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_batches", type=int, default=8, help="8x16=128 samples (matches old 0a)")
    p.add_argument("--svd_target_N", type=int, default=32)
    p.add_argument("--max_batches", type=int, default=64)
    p.add_argument("--representative-t", type=int, default=-1)
    p.add_argument("--energy-threshold", type=float, default=0.98)
    p.add_argument(
        "--output_root",
        type=str,
        default="ldm_S3cache/cache_method/stage0_output_qldm/evidence_unified",
    )
    p.add_argument(
        "--unified_npz",
        type=str,
        default=None,
        help="Optional unified NPZ path (default: <output_root>/evidence_qldm_T{n}.npz)",
    )
    p.add_argument(
        "--cali_ckpt",
        type=str,
        required=True,
        help="TFMQ-DM calibration checkpoint (required for Q-LDM evidence)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument("--skip_verify", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    _setup_logger(args.log_file)
    _seed_all(args.seed)
    torch.set_grad_enabled(False)

    n_batches = min(args.n_batches, args.max_batches)
    output_root = _resolve_repo_path(args.output_root)
    unified_npz = _resolve_repo_path(
        args.unified_npz or str(output_root / f"evidence_qldm_T{args.num_steps}.npz")
    )

    LOGGER.info("=" * 80)
    LOGGER.info("LDM unified evidence collection (Q-LDM)")
    LOGGER.info("T=%d batch_size=%d n_batches=%d target_N=%d", args.num_steps, args.batch_size, n_batches, args.svd_target_N)
    LOGGER.info("output_root=%s", output_root)
    LOGGER.info("=" * 80)

    logdir, ckpt = _resolve_resume_to_logdir_and_ckpt(args.resume)
    config = _load_config(logdir)
    model = _load_model(config, ckpt)

    use_ema_scope = False
    if not args.cali_ckpt:
        raise ValueError("Q-LDM evidence requires --cali_ckpt (TFMQ-DM calibration checkpoint).")
    model = setup_quantized_ldm(model, args.cali_ckpt)
    LOGGER.info("Q-LDM mode: EMA applied before quant via setup_quantized_ldm, use_ema_scope=False")

    collector = UnifiedEvidenceCollector(
        max_timesteps=args.num_steps,
        target_n=args.svd_target_N,
        block_names=list(DEFAULT_BLOCKS),
        representative_t=args.representative_t,
        energy_threshold=args.energy_threshold,
    )

    run_collection(
        model=model,
        collector=collector,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        n_batches=n_batches,
        use_ema_scope=use_ema_scope,
    )

    similarity = collector.finalize_similarity()
    svd_results = collector.finalize_svd()

    metadata = {
        "format": "ldm_s3cache_evidence_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume": str(ckpt),
        "cali_ckpt": args.cali_ckpt,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "n_batches": n_batches,
        "svd_target_N": args.svd_target_N,
        "seed": args.seed,
        "n_blocks": len(DEFAULT_BLOCKS),
        "block_names": list(DEFAULT_BLOCKS),
        "ema_scope_during_sampling": use_ema_scope,
        "collection_mode": "one_pass_similarity_plus_svd",
    }

    save_unified_evidence_npz(
        unified_npz,
        similarity=similarity,
        svd_results=svd_results,
        block_names=list(DEFAULT_BLOCKS),
        metadata=metadata,
    )
    LOGGER.info("Wrote unified NPZ: %s", unified_npz)

    npz_dir, svd_dir = export_legacy_stage0_inputs(
        output_root=output_root,
        num_steps=args.num_steps,
        similarity=similarity,
        svd_results=svd_results,
    )
    LOGGER.info("Exported legacy NPZ dir: %s", npz_dir)
    LOGGER.info("Exported legacy SVD dir: %s", svd_dir)

    manifest = {
        "unified_npz": str(unified_npz),
        "legacy_npz_dir": str(npz_dir),
        "legacy_svd_dir": str(svd_dir),
        "metadata": metadata,
    }
    manifest_path = output_root / f"manifest_T{args.num_steps}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    LOGGER.info("Wrote manifest: %s", manifest_path)

    if not args.skip_verify:
        report = verify_evidence_outputs(
            npz_dir=npz_dir,
            svd_dir=svd_dir,
            expected_blocks=EXPECTED_NUM_BLOCKS,
            expected_steps=args.num_steps,
        )
        LOGGER.info("Evidence verification passed: %s", report)

    torch.cuda.empty_cache()
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
