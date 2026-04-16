import argparse
import glob
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf


def _bootstrap_local_paths() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate_paths = [
        repo_root,
        os.path.join(repo_root, "src", "taming-transformers"),
        os.path.join(repo_root, "src", "clip"),
    ]
    for path in candidate_paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_local_paths()

from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.openaimodel import (
    Downsample,
    ResBlock,
    TimestepEmbedSequential,
    Upsample,
)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class BlockInfo:
    group: str
    index: int
    canonical_name: str
    runtime_name: str
    module_path: str
    ds: int
    channels: int


def _resolve_resume_to_logdir_and_ckpt(resume: str) -> Tuple[str, str]:
    if not os.path.exists(resume):
        raise FileNotFoundError(f"Cannot find resume path: {resume}")
    if os.path.isfile(resume):
        logdir = os.path.dirname(resume)
        ckpt = resume
    else:
        logdir = resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt}")
    return logdir, ckpt


def _load_config_from_logdir(logdir: str) -> Any:
    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    if not base_configs:
        raise FileNotFoundError(f"Cannot find config.yaml in: {logdir}")
    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    return OmegaConf.merge(*configs)


def _load_model(config: Any, ckpt: str) -> torch.nn.Module:
    print(f"[load] Loading checkpoint: {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = pl_sd.get("state_dict", pl_sd)
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    model.eval()
    return model


def _is_downsample_block(block: TimestepEmbedSequential) -> bool:
    for layer in block:
        if isinstance(layer, Downsample):
            return True
        if isinstance(layer, ResBlock) and hasattr(layer, "h_upd") and isinstance(layer.h_upd, Downsample):
            return True
    return False


def _is_upsample_block(block: TimestepEmbedSequential) -> bool:
    for layer in block:
        if isinstance(layer, Upsample):
            return True
        if isinstance(layer, ResBlock) and hasattr(layer, "h_upd") and isinstance(layer.h_upd, Upsample):
            return True
    return False


def _infer_block_channels(block: TimestepEmbedSequential) -> int:
    for layer in reversed(list(block)):
        if isinstance(layer, ResBlock):
            return int(layer.out_channels)
        if isinstance(layer, (Downsample, Upsample)) and hasattr(layer, "out_channels"):
            return int(layer.out_channels)
        if hasattr(layer, "out_channels"):
            try:
                return int(layer.out_channels)
            except Exception:
                pass
    raise ValueError("Unable to infer out_channels for block")


def _collect_timestep_embed_paths(diffusion_model: torch.nn.Module) -> List[str]:
    paths: List[str] = []
    for name, module in diffusion_model.named_modules():
        if isinstance(module, TimestepEmbedSequential):
            full = f"model.model.diffusion_model.{name}" if name else "model.model.diffusion_model"
            paths.append(full)
    return paths


def _build_runtime_block_map(diffusion_model: torch.nn.Module) -> List[BlockInfo]:
    infos: List[BlockInfo] = []

    input_blocks = diffusion_model.input_blocks
    middle_block = diffusion_model.middle_block
    output_blocks = diffusion_model.output_blocks

    if len(input_blocks) != 12:
        raise ValueError(f"Expected 12 input_blocks, got {len(input_blocks)}")
    if len(output_blocks) != 12:
        raise ValueError(f"Expected 12 output_blocks, got {len(output_blocks)}")

    ds = 1
    for i, blk in enumerate(input_blocks):
        if not isinstance(blk, TimestepEmbedSequential):
            raise TypeError(f"input_blocks[{i}] is not TimestepEmbedSequential")
        ch = _infer_block_channels(blk)
        infos.append(
            BlockInfo(
                group="encoder",
                index=i,
                canonical_name=f"model.input_blocks.{i}",
                runtime_name=f"encoder_layer_{i}",
                module_path=f"model.model.diffusion_model.input_blocks.{i}",
                ds=ds,
                channels=ch,
            )
        )
        if _is_downsample_block(blk):
            ds *= 2

    if not isinstance(middle_block, TimestepEmbedSequential):
        raise TypeError("middle_block is not TimestepEmbedSequential")
    infos.append(
        BlockInfo(
            group="middle",
            index=0,
            canonical_name="model.middle_block",
            runtime_name="middle_layer",
            module_path="model.model.diffusion_model.middle_block",
            ds=ds,
            channels=_infer_block_channels(middle_block),
        )
    )

    for i, blk in enumerate(output_blocks):
        if not isinstance(blk, TimestepEmbedSequential):
            raise TypeError(f"output_blocks[{i}] is not TimestepEmbedSequential")
        ch = _infer_block_channels(blk)
        infos.append(
            BlockInfo(
                group="decoder",
                index=i,
                canonical_name=f"model.output_blocks.{i}",
                runtime_name=f"decoder_layer_{i}",
                module_path=f"model.model.diffusion_model.output_blocks.{i}",
                ds=ds,
                channels=ch,
            )
        )
        if _is_upsample_block(blk):
            ds //= 2

    return infos


def _validate_inventory(infos: List[BlockInfo]) -> None:
    if len(infos) != 25:
        raise ValueError(f"Expected 25 blocks total, got {len(infos)}")

    enc = [x for x in infos if x.group == "encoder"]
    mid = [x for x in infos if x.group == "middle"]
    dec = [x for x in infos if x.group == "decoder"]
    if len(enc) != 12 or len(mid) != 1 or len(dec) != 12:
        raise ValueError(
            f"Group count mismatch: encoder={len(enc)} middle={len(mid)} decoder={len(dec)}"
        )

    ds4 = [x for x in infos if x.ds == 4]
    ds8 = [x for x in infos if x.ds == 8]
    if ds4 and any(x.channels != 672 for x in ds4):
        bad = [(x.runtime_name, x.channels) for x in ds4 if x.channels != 672]
        raise ValueError(f"Expected ds=4 channels=672, mismatches: {bad}")
    if ds8 and any(x.channels != 896 for x in ds8):
        bad = [(x.runtime_name, x.channels) for x in ds8 if x.channels != 896]
        raise ValueError(f"Expected ds=8 channels=896, mismatches: {bad}")


def _to_json_dict(
    *,
    resume: str,
    ckpt: str,
    config_path: str,
    infos: List[BlockInfo],
) -> Dict[str, Any]:
    blocks = []
    runtime_name_to_module_path: Dict[str, str] = {}
    canonical_name_to_runtime_name: Dict[str, str] = {}

    for block_id, bi in enumerate(infos):
        row = {
            "block_id": block_id,
            "canonical_runtime_block_id": block_id,
            "canonical_name": bi.canonical_name,
            "runtime_name": bi.runtime_name,
            "module_path": bi.module_path,
            "group": bi.group,
            "group_index": bi.index,
            "ds": bi.ds,
            "channels": bi.channels,
        }
        blocks.append(row)
        runtime_name_to_module_path[bi.runtime_name] = bi.module_path
        canonical_name_to_runtime_name[bi.canonical_name] = bi.runtime_name

    return {
        "version": "ldm_block_inventory_v1",
        "source": {
            "resume": os.path.abspath(resume),
            "ckpt": os.path.abspath(ckpt),
            "config": os.path.abspath(config_path),
        },
        "counts": {
            "encoder": 12,
            "middle": 1,
            "decoder": 12,
            "total": len(blocks),
        },
        "block_names": [b["canonical_name"] for b in blocks],
        "blocks": blocks,
        "runtime_name_to_module_path": runtime_name_to_module_path,
        "canonical_name_to_runtime_name": canonical_name_to_runtime_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory LDM diffusion_model TimestepEmbedSequential blocks and export runtime map JSON"
    )
    parser.add_argument(
        "--resume",
        type=str,
        required=True,
        help="Checkpoint path or logdir containing model.ckpt + config.yaml (same style as scripts/sample_diffusion.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ldm_block_map.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    args = parser.parse_args()

    _seed_all(args.seed)

    logdir, ckpt = _resolve_resume_to_logdir_and_ckpt(args.resume)
    config_path = os.path.join(logdir, "config.yaml")
    config = _load_config_from_logdir(logdir)
    model = _load_model(config, ckpt)

    diffusion_model = model.model.diffusion_model

    print("\n[TimestepEmbedSequential inventory]")
    paths = _collect_timestep_embed_paths(diffusion_model)
    for p in paths:
        print(p)

    infos = _build_runtime_block_map(diffusion_model)
    _validate_inventory(infos)

    payload = _to_json_dict(
        resume=args.resume,
        ckpt=ckpt,
        config_path=config_path,
        infos=infos,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n[summary]")
    print(f"total blocks: {len(infos)} (encoder=12, middle=1, decoder=12)")
    print(f"output json: {out_path.resolve()}")


if __name__ == "__main__":
    main()
