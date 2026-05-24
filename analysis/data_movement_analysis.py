#!/usr/bin/env python3
"""LDM-FFHQ256 data-movement and CIM array analysis.

Run with:
  /home/jimmy/anaconda3/envs/ldm/bin/python data_movement_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MODEL_KEY = "ldm_ffhq256"
REPO_ROOT = Path("/home/jimmy/latent-diffusion")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "output" / "data_movement"

CONFIG_PATH = REPO_ROOT / "models/ldm/ffhq256/config.yaml"
CKPT_PATH = REPO_ROOT / "models/ldm/ffhq256/model.ckpt"
SCHEDULER_PATH = (
    REPO_ROOT
    / "ldm_S3cache/cache_method/Stage2/stage2_output_ldm/"
    / "src_K15_sw3_lam1.0/02_refined_blockwise/stage2_refined_scheduler_config.json"
)

T_EXPECTED = 200
BYTES_PER_ELEMENT = 4
LATENT_RESOLUTION = 64
BATCH_SIZE = 1


@dataclass(frozen=True)
class Shape:
    c: int
    h: int
    w: int


@dataclass
class LayerRecord:
    model: str
    block_id: int
    block_name: str
    layer_name: str
    layer_type: str
    weight_shape: str
    weight_bytes: int
    act_bytes: int
    dm_per_exec: int
    cim_rows: int
    cim_cols: int
    cim_max_dim: int
    cim_array_size: int
    is_quantized: bool
    seq_len: int


@dataclass
class BlockRecord:
    model: str
    block_id: int
    block_name: str
    canonical_name: str
    spatial_h: int
    spatial_w: int
    input_channels: int
    output_channels: int
    weight_bytes_per_exec: int
    act_bytes_per_exec: int
    dm_per_exec: int
    exec_count_baseline: int
    exec_count_cached: int
    dm_baseline: int
    dm_cached: int
    cim_block_max_dim: int
    cim_block_max_array_size: int


def format_bytes(n: int | float) -> str:
    n = int(n)
    if n >= 1 << 30:
        return f"{n / (1 << 30):.4f} GB ({n:,} B)"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.4f} MB ({n:,} B)"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.4f} KB ({n:,} B)"
    return f"{n:,} B"


def byte_summary_fields(prefix: str, n: int | float) -> dict[str, Any]:
    n = float(n)
    return {
        f"{prefix}_readable": format_bytes(n),
        f"{prefix}_KB": n / (1 << 10),
        f"{prefix}_MB": n / (1 << 20),
        f"{prefix}_GB": n / (1 << 30),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(title: str, headers: list[str], rows: Iterable[Iterable[Any]]) -> None:
    rows = [[str(x) for x in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def runtime_name_from_id(block_id: int) -> str:
    if block_id < 12:
        return f"encoder_layer_{block_id}"
    if block_id == 12:
        return "middle_layer"
    return f"decoder_layer_{block_id - 13}"


def load_scheduler(path: Path) -> tuple[int, list[dict[str, Any]]]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    t = int(cfg["T"])
    if t != T_EXPECTED:
        raise ValueError(f"Scheduler T={t}, expected {T_EXPECTED}")
    blocks = sorted(cfg["blocks"], key=lambda b: int(b["canonical_runtime_block_id"]))
    seen = [int(b["canonical_runtime_block_id"]) for b in blocks]
    if seen != list(range(len(blocks))):
        raise ValueError(f"canonical_runtime_block_id must be contiguous, got {seen}")
    for b in blocks:
        runtime = str(b["runtime_name"])
        if runtime != runtime_name_from_id(int(b["canonical_runtime_block_id"])):
            raise ValueError(f"runtime name/id mismatch in scheduler block: {b}")
        if len(b["expanded_mask"]) != t:
            raise ValueError(f"{runtime}: mask length mismatch")
    return t, blocks


def module_for_runtime(model: Any, runtime_name: str) -> Any:
    if runtime_name.startswith("encoder_layer_"):
        return model.input_blocks[int(runtime_name.rsplit("_", 1)[1])]
    if runtime_name == "middle_layer":
        return model.middle_block
    if runtime_name.startswith("decoder_layer_"):
        return model.output_blocks[int(runtime_name.rsplit("_", 1)[1])]
    raise ValueError(f"Unknown runtime name: {runtime_name}")


def _weight(module: Any) -> Any | None:
    w = getattr(module, "org_weight", None)
    if w is not None:
        return w
    w = getattr(module, "weight", None)
    return w if w is not None else None


def _conv2d_record(
    *, block_id: int, block_name: str, layer_name: str, module: Any, shape: Shape
) -> tuple[LayerRecord, Shape]:
    w = _weight(module)
    if w is None or int(w.ndim) != 4:
        raise ValueError(f"{layer_name}: expected 4D weight")
    c_out, c_in, kh, kw = [int(x) for x in w.shape]
    weight_bytes = c_out * c_in * kh * kw * BYTES_PER_ELEMENT
    act_bytes = BATCH_SIZE * c_in * shape.h * shape.w * BYTES_PER_ELEMENT
    rows = c_in * kh * kw
    cols = c_out
    rec = LayerRecord(
        model=MODEL_KEY,
        block_id=block_id,
        block_name=block_name,
        layer_name=layer_name,
        layer_type="conv2d",
        weight_shape=str((c_out, c_in, kh, kw)),
        weight_bytes=weight_bytes,
        act_bytes=act_bytes,
        dm_per_exec=weight_bytes + act_bytes,
        cim_rows=rows,
        cim_cols=cols,
        cim_max_dim=max(rows, cols),
        cim_array_size=rows * cols,
        is_quantized=False,
        seq_len=1,
    )
    return rec, Shape(c_out, shape.h, shape.w)


def _conv1d_record(*, block_id: int, block_name: str, layer_name: str, module: Any, shape: Shape) -> LayerRecord:
    w = _weight(module)
    if w is None or int(w.ndim) != 3:
        raise ValueError(f"{layer_name}: expected 3D weight")
    c_out, c_in, k = [int(x) for x in w.shape]
    length = shape.h * shape.w
    weight_bytes = c_out * c_in * k * BYTES_PER_ELEMENT
    act_bytes = BATCH_SIZE * c_in * length * BYTES_PER_ELEMENT
    rows = c_in * k
    cols = c_out
    return LayerRecord(
        model=MODEL_KEY,
        block_id=block_id,
        block_name=block_name,
        layer_name=layer_name,
        layer_type="conv1d",
        weight_shape=str((c_out, c_in, k)),
        weight_bytes=weight_bytes,
        act_bytes=act_bytes,
        dm_per_exec=weight_bytes + act_bytes,
        cim_rows=rows,
        cim_cols=cols,
        cim_max_dim=max(rows, cols),
        cim_array_size=rows * cols,
        is_quantized=False,
        seq_len=1,
    )


def _linear_record(*, block_id: int, block_name: str, layer_name: str, module: Any) -> LayerRecord:
    w = _weight(module)
    if w is None or int(w.ndim) != 2:
        raise ValueError(f"{layer_name}: expected 2D weight")
    out_f, in_f = [int(x) for x in w.shape]
    weight_bytes = out_f * in_f * BYTES_PER_ELEMENT
    act_bytes = BATCH_SIZE * in_f * BYTES_PER_ELEMENT
    return LayerRecord(
        model=MODEL_KEY,
        block_id=block_id,
        block_name=block_name,
        layer_name=layer_name,
        layer_type="linear",
        weight_shape=str((out_f, in_f)),
        weight_bytes=weight_bytes,
        act_bytes=act_bytes,
        dm_per_exec=weight_bytes + act_bytes,
        cim_rows=in_f,
        cim_cols=out_f,
        cim_max_dim=max(in_f, out_f),
        cim_array_size=in_f * out_f,
        is_quantized=False,
        seq_len=1,
    )


def _module_out_channels(module: Any, fallback: int) -> int:
    w = _weight(module)
    if w is not None and int(w.ndim) in (3, 4):
        return int(w.shape[0])
    return int(getattr(module, "out_channels", fallback))


def _is_instance(obj: Any, cls: Any) -> bool:
    return isinstance(obj, cls)


def _enumerate_resblock(
    rb: Any,
    *,
    block_id: int,
    block_name: str,
    prefix: str,
    shape: Shape,
    classes: dict[str, Any],
) -> tuple[list[LayerRecord], Shape]:
    records: list[LayerRecord] = []
    up = hasattr(rb, "h_upd") and _is_instance(rb.h_upd, classes["Upsample"])
    down = hasattr(rb, "h_upd") and _is_instance(rb.h_upd, classes["Downsample"])
    out_channels = int(getattr(rb, "out_channels"))

    conv_shape = shape
    if up:
        conv_shape = Shape(shape.c, shape.h * 2, shape.w * 2)
    elif down:
        conv_shape = Shape(shape.c, shape.h // 2, shape.w // 2)

    rec, _ = _conv2d_record(
        block_id=block_id,
        block_name=block_name,
        layer_name=f"{prefix}.in_layers.conv",
        module=rb.in_layers[-1],
        shape=conv_shape,
    )
    records.append(rec)
    records.append(
        _linear_record(
            block_id=block_id,
            block_name=block_name,
            layer_name=f"{prefix}.emb_layers.linear",
            module=rb.emb_layers[-1],
        )
    )
    rec, _ = _conv2d_record(
        block_id=block_id,
        block_name=block_name,
        layer_name=f"{prefix}.out_layers.conv",
        module=rb.out_layers[-1],
        shape=Shape(out_channels, conv_shape.h, conv_shape.w),
    )
    records.append(rec)
    if _weight(rb.skip_connection) is not None:
        rec, _ = _conv2d_record(
            block_id=block_id,
            block_name=block_name,
            layer_name=f"{prefix}.skip_connection",
            module=rb.skip_connection,
            shape=conv_shape,
        )
        records.append(rec)
    return records, Shape(out_channels, conv_shape.h, conv_shape.w)


def _enumerate_attention(
    attn: Any, *, block_id: int, block_name: str, prefix: str, shape: Shape
) -> list[LayerRecord]:
    return [
        _conv1d_record(
            block_id=block_id,
            block_name=block_name,
            layer_name=f"{prefix}.qkv",
            module=attn.qkv,
            shape=shape,
        ),
        _conv1d_record(
            block_id=block_id,
            block_name=block_name,
            layer_name=f"{prefix}.proj_out",
            module=attn.proj_out,
            shape=shape,
        ),
    ]


def enumerate_block(
    block: Any,
    *,
    block_id: int,
    block_name: str,
    in_shape: Shape,
    classes: dict[str, Any],
) -> tuple[list[LayerRecord], Shape]:
    records: list[LayerRecord] = []
    shape = in_shape
    for idx, layer in enumerate(block):
        prefix = f"{block_name}[{idx}]"
        if isinstance(layer, classes["ResBlock"]):
            recs, shape = _enumerate_resblock(
                layer, block_id=block_id, block_name=block_name, prefix=prefix, shape=shape, classes=classes
            )
            records.extend(recs)
        elif isinstance(layer, classes["AttentionBlock"]):
            records.extend(
                _enumerate_attention(layer, block_id=block_id, block_name=block_name, prefix=prefix, shape=shape)
            )
        elif isinstance(layer, classes["Downsample"]):
            target = getattr(layer, "op", layer)
            if _weight(target) is not None:
                rec, _ = _conv2d_record(
                    block_id=block_id,
                    block_name=block_name,
                    layer_name=f"{prefix}.downsample",
                    module=target,
                    shape=shape,
                )
                records.append(rec)
            shape = Shape(_module_out_channels(target, shape.c), shape.h // 2, shape.w // 2)
        elif isinstance(layer, classes["Upsample"]):
            target = getattr(layer, "conv", layer)
            up_shape = Shape(shape.c, shape.h * 2, shape.w * 2)
            if _weight(target) is not None:
                rec, _ = _conv2d_record(
                    block_id=block_id,
                    block_name=block_name,
                    layer_name=f"{prefix}.upsample",
                    module=target,
                    shape=up_shape,
                )
                records.append(rec)
            shape = Shape(_module_out_channels(target, shape.c), up_shape.h, up_shape.w)
        elif _weight(layer) is not None:
            w = _weight(layer)
            if int(w.ndim) == 4:
                rec, shape = _conv2d_record(
                    block_id=block_id,
                    block_name=block_name,
                    layer_name=f"{prefix}.conv",
                    module=layer,
                    shape=shape,
                )
                records.append(rec)
            elif int(w.ndim) == 2:
                records.append(
                    _linear_record(
                        block_id=block_id,
                        block_name=block_name,
                        layer_name=f"{prefix}.linear",
                        module=layer,
                    )
                )
    return records, shape


def load_model() -> Any:
    for sub in ["", "src/taming-transformers", "src/clip"]:
        p = REPO_ROOT / sub if sub else REPO_ROOT
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.chdir(REPO_ROOT)
    import torch
    from ldm.util import instantiate_from_config
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(CONFIG_PATH))
    model = instantiate_from_config(cfg.model)
    sd = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    state = sd.get("state_dict", sd)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.model.diffusion_model


def analyze() -> tuple[list[BlockRecord], list[LayerRecord], dict[str, Any]]:
    for sub in ["", "src/taming-transformers", "src/clip"]:
        p = REPO_ROOT / sub if sub else REPO_ROOT
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, Downsample, ResBlock, Upsample

    t, sched_blocks = load_scheduler(SCHEDULER_PATH)
    model = load_model()
    classes = {
        "ResBlock": ResBlock,
        "AttentionBlock": AttentionBlock,
        "Downsample": Downsample,
        "Upsample": Upsample,
    }
    layer_records: list[LayerRecord] = []
    block_records: list[BlockRecord] = []
    current = Shape(3, LATENT_RESOLUTION, LATENT_RESOLUTION)
    encoder_outputs: list[Shape] = []

    for b in sched_blocks:
        block_id = int(b["canonical_runtime_block_id"])
        runtime = str(b["runtime_name"])
        block = module_for_runtime(model, runtime)
        if runtime.startswith("decoder_layer_"):
            skip = encoder_outputs.pop()
            current = Shape(current.c + skip.c, current.h, current.w)
        in_shape = current
        recs, out_shape = enumerate_block(
            block, block_id=block_id, block_name=runtime, in_shape=in_shape, classes=classes
        )
        if not recs:
            raise ValueError(f"No compute layers found for {runtime}")
        layer_records.extend(recs)
        if runtime.startswith("encoder_layer_"):
            encoder_outputs.append(out_shape)
        current = out_shape

        weight_sum = sum(r.weight_bytes for r in recs)
        act_sum = sum(r.act_bytes for r in recs)
        exec_cached = sum(bool(x) for x in b["expanded_mask"])
        dm_exec = weight_sum + act_sum
        block_records.append(
            BlockRecord(
                model=MODEL_KEY,
                block_id=block_id,
                block_name=runtime,
                canonical_name=str(b["name"]),
                spatial_h=in_shape.h,
                spatial_w=in_shape.w,
                input_channels=in_shape.c,
                output_channels=out_shape.c,
                weight_bytes_per_exec=weight_sum,
                act_bytes_per_exec=act_sum,
                dm_per_exec=dm_exec,
                exec_count_baseline=t,
                exec_count_cached=exec_cached,
                dm_baseline=dm_exec * t,
                dm_cached=dm_exec * exec_cached,
                cim_block_max_dim=max(r.cim_max_dim for r in recs),
                cim_block_max_array_size=max(r.cim_array_size for r in recs),
            )
        )

    if encoder_outputs:
        raise ValueError(f"Encoder skip stack not fully consumed: {len(encoder_outputs)}")
    validate_records(t, block_records, layer_records)
    return block_records, layer_records, make_summary(t, block_records, layer_records)


def validate_records(t: int, blocks: list[BlockRecord], layers: list[LayerRecord]) -> None:
    if len(blocks) != 25:
        raise ValueError(f"Expected 25 blocks, got {len(blocks)}")
    expected = [runtime_name_from_id(i) for i in range(25)]
    if [b.block_name for b in blocks] != expected:
        raise ValueError("Runtime order mismatch")
    for r in layers:
        if not r.layer_name.startswith(r.block_name):
            raise ValueError(f"Layer name not under block: {r.block_name} vs {r.layer_name}")
        if r.dm_per_exec != r.weight_bytes + r.act_bytes:
            raise ValueError(f"Invalid total bytes for {r.layer_name}")
    for b in blocks:
        child = [r for r in layers if r.block_id == b.block_id]
        if b.weight_bytes_per_exec != sum(r.weight_bytes for r in child):
            raise ValueError(f"Weight sum mismatch for {b.block_name}")
        if b.act_bytes_per_exec != sum(r.act_bytes for r in child):
            raise ValueError(f"Activation sum mismatch for {b.block_name}")
        if b.dm_baseline != b.dm_per_exec * t:
            raise ValueError(f"Baseline mismatch for {b.block_name}")


def make_summary(t: int, blocks: list[BlockRecord], layers: list[LayerRecord]) -> dict[str, Any]:
    baseline = sum(b.dm_baseline for b in blocks)
    cached = sum(b.dm_cached for b in blocks)
    baseline_per_step = baseline / t
    cached_per_step = cached / t
    summary = {
        "model": MODEL_KEY,
        "T": t,
        "num_blocks": len(blocks),
        "num_layers": len(layers),
        "num_quantized_layers": 0,
        "bytes_per_element": BYTES_PER_ELEMENT,
        "baseline_bytes": baseline,
        "cached_bytes": cached,
        "reduction_ratio": (baseline - cached) / baseline,
        "baseline_bytes_per_step": baseline_per_step,
        "cached_bytes_per_step": cached_per_step,
        "global_cim_max_dim": max(r.cim_max_dim for r in layers),
        "global_cim_max_array_size": max(r.cim_array_size for r in layers),
    }
    summary.update(byte_summary_fields("baseline", baseline))
    summary.update(byte_summary_fields("cached", cached))
    summary.update(byte_summary_fields("baseline_per_step", baseline_per_step))
    summary.update(byte_summary_fields("cached_per_step", cached_per_step))
    return summary


def save_outputs(output_dir: Path, blocks: list[BlockRecord], layers: list[LayerRecord], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "exp1_block_detail.csv", [asdict(b) for b in blocks])
    write_csv(output_dir / "exp1_layer_detail.csv", [asdict(r) for r in layers])
    write_csv(output_dir / "exp2_cim_layer_detail.csv", [asdict(r) for r in layers])
    write_csv(
        output_dir / "exp2_cim_block_summary.csv",
        [
            {
                "model": b.model,
                "block_id": b.block_id,
                "block_name": b.block_name,
                "num_layers": sum(1 for r in layers if r.block_id == b.block_id),
                "block_max_dim": b.cim_block_max_dim,
                "block_max_array_size": b.cim_block_max_array_size,
                "spatial_h": b.spatial_h,
                "spatial_w": b.spatial_w,
                "input_channels": b.input_channels,
                "output_channels": b.output_channels,
            }
            for b in blocks
        ],
    )
    write_csv(output_dir / "exp1_summary.csv", [summary])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_report(blocks: list[BlockRecord], layers: list[LayerRecord], summary: dict[str, Any]) -> None:
    print(f"\nModel: {MODEL_KEY}")
    print(f"Baseline: {format_bytes(summary['baseline_bytes'])}")
    print(f"Cached:   {format_bytes(summary['cached_bytes'])}")
    print(f"Reduction: {summary['reduction_ratio']:.2%}")
    print(f"CIM max dim: {summary['global_cim_max_dim']}")
    print(f"CIM max array size: {summary['global_cim_max_array_size']:,}")
    print_table(
        "Block-Level Data Movement",
        ["id", "block", "shape", "DM/exec", "exec", "cached DM", "CIM"],
        [
            [
                b.block_id,
                b.block_name,
                f"{b.input_channels}x{b.spatial_h}x{b.spatial_w}",
                format_bytes(b.dm_per_exec),
                b.exec_count_cached,
                format_bytes(b.dm_cached),
                b.cim_block_max_dim,
            ]
            for b in blocks
        ],
    )
    print_table(
        "Layer-Level Data Movement",
        ["block", "layer", "type", "weight", "act", "total", "CIM(r,c)"],
        [
            [
                r.block_name,
                r.layer_name,
                r.layer_type,
                format_bytes(r.weight_bytes),
                format_bytes(r.act_bytes),
                format_bytes(r.dm_per_exec),
                f"({r.cim_rows},{r.cim_cols})",
            ]
            for r in layers
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    blocks, layers, summary = analyze()
    save_outputs(Path(args.output_dir), blocks, layers, summary)
    print_report(blocks, layers, summary)
    print(f"\n[Done] Results written to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
