"""
Stage2 (LDM): 將 Stage1 的 scheduler_config.json 轉成 runtime 可用的 cache scheduler。

時間軸（與 Stage1-LDM 一致）：
- DDIM 採樣步序 index i=0..T-1，其中 i=0 對應 DDIM t=T-1，i=T-1 對應 t=0。
- Stage1 的 expanded_mask 使用同一個步序 index。
- runtime cache scheduler 的 value 是「需要 full compute / recompute 的 DDIM timestep t」集合。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ldm_S3cache.cache_method.Stage1.stage1_scheduler_ldm import (
    expand_zone_mask_ddim,
    or_expanded_with_zone_mask,
    validate_shared_zones_ddim,
)

EXPECTED_NUM_BLOCKS = 25
TIME_ORDER_EXPECTED = "ddim_descending_Tminus1_to_0"
FIRST_INPUT_RUNTIME_BLOCK_NAME = "encoder_layer_0"

RUNTIME_LAYER_NAMES: Tuple[str, ...] = tuple(
    [f"encoder_layer_{i}" for i in range(12)]
    + ["middle_layer"]
    + [f"decoder_layer_{i}" for i in range(12)]
)
assert len(RUNTIME_LAYER_NAMES) == EXPECTED_NUM_BLOCKS


def load_stage1_scheduler_config(path: str | Path) -> Dict[str, Any]:
    """讀取 Stage1 scheduler_config.json。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"scheduler_config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise TypeError("scheduler_config root must be a JSON object")
    return cfg


def runtime_name_to_block_id(runtime: str) -> int:
    """Runtime 名稱 -> canonical runtime block id。"""
    s = runtime.strip()
    try:
        return RUNTIME_LAYER_NAMES.index(s)
    except ValueError as e:
        raise ValueError(f"unrecognized runtime block name: {runtime!r}") from e


def runtime_block_to_stage1_name(runtime: str) -> str:
    """Runtime 名稱 -> Stage1 canonical block name。"""
    s = runtime.strip()
    m = re.match(r"^encoder_layer_(\d+)$", s)
    if m:
        return f"model.input_blocks.{int(m.group(1))}"
    if s == "middle_layer":
        return "model.middle_block"
    m = re.match(r"^decoder_layer_(\d+)$", s)
    if m:
        return f"model.output_blocks.{int(m.group(1))}"
    raise ValueError(f"unrecognized runtime block name: {runtime!r}")


def stage1_block_to_runtime_block(stage1_name: str) -> str:
    """Stage1 canonical block name -> Runtime 名稱。"""
    s = stage1_name.strip()
    m = re.match(r"^model\.input_blocks\.(\d+)$", s)
    if m:
        return f"encoder_layer_{int(m.group(1))}"
    if s == "model.middle_block":
        return "middle_layer"
    m = re.match(r"^model\.output_blocks\.(\d+)$", s)
    if m:
        return f"decoder_layer_{int(m.group(1))}"
    raise ValueError(f"unrecognized Stage1 block name: {stage1_name!r}")


def ddim_timestep_to_step_index(ddim_t: int, T: int) -> int:
    """DDIM timestep t -> Stage1 expanded_mask 步序 index i。"""
    return (T - 1) - int(ddim_t)


def rebuild_expanded_mask_from_shared_zones_and_k_per_zone(
    shared_zones: List[Dict[str, Any]],
    k_per_zone: List[int],
    T: int,
    *,
    block_id: int = 0,
) -> List[bool]:
    """
    由 shared_zones + k_per_zone 重建單一 block 的 expanded_mask（長度 T）。
    規則對齊 Stage1：zone union 後強制 expanded_mask[0] = True。
    """
    if len(k_per_zone) != len(shared_zones):
        raise ValueError(
            f"k_per_zone len {len(k_per_zone)} != shared_zones len {len(shared_zones)}"
        )
    row = np.zeros(T, dtype=bool)
    for z, k in zip(shared_zones, k_per_zone):
        ms, _, _ = expand_zone_mask_ddim(int(z["t_start"]), int(z["t_end"]), int(k), T)
        or_expanded_with_zone_mask(row, ms, block_id=block_id, zone_id=int(z["id"]))
    row[0] = True
    return row.tolist()


def validate_stage1_scheduler_config(
    cfg: Dict[str, Any],
    *,
    require_full_coverage: bool = True,
    require_k_per_zone: bool = True,
) -> None:
    """
    Stage1/Stage2 scheduler 結構驗證。

    require_k_per_zone=False 時，只檢查 expanded_mask 合法，不驗證 rebuild(k) 一致性。
    """
    if cfg.get("time_order") != TIME_ORDER_EXPECTED:
        raise ValueError(
            f"time_order must be {TIME_ORDER_EXPECTED!r}, got {cfg.get('time_order')!r}"
        )

    T = int(cfg["T"])
    if T < 2:
        raise ValueError(f"T must be >= 2, got {T}")

    blocks = cfg.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("blocks must be a list")
    if require_full_coverage and len(blocks) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"blocks must be length {EXPECTED_NUM_BLOCKS}, got {len(blocks)}"
        )
    if not require_full_coverage and len(blocks) < 1:
        raise ValueError("blocks must be non-empty when require_full_coverage=False")

    ids = [int(b["id"]) for b in blocks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"block ids must be unique, got {ids}")
    if require_full_coverage:
        ids_sorted = sorted(ids)
        if ids_sorted != list(range(EXPECTED_NUM_BLOCKS)):
            raise ValueError(
                f"block ids must be 0..{EXPECTED_NUM_BLOCKS - 1} exactly once, got {ids_sorted}"
            )

    shared = cfg.get("shared_zones")
    if not isinstance(shared, list) or len(shared) < 1:
        raise ValueError("shared_zones must be a non-empty list")
    validate_shared_zones_ddim(shared, T)
    nz = len(shared)

    mapped_runtime_names: List[str] = []
    for b in blocks:
        bid = int(b.get("id"))

        mask = b.get("expanded_mask")
        if not isinstance(mask, list) or len(mask) != T:
            raise ValueError(
                f"block id={bid}: expanded_mask length must be T={T}, "
                f"got {len(mask) if isinstance(mask, list) else type(mask)}"
            )
        row = np.asarray(mask, dtype=bool)
        if row.shape != (T,):
            raise ValueError(
                f"block id={bid}: expanded_mask shape must be ({T},), got {row.shape}"
            )
        if not bool(row[0]):
            raise ValueError(
                f"block id={bid}: expanded_mask[0] must be True "
                "(step_idx=0 <-> DDIM t=T-1)"
            )

        name = str(b.get("name", ""))
        rt = stage1_block_to_runtime_block(name)
        rid = runtime_name_to_block_id(rt)
        mapped_runtime_names.append(rt)

        rt_declared = b.get("runtime_name", None)
        if rt_declared is not None and str(rt_declared) != rt:
            raise ValueError(
                f"block id={bid}: runtime_name {rt_declared!r} contradicts name {name!r} -> {rt!r}"
            )
        rid_declared = b.get("canonical_runtime_block_id", None)
        if rid_declared is not None and int(rid_declared) != rid:
            raise ValueError(
                f"block id={bid}: canonical_runtime_block_id {rid_declared!r} contradicts name {name!r} -> {rid}"
            )
        local_bid_declared = b.get("scheduler_local_block_id", None)
        if local_bid_declared is not None and int(local_bid_declared) != bid:
            raise ValueError(
                f"block id={bid}: scheduler_local_block_id {local_bid_declared!r} must equal id"
            )

        if require_k_per_zone:
            kz = b.get("k_per_zone")
            if not isinstance(kz, list) or len(kz) != nz:
                raise ValueError(
                    f"block id={bid}: len(k_per_zone) must be {nz}, "
                    f"got {len(kz) if isinstance(kz, list) else type(kz)}"
                )
            kz_int = [int(x) for x in kz]
            for zi, kv in enumerate(kz_int):
                if kv < 1:
                    raise ValueError(f"block id={bid}: k_per_zone[{zi}] must be >= 1, got {kv}")
            rebuilt = np.asarray(
                rebuild_expanded_mask_from_shared_zones_and_k_per_zone(
                    shared,
                    kz_int,
                    T,
                    block_id=bid,
                ),
                dtype=bool,
            )
            if not np.all(row >= rebuilt):
                bad = np.where(~row & rebuilt)[0].tolist()
                raise ValueError(
                    f"block id={bid}: expanded_mask must be >= rebuild(shared_zones,k_per_zone), "
                    f"missing steps {bad[:24]}" + (" ..." if len(bad) > 24 else "")
                )

    if len(set(mapped_runtime_names)) != len(mapped_runtime_names):
        raise ValueError("mapped runtime block names from blocks[].name must be unique")
    if require_full_coverage and set(mapped_runtime_names) != set(RUNTIME_LAYER_NAMES):
        missing = sorted(set(RUNTIME_LAYER_NAMES) - set(mapped_runtime_names))
        extra = sorted(set(mapped_runtime_names) - set(RUNTIME_LAYER_NAMES))
        raise ValueError(
            "mapped runtime block set mismatch: "
            f"missing={missing}, extra={extra}"
        )


def expanded_mask_row_to_recompute_ddim_timesteps(
    expanded_mask_row: List[bool] | np.ndarray,
    T: int,
) -> Set[int]:
    """
    單一 block expanded_mask（步序 i）-> recompute 的 DDIM timestep t 集合。
    expanded_mask[i] 的對應為 t=(T-1)-i。
    """
    row = np.asarray(expanded_mask_row, dtype=bool)
    if row.shape != (T,):
        raise ValueError(f"expanded_mask row must have shape ({T},), got {row.shape}")
    out: Set[int] = set()
    for t in range(T):
        si = ddim_timestep_to_step_index(t, T)
        if bool(row[si]):
            out.add(t)
    return out


def stage1_mask_to_runtime_cache_scheduler(
    cfg: Dict[str, Any],
    *,
    require_k_per_zone: bool = True,
) -> Dict[str, Set[int]]:
    """
    由 Stage1 config 產生完整 runtime cache scheduler。
    key: runtime layer；value: 需 recompute 的 DDIM timestep t 集合。
    """
    validate_stage1_scheduler_config(cfg, require_k_per_zone=require_k_per_zone)
    T = int(cfg["T"])
    blocks = sorted(cfg["blocks"], key=lambda b: int(b["id"]))
    sched: Dict[str, Set[int]] = {}
    for b in blocks:
        rt = stage1_block_to_runtime_block(str(b["name"]))
        if rt in sched:
            raise ValueError(f"duplicate runtime block {rt}")
        sched[rt] = expanded_mask_row_to_recompute_ddim_timesteps(b["expanded_mask"], T)
    if len(sched) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"internal error: expected {EXPECTED_NUM_BLOCKS} runtime keys, got {len(sched)}"
        )
    return sched


def cache_scheduler_to_jsonable(sched: Dict[str, Set[int]]) -> Dict[str, List[int]]:
    """set 轉 JSON friendly sorted list。"""
    return {k: sorted(v) for k, v in sorted(sched.items())}


def prefix_ddim_timesteps_first_n(T: int, n: int) -> Set[int]:
    """
    取前 N 個 DDIM timestep（依採樣順序先 t=T-1，再往下）。
    回傳 t 集合。
    """
    if int(n) < 0:
        raise ValueError(f"prefix steps N must be >= 0, got {n!r}")
    if n <= 0:
        return set()
    T = int(T)
    n_eff = min(int(n), T)
    return set(range(T - n_eff, T))


def apply_cache_scheduler_runtime_overrides(
    sched: Dict[str, Set[int]],
    T: int,
    *,
    force_full_prefix_steps: int = 0,
    force_full_runtime_blocks: Optional[List[str]] = None,
) -> Tuple[Dict[str, Set[int]], Dict[str, Any]]:
    """
    對既有 runtime scheduler 做聯集式保守覆寫（只會增加 recompute，不會減少）。

    - force_full_prefix_steps: 所有 block 前 N 步強制 full（以 DDIM t 表示）。
    - force_full_runtime_blocks: 指定 block 全 timestep 強制 full。
    """
    T = int(T)
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if int(force_full_prefix_steps) < 0:
        raise ValueError(
            f"force_full_prefix_steps must be >= 0, got {force_full_prefix_steps!r}"
        )
    if set(sched.keys()) != set(RUNTIME_LAYER_NAMES):
        raise ValueError(
            "cache_scheduler keys must match full runtime inventory; "
            f"got {sorted(sched.keys())[:8]}... (len={len(sched)})"
        )

    force_full_runtime_blocks = list(force_full_runtime_blocks or [])
    for name in force_full_runtime_blocks:
        if name not in RUNTIME_LAYER_NAMES:
            raise ValueError(
                f"unknown runtime block in force_full_runtime_blocks: {name!r}"
            )

    out: Dict[str, Set[int]] = {k: set(v) for k, v in sched.items()}
    prefix_ts = prefix_ddim_timesteps_first_n(T, force_full_prefix_steps)
    all_ts = set(range(T))

    for k in out:
        out[k] |= prefix_ts
    for name in force_full_runtime_blocks:
        out[name] |= all_ts

    meta: Dict[str, Any] = {
        "force_full_prefix_steps": int(force_full_prefix_steps),
        "prefix_ddim_timesteps": sorted(prefix_ts),
        "force_full_runtime_blocks": list(force_full_runtime_blocks),
        "first_input_runtime_block_name": FIRST_INPUT_RUNTIME_BLOCK_NAME,
        "note": (
            "Overrides are unions on top of Stage1-expanded recompute sets; "
            "they do not replace Stage1 scheduler JSON."
        ),
    }
    return out, meta


def cache_runtime_override_variant_label(
    *,
    force_full_prefix_steps: int,
    force_full_runtime_blocks: List[str],
    safety_first_input_block: bool,
) -> str:
    """回傳簡化的覆寫類別標籤（僅供 log/json 記錄）。"""
    prefix = int(force_full_prefix_steps)
    blocks = list(force_full_runtime_blocks or [])
    if safety_first_input_block and FIRST_INPUT_RUNTIME_BLOCK_NAME not in blocks:
        blocks = blocks + [FIRST_INPUT_RUNTIME_BLOCK_NAME]
    has_prefix = prefix > 0
    has_first = FIRST_INPUT_RUNTIME_BLOCK_NAME in blocks
    extra_blocks = [b for b in blocks if b != FIRST_INPUT_RUNTIME_BLOCK_NAME]

    if not has_prefix and not blocks and not safety_first_input_block:
        return "baseline"
    if has_prefix and not has_first and not extra_blocks:
        return "prefix_only"
    if has_first and not has_prefix and not extra_blocks:
        return "first_input_only"
    if has_prefix and has_first and not extra_blocks:
        return "combined"
    return "custom"
