#!/usr/bin/env python3
"""
Export Stage2 runtime diagnostics JSON files into a single AI-friendly CSV.

Row scopes:
- global: one row per diagnostics file
- block_summary: one row per block (aggregated from per-step metrics)
- block_zone: one row per block per zone
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_source_params(source_name: str) -> Tuple[Optional[int], Optional[int], Optional[float], str]:
    """
    Parse source like:
      src_K15_sw3_lam1.0
    """
    m = re.fullmatch(r"src_K(?P<K>\d+)_sw(?P<sw>\d+)_lam(?P<lam>[0-9.]+)", source_name)
    if not m:
        return None, None, None, source_name
    return int(m.group("K")), int(m.group("sw")), float(m.group("lam")), f"K{m.group('K')}_sw{m.group('sw')}_lam{m.group('lam')}"


def _safe_mean(xs: Iterable[float]) -> Optional[float]:
    xs_list = list(xs)
    if not xs_list:
        return None
    return float(mean(xs_list))


def _safe_argmax(pairs: List[Tuple[int, float]]) -> Tuple[Optional[int], Optional[float]]:
    if not pairs:
        return None, None
    idx, value = max(pairs, key=lambda x: x[1])
    return idx, float(value)


def _safe_argmin(pairs: List[Tuple[int, float]]) -> Tuple[Optional[int], Optional[float]]:
    if not pairs:
        return None, None
    idx, value = min(pairs, key=lambda x: x[1])
    return idx, float(value)


def _zone_sort_key(zone_id: str) -> Tuple[int, str]:
    try:
        return int(zone_id), zone_id
    except Exception:
        return 10**9, zone_id


def _collect_rows(diag_path: Path) -> List[Dict[str, Any]]:
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    source_name = diag_path.parents[1].name
    pass_name = diag_path.parents[0].name
    K, sw, lam, scheduler_id = _parse_source_params(source_name)

    common = {
        "source_name": source_name,
        "scheduler_id": scheduler_id,
        "K": K,
        "sw": sw,
        "lambda": lam,
        "pass_name": pass_name,
        "diagnostics_path": str(diag_path),
        "scheduler_config_path": diag.get("scheduler_config_path"),
        "T": diag.get("T"),
        "threshold_meta_json": json.dumps(diag.get("stage2_threshold_meta", {}), ensure_ascii=False),
        "time_axis_note": diag.get("time_axis_note"),
    }

    rows: List[Dict[str, Any]] = []

    gs = diag.get("global_summary", {})
    rows.append(
        {
            **common,
            "row_scope": "global",
            "block_id": "",
            "zone_id": "",
            "mean_l1": gs.get("mean_l1"),
            "mean_l2": gs.get("mean_l2"),
            "mean_cosine": gs.get("mean_cosine"),
            "num_entries": gs.get("num_entries"),
            "note": gs.get("note"),
        }
    )

    per_step = diag.get("per_block_step_error", {})
    per_zone = diag.get("per_block_zone_error", {})
    refined = diag.get("refined_cache_scheduler", {})

    for block_id, step_dict in per_step.items():
        step_pairs_l1: List[Tuple[int, float]] = []
        step_pairs_l2: List[Tuple[int, float]] = []
        step_pairs_cos: List[Tuple[int, float]] = []
        for t_str, metrics in step_dict.items():
            try:
                t = int(t_str)
            except Exception:
                continue
            l1 = metrics.get("l1")
            l2 = metrics.get("l2")
            cosine = metrics.get("cosine")
            if isinstance(l1, (int, float)):
                step_pairs_l1.append((t, float(l1)))
            if isinstance(l2, (int, float)):
                step_pairs_l2.append((t, float(l2)))
            if isinstance(cosine, (int, float)):
                step_pairs_cos.append((t, float(cosine)))

        max_l1_t, max_l1 = _safe_argmax(step_pairs_l1)
        max_l2_t, max_l2 = _safe_argmax(step_pairs_l2)
        min_cos_t, min_cos = _safe_argmin(step_pairs_cos)

        sched_block = refined.get(block_id, {})
        full_steps = None
        reuse_steps = None
        full_compute_step_indices_json = ""
        k_per_zone_json = ""
        T = diag.get("T")
        if isinstance(sched_block, list):
            full_steps = int(len(sched_block))
            if isinstance(T, int):
                reuse_steps = int(T - full_steps)
            full_compute_step_indices_json = json.dumps(sched_block, ensure_ascii=False)
        elif isinstance(sched_block, dict):
            mask = sched_block.get("expanded_mask", [])
            if isinstance(mask, list):
                full_steps = int(sum(bool(x) for x in mask))
                reuse_steps = int(len(mask) - full_steps)
            k_per_zone_json = json.dumps(sched_block.get("k_per_zone", []), ensure_ascii=False)

        rows.append(
            {
                **common,
                "row_scope": "block_summary",
                "block_id": block_id,
                "zone_id": "",
                "step_count": len(step_dict),
                "mean_l1": _safe_mean(v for _, v in step_pairs_l1),
                "mean_l2": _safe_mean(v for _, v in step_pairs_l2),
                "mean_cosine": _safe_mean(v for _, v in step_pairs_cos),
                "max_l1": max_l1,
                "step_at_max_l1": max_l1_t,
                "max_l2": max_l2,
                "step_at_max_l2": max_l2_t,
                "min_cosine": min_cos,
                "step_at_min_cosine": min_cos_t,
                "num_entries": len(step_dict),
                "full_compute_steps": full_steps,
                "reuse_steps": reuse_steps,
                "k_per_zone_json": k_per_zone_json,
                "full_compute_step_indices_json": full_compute_step_indices_json,
            }
        )

        zone_dict = per_zone.get(block_id, {})
        for zone_id in sorted(zone_dict.keys(), key=_zone_sort_key):
            z = zone_dict[zone_id]
            rows.append(
                {
                    **common,
                    "row_scope": "block_zone",
                    "block_id": block_id,
                    "zone_id": zone_id,
                    "mean_l1": z.get("mean_l1"),
                    "mean_l2": z.get("mean_l2"),
                    "mean_cosine": z.get("mean_cosine"),
                    "num_steps": z.get("num_steps"),
                    "num_compared_in_zone": z.get("num_compared_in_zone"),
                    "num_entries": z.get("num_compared_in_zone"),
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2_output_root",
        type=Path,
        default=Path("ldm_S3cache/cache_method/Stage2/stage2_output_ldm"),
        help="Root directory containing src_*/00_global_refine and 02_refined_blockwise outputs.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <stage2_output_root>/csv_exports/stage2_runtime_diagnostics_combined.csv",
    )
    args = parser.parse_args()

    output_root = args.stage2_output_root
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = output_root / "csv_exports" / "stage2_runtime_diagnostics_combined.csv"

    diag_paths = sorted(output_root.glob("src_*/**/stage2_runtime_diagnostics.json"))
    if not diag_paths:
        raise SystemExit(f"No diagnostics found under: {output_root}")

    all_rows: List[Dict[str, Any]] = []
    for p in diag_paths:
        all_rows.extend(_collect_rows(p))

    fieldnames = [
        "source_name",
        "scheduler_id",
        "K",
        "sw",
        "lambda",
        "pass_name",
        "row_scope",
        "block_id",
        "zone_id",
        "mean_l1",
        "mean_l2",
        "mean_cosine",
        "max_l1",
        "step_at_max_l1",
        "max_l2",
        "step_at_max_l2",
        "min_cosine",
        "step_at_min_cosine",
        "num_steps",
        "num_compared_in_zone",
        "step_count",
        "num_entries",
        "full_compute_steps",
        "reuse_steps",
        "k_per_zone_json",
        "full_compute_step_indices_json",
        "T",
        "scheduler_config_path",
        "diagnostics_path",
        "threshold_meta_json",
        "time_axis_note",
        "note",
    ]

    scope_rank = {"global": 0, "block_summary": 1, "block_zone": 2}

    def _sort_key(r: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            r.get("source_name", ""),
            r.get("pass_name", ""),
            scope_rank.get(str(r.get("row_scope", "")), 9),
            r.get("block_id", ""),
            _zone_sort_key(str(r.get("zone_id", ""))),
        )

    all_rows.sort(key=_sort_key)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"✅ Wrote CSV: {output_csv}")
    print(f"   Diagnostics files: {len(diag_paths)}")
    print(f"   Total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
