#!/usr/bin/env python3
"""Export Stage-1 sweep diagnostics to csv_exports (summary / per_block / per_zone)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

SUMMARY_FIELDS = [
    "run",
    "K",
    "sw",
    "lambda",
    "T",
    "B",
    "Z",
    "zone_len_min",
    "zone_len_max",
    "zone_len_mean",
    "change_points_count",
    "merged_zones_count",
    "total_F",
    "total_R",
    "F_ratio",
    "F_mean_per_block",
    "F_std_per_block",
    "R_mean_per_block",
    "cost_mean_per_block",
    "cost_std_per_block",
    "cost_sum_all_blocks",
    "k_mean",
    "k1_count",
    "k2_count",
    "k3_count",
    "k4_count",
    "I_cut_mean",
    "I_cut_std",
    "I_l1cos_mean",
    "I_l1cos_std",
    "Delta_abs_mean",
    "Delta_abs_max",
]

PER_BLOCK_FIELDS = [
    "run",
    "K",
    "sw",
    "lambda",
    "block_id",
    "block_name",
    "num_F",
    "num_R",
    "F_ratio_block",
    "total_cost_J_sum_zones",
    "k_per_zone",
]

PER_ZONE_FIELDS = [
    "run",
    "K",
    "sw",
    "lambda",
    "zone_id",
    "t_start",
    "t_end",
    "length",
    "candidate_k",
    "k1_count",
    "k2_count",
    "k3_count",
    "k4_count",
    "selected_k_mean",
    "selected_J_mean",
    "selected_J_std",
    "selected_k_per_block",
]


def _parse_tag(tag: str) -> Tuple[int, int, float]:
    m = re.match(r"K(\d+)_sw(\d+)_lam([\d.]+)$", tag)
    if not m:
        raise ValueError(f"Unrecognized sweep tag: {tag}")
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def _discover_runs(stage1_output_dir: Path, run_names: Sequence[str] | None = None) -> List[Path]:
    if run_names:
        runs = []
        for name in run_names:
            d = stage1_output_dir / name
            if not d.is_dir():
                raise FileNotFoundError(f"Missing sweep dir: {d}")
            runs.append(d)
        return runs
    return sorted(stage1_output_dir.glob("sweep_*"))


def _load_run(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    tag = run_dir.name.replace("sweep_", "")
    k, sw, lam = _parse_tag(tag)
    with open(run_dir / "scheduler_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(run_dir / "verification_summary.json", encoding="utf-8") as f:
        ver = json.load(f)
    with open(run_dir / "scheduler_diagnostics.json", encoding="utf-8") as f:
        diag = json.load(f)

    blocks = cfg["blocks"]
    b = len(blocks)
    zones = cfg["shared_zones"]
    zone_lens = [int(z["length"]) for z in zones]
    per_block = ver["per_block"]
    total_f = sum(int(p["num_F"]) for p in per_block)
    total_r = sum(int(p["num_R"]) for p in per_block)
    costs = [float(p["total_cost_J_sum_zones"]) for p in per_block]
    ks = [int(kk) for p in per_block for kk in p["k_per_zone"]]
    delta = np.array(diag["Delta_processing_order"], dtype=float)

    summary = {
        "run": run_dir.name,
        "K": k,
        "sw": sw,
        "lambda": lam,
        "T": int(cfg["T"]),
        "B": b,
        "Z": len(zones),
        "zone_len_min": min(zone_lens),
        "zone_len_max": max(zone_lens),
        "zone_len_mean": sum(zone_lens) / len(zone_lens),
        "change_points_count": len(diag["change_points_step_index"]),
        "merged_zones_count": len(diag["merged_step_zones"]),
        "total_F": total_f,
        "total_R": total_r,
        "F_ratio": round(total_f / (total_f + total_r), 4),
        "F_mean_per_block": total_f / b,
        "F_std_per_block": statistics.pstdev([int(p["num_F"]) for p in per_block]) if b > 1 else 0.0,
        "R_mean_per_block": total_r / b,
        "cost_mean_per_block": sum(costs) / b,
        "cost_std_per_block": statistics.pstdev(costs) if b > 1 else 0.0,
        "cost_sum_all_blocks": sum(costs),
        "k_mean": sum(ks) / len(ks),
        "k1_count": sum(1 for kk in ks if kk == 1),
        "k2_count": sum(1 for kk in ks if kk == 2),
        "k3_count": sum(1 for kk in ks if kk == 3),
        "k4_count": sum(1 for kk in ks if kk == 4),
        "I_cut_mean": float(diag["I_cut_stats"]["mean"]),
        "I_cut_std": float(diag["I_cut_stats"]["std"]),
        "I_l1cos_mean": float(diag["I_l1cos_stats"]["mean"]),
        "I_l1cos_std": float(diag["I_l1cos_stats"]["std"]),
        "Delta_abs_mean": float(np.mean(np.abs(delta))),
        "Delta_abs_max": float(np.max(np.abs(delta))),
    }
    return summary, ver, diag, cfg, blocks


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_stage1_sweep_csv(
    stage1_output_dir: str | Path,
    output_subdir: str = "csv_exports/lambda",
    run_names: Sequence[str] | None = None,
    tag_suffix: str | None = None,
) -> Dict[str, Path]:
    root = Path(stage1_output_dir)
    runs = _discover_runs(root, run_names)
    if not runs:
        raise ValueError(f"No sweep runs found under {root}")

    n_runs = len(runs)
    suffix = tag_suffix if tag_suffix is not None else f"{n_runs}runs"
    out_dir = root / output_subdir
    paths = {
        "summary": out_dir / f"stage1_sweep_summary_lambda_{suffix}.csv",
        "per_block": out_dir / f"stage1_sweep_per_block_lambda_{suffix}.csv",
        "per_zone": out_dir / f"stage1_sweep_per_zone_lambda_{suffix}.csv",
    }

    summary_rows: List[Dict[str, Any]] = []
    per_block_rows: List[Dict[str, Any]] = []
    per_zone_rows: List[Dict[str, Any]] = []

    for run_dir in runs:
        summary, ver, _diag, cfg, blocks = _load_run(run_dir)
        summary_rows.append(summary)
        k, sw, lam = summary["K"], summary["sw"], summary["lambda"]

        for pb in ver["per_block"]:
            bid = int(pb["block_id"])
            num_f = int(pb["num_F"])
            num_r = int(pb["num_R"])
            per_block_rows.append(
                {
                    "run": summary["run"],
                    "K": k,
                    "sw": sw,
                    "lambda": lam,
                    "block_id": bid,
                    "block_name": blocks[bid]["name"],
                    "num_F": num_f,
                    "num_R": num_r,
                    "F_ratio_block": round(num_f / (num_f + num_r), 2),
                    "total_cost_J_sum_zones": float(pb["total_cost_J_sum_zones"]),
                    "k_per_zone": ";".join(str(x) for x in pb["k_per_zone"]),
                }
            )

        boundaries = ver["shared_zone_boundaries"]
        for pz in ver["per_zone"]:
            zi = int(pz["zone_id"])
            selected_k = [int(x) for x in pz["selected_k_per_block"]]
            selected_j = [float(x) for x in pz["selected_J_per_block"]]
            per_zone_rows.append(
                {
                    "run": summary["run"],
                    "K": k,
                    "sw": sw,
                    "lambda": lam,
                    "zone_id": zi,
                    "t_start": int(boundaries[zi]["t_start"]),
                    "t_end": int(boundaries[zi]["t_end"]),
                    "length": int(pz["length"]),
                    "candidate_k": ";".join(str(x) for x in pz["candidate_k"]),
                    "k1_count": sum(1 for x in selected_k if x == 1),
                    "k2_count": sum(1 for x in selected_k if x == 2),
                    "k3_count": sum(1 for x in selected_k if x == 3),
                    "k4_count": sum(1 for x in selected_k if x == 4),
                    "selected_k_mean": sum(selected_k) / len(selected_k),
                    "selected_J_mean": statistics.mean(selected_j),
                    "selected_J_std": statistics.pstdev(selected_j) if len(selected_j) > 1 else 0.0,
                    "selected_k_per_block": ";".join(str(x) for x in selected_k),
                }
            )

    _write_csv(paths["summary"], SUMMARY_FIELDS, summary_rows)
    _write_csv(paths["per_block"], PER_BLOCK_FIELDS, per_block_rows)
    _write_csv(paths["per_zone"], PER_ZONE_FIELDS, per_zone_rows)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Export Stage-1 sweep CSV summaries (LDM)")
    p.add_argument(
        "--stage1_output_dir",
        type=str,
        default="ldm_S3cache/cache_method/Stage1/stage1_output_ldm",
    )
    p.add_argument(
        "--output_subdir",
        type=str,
        default="csv_exports/lambda",
        help="Subdirectory under stage1_output_dir for CSV files",
    )
    p.add_argument(
        "--runs",
        type=str,
        default="",
        help="Comma-separated sweep dir names (default: all sweep_* under output dir)",
    )
    p.add_argument(
        "--tag_suffix",
        type=str,
        default="",
        help="Filename suffix, e.g. 3runs (default: <N>runs)",
    )
    args = p.parse_args()

    run_names = [x.strip() for x in args.runs.split(",") if x.strip()] or None
    tag_suffix = args.tag_suffix or None
    paths = export_stage1_sweep_csv(
        args.stage1_output_dir,
        output_subdir=args.output_subdir,
        run_names=run_names,
        tag_suffix=tag_suffix,
    )
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
