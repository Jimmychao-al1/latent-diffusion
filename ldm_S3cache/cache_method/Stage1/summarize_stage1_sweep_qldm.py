#!/usr/bin/env python3
"""Build sweep_summary.csv from Q-LDM Stage-1 sweep directories."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

SUMMARY_FIELDS = [
    "K",
    "sw",
    "lambda",
    "cost_J",
    "F_ratio",
    "F_ratio_pct",
    "num_blocks",
    "k_min",
    "k_max",
    "k_mean",
    "error",
]

FP_LDM_BEST = {
    "K": 15,
    "sw": 3,
    "lambda": 1.0,
    "cost_J": 98.683665,
    "F_ratio": 0.2702,
    "F_ratio_pct": 27.02,
}


def _parse_tag(dirname: str) -> tuple[int, int, float]:
    m = re.match(r"(?:sweep_)?K(\d+)_sw(\d+)_lam([\d.]+)$", dirname)
    if not m:
        raise ValueError(f"Unrecognized sweep dir name: {dirname}")
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def _load_run(run_dir: Path) -> Dict[str, Any]:
    k, sw, lam = _parse_tag(run_dir.name)
    with open(run_dir / "verification_summary.json", encoding="utf-8") as f:
        ver = json.load(f)

    per_block = ver["per_block"]
    num_blocks = len(per_block)
    total_f = sum(int(p["num_F"]) for p in per_block)
    total_r = sum(int(p["num_R"]) for p in per_block)
    f_ratio = total_f / (total_f + total_r) if (total_f + total_r) > 0 else 0.0
    costs = [float(p["total_cost_J_sum_zones"]) for p in per_block]
    ks = [int(x) for p in per_block for x in p["k_per_zone"]]

    return {
        "K": k,
        "sw": sw,
        "lambda": lam,
        "cost_J": round(sum(costs), 6),
        "F_ratio": round(f_ratio, 4),
        "F_ratio_pct": round(f_ratio * 100.0, 2),
        "num_blocks": num_blocks,
        "k_min": min(ks) if ks else "",
        "k_max": max(ks) if ks else "",
        "k_mean": round(sum(ks) / len(ks), 4) if ks else "",
        "error": "",
    }


def discover_runs(stage1_output_dir: Path) -> List[Path]:
    runs = []
    for pattern in ("sweep_K*", "K*_sw*"):
        runs.extend(stage1_output_dir.glob(pattern))
    uniq = sorted({p.resolve() for p in runs if p.is_dir()})
    return uniq


def build_summary(stage1_output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in discover_runs(stage1_output_dir):
        try:
            if not (run_dir / "verification_summary.json").is_file():
                raise FileNotFoundError("missing verification_summary.json")
            rows.append(_load_run(run_dir))
        except Exception as exc:
            k, sw, lam = (0, 0, 0.0)
            try:
                k, sw, lam = _parse_tag(run_dir.name)
            except ValueError:
                pass
            rows.append(
                {
                    "K": k,
                    "sw": sw,
                    "lambda": lam,
                    "cost_J": "",
                    "F_ratio": "",
                    "F_ratio_pct": "",
                    "num_blocks": "",
                    "k_min": "",
                    "k_max": "",
                    "k_mean": "",
                    "error": str(exc),
                }
            )
    rows.sort(key=lambda r: (r["cost_J"] == "", float(r["cost_J"]) if r["cost_J"] != "" else float("inf")))
    return rows


def _print_table(rows: List[Dict[str, Any]]) -> None:
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        print("No successful runs to display.")
        return

    headers = ["K", "sw", "lambda", "cost_J", "F_ratio_pct", "k_min", "k_max", "k_mean"]
    widths = {h: max(len(h), *(len(str(r[h])) for r in ok)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("-" * len(line))
    for r in ok:
        print("  ".join(str(r[h]).ljust(widths[h]) for h in headers))


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize Q-LDM Stage-1 sweep")
    p.add_argument(
        "--stage1_output_dir",
        type=str,
        default="ldm_S3cache/cache_method/Stage1/stage1_output_qldm",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Default: <stage1_output_dir>/sweep_summary.csv",
    )
    args = p.parse_args()

    root = Path(args.stage1_output_dir)
    out_csv = Path(args.output_csv) if args.output_csv else root / "sweep_summary.csv"
    rows = build_summary(root)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {out_csv}")
    print("\n=== Full sweep table (sorted by cost_J) ===")
    _print_table(rows)

    ok = [r for r in rows if not r.get("error")]
    print("\n=== Top 5 configurations ===")
    for i, r in enumerate(ok[:5], 1):
        print(
            f"  {i}. K={r['K']} sw={r['sw']} lam={r['lambda']}  "
            f"cost_J={r['cost_J']}  F_ratio={r['F_ratio_pct']}%"
        )

    if ok:
        best = ok[0]
        print("\n=== vs FP LDM best (K15 sw3 lam1.0) ===")
        print(
            f"  FP LDM:  cost_J={FP_LDM_BEST['cost_J']:.4f}  "
            f"F_ratio={FP_LDM_BEST['F_ratio_pct']:.2f}%"
        )
        print(
            f"  Q-LDM:   cost_J={best['cost_J']}  "
            f"F_ratio={best['F_ratio_pct']}%  "
            f"(K={best['K']} sw={best['sw']} lam={best['lambda']})"
        )
        delta_j = float(best["cost_J"]) - FP_LDM_BEST["cost_J"]
        delta_f = float(best["F_ratio_pct"]) - FP_LDM_BEST["F_ratio_pct"]
        print(f"  Delta:   cost_J {delta_j:+.4f}  F_ratio {delta_f:+.2f} pp")


if __name__ == "__main__":
    main()
