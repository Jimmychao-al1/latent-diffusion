#!/usr/bin/env python3
"""
LDM b_SVD - Stage C: correlate SVD subspace drift with a_L1 similarity metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_svd_metrics(svd_json_path: Path) -> Dict:
    if not svd_json_path.exists():
        raise FileNotFoundError(f"SVD json not found: {svd_json_path}")
    with open(svd_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_similarity_npz(npz_path: Path) -> Dict:
    if not npz_path.exists():
        raise FileNotFoundError(f"similarity npz not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)
    if "l1_step_mean" not in keys or "cos_step_mean" not in keys:
        raise KeyError(f"Required keys missing in {npz_path}: l1_step_mean/cos_step_mean")

    return {
        "l1_step_mean": data["l1_step_mean"],
        "cos_step_mean": data["cos_step_mean"],
        "t_curr_interval": data["t_curr_interval"] if "t_curr_interval" in keys else None,
    }


def compute_correlations(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if len(x) != len(y):
        raise ValueError(f"length mismatch: len(x)={len(x)} len(y)={len(y)}")
    if len(x) < 2:
        raise ValueError(f"insufficient pairs: {len(x)}")
    if not np.isfinite(x).all():
        bad = int((~np.isfinite(x)).sum())
        raise ValueError(f"x contains non-finite values: count={bad}")
    if not np.isfinite(y).all():
        bad = int((~np.isfinite(y)).sum())
        raise ValueError(f"y contains non-finite values: count={bad}")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        raise ValueError("constant input sequence; correlation undefined")

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    vals = [pearson_r, spearman_r, pearson_p, spearman_p]
    if not all(np.isfinite(v) for v in vals):
        raise ValueError(f"correlation result is non-finite: {vals}")

    return {
        "pearson": float(pearson_r),
        "spearman": float(spearman_r),
        "pearson_pvalue": float(pearson_p),
        "spearman_pvalue": float(spearman_p),
        "valid_pairs": int(len(x)),
        "status": "ok",
    }


def plot_alignment(
    svd_dist: np.ndarray,
    l1: np.ndarray,
    cos_dist: np.ndarray,
    block_slug: str,
    output_path: Path,
    t_curr_interval: Optional[np.ndarray] = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    l = len(svd_dist)
    x = np.arange(l)

    xticks = list(range(0, l, 10))
    if (l - 1) not in xticks:
        xticks.append(l - 1)
    if t_curr_interval is not None and len(t_curr_interval) == l:
        xticklabels = [str(int(t_curr_interval[i])) for i in xticks]
    else:
        xticklabels = [str(int((l - 1) - i)) for i in xticks]

    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(x, l1, "b-", linewidth=2, label="L1 step mean", alpha=0.8)
    ax1_twin.plot(x, svd_dist, "r--", linewidth=2, label="SVD Subspace Dist", alpha=0.8)
    ax1.set_xlabel("DDIM current timestep t_curr")
    ax1.set_ylabel("L1 step mean", color="b")
    ax1_twin.set_ylabel("SVD Subspace Distance", color="r")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1_twin.tick_params(axis="y", labelcolor="r")
    ax1.set_title(f"{block_slug} - L1 step mean vs SVD Subspace Distance", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.plot(x, cos_dist, "g-", linewidth=2, label="Cosine distance (1 − cosine similarity)", alpha=0.8)
    ax2_twin.plot(x, svd_dist, "r--", linewidth=2, label="SVD Subspace Dist", alpha=0.8)
    ax2.set_xlabel("DDIM current timestep t_curr")
    ax2.set_ylabel("Cosine distance (1 − cosine similarity)", color="g")
    ax2_twin.set_ylabel("SVD Subspace Distance", color="r")
    ax2.tick_params(axis="y", labelcolor="g")
    ax2_twin.tick_params(axis="y", labelcolor="r")
    ax2.set_title(
        f"{block_slug} - Cosine distance (1 − cosine similarity) vs SVD Subspace Distance",
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scatter(
    svd_dist: np.ndarray,
    l1: np.ndarray,
    cos_dist: np.ndarray,
    block_slug: str,
    output_path: Path,
    l1_corr: Dict,
    cos_corr: Dict,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    l1_pearson = l1_corr["pearson"] if l1_corr["pearson"] is not None else float("nan")
    l1_spearman = l1_corr["spearman"] if l1_corr["spearman"] is not None else float("nan")
    cos_pearson = cos_corr["pearson"] if cos_corr["pearson"] is not None else float("nan")
    cos_spearman = cos_corr["spearman"] if cos_corr["spearman"] is not None else float("nan")

    ax1 = axes[0]
    ax1.scatter(l1, svd_dist, alpha=0.6, s=30, c="blue")
    ax1.set_xlabel("L1 step mean")
    ax1.set_ylabel("SVD Subspace Distance")
    ax1.set_title(
        f"{block_slug} - L1 step mean vs SVD\nPearson: {l1_pearson:.4f}, Spearman: {l1_spearman:.4f}",
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(cos_dist, svd_dist, alpha=0.6, s=30, c="green")
    ax2.set_xlabel("Cosine distance (1 − cosine similarity)")
    ax2.set_ylabel("SVD Subspace Distance")
    ax2.set_title(
        f"{block_slug} - Cosine Distance vs SVD\nPearson: {cos_pearson:.4f}, Spearman: {cos_spearman:.4f}",
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def process_single_correlation(
    svd_json_path: Path,
    similarity_npz_path: Path,
    output_dir: Path,
    plot_figures: bool = True,
) -> Dict:
    svd_data = load_svd_metrics(svd_json_path)
    block_slug = svd_data["block"]
    svd_dist = np.array(svd_data["subspace_dist"])

    sim_data = load_similarity_npz(similarity_npz_path)
    l1_step_mean = np.asarray(sim_data["l1_step_mean"])
    cos_step_mean = np.asarray(sim_data["cos_step_mean"])
    t_curr_interval = sim_data.get("t_curr_interval", None)

    interval_len = len(l1_step_mean)
    lengths = [interval_len, len(cos_step_mean)]
    if t_curr_interval is not None:
        t_curr_interval = np.asarray(t_curr_interval)
        lengths.append(len(t_curr_interval))

    min_interval_len = min(lengths)
    if min_interval_len <= 0:
        raise ValueError("similarity interval sequence is empty")

    l = min(min_interval_len, max(len(svd_dist) - 1, 0))
    if l <= 1:
        raise ValueError(f"Aligned length too short: {l}")

    svd_seq = svd_dist[1 : 1 + l]
    l1_seq = l1_step_mean[:l]
    cos_dist_seq = 1.0 - cos_step_mean[:l]
    if t_curr_interval is not None:
        t_curr_interval = t_curr_interval[:l]

    if not np.isfinite(l1_seq).all():
        raise ValueError(f"l1_step_mean has non-finite values: {(~np.isfinite(l1_seq)).sum()}")
    if not np.isfinite(cos_dist_seq).all():
        raise ValueError(f"cos_dist_seq has non-finite values: {(~np.isfinite(cos_dist_seq)).sum()}")
    if not np.isfinite(svd_seq).all():
        raise ValueError(f"svd_seq has non-finite values: {(~np.isfinite(svd_seq)).sum()}")

    l1_vs_svd = compute_correlations(l1_seq, svd_seq)
    cos_vs_svd = compute_correlations(cos_dist_seq, svd_seq)

    print("L1 vs SVD:")
    print(f"  Pearson: {l1_vs_svd['pearson']:.4f} (p={l1_vs_svd['pearson_pvalue']:.4e})")
    print(f"  Spearman: {l1_vs_svd['spearman']:.4f} (p={l1_vs_svd['spearman_pvalue']:.4e})")

    print("Cosine Distance vs SVD:")
    print(f"  Pearson: {cos_vs_svd['pearson']:.4f} (p={cos_vs_svd['pearson_pvalue']:.4e})")
    print(f"  Spearman: {cos_vs_svd['spearman']:.4f} (p={cos_vs_svd['spearman_pvalue']:.4e})")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"{block_slug}.json"

    result = {
        "block": block_slug,
        "T_svd": int(svd_data["T"]),
        "interval_length_used": int(l),
        "rank_r": int(svd_data["rank_r"]),
        "x_axis_def": "interval-wise t_curr (left=noise side, right=clear side)",
        "correlation": {
            "L1_vs_SVD": l1_vs_svd,
            "CosDist_vs_SVD": cos_vs_svd,
        },
    }
    if t_curr_interval is not None:
        result["t_curr_interval"] = t_curr_interval.tolist()

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if plot_figures:
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_alignment(
            svd_dist=svd_seq,
            l1=l1_seq,
            cos_dist=cos_dist_seq,
            block_slug=block_slug,
            output_path=fig_dir / f"{block_slug}_alignment.png",
            t_curr_interval=t_curr_interval,
        )
        plot_scatter(
            svd_dist=svd_seq,
            l1=l1_seq,
            cos_dist=cos_dist_seq,
            block_slug=block_slug,
            output_path=fig_dir / f"{block_slug}_scatter.png",
            l1_corr=l1_vs_svd,
            cos_corr=cos_vs_svd,
        )

    print(f"[CORR] output: {output_json}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="LDM SVD vs Similarity Correlation")
    p.add_argument("--svd_metrics", type=str, help="single block svd metrics json")
    p.add_argument("--similarity_npz", type=str, help="single block similarity npz")
    p.add_argument(
        "--svd_metrics_dir",
        type=str,
        default="ldm_S3cache/cache_method/b_SVD/T_200/svd_metrics",
        help="svd metrics dir for --all",
    )
    p.add_argument(
        "--similarity_npz_dir",
        type=str,
        default="ldm_S3cache/cache_method/a_L1_L2_cosine/T_200/v2_latest/result_npz",
        help="similarity npz dir for --all",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="ldm_S3cache/cache_method/b_SVD/T_200/correlation",
        help="correlation output root",
    )
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_root)

    if args.all:
        svd_dir = Path(args.svd_metrics_dir)
        sim_dir = Path(args.similarity_npz_dir)
        if not svd_dir.exists():
            raise FileNotFoundError(f"svd_metrics_dir not found: {svd_dir}")
        if not sim_dir.exists():
            raise FileNotFoundError(f"similarity_npz_dir not found: {sim_dir}")

        svd_jsons = sorted(svd_dir.glob("*.json"))
        ok = 0
        for svd_json in svd_jsons:
            block_slug = svd_json.stem
            sim_npz = sim_dir / f"{block_slug}.npz"
            if not sim_npz.exists():
                print(f"[SKIP] missing similarity npz for {block_slug}")
                continue
            try:
                process_single_correlation(
                    svd_json_path=svd_json,
                    similarity_npz_path=sim_npz,
                    output_dir=output_dir,
                    plot_figures=args.plot,
                )
                ok += 1
            except Exception as e:
                print(f"[WARN] failed on {block_slug}: {e}")
        print(f"Done: {ok}/{len(svd_jsons)}")
        return

    if args.svd_metrics and args.similarity_npz:
        process_single_correlation(
            svd_json_path=Path(args.svd_metrics),
            similarity_npz_path=Path(args.similarity_npz),
            output_dir=output_dir,
            plot_figures=args.plot,
        )
        return

    p.print_help()


if __name__ == "__main__":
    main()
