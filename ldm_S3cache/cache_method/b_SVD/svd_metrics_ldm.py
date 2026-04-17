#!/usr/bin/env python3
"""
LDM b_SVD - Stage B: compute SVD metrics from collected block features.

Supports:
- file mode: load svd_features/<block_slug>/t_{t}.pt + meta.json
- in-memory mode: process feature_buffers directly (used by Stage A->B->C pipeline)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm


def load_features(feature_dir: Path) -> Tuple[List[torch.Tensor], Dict]:
    meta_path = feature_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    T = int(meta["T"])
    features: List[torch.Tensor] = []
    for t in tqdm(range(T), desc=f"Loading {meta.get('block', feature_dir.name)}"):
        pt_path = feature_dir / f"t_{t}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing feature file: {pt_path}")
        features.append(torch.load(pt_path, map_location="cpu"))

    return features, meta


def compute_covariance_eigen(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute eigenvalues/vectors of uncentered channel second-moment matrix."""
    n, c, h, w = x.shape
    m = n * h * w
    x_reshaped = x.permute(1, 0, 2, 3).reshape(c, m).double()
    sigma = (x_reshaped @ x_reshaped.T) / m
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvectors = torch.flip(eigenvectors, dims=[1])
    return eigenvalues, eigenvectors


def compute_rank_r(eigenvalues: torch.Tensor, energy_threshold: float = 0.98) -> int:
    total = eigenvalues.sum()
    cumulative = torch.cumsum(eigenvalues, dim=0)
    r = torch.searchsorted(cumulative, energy_threshold * total).item() + 1
    r = max(1, min(r, len(eigenvalues)))
    return int(r)


def compute_subspace_distance(u_t: torch.Tensor, u_prev: torch.Tensor, r: int) -> float:
    u_t_r = u_t[:, :r]
    u_prev_r = u_prev[:, :r]
    m = u_t_r.T @ u_prev_r
    norm_sq = (m * m).sum()
    dist = 1.0 - (norm_sq / r).item()
    return float(dist)


def compute_energy_ratios(
    eigenvalues_list: List[torch.Tensor],
    k_values: List[int] = [4, 8, 16, 32, 64],
) -> Dict[str, List[float]]:
    energy_ratio: Dict[str, List[float]] = {f"k{k}": [] for k in k_values}
    for eigenvalues in eigenvalues_list:
        total = eigenvalues.sum()
        for k in k_values:
            k_actual = min(k, len(eigenvalues))
            ratio = (eigenvalues[:k_actual].sum() / total).item()
            energy_ratio[f"k{k}"].append(float(ratio))
    return energy_ratio


def process_features_in_memory(
    features: List[torch.Tensor],
    meta: Dict,
    output_dir: Path,
    representative_t: int,
    energy_threshold: float,
    compute_energy: bool = True,
) -> Dict:
    t_total = int(meta["T"])
    c = int(meta["C"])
    block_slug = str(meta["block"])

    if len(features) != t_total:
        raise ValueError(f"Feature length mismatch: len(features)={len(features)}, T={t_total}")

    eigenvalues_list: List[torch.Tensor] = []
    eigenvectors_list: List[torch.Tensor] = []

    for t in tqdm(range(t_total), desc="SVD"):
        eigenvalues, eigenvectors = compute_covariance_eigen(features[t])
        eigenvalues_list.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)

    if representative_t < 0 or representative_t >= t_total:
        representative_t = t_total - 1

    eigenvalues_ref = eigenvalues_list[representative_t]
    rank_r = compute_rank_r(eigenvalues_ref, energy_threshold)
    cumulative_ref = torch.cumsum(eigenvalues_ref, dim=0) / eigenvalues_ref.sum()
    actual_energy = float(cumulative_ref[rank_r - 1].item())

    subspace_dist: List[float] = [0.0]
    for t in tqdm(range(1, t_total), desc="Subspace distance"):
        dist = compute_subspace_distance(eigenvectors_list[t], eigenvectors_list[t - 1], rank_r)
        subspace_dist.append(float(dist))

    energy_ratio = compute_energy_ratios(eigenvalues_list) if compute_energy else None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{block_slug}.json"

    result = {
        "block": block_slug,
        "target_block_name": meta.get("target_block_name", block_slug),
        "T": t_total,
        "C": c,
        "N": int(meta["N"]),
        "H": int(meta["H"]),
        "W": int(meta["W"]),
        "rank_r": int(rank_r),
        "representative_t": int(representative_t),
        "energy_threshold": float(energy_threshold),
        "actual_energy_at_r": float(actual_energy),
        "timesteps": list(range(t_total)),
        "subspace_dist": subspace_dist,
    }

    if energy_ratio is not None:
        result["energy_ratio"] = energy_ratio

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[SVD] output: {output_path}")
    return result


def process_feature_buffers_in_memory(
    feature_buffers: Dict[int, List[torch.Tensor]],
    meta: Dict,
    target_n: int,
    output_dir: Path,
    representative_t: int,
    energy_threshold: float,
    compute_energy: bool = True,
) -> Dict:
    t_total = int(meta["T"])
    c = int(meta["C"])
    block_slug = str(meta["block"])

    eigenvalues_list: List[torch.Tensor] = []
    eigenvectors_list: List[torch.Tensor] = []
    actual_n = None

    for t in tqdm(range(t_total), desc="SVD"):
        chunks = feature_buffers.get(t, [])
        if len(chunks) == 0:
            raise RuntimeError(f"No feature chunks for t={t}")

        tensor_t = torch.cat(chunks, dim=0)
        if tensor_t.shape[0] > target_n:
            tensor_t = tensor_t[:target_n]

        n_t = int(tensor_t.shape[0])
        if n_t <= 0:
            raise RuntimeError(f"No valid samples at t={t}")
        if actual_n is None:
            actual_n = n_t

        eigenvalues, eigenvectors = compute_covariance_eigen(tensor_t)
        eigenvalues_list.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)

        feature_buffers[t] = []
        del tensor_t

    if representative_t < 0 or representative_t >= t_total:
        representative_t = t_total - 1

    eigenvalues_ref = eigenvalues_list[representative_t]
    rank_r = compute_rank_r(eigenvalues_ref, energy_threshold)
    cumulative_ref = torch.cumsum(eigenvalues_ref, dim=0) / eigenvalues_ref.sum()
    actual_energy = float(cumulative_ref[rank_r - 1].item())

    subspace_dist: List[float] = [0.0]
    for t in tqdm(range(1, t_total), desc="Subspace distance"):
        dist = compute_subspace_distance(eigenvectors_list[t], eigenvectors_list[t - 1], rank_r)
        subspace_dist.append(float(dist))

    energy_ratio = compute_energy_ratios(eigenvalues_list) if compute_energy else None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{block_slug}.json"

    result = {
        "block": block_slug,
        "target_block_name": meta.get("target_block_name", block_slug),
        "T": t_total,
        "C": c,
        "N": int(actual_n if actual_n is not None else target_n),
        "H": int(meta["H"]),
        "W": int(meta["W"]),
        "rank_r": int(rank_r),
        "representative_t": int(representative_t),
        "energy_threshold": float(energy_threshold),
        "actual_energy_at_r": float(actual_energy),
        "timesteps": list(range(t_total)),
        "subspace_dist": subspace_dist,
    }

    if energy_ratio is not None:
        result["energy_ratio"] = energy_ratio

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[SVD] output: {output_path}")
    return result


def process_single_block(
    feature_dir: Path,
    output_dir: Path,
    representative_t: int,
    energy_threshold: float,
    compute_energy: bool = True,
) -> Dict:
    features, meta = load_features(feature_dir)
    if representative_t < 0:
        representative_t = int(meta["T"]) - 1
    return process_features_in_memory(
        features=features,
        meta=meta,
        output_dir=output_dir,
        representative_t=representative_t,
        energy_threshold=energy_threshold,
        compute_energy=compute_energy,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="LDM SVD Metrics")
    p.add_argument("--feature_dir", type=str, help="Single block feature dir")
    p.add_argument(
        "--feature_root",
        type=str,
        default="ldm_S3cache/cache_method/b_SVD/T_200/svd_features",
        help="Feature root for --all",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="ldm_S3cache/cache_method/b_SVD/T_200/svd_metrics",
        help="Output dir for metrics json",
    )
    p.add_argument("--representative-t", type=int, default=-1)
    p.add_argument("--energy-threshold", type=float, default=0.98)
    p.add_argument("--no-compute-energy", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_root)
    compute_energy = not args.no_compute_energy

    if args.all:
        feature_root = Path(args.feature_root)
        if not feature_root.exists():
            raise FileNotFoundError(f"feature_root not found: {feature_root}")
        block_dirs = sorted([d for d in feature_root.iterdir() if d.is_dir()])
        print(f"Found {len(block_dirs)} blocks")
        for block_dir in block_dirs:
            try:
                process_single_block(
                    feature_dir=block_dir,
                    output_dir=output_dir,
                    representative_t=args.representative_t,
                    energy_threshold=args.energy_threshold,
                    compute_energy=compute_energy,
                )
            except Exception as e:
                print(f"[WARN] failed on {block_dir.name}: {e}")
        return

    if args.feature_dir:
        process_single_block(
            feature_dir=Path(args.feature_dir),
            output_dir=output_dir,
            representative_t=args.representative_t,
            energy_threshold=args.energy_threshold,
            compute_energy=compute_energy,
        )
        return

    p.print_help()


if __name__ == "__main__":
    main()
