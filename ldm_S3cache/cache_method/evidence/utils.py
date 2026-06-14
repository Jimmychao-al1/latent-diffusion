"""Evidence metrics, accumulators, and export helpers for LDM/Q-LDM S3-Cache."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def compute_ldm_similarity_batch(
    t_prev: torch.Tensor,
    t_curr: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match ``LDMSimilarityCollector._calc_metrics_batch`` (symmetric L1 + cosine)."""

    t1f = t_prev.detach().to(torch.float32)
    t2f = t_curr.detach().to(torch.float32)

    diff = torch.abs(t1f - t2f)
    l1_diff = diff.mean(dim=(1, 2, 3))
    l1_ref = (torch.abs(t1f).mean(dim=(1, 2, 3)) + torch.abs(t2f).mean(dim=(1, 2, 3))) / 2.0 + eps
    l1_vals = l1_diff / l1_ref

    l1_rate_diff = diff.sum(dim=(1, 2, 3))
    l1_rate_ref = torch.abs(t1f).sum(dim=(1, 2, 3)) + eps
    l1_rate_vals = l1_rate_diff / l1_rate_ref

    t1_flat = t1f.reshape(t1f.size(0), -1)
    t2_flat = t2f.reshape(t2f.size(0), -1)
    dot = (t1_flat * t2_flat).sum(dim=1)
    denom = torch.clamp(t1_flat.norm(dim=1) * t2_flat.norm(dim=1), min=eps)
    cos_vals = dot / denom
    cos_vals = torch.nan_to_num(cos_vals, nan=0.0, posinf=1.0, neginf=-1.0)
    cos_vals = torch.clamp(cos_vals, min=-1.0, max=1.0)

    return l1_vals, l1_rate_vals, cos_vals


@dataclass
class SimilarityAccumulator:
    """Per-block running average for interval-wise L1 / cosine metrics."""

    n_blocks: int
    n_intervals: int
    block_names: list[str]
    l1_sum: np.ndarray = field(init=False)
    l1_rate_sum: np.ndarray = field(init=False)
    cos_sum: np.ndarray = field(init=False)
    counts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        shape = (self.n_blocks, self.n_intervals)
        self.l1_sum = np.zeros(shape, dtype=np.float64)
        self.l1_rate_sum = np.zeros(shape, dtype=np.float64)
        self.cos_sum = np.zeros(shape, dtype=np.float64)
        self.counts = np.zeros(shape, dtype=np.int64)

    def block_index(self, block_name: str) -> int:
        return self.block_names.index(block_name)

    def add_interval(
        self,
        block_idx: int,
        interval_idx: int,
        l1_vals: torch.Tensor,
        l1_rate_vals: torch.Tensor,
        cos_vals: torch.Tensor,
    ) -> None:
        if interval_idx < 0 or interval_idx >= self.n_intervals:
            return
        n = int(l1_vals.numel())
        if n <= 0:
            return
        self.l1_sum[block_idx, interval_idx] += float(l1_vals.sum().item())
        self.l1_rate_sum[block_idx, interval_idx] += float(l1_rate_vals.sum().item())
        self.cos_sum[block_idx, interval_idx] += float(cos_vals.sum().item())
        self.counts[block_idx, interval_idx] += n

    def finalize(self) -> dict[str, dict[str, np.ndarray]]:
        out: dict[str, dict[str, np.ndarray]] = {}
        for idx, block_name in enumerate(self.block_names):
            counts = self.counts[idx]
            safe = np.maximum(counts, 1)
            out[block_name] = {
                "l1_step_mean": np.divide(self.l1_sum[idx], safe, where=counts > 0),
                "l1_rate_step_mean": np.divide(self.l1_rate_sum[idx], safe, where=counts > 0),
                "cos_step_mean": np.divide(self.cos_sum[idx], safe, where=counts > 0),
                "pair_count": counts.copy(),
            }
            for key in ("l1_step_mean", "l1_rate_step_mean", "cos_step_mean"):
                arr = out[block_name][key]
                arr[counts == 0] = np.nan
        return out


def save_unified_evidence_npz(
    path: Path,
    *,
    similarity: dict[str, dict[str, np.ndarray]],
    svd_results: dict[str, dict[str, Any]],
    block_names: list[str],
    metadata: dict[str, Any],
) -> None:
    """Save one-shot evidence artifact (DiT-style unified NPZ)."""

    n_blocks = len(block_names)
    n_intervals = next(iter(similarity.values()))["l1_step_mean"].shape[0]
    n_steps = n_intervals + 1

    l1_diff = np.stack([similarity[b]["l1_step_mean"] for b in block_names], axis=0)
    cos_sim = np.stack([similarity[b]["cos_step_mean"] for b in block_names], axis=0)
    subspace_dist = np.stack(
        [np.asarray(svd_results[b.replace(".", "_")]["subspace_dist"], dtype=np.float64) for b in block_names],
        axis=0,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        l1_diff=l1_diff,
        cos_sim=cos_sim,
        subspace_dist=subspace_dist,
        block_names=np.array(block_names, dtype=object),
        timestep_map=np.arange(n_steps, dtype=np.int32),
        metadata_json=np.array(json.dumps(metadata, indent=2, sort_keys=True)),
    )


def export_legacy_stage0_inputs(
    *,
    output_root: Path,
    num_steps: int,
    similarity: dict[str, dict[str, np.ndarray]],
    svd_results: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    """Write per-block NPZ + SVD JSON for ``stage0_normalization_ldm.py``."""

    t_root = output_root / f"T_{num_steps}"
    npz_dir = t_root / "v2_latest" / "result_npz"
    svd_dir = t_root / "svd_metrics"
    npz_dir.mkdir(parents=True, exist_ok=True)
    svd_dir.mkdir(parents=True, exist_ok=True)

    t_axis = np.arange(num_steps, dtype=np.int32)
    t_curr_interval = (num_steps - 2) - np.arange(num_steps - 1, dtype=np.int32)

    for block_name, sim in similarity.items():
        block_slug = block_name.replace(".", "_")
        l1_step_mean = sim["l1_step_mean"].astype(np.float32)
        cos_step_mean = sim["cos_step_mean"].astype(np.float32)
        cosine = np.eye(num_steps, dtype=np.float32)

        np.savez(
            npz_dir / f"{block_slug}.npz",
            l1_step_mean=l1_step_mean,
            l1_step_std=np.zeros_like(l1_step_mean),
            l1_rate_step_mean=sim["l1_rate_step_mean"].astype(np.float32),
            l1_rate_step_std=np.zeros_like(l1_step_mean),
            cos_step_mean=cos_step_mean,
            cos_step_std=np.zeros_like(cos_step_mean),
            l1rel=cosine,
            l1rel_rate=cosine,
            cosine=cosine,
            step_idx=t_axis,
            t_pointwise=t_axis,
            t_curr_interval=t_curr_interval,
            mapped_t=np.full((num_steps,), -1, dtype=np.int32),
            axis_convention=np.array(
                "interval-wise: analysis interval index j (0..T-2)",
                dtype=object,
            ),
        )

        svd_payload = svd_results[block_slug]
        with open(svd_dir / f"{block_slug}.json", "w", encoding="utf-8") as f:
            json.dump(svd_payload, f, indent=2)

    return npz_dir, svd_dir


def verify_evidence_outputs(
    *,
    npz_dir: Path,
    svd_dir: Path,
    expected_blocks: int,
    expected_steps: int,
) -> dict[str, Any]:
    """Lightweight gate check before Stage 0."""

    npz_files = sorted(npz_dir.glob("*.npz"))
    svd_files = sorted(svd_dir.glob("*.json"))
    if len(npz_files) != expected_blocks:
        raise ValueError(f"Expected {expected_blocks} npz files, got {len(npz_files)}")
    if len(svd_files) != expected_blocks:
        raise ValueError(f"Expected {expected_blocks} svd json files, got {len(svd_files)}")

    issues: list[str] = []
    for npz_path in npz_files:
        data = np.load(npz_path)
        l1 = data["l1_step_mean"]
        cos = data["cos_step_mean"]
        if l1.shape != (expected_steps - 1,):
            issues.append(f"{npz_path.name}: l1 shape {l1.shape}")
        if cos.shape != (expected_steps - 1,):
            issues.append(f"{npz_path.name}: cos shape {cos.shape}")
        if not np.isfinite(l1).all():
            issues.append(f"{npz_path.name}: l1 has NaN/Inf")
        if not np.isfinite(cos).all():
            issues.append(f"{npz_path.name}: cos has NaN/Inf")

    for svd_path in svd_files:
        with open(svd_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        dist = np.asarray(payload["subspace_dist"], dtype=np.float64)
        if dist.shape != (expected_steps,):
            issues.append(f"{svd_path.name}: subspace_dist shape {dist.shape}")

    if issues:
        raise ValueError("Evidence verification failed:\n" + "\n".join(issues))

    return {
        "npz_count": len(npz_files),
        "svd_count": len(svd_files),
        "expected_steps": expected_steps,
        "expected_blocks": expected_blocks,
    }
