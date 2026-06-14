"""Unified one-pass evidence collector for LDM UNet blocks."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential
from tqdm.auto import tqdm

from evidence.utils import SimilarityAccumulator, compute_ldm_similarity_batch

LOGGER = logging.getLogger("LDM_Evidence")


def _is_supported_unet_block_name(name: str) -> bool:
    return name.startswith("input_blocks.") or name == "middle_block" or name.startswith("output_blocks.")


class UnifiedEvidenceCollector:
    """Collect L1/cosine similarity and SVD drift for all UNet blocks in one DDIM run."""

    def __init__(
        self,
        *,
        max_timesteps: int,
        target_n: int,
        block_names: List[str],
        representative_t: int = -1,
        energy_threshold: float = 0.98,
    ) -> None:
        self.max_timesteps = int(max_timesteps)
        self.target_n = int(target_n)
        self.block_names = list(block_names)
        self.representative_t = int(representative_t)
        self.energy_threshold = float(energy_threshold)

        self.hooks: List[Any] = []
        self._step_counter = -1
        self._prev_outputs: Dict[str, Optional[torch.Tensor]] = {b: None for b in self.block_names}

        self.sim = SimilarityAccumulator(
            n_blocks=len(self.block_names),
            n_intervals=max(0, self.max_timesteps - 1),
            block_names=self.block_names,
        )

        self.svd_states: Dict[str, Dict[str, Any]] = {}
        for block_name in self.block_names:
            self.svd_states[block_name] = {
                "C": None,
                "H": None,
                "W": None,
                "cov_sums": [None] * self.max_timesteps,
                "sample_counts": np.zeros((self.max_timesteps,), dtype=np.int32),
                "token_counts": np.zeros((self.max_timesteps,), dtype=np.int64),
            }

    def _create_step_pre_hook(self):
        def pre_hook(module, args, kwargs):
            del module, args, kwargs
            self._step_counter = (self._step_counter + 1) % self.max_timesteps

        return pre_hook

    def _create_block_hook(self, block_name: str):
        def hook_fn(module, inputs, output):
            del module, inputs
            step_idx = int(self._step_counter)
            if step_idx < 0 or step_idx >= self.max_timesteps:
                return
            if not torch.is_tensor(output) or output.ndim != 4:
                return

            out = output.detach()
            block_idx = self.sim.block_index(block_name)

            prev = self._prev_outputs[block_name]
            if prev is not None and step_idx >= 1:
                l1_vals, l1_rate_vals, cos_vals = compute_ldm_similarity_batch(prev, out)
                self.sim.add_interval(block_idx, step_idx - 1, l1_vals, l1_rate_vals, cos_vals)

            st = self.svd_states[block_name]
            remain = self.target_n - int(st["sample_counts"][step_idx])
            if remain > 0:
                chunk = out.to(dtype=torch.float32)
                if chunk.shape[0] > remain:
                    chunk = chunk[:remain]
                if chunk.shape[0] > 0:
                    n, c, h, w = chunk.shape
                    if st["C"] is None:
                        st["C"], st["H"], st["W"] = int(c), int(h), int(w)
                    flat = chunk.permute(1, 0, 2, 3).reshape(c, n * h * w)
                    cov_chunk = flat @ flat.T
                    if st["cov_sums"][step_idx] is None:
                        st["cov_sums"][step_idx] = cov_chunk
                    else:
                        st["cov_sums"][step_idx].add_(cov_chunk)
                    st["sample_counts"][step_idx] += int(n)
                    st["token_counts"][step_idx] += int(n * h * w)

            self._prev_outputs[block_name] = out.clone()

        return hook_fn

    def register_hooks(self, unet: nn.Module) -> None:
        registered: List[str] = []
        for name, module in unet.named_modules():
            if not isinstance(module, TimestepEmbedSequential):
                continue
            if not _is_supported_unet_block_name(name):
                continue

            canonical = f"model.{name}"
            if canonical not in self.block_names:
                continue

            self.hooks.append(module.register_forward_hook(self._create_block_hook(canonical)))
            registered.append(canonical)
            LOGGER.info("Register block hook: %s", canonical)

        if len(registered) != len(self.block_names):
            missing = sorted(set(self.block_names) - set(registered))
            raise ValueError(f"Failed to register all blocks. Missing: {missing}")

        self.hooks.append(unet.register_forward_pre_hook(self._create_step_pre_hook(), with_kwargs=True))
        LOGGER.info("Registered %d block hooks", len(registered))

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def reset_batch_state(self) -> None:
        self._prev_outputs = {b: None for b in self.block_names}

    def min_collected(self) -> int:
        mins = [int(st["sample_counts"].min()) for st in self.svd_states.values()]
        return int(min(mins)) if mins else 0

    def has_enough_svd(self) -> bool:
        return self.min_collected() >= self.target_n

    def finalize_similarity(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self.sim.finalize()

    def finalize_svd(self) -> Dict[str, Dict[str, Any]]:
        from b_SVD.svd_metrics_ldm import compute_energy_ratios, compute_rank_r, compute_subspace_distance

        results: Dict[str, Dict[str, Any]] = {}
        for block_name in tqdm(self.block_names, desc="Finalize SVD"):
            st = self.svd_states[block_name]
            c, h, w = st["C"], st["H"], st["W"]
            if c is None:
                raise RuntimeError(f"No SVD features collected for {block_name}")

            eigenvalues_list: List[torch.Tensor] = []
            eigenvectors_list: List[torch.Tensor] = []
            for t in range(self.max_timesteps):
                cov = st["cov_sums"][t]
                tokens = int(st["token_counts"][t])
                if cov is None or tokens <= 0:
                    raise RuntimeError(f"Missing covariance for block={block_name} t={t}")
                sigma = (cov / float(tokens)).double()
                eigvals, eigvecs = torch.linalg.eigh(sigma)
                eigvals = torch.flip(eigvals, dims=[0])
                eigvecs = torch.flip(eigvecs, dims=[1])
                eigenvalues_list.append(eigvals)
                eigenvectors_list.append(eigvecs)
                st["cov_sums"][t] = None

            rep_t = self.representative_t
            if rep_t < 0 or rep_t >= self.max_timesteps:
                rep_t = self.max_timesteps - 1

            eigenvalues_ref = eigenvalues_list[rep_t]
            rank_r = compute_rank_r(eigenvalues_ref, self.energy_threshold)
            cumulative_ref = torch.cumsum(eigenvalues_ref, dim=0) / eigenvalues_ref.sum()
            actual_energy = float(cumulative_ref[rank_r - 1].item())

            subspace_dist: List[float] = [0.0]
            for t in range(1, self.max_timesteps):
                dist = compute_subspace_distance(eigenvectors_list[t], eigenvectors_list[t - 1], rank_r)
                subspace_dist.append(float(dist))

            block_slug = block_name.replace(".", "_")
            results[block_slug] = {
                "block": block_slug,
                "target_block_name": block_name,
                "T": int(self.max_timesteps),
                "C": int(c),
                "N": int(st["sample_counts"].min()),
                "H": int(h),
                "W": int(w),
                "rank_r": int(rank_r),
                "representative_t": int(rep_t),
                "energy_threshold": float(self.energy_threshold),
                "actual_energy_at_r": float(actual_energy),
                "timesteps": list(range(self.max_timesteps)),
                "subspace_dist": subspace_dist,
                "energy_ratio": compute_energy_ratios(eigenvalues_list),
            }

        return results
