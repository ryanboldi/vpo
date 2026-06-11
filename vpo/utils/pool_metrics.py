"""Bridge between raw sub_scores/uids in the training batch and compute_pool_metrics.

Works uniformly for all methods:
  - Single-sol (GRPO, GDPO): sub_scores[i] is a flat list of k floats.
    Each rollout contributes 1 solution. Pool = n rollouts per prompt.
  - Multi-sol (VPO, MULTI_RLVR): sub_scores[i] is a list of lists
    (m x k matrix). Each rollout contributes m solutions.
    Pool = n*m solutions per prompt.

Shape is auto-detected. Output is a flat dict of "pool/{metric}/mean"
keys ready to merge into the trainer's metric dict.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from vpo.utils.eval_metrics import compute_pool_metrics


def _is_multi_sol(entry) -> bool:
    """True if entry is a matrix (list of lists), False if flat vector."""
    if entry is None or not hasattr(entry, "__len__") or len(entry) == 0:
        return False
    first = entry[0]
    return isinstance(first, (list, tuple, np.ndarray))


def build_pool_and_compute(
    sub_scores,
    uids,
    rng_seed: int = 42,
    n_selections: int = 100,
) -> dict[str, float]:
    """Compute pool metrics across all prompt groups, return averaged dict.

    Args:
        sub_scores: Length-bs sequence. Each element is either:
            - A flat list/array of k floats (single-sol)
            - A list of lists / 2D array of shape (m, k) (multi-sol)
        uids: Length-bs sequence of prompt UIDs for grouping.
        rng_seed: Seed for stochastic subset selection reproducibility.
        n_selections: Number of random subset draws (matches lexicase default).

    Returns:
        Dict like {"pool/max_sum/mean": ..., "pool/witness_frac/mean": ...}.
        Empty dict if no valid groups.
    """
    uid_to_indices: dict = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_indices[uid if not isinstance(uid, np.generic) else uid.item()].append(i)

    per_group: list[dict[str, float]] = []

    for uid, indices in uid_to_indices.items():
        pool_vectors: list[list[float]] = []
        rollout_ids: list[int] = []
        positions: list[int] = []

        for rollout_idx in indices:
            entry = sub_scores[rollout_idx]
            if entry is None:
                continue
            # Handle numpy object arrays
            if isinstance(entry, np.ndarray) and entry.dtype == object:
                entry = entry.tolist()
            if hasattr(entry, "__len__") and len(entry) == 0:
                continue

            if _is_multi_sol(entry):
                for pos, row in enumerate(entry):
                    pool_vectors.append(
                        row.tolist() if isinstance(row, np.ndarray) else list(row)
                    )
                    rollout_ids.append(rollout_idx)
                    positions.append(pos)
            else:
                pool_vectors.append(
                    entry.tolist() if isinstance(entry, np.ndarray) else list(entry)
                )
                rollout_ids.append(rollout_idx)
                positions.append(0)

        if not pool_vectors:
            continue

        pm = compute_pool_metrics(
            pool_vectors,
            rollout_ids=rollout_ids,
            positions=positions,
            n_selections=n_selections,
            rng_seed=rng_seed,
        )
        per_group.append(pm)

    # Average across groups, skipping NaN
    result: dict[str, float] = {}
    if per_group:
        keys = list(per_group[0].keys())
        for key in keys:
            vals = [
                pm[key] for pm in per_group
                if not (isinstance(pm[key], float) and math.isnan(pm[key]))
            ]
            if vals:
                result[f"pool/{key}/mean"] = float(np.mean(vals))

    return result
