"""VPO — Vector Policy Optimization.

Advantage estimator for vector-valued (multi-objective) RL. Rather than
collapsing the reward vector under one fixed weighting, VPO samples random
*scalarizations* w from Dirichlet(alpha,...,alpha) — uniform over the simplex
when alpha=1 — scores each solution by its scalarized reward w·r, and uses the
best-of-m under each scalarization to shape advantages. The deployment goal is
still a fixed objective; the stochastic scalarization is simply what drives the
policy to keep a reward-diverse set of solutions (it is NOT goal-conditioning on
a user-supplied preference).

``vpo_advantage`` (the only variant): own-pool objective. Each rollout's
raw score is ``E_w[max_{s in r} w·r_s]`` over its own m solutions;
advantages are GRPO-style z-norm of the raw scores within the prompt
group. Directly optimizes the test-time ``E_w[best-of-m]`` metric.

Sampler selection:

  - ``naive`` (default): iid Dirichlet(alpha,...,alpha). Classic MC.
  - ``sobol`` / ``sobol_qmc`` / ``qmc``: Quasi-MC via Sobol in the
    (k-1)-cube mapped to the simplex with the sorted-uniforms trick.
    Empirically ~2-6x lower advantage noise per sample.

Select at runtime by setting the ``VPO_SAMPLER`` env var, or by passing
``sampler=`` to ``vpo_advantage``. Concentration via ``VPO_ALPHA``.
"""

import os
import warnings
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import gamma as _gamma_dist
from scipy.stats import qmc


_ALPHA_ENV_VAR = "VPO_ALPHA"
_SEED_ENV_VAR = "VPO_SEED"

_GLOBAL_RNG: np.random.Generator | None = None


def _get_global_rng() -> np.random.Generator:
    """Process-persistent RNG seeded from ``VPO_SEED`` (default 0).

    Reused across ``vpo_advantage`` calls so the weight-sampling stream is
    deterministic for a given run — re-running with the same seed reproduces
    the same advantages — while still advancing each step. Pass ``rng=`` to
    ``vpo_advantage`` to override (e.g. in tests).
    """
    global _GLOBAL_RNG
    if _GLOBAL_RNG is None:
        raw = os.environ.get(_SEED_ENV_VAR, "0")
        try:
            seed = int(raw) if str(raw).strip() else 0
        except ValueError:
            seed = 0
        _GLOBAL_RNG = np.random.default_rng(seed)
    return _GLOBAL_RNG


def _resolve_alpha(k: int) -> np.ndarray:
    """Resolve concentration parameter for Dir(alpha,...,alpha).

    Reads ``VPO_ALPHA`` env var. Default is 1.0 (uniform on
    the simplex). Lower values (e.g. 0.3) sharpen the distribution
    toward the corners — most samples favor a single objective heavily;
    useful for ablations on Pareto coverage.
    """
    raw = os.environ.get(_ALPHA_ENV_VAR)
    if raw is None or not raw.strip():
        return np.ones(k)
    try:
        alpha = float(raw)
    except ValueError:
        raise ValueError(
            f"{_ALPHA_ENV_VAR}={raw!r} is not a float"
        )
    if alpha <= 0:
        raise ValueError(f"{_ALPHA_ENV_VAR}={alpha} must be > 0")
    return np.full(k, alpha)


def _sample_naive_dirichlet(
    n_samples: int, k: int, rng: np.random.Generator
) -> np.ndarray:
    """iid Dirichlet(alpha,...,alpha); shape (n_samples, k).

    alpha defaults to 1.0; override via VPO_ALPHA env var.
    """
    return rng.dirichlet(_resolve_alpha(k), size=n_samples)


def _sample_sobol_qmc(
    n_samples: int, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Quasi-MC Dirichlet(alpha,...,alpha) on the (k-1)-simplex via Sobol.

    alpha=1 (the VPO_ALPHA default): sorted-uniforms trick in the (k-1)-cube —
    exactly uniform on the simplex. alpha!=1: per-coordinate Gamma(alpha)
    inverse-CDF of a k-dim Sobol stream, normalized to the simplex (the
    standard Gamma-ratio construction), so VPO_ALPHA concentrates sobol draws
    the same way it concentrates naive draws.

    Each row: marginally Dirichlet(alpha,...,alpha), jointly low-discrepancy.
    Variance reduction relative to naive MC grows super-linearly in N; see
    ``notebooks/dirichlet_sampling.py``.
    """
    if k == 1:
        return np.ones((n_samples, 1))
    alpha = _resolve_alpha(k)
    seed = int(rng.integers(0, 2**31 - 1))
    uniform_alpha = bool(np.all(alpha == 1.0))
    sampler = qmc.Sobol(d=k - 1 if uniform_alpha else k, scramble=True, seed=seed)
    # scipy emits a UserWarning when n is not a power of 2 (balance property
    # is weaker). That's a training-signal tradeoff the user already accepted
    # by selecting sobol; silence the warning to avoid log spam at each step.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        u = sampler.random(n_samples)
    if uniform_alpha:
        u = np.sort(u, axis=1)
        padded = np.concatenate(
            [np.zeros((n_samples, 1)), u, np.ones((n_samples, 1))], axis=1
        )
        return np.diff(padded, axis=1)
    # Keep ppf finite and rows strictly positive-sum.
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    g = _gamma_dist.ppf(u, a=alpha)
    total = g.sum(axis=1, keepdims=True)
    # Tiny alpha can underflow every coordinate to 0; fall back to uniform.
    degenerate = (total <= 0.0).ravel()
    if degenerate.any():
        g[degenerate] = 1.0
        total = g.sum(axis=1, keepdims=True)
    return g / total


_SAMPLERS = {
    "naive": _sample_naive_dirichlet,
    "sobol": _sample_sobol_qmc,
    "sobol_qmc": _sample_sobol_qmc,
    "qmc": _sample_sobol_qmc,
}
_SAMPLER_ENV_VAR = "VPO_SAMPLER"
_DEFAULT_SAMPLER = "naive"


def _resolve_sampler(name: str | None = None) -> tuple[str, callable]:
    """Return (canonical_name, sampler_fn). Precedence: arg > env > default."""
    if name is None:
        name = os.environ.get(_SAMPLER_ENV_VAR, _DEFAULT_SAMPLER)
    key = name.strip().lower()
    if key not in _SAMPLERS:
        raise ValueError(
            f"Unknown VPO sampler {name!r}. "
            f"Valid options: {sorted(set(_SAMPLERS))}."
        )
    return key, _SAMPLERS[key]


def vpo_advantage(
    sub_scores,
    uids: np.ndarray,
    n_selections: int = 100,
    eps: float = 1e-6,
    sampler: str | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """VPO advantage: own-pool ``E_w[max_{s in r} w · r_s]`` per rollout.

    Per prompt group:
      1. Draw n_selections random w ~ Dirichlet(alpha, ..., alpha).
      2. For each rollout r, raw score = (1/n_sel) * sum_w max_{s in r} (w · r_s).
      3. Z-normalize raw scores across the n rollouts in the group (GRPO-style).

    Directly optimizes the test metric ``E_w[max_{s} w · r_s]`` evaluated
    on the rollout's own m-pool.

    Args:
        sub_scores: Array-like of length bs. Each element is a list/array of
            shape (m_i, k_g) — score matrix for that rollout's solutions.
        uids: (bs,) array of prompt UIDs for grouping.
        n_selections: Number of Dirichlet samples per group.
        eps: Numerical-stability epsilon for the z-norm.
        sampler: Simplex sampler name. ``None`` uses ``VPO_SAMPLER`` env var
            (default ``"naive"``).
            Options: ``"naive"``, ``"sobol"``/``"sobol_qmc"``/``"qmc"``.
        rng: Optional numpy Generator for the weight draws. ``None`` uses the
            process-persistent RNG seeded from ``VPO_SEED`` (reproducible).

    Returns:
        advantages: (bs,) tensor of group-normalized advantages.
        diagnostics: dict of batch-level scalar metrics for wandb.
    """
    bs = len(sub_scores)
    sampler_name, sample_fn = _resolve_sampler(sampler)

    uid_to_rollout_indices: dict = defaultdict(list)
    for i in range(bs):
        uid_to_rollout_indices[uids[i]].append(i)

    advantages = torch.zeros(bs)

    all_pool_sizes: list[int] = []
    all_group_stds: list[float] = []
    all_own_pool_expected_max: list[float] = []
    all_pool_expected_best: list[float] = []
    rng = rng if rng is not None else _get_global_rng()

    for uid, rollout_indices in uid_to_rollout_indices.items():
        # Collect rollout-local score matrices
        per_rollout_vecs: list[list[list[float]]] = []
        rollout_has_sols: list[bool] = []
        k_g = 0
        pool_size = 0
        for rollout_idx in rollout_indices:
            matrix = sub_scores[rollout_idx]
            if matrix is None or len(matrix) == 0:
                per_rollout_vecs.append([])
                rollout_has_sols.append(False)
                continue
            vecs = [list(row) if not isinstance(row, list) else row for row in matrix]
            per_rollout_vecs.append(vecs)
            rollout_has_sols.append(True)
            if k_g == 0:
                k_g = len(vecs[0])
            pool_size += len(vecs)

        all_pool_sizes.append(pool_size)

        if k_g == 0 or pool_size == 0:
            all_group_stds.append(0.0)
            all_own_pool_expected_max.append(0.0)
            all_pool_expected_best.append(0.0)
            continue

        # Sample weights once, shared across rollouts for low-variance
        # within-group comparison (reduces z-norm noise).
        w_mat = sample_fn(n_selections, k_g, rng)  # (n_sel, k_g)

        raw_scores = torch.zeros(len(rollout_indices))
        pool_per_w_max = np.full(n_selections, -np.inf)

        for j, vecs in enumerate(per_rollout_vecs):
            if not vecs:
                continue
            # (m_r, k_g) @ (k_g, n_sel) -> (m_r, n_sel)
            R = np.asarray(vecs, dtype=np.float64)
            scored = R @ w_mat.T  # (m_r, n_sel)
            own_max = scored.max(axis=0)  # (n_sel,)
            raw_scores[j] = float(own_max.mean())
            pool_per_w_max = np.maximum(pool_per_w_max, own_max)

        mean = raw_scores.mean()
        # Population std (correction=0), matching GRPO's z-norm convention and
        # avoiding NaN when a group collapses to a single effective rollout.
        std = raw_scores.std(unbiased=False)
        if std < eps:
            normalized = torch.zeros_like(raw_scores)
        else:
            normalized = (raw_scores - mean) / (std + eps)

        for j, rollout_idx in enumerate(rollout_indices):
            advantages[rollout_idx] = normalized[j]

        all_group_stds.append(float(std))
        all_own_pool_expected_max.append(float(raw_scores.mean().item()))
        # Mean over w of pool-wide max (comparable across rollouts/methods).
        pool_eb = float(pool_per_w_max[np.isfinite(pool_per_w_max)].mean()) \
            if np.isfinite(pool_per_w_max).any() else 0.0
        all_pool_expected_best.append(pool_eb)

    n_groups = len(all_pool_sizes) if all_pool_sizes else 1
    # Numeric sampler code so wandb can plot it. 0=naive, 1=sobol (any alias).
    sampler_code = 0 if sampler_name == "naive" else 1
    diagnostics = {
        "vpo/pool_size_mean": sum(all_pool_sizes) / n_groups,
        "vpo/group_std_mean": sum(all_group_stds) / n_groups,
        "vpo/own_pool_expected_max_mean": (
            sum(all_own_pool_expected_max) / n_groups
            if all_own_pool_expected_max else 0.0
        ),
        "vpo/pool_expected_best_mean": (
            sum(all_pool_expected_best) / n_groups
            if all_pool_expected_best else 0.0
        ),
        "vpo/sampler_code": sampler_code,
    }

    return advantages, diagnostics
