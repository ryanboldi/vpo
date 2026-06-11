"""Tests for the shared, task-agnostic VPO/PO eval metrics.

These guard the single definition of every general metric (pass@k, the VPO
E_w[best-of-pool] objective, diversity, Pareto fraction, best-/solve-at-k
curves) that all tasks share via ``vpo.utils.eval_metrics`` — so a change to
one of them can never silently disagree with a stale per-task copy again.

Run directly (``python tests/test_eval_metrics.py``) or under pytest.
"""

import math

import numpy as np

from vpo.utils.eval_metrics import (
    best_of_k_curve,
    compute_pool_metrics,
    dirichlet_weights,
    diversity,
    expected_max_weighted,
    mean_weighted,
    pareto_frac,
    pass_at_k,
    solve_at_k_curve,
)


# ─── pass@k (unbiased Codex/HumanEval estimator) ─────────────────────────────


def test_pass_at_k_known_values():
    assert pass_at_k(1, 1, 1) == 1.0
    assert pass_at_k(2, 0, 1) == 0.0          # no solver -> 0
    assert abs(pass_at_k(2, 1, 1) - 0.5) < 1e-12
    assert abs(pass_at_k(4, 1, 2) - 0.5) < 1e-12  # 1 - C(3,2)/C(4,2) = 1 - 3/6
    assert pass_at_k(5, 2, 5) == 1.0          # n-c < k -> certain


def test_pass_at_k_degenerate():
    assert pass_at_k(0, 0, 1) == 0.0
    assert pass_at_k(3, 1, 0) == 0.0
    assert pass_at_k(3, 0, 2) == 0.0


def _legacy_lcb_pass_at_k(n, c, k):
    """The old tasks/livecodebench.py copy (clamps k>n; product form)."""
    if k <= 0 or n <= 0 or c <= 0:
        return 0.0
    if k > n:
        k = n
    if (n - c) < k:
        return 1.0
    num = 1.0
    for i in range(k):
        num *= (n - c - i) / (n - i)
    return 1.0 - num


def test_pass_at_k_matches_legacy_lcb_when_k_le_n():
    # Equivalent to the deleted LCB copy on every (n, c, k) with k <= n — the
    # only regime LCB eval ever uses (k in {1,5,10}, n_total_sols >= k).
    for n in range(1, 13):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                assert abs(pass_at_k(n, c, k) - _legacy_lcb_pass_at_k(n, c, k)) < 1e-12


# ─── E_w[best-of-pool] : the VPO test-time objective ─────────────────────────


def test_expected_max_weighted_known():
    M = [[1.0, 0.0], [0.0, 1.0]]               # two specialist solutions
    W = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]   # three preference vectors
    # w=[1,0]->max(1,0)=1 ; w=[0,1]->max(0,1)=1 ; w=[.5,.5]->max(.5,.5)=.5
    assert abs(expected_max_weighted(M, W) - (1 + 1 + 0.5) / 3) < 1e-12


def test_mean_weighted_known():
    M = [[1.0, 0.0], [0.0, 1.0]]
    W = [[0.5, 0.5]]
    # mean over pool of w·v = mean(0.5, 0.5) = 0.5
    assert abs(mean_weighted(M, W) - 0.5) < 1e-12


def test_expected_max_ge_mean_weighted():
    rng = np.random.default_rng(0)
    M = rng.random((8, 4))
    W = dirichlet_weights(4, 16, seed=1)
    assert expected_max_weighted(M, W) >= mean_weighted(M, W) - 1e-12


def test_weighted_metrics_empty():
    assert expected_max_weighted([], [[1.0]]) == 0.0
    assert mean_weighted([[1.0]], []) == 0.0


def test_weighted_metrics_accept_1d_as_single_solution():
    # A 1-D score vector is a pool of one — must NOT collapse over weights.
    M1 = [1.0, 0.0]
    W = [[1.0, 0.0], [0.0, 1.0]]            # w=[1,0]->1 ; w=[0,1]->0 ; mean=0.5
    assert abs(expected_max_weighted(M1, W) - 0.5) < 1e-12
    assert abs(expected_max_weighted(M1, W) - expected_max_weighted([M1], W)) < 1e-12
    assert abs(mean_weighted(M1, W) - mean_weighted([M1], W)) < 1e-12


def _legacy_musique_expected_max(pool, weights):
    if pool.size == 0:
        return 0.0
    return float((pool @ weights.T).max(axis=0).mean())


def test_expected_max_matches_legacy_musique():
    rng = np.random.default_rng(3)
    M = rng.random((10, 5))
    W = dirichlet_weights(5, 20, seed=7)
    assert abs(expected_max_weighted(M, W) - _legacy_musique_expected_max(M, W)) < 1e-12


# ─── diversity / pareto ──────────────────────────────────────────────────────


def test_diversity_known():
    assert diversity([[0.0, 0.0], [1.0, 1.0]]) == 2.0   # one pair, L1 = 2
    assert diversity([[1.0, 1.0]]) == 0.0               # singleton -> 0


def _legacy_eureqa_pairwise_l1(vecs):
    nv = len(vecs)
    if nv < 2:
        return 0.0
    diffs = vecs[:, None] - vecs[None, :]
    return float(np.abs(diffs).sum(axis=-1).sum() / (nv * (nv - 1)))


def test_diversity_matches_eureqa_legacy():
    rng = np.random.default_rng(11)
    M = rng.random((7, 5))
    assert abs(diversity(M) - _legacy_eureqa_pairwise_l1(M)) < 1e-12


def test_pareto_frac_known():
    # [1,0] and [0,1] are both non-dominated; [0,0] is dominated by both.
    assert abs(pareto_frac([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]) - 2 / 3) < 1e-12
    assert pareto_frac([[1.0, 1.0]]) == 1.0


# ─── stochastic subset selection ─────────────────────────────────────────────


def test_subset_winner_share_max_discriminates():
    # One rollout dominating every subset draw vs. wins spread across
    # specialists must yield different values (the old subset_score mean was
    # identically 1/n_rollouts for both — a vacuous metric).
    dominant = [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    spread = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    m_dom = compute_pool_metrics(dominant, rng_seed=0)
    m_spr = compute_pool_metrics(spread, rng_seed=0)
    assert m_dom["subset_winner_share_max"] == 1.0
    assert m_spr["subset_winner_share_max"] < 1.0
    assert "subset_score" not in m_dom  # the constant metric is gone


# ─── best-of-k / solve-at-k curves ───────────────────────────────────────────


def test_best_of_k_curve_properties():
    scalars = [0.2, 0.9, 0.5, 0.1]
    perm = np.arange(len(scalars))
    curve = best_of_k_curve(scalars, perm)
    assert len(curve) == len(scalars)
    assert curve[0] == scalars[0]                       # best-of-1 = first sampled
    assert curve[-1] == max(scalars)                    # best-of-all = global max
    assert all(curve[i] <= curve[i + 1] for i in range(len(curve) - 1))  # nondecreasing


def test_best_and_solve_share_permutation():
    scalars = [0.2, 1.0, 0.5]
    solved = [0, 1, 0]
    perm = np.random.default_rng(0).permutation(len(scalars))
    best = best_of_k_curve(scalars, perm)
    solve = solve_at_k_curve(solved, perm)
    # The solving solution (index 1, scalar 1.0) becomes a solver at exactly the
    # same k at which it becomes the best — they ride the same permutation.
    first_best = int(np.argmax(best >= 1.0))
    first_solve = int(np.argmax(solve >= 1.0))
    assert first_best == first_solve


# ─── dirichlet weights ───────────────────────────────────────────────────────


def test_dirichlet_weights_shape_and_simplex():
    W = dirichlet_weights(4, 16, seed=0)
    assert W.shape == (16, 4)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert (W >= 0).all()


def test_dirichlet_weights_reproducible():
    assert np.array_equal(dirichlet_weights(3, 5, seed=42), dirichlet_weights(3, 5, seed=42))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} eval-metric tests passed.")
