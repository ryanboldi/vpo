"""Unified evaluation metrics for multi-solution rollouts.

Task-agnostic metrics that work on any score matrix (binary or real-valued).
Designed to be comparable across GRPO (m=1 per rollout, n rollouts) and
par-grpo (m>1 per rollout via chaining).

Input: a **pool** of solutions for a single prompt, each with:
  - score vector  v ∈ R^N  (N objectives)
  - rollout_id    which rollout produced it
  - position      index within the rollout chain (0-based)

Metrics returned:
  Quality (absolute):
    max_sum            sum_j max_i M[i][j]  — union coverage
    best_individual    max_i sum_j M[i][j]  — best single solution
    pass_at_k          1 if any solution fully solves all objectives
    partial_pass_at_k  mean solution quality (avg entry in M)

  Diversity / structure (relative to pool):
    witness_count      how many solutions are witnesses
    witness_frac       witness_count / pool_size
    witness_ndcg       avg per-rollout NDCG (rewards ordering in chains)
    diversity          mean pairwise L1 between score vectors

  Stochastic subset selection (scalable alternative to witness):
    subset_winner_share_max  fraction of random subset draws won by the
                       pool's most-winning rollout (1.0 = one rollout
                       dominates every draw; ~1/n_rollouts = wins spread
                       evenly). NB: the *mean* per-rollout credit is
                       identically 1/n_rollouts (each draw hands out exactly
                       1.0 of credit), so it is not reported.
    subset_score_pos   per-rollout mean position-discounted credit (earlier
                       solutions contribute more)

Cap: N > MAX_OBJECTIVES raises ValueError for witness metrics (2^N subsets).
Stochastic subset metrics have no cap on N.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np

MAX_OBJECTIVES = 12
DEFAULT_N_SELECTIONS = 100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_pool_metrics(
    score_vectors: list[list[float]],
    rollout_ids: list[int] | None = None,
    positions: list[int] | None = None,
    n_selections: int = DEFAULT_N_SELECTIONS,
    rng_seed: int | None = None,
) -> dict[str, float]:
    """Compute all metrics for a pool of solutions.

    Args:
        score_vectors: (pool_size, N) score matrix. Each row is one
            solution's objective scores.
        rollout_ids: Which rollout each solution belongs to. If None,
            every solution is treated as its own rollout (GRPO-style).
        positions: Position within the rollout chain (0-based). If None,
            uses the order within each rollout_id group as given.
        n_selections: Number of random subset draws for stochastic
            metrics. Matches lexicase's n_selections default (100).
        rng_seed: Optional seed for reproducibility of stochastic metrics.

    Returns:
        Dict of metric_name -> float.
    """
    pool_size = len(score_vectors)
    if pool_size == 0:
        return _empty_metrics()

    n_obj = len(score_vectors[0])
    if n_obj == 0:
        return _empty_metrics()

    # Default: each solution is its own rollout
    if rollout_ids is None:
        rollout_ids = list(range(pool_size))
    if positions is None:
        # Assign positions within each rollout by input order
        positions = _assign_positions(rollout_ids)

    # --- Quality metrics ---
    ms = _max_sum(score_vectors)
    bi = _best_individual(score_vectors)
    pak = _pass_at_k(score_vectors)
    ppak = _partial_pass_at_k(score_vectors)

    # --- Unbiased pass@k for k in PASS_AT_K_KS ---
    # Per-solution view: pool of all rows, c = number fully solved.
    c_sol = _count_solved(score_vectors)
    n_sol = pool_size
    pass_at_k_sol = {
        f"pass@{k}_sol": _pass_at_k_unbiased(n_sol, c_sol, k)
        for k in PASS_AT_K_KS
    }
    # Per-chain view: a chain is "correct" if ANY of its solutions fully solved.
    # Lets us report apples-to-apples pass@k where one chain = one inference,
    # regardless of how many solutions the chain emitted (1 for GRPO, m for VPO).
    chain_solved: dict = {}
    for row, rid in zip(score_vectors, rollout_ids):
        if rid not in chain_solved:
            chain_solved[rid] = False
        if all(v >= 1.0 for v in row):
            chain_solved[rid] = True
    n_chain = len(chain_solved)
    c_chain = sum(chain_solved.values())
    pass_at_k_chain = {
        f"pass@{k}_chain": _pass_at_k_unbiased(n_chain, c_chain, k)
        for k in PASS_AT_K_KS
    }

    # --- Witness metrics (only when N is small enough) ---
    if n_obj <= MAX_OBJECTIVES:
        gains = _compute_gains(score_vectors)
        w_count = sum(1 for g in gains if g > 0)
        w_frac = w_count / pool_size
        w_ndcg = _average_ndcg(gains, rollout_ids, positions)
    else:
        w_count = float("nan")
        w_frac = float("nan")
        w_ndcg = float("nan")

    # --- Stochastic subset selection (works for any N) ---
    ss, ss_pos = _stochastic_subset_scores(
        score_vectors, rollout_ids, positions,
        n_selections=n_selections, rng_seed=rng_seed,
    )

    # --- Pareto fraction ---
    pf = _pareto_frac(score_vectors)

    # --- Diversity ---
    div = _diversity(score_vectors)

    return {
        "max_sum": ms,
        "best_individual": bi,
        "pass_at_k": pak,
        "partial_pass_at_k": ppak,
        **pass_at_k_sol,
        **pass_at_k_chain,
        "witness_count": float(w_count),
        "witness_frac": w_frac,
        "witness_ndcg": w_ndcg,
        "subset_winner_share_max": ss,
        "subset_score_pos": ss_pos,
        "pareto_frac": pf,
        "diversity": div,
    }


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def _max_sum(M: list[list[float]]) -> float:
    """sum_j max_i M[i][j] — per-objective best, summed."""
    n_obj = len(M[0])
    total = 0.0
    for j in range(n_obj):
        col_max = M[0][j]
        for i in range(1, len(M)):
            if M[i][j] > col_max:
                col_max = M[i][j]
        total += col_max
    return total


def _best_individual(M: list[list[float]]) -> float:
    """max_i sum_j M[i][j] — best single solution's total."""
    return max(sum(row) for row in M)


def _pass_at_k(M: list[list[float]]) -> float:
    """1.0 if any solution fully solves all objectives (all >= 1.0)."""
    for row in M:
        if all(v >= 1.0 for v in row):
            return 1.0
    return 0.0


# k values to report for unbiased pass@k. Capped at 3 = m (number of solutions
# emitted per inference by multi-sol methods) so the comparison is fair:
# k=3 is one multi-sol inference vs three single-sol inferences.
PASS_AT_K_KS = (1, 2, 3)


def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Codex/HumanEval unbiased pass@k estimator.

    n: total samples drawn, c: number that fully solved, k: pass@k sample size.
    Returns probability that a random k-subset of the n samples contains at
    least one correct solution. NaN-safe: returns 0.0 if c==0, 1.0 if n-c<k.
    """
    if c <= 0 or n <= 0 or k <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    # 1 - prod_{i=n-c+1..n} (1 - k/i)  — pure-Python, avoids numpy dep here
    p = 1.0
    for i in range(n - c + 1, n + 1):
        p *= 1.0 - k / i
    return float(1.0 - p)


def _count_solved(M: list[list[float]]) -> int:
    """Number of rows in M that fully solve (all entries >= 1.0)."""
    return sum(1 for row in M if all(v >= 1.0 for v in row))


def _partial_pass_at_k(M: list[list[float]]) -> float:
    """Mean per-solution score, averaged across solutions.

    = (1/pool_size) * sum_i (1/N) * sum_j M[i][j]
    = mean entry of M.
    """
    total = 0.0
    count = 0
    for row in M:
        for v in row:
            total += v
            count += 1
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Witness computation (Steps 1-3 from the spec)
# ---------------------------------------------------------------------------


def _compute_gains(score_vectors: list[list[float]]) -> list[float]:
    """Compute per-solution gain via witness/dedup.

    Returns a list of floats the same length as score_vectors.
    """
    pool_size = len(score_vectors)
    n_obj = len(score_vectors[0])

    # Step 1: Dedupe into equivalence classes by score vector.
    # Use tuple of scores as key.
    class_map: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for idx, sv in enumerate(score_vectors):
        key = tuple(sv)
        class_map[key].append(idx)

    classes = list(class_map.values())  # each is a list of solution indices
    class_vectors = [score_vectors[members[0]] for members in classes]
    n_classes = len(classes)

    # Step 2: For each non-empty subset of objectives, find unique winner.
    is_witness = [False] * n_classes

    # Precompute subset sums for each class: subset_sums[c][subset_mask]
    # For efficiency, iterate subsets and compute on the fly.
    n_subsets = (1 << n_obj) - 1  # 2^N - 1

    for mask in range(1, n_subsets + 1):
        # Compute sum over objectives in mask for each class
        best_val = -math.inf
        second_val = -math.inf
        best_class = -1

        for c_idx in range(n_classes):
            cv = class_vectors[c_idx]
            s = 0.0
            m = mask
            j = 0
            while m:
                if m & 1:
                    s += cv[j]
                m >>= 1
                j += 1

            if s > best_val:
                second_val = best_val
                best_val = s
                best_class = c_idx
            elif s > second_val:
                second_val = s

        # Unique winner: strictly better than second best
        if best_val > second_val:
            is_witness[best_class] = True

    # Step 3: Assign gains. Witness credit (1) split among class members.
    gains = [0.0] * pool_size
    for c_idx, members in enumerate(classes):
        if is_witness[c_idx]:
            g = 1.0 / len(members)
            for idx in members:
                gains[idx] = g

    return gains


# ---------------------------------------------------------------------------
# NDCG (Step 4)
# ---------------------------------------------------------------------------


def _average_ndcg(
    gains: list[float],
    rollout_ids: list[int],
    positions: list[int],
) -> float:
    """Average per-rollout NDCG.

    For each rollout, order solutions by position, compute DCG from gains,
    normalize by the ideal DCG (M slots all with gain=1).
    """
    # Group solutions by rollout
    rollouts: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for idx, (rid, pos) in enumerate(zip(rollout_ids, positions)):
        rollouts[rid].append((pos, gains[idx]))

    ndcg_sum = 0.0
    n_rollouts = len(rollouts)

    for rid, items in rollouts.items():
        # Sort by position within chain
        items.sort(key=lambda x: x[0])
        m = len(items)

        # DCG
        dcg = 0.0
        for rank, (pos, g) in enumerate(items):
            dcg += g / math.log2(rank + 2)  # rank 0 -> log2(2) = 1

        # IDCG: all M slots filled with gain=1
        idcg = sum(1.0 / math.log2(i + 2) for i in range(m))

        if idcg > 0:
            ndcg_sum += dcg / idcg
        # If idcg == 0 (m=0), skip — shouldn't happen

    return ndcg_sum / n_rollouts if n_rollouts > 0 else 0.0


# ---------------------------------------------------------------------------
# Stochastic subset selection
# ---------------------------------------------------------------------------


def _stochastic_subset_scores(
    score_vectors: list[list[float]],
    rollout_ids: list[int],
    positions: list[int],
    n_selections: int,
    rng_seed: int | None = None,
) -> tuple[float, float]:
    """Per-rollout credit via random subset selection, averaged.

    For each draw:
      1. Sample a random non-empty subset T of objectives.
      2. For each solution, compute sum of scores on T.
      3. Find the global max score. Identify all solutions achieving it.
      4. Credit the rollouts containing those solutions, split evenly
         among tied rollouts.
      5. For position-discounted version: weight each winning solution's
         credit by 1/log2(position + 2) before aggregating to rollouts.

    Returns:
        (subset_winner_share_max, subset_score_pos):
            subset_winner_share_max: fraction of draws won by the rollout
                with the most wins. (The unweighted *mean* per-rollout
                credit is identically 1/n_rollouts — every draw distributes
                exactly 1.0 of credit — so it carries no information and is
                not returned.)
            subset_score_pos: mean per-rollout position-discounted credit.
    """
    pool_size = len(score_vectors)
    n_obj = len(score_vectors[0])
    rng = random.Random(rng_seed)

    # Build rollout structure: rollout_id -> list of (pool_idx, position)
    rollout_map: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for idx in range(pool_size):
        rollout_map[rollout_ids[idx]].append((idx, positions[idx]))
    unique_rollouts = sorted(rollout_map.keys())
    n_rollouts = len(unique_rollouts)

    # Accumulators: per-rollout total credit
    credit = defaultdict(float)       # unweighted
    credit_pos = defaultdict(float)   # position-discounted

    for _ in range(n_selections):
        # Sample a random non-empty subset of objectives
        mask = rng.randint(1, (1 << n_obj) - 1)

        # Extract which objective indices are in the mask
        obj_indices = []
        m = mask
        j = 0
        while m:
            if m & 1:
                obj_indices.append(j)
            m >>= 1
            j += 1

        # Score each solution on this subset
        scores = [0.0] * pool_size
        for i in range(pool_size):
            s = 0.0
            for oj in obj_indices:
                s += score_vectors[i][oj]
            scores[i] = s

        # Find global max
        best = max(scores)

        # Find all solutions achieving the max
        winners = [i for i in range(pool_size) if scores[i] >= best - 1e-12]

        # Which rollouts contain a winner?
        winning_rollouts: set[int] = set()
        for w in winners:
            winning_rollouts.add(rollout_ids[w])
        n_tied = len(winning_rollouts)
        rollout_credit = 1.0 / n_tied

        # Unweighted: each winning rollout gets equal credit
        for rid in winning_rollouts:
            credit[rid] += rollout_credit

        # Position-discounted: each winning rollout gets credit scaled
        # by the position of its earliest winner. A rollout whose winner
        # is at position 0 gets full credit (1/log2(2) = 1.0); position 1
        # gets ~0.63; position 2 gets 0.5, etc. Credit is still split
        # among tied rollouts, but the absolute value is lower when
        # winners appear later in the chain.
        for rid in winning_rollouts:
            # Earliest winning position in this rollout
            best_pos = min(
                positions[w] for w in winners if rollout_ids[w] == rid
            )
            pos_discount = 1.0 / math.log2(best_pos + 2)
            credit_pos[rid] += pos_discount / n_tied

    # Max win share across rollouts (the mean is identically 1/n_rollouts).
    winner_share_max = (
        max(credit[rid] for rid in unique_rollouts) / n_selections
        if n_selections > 0
        else 0.0
    )
    subset_score_pos = sum(credit_pos[rid] for rid in unique_rollouts) / (
        n_rollouts * max(n_selections, 1)
    )

    return winner_share_max, subset_score_pos


# ---------------------------------------------------------------------------
# Pareto fraction
# ---------------------------------------------------------------------------


def _pareto_count(score_vectors: list[list[float]], eps: float = 1e-12) -> int:
    """Number of Pareto-optimal (non-dominated) solutions.

    Solution j dominates solution i if j >= i on all objectives (within eps)
    and j > i on at least one (by more than eps).
    """
    n = len(score_vectors)
    if n == 0:
        return 0
    if n == 1:
        return 1
    n_obj = len(score_vectors[0])
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            # Check if j dominates i
            all_geq = True
            any_gt = False
            for k in range(n_obj):
                if score_vectors[j][k] < score_vectors[i][k] - eps:
                    all_geq = False
                    break
                if score_vectors[j][k] > score_vectors[i][k] + eps:
                    any_gt = True
            if all_geq and any_gt:
                dominated[i] = True
                break
    return sum(1 for d in dominated if not d)


def _pareto_frac(score_vectors: list[list[float]], eps: float = 1e-12) -> float:
    """Fraction of solutions that are Pareto-optimal (see ``_pareto_count``)."""
    n = len(score_vectors)
    if n <= 1:
        return 1.0
    return _pareto_count(score_vectors, eps=eps) / n


# ---------------------------------------------------------------------------
# Diversity / spread
# ---------------------------------------------------------------------------


def _feature_spread(M: list[list[float]]) -> float:
    """Mean per-objective population std across solutions."""
    n = len(M)
    if n < 2:
        return 0.0
    k = len(M[0])
    total_std = 0.0
    for j in range(k):
        col = [M[i][j] for i in range(n)]
        mean = sum(col) / n
        var = sum((v - mean) ** 2 for v in col) / n
        total_std += var ** 0.5
    return total_std / k if k else 0.0


def _diversity(M: list[list[float]]) -> float:
    """Mean pairwise L1 distance between score vectors."""
    n = len(M)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sum(abs(M[i][k] - M[j][k]) for k in range(len(M[i])))
            pairs += 1
    return total / pairs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assign_positions(rollout_ids: list[int]) -> list[int]:
    """Assign 0-based positions within each rollout by input order."""
    counters: dict[int, int] = defaultdict(int)
    positions = []
    for rid in rollout_ids:
        positions.append(counters[rid])
        counters[rid] += 1
    return positions


# ---------------------------------------------------------------------------
# General VPO/PO eval quantities (the numbers the paper reports across tasks).
#
# These live HERE, not in any tasks/<task>.py, so every task computes them from
# ONE definition — no per-task reimplementation that can silently drift. A task
# file should keep only task-specific scoring (turning a generation into a score
# vector / answer); all task-agnostic aggregation goes through this module.
# Covered by tests/test_eval_metrics.py.
# ---------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Codex/HumanEval pass@k — public alias of the estimator above.

    n samples drawn, c fully solved; probability a random k-subset contains a
    solver. Does NOT clamp k>n (returns 1.0 when n-c<k, which is the correct
    value). Callers wanting "pass@min(k,n)" should clamp k themselves.
    """
    return _pass_at_k_unbiased(n, c, k)


def diversity(score_matrix) -> float:
    """Mean pairwise L1 distance between score vectors (public alias)."""
    return _diversity([list(v) for v in score_matrix])


def pareto_frac(score_matrix, eps: float = 1e-12) -> float:
    """Fraction of Pareto-optimal (non-dominated) solutions (public alias)."""
    return _pareto_frac([list(v) for v in score_matrix], eps=eps)


def pareto_count(score_matrix, eps: float = 1e-12) -> int:
    """Number of Pareto-optimal (non-dominated) solutions (public alias)."""
    return _pareto_count([list(v) for v in score_matrix], eps=eps)


def feature_spread(score_matrix) -> float:
    """Mean per-objective population std across solutions (public alias)."""
    return _feature_spread([list(v) for v in score_matrix])


def dirichlet_weights(n_obj: int, n: int, seed: int | None = None) -> np.ndarray:
    """Sample ``n`` weight vectors ~ Dir(1,...,1) over ``n_obj`` objectives."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(n_obj), size=n)


def expected_max_weighted(score_matrix, weights) -> float:
    """E_w[ max_{v in pool} (w · v) ] — the VPO test-time objective.

    score_matrix: (pool_size, N). weights: (n_weights, N). Returns the mean over
    weight draws of the best weighted score achievable from the pool. Empty pool
    or no weights -> 0.0.
    """
    M = np.asarray(score_matrix, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float64)
    if M.size == 0 or W.size == 0:
        return 0.0
    if M.ndim == 1:  # a single solution vector -> a pool of one
        M = M[None, :]
    return float((M @ W.T).max(axis=0).mean())


def mean_weighted(score_matrix, weights) -> float:
    """E_w[ mean_{v in pool} (w · v) ] — pool-average weighted score."""
    M = np.asarray(score_matrix, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float64)
    if M.size == 0 or W.size == 0:
        return 0.0
    if M.ndim == 1:
        M = M[None, :]
    return float((M @ W.T).mean())


def best_of_k_curve(scalars, perm) -> np.ndarray:
    """Best-of-k curve: ``curve[k-1]`` = max of the first k scalars under ``perm``.

    ``scalars`` is a 1-D per-solution scalar (e.g. sum or mean over objectives —
    that choice stays with the task). ``perm`` is a permutation of
    ``range(len(scalars))``; pass the SAME perm to ``solve_at_k_curve`` so both
    curves describe one random ordering of the pool.
    """
    s = np.asarray(scalars, dtype=np.float64)[perm]
    return np.maximum.accumulate(s)


def solve_at_k_curve(solved, perm) -> np.ndarray:
    """Full-solve-at-k curve over ``perm`` (``solved`` is a 0/1 or bool 1-D array)."""
    f = np.asarray(solved, dtype=np.float64)[perm]
    return np.maximum.accumulate(f)


def _empty_metrics() -> dict[str, float]:
    return {
        "max_sum": 0.0,
        "best_individual": 0.0,
        "pass_at_k": 0.0,
        "partial_pass_at_k": 0.0,
        **{f"pass@{k}_sol": 0.0 for k in PASS_AT_K_KS},
        **{f"pass@{k}_chain": 0.0 for k in PASS_AT_K_KS},
        "witness_count": 0.0,
        "witness_frac": 0.0,
        "witness_ndcg": 0.0,
        "subset_winner_share_max": 0.0,
        "subset_score_pos": 0.0,
        "pareto_frac": 0.0,
        "diversity": 0.0,
    }
