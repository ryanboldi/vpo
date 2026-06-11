"""Scalar reward variants for multi-solution rollouts.

A multi-solution rollout produces a (m', k) matrix of pass/fail values
where m' is the number of dedup'd solutions and k is the number of
sub-objectives (test cases / countdown sub-rewards / LCB tests).

Different scalarization strategies turn that matrix into a single number
used as the GRPO advantage signal:

  sum_all         multi-RLVR baseline. Sum every entry. Rewards both
                  per-solution correctness AND solution count.
                      score = sum_i sum_j matrix[i][j]

  max_sum_naive   "team coverage". For each test case, take the max over
                  the m solutions (= "did at least one solution pass it"),
                  then sum across test cases. Rewards UNION coverage but
                  not redundancy.
                      score = sum_j max_i matrix[i][j]

  set_coverage    "downward-closed coverage". A solution with bitmask b
                  covers every subset of b. Score = number of distinct
                  nonempty subsets in the union of these downward-closures
                  across all m solutions. Rewards diverse partial successes
                  without rewarding duplicates.
                      score = |{ s ≠ ∅ : s ⊆ b_i for some i }|

The reward variant is selected via the PAR_GRPO_SCORE_VARIANT env var
(set per-method by train_eval_one.sh). Default = "sum_all" so existing
multi_rlvr behavior is unchanged.
"""

import os

# Env var consulted by reward functions to pick which scalar becomes "score".
ENV_VAR = "PAR_GRPO_SCORE_VARIANT"
DEFAULT_VARIANT = "sum_all"
KNOWN_VARIANTS = ("sum_all", "max_sum_naive", "set_coverage")


def get_active_variant() -> str:
    v = os.environ.get(ENV_VAR, DEFAULT_VARIANT)
    if v not in KNOWN_VARIANTS:
        raise ValueError(
            f"{ENV_VAR}={v!r} is not a known variant. "
            f"Options: {KNOWN_VARIANTS}"
        )
    return v


# ─── Scalar formulas ────────────────────────────────────────────────────────


def sum_all(matrix) -> float:
    """Sum every entry. Equivalent to multi-RLVR's existing scalar."""
    return float(sum(sum(row) for row in matrix))


def max_sum_naive(matrix) -> float:
    """Per-test-case max across solutions, summed across test cases.

    score = sum_j max_i matrix[i][j]

    With binary entries, this is the number of test cases that AT LEAST ONE
    solution passes (the rollout's "team coverage"). No reward for two
    solutions that pass the same test.
    """
    if not matrix:
        return 0.0
    k = len(matrix[0])
    if k == 0:
        return 0.0
    total = 0.0
    for j in range(k):
        col_max = 0.0
        for row in matrix:
            v = row[j]
            if v > col_max:
                col_max = v
        total += col_max
    return float(total)


def set_coverage(matrix) -> float:
    """Number of distinct nonempty subsets covered by the union of solutions.

    A solution with bitmask b covers every subset of b. The score is the
    cardinality of the union of these downward-closures over all solutions
    in the rollout, excluding the empty set.

    Examples (k=4):
      [[1,1,0,0]]              -> 3   ({0}, {1}, {0,1})
      [[1,1,0,0],[1,1,0,0]]    -> 3   (no double-dip)
      [[1,1,0,0],[0,0,1,0]]    -> 4   ({0}, {1}, {0,1}, {2})
      [[1,1,1,1]]              -> 15  (all nonempty subsets of 4 elts)

    Implementation: inclusion–exclusion over the m solutions, using
    |down(b_i) ∩ down(b_j)| = |down(b_i & b_j)| = 2^popcount(b_i & b_j).
    O(2^m) terms with m = number of solutions (≤ ~5 after dedup) — cheap for
    any k, unlike enumerating the up-to-2^k covered subsets (which cost ~5ms
    per fully-passing LCB sample at k=16, on the per-rollout reward path).
    """
    if not matrix:
        return 0.0
    masks = []
    for row in matrix:
        b = 0
        for i, v in enumerate(row):
            if v >= 0.5:
                b |= 1 << i
        masks.append(b)
    # Dedup identical masks: redundant inclusion–exclusion terms cancel, but
    # dropping them keeps the term count at 2^(distinct masks).
    masks = list(set(masks))
    m = len(masks)
    total = 0.0
    for sel in range(1, 1 << m):
        inter = ~0
        bits = 0
        s = sel
        j = 0
        while s:
            if s & 1:
                inter &= masks[j]
                bits += 1
            s >>= 1
            j += 1
        term = float(2 ** int(inter).bit_count() - 1) if inter else 0.0
        total += term if bits % 2 == 1 else -term
    return total


# ─── Dispatch ───────────────────────────────────────────────────────────────


_VARIANT_FNS = {
    "sum_all": sum_all,
    "max_sum_naive": max_sum_naive,
    "set_coverage": set_coverage,
}


def all_scalars(matrix) -> dict:
    """Compute every variant. Used by reward functions to log all of them
    so we can see what each rollout would have looked like under any
    variant, regardless of which one drove training."""
    return {name: fn(matrix) for name, fn in _VARIANT_FNS.items()}


def compute_scalar(matrix, variant: str | None = None) -> float:
    """Compute the scalar selected by `variant`, or by env var if None."""
    name = variant if variant is not None else get_active_variant()
    if name not in _VARIANT_FNS:
        raise ValueError(f"unknown score variant {name!r}")
    return _VARIANT_FNS[name](matrix)
