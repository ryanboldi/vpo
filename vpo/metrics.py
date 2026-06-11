"""Builders for the reward metric dict.

  - ``build_single_metrics(sub_scores, ...)``  — for a single-solution rollout
  - ``build_multi_metrics(score_matrix, ...)`` — for a multi-solution rollout

Each returns the common reward-dict fields (``score``, ``sub_scores``, counts,
solve flags, ``task_id``). The reward dispatcher merges per-task objective
channels and extra fields on top.
"""

import hashlib

from vpo.utils.score_variants import all_scalars, get_active_variant


def _row_fully_solved(row) -> bool:
    """A row is fully solved iff every entry is >= 1.0."""
    return bool(row) and all(s >= 1.0 for s in row)


def _coerce_task_id(task_id) -> int:
    """Coerce a task_id to a stable int for logging.

    MBPP task_ids are already ints. LCB task_ids are AtCoder-style strings
    like "abc394_a" which can't be parsed as int — fall back to a stable
    hash so downstream code can still use task_id as a join key.
    """
    try:
        return int(task_id)
    except (ValueError, TypeError):
        # Stable across processes and fits in int32. NB: builtin hash() of a str
        # is salted by PYTHONHASHSEED and differs between processes, so it would
        # break the cross-step/cross-process join this key exists for.
        digest = hashlib.sha1(str(task_id).encode("utf-8")).hexdigest()
        return int(digest, 16) % (2**31)


def build_multi_metrics(
    score_matrix: list[list[float]],
    *,
    num_parsed: int,
    num_test_executions: int,
    task_id: int = -1,
) -> dict:
    """Standard derived metrics for a multi-solution rollout.

    Args:
        score_matrix: (m', k) post-dedup per-solution × per-test-case scores.
        num_parsed: Solutions parsed before dedup.
        num_test_executions: Sandbox calls (for throughput logging).
        task_id: Stable join key across steps. -1 if missing.

    Note:
        first_solution_solves is computed from score_matrix[0] (the first
        DEDUP'D solution). Callers that need a different definition should
        override the key after calling this builder.
    """
    scalars = all_scalars(score_matrix)
    first_fully = float(
        bool(score_matrix) and _row_fully_solved(score_matrix[0])
    )
    any_fully = float(any(_row_fully_solved(r) for r in score_matrix))

    return {
        "score": scalars[get_active_variant()],
        "sub_scores": score_matrix,
        "score_sum_all": scalars["sum_all"],
        "score_max_sum_naive": scalars["max_sum_naive"],
        "score_set_coverage": scalars["set_coverage"],
        "num_unique": len(score_matrix),
        "num_parsed": int(num_parsed),
        "num_programs_generated": int(num_parsed),
        "num_programs_executed": len(score_matrix),
        "num_test_executions": int(num_test_executions),
        "first_solution_solves": first_fully,
        "any_solution_solves": any_fully,
        "task_id": _coerce_task_id(task_id),
    }


def build_single_metrics(
    sub_scores: list[float],
    *,
    num_test_executions: int,
    num_programs_executed: int = 1,
    task_id: int = -1,
) -> dict:
    """Standard derived metrics for a single-solution rollout.

    Args:
        sub_scores: Per-test-case (or per-sub-objective) binary scores.
        num_test_executions: Sandbox calls (0 for non-executable tasks).
        num_programs_executed: 1 by default; pass 0 if code extraction failed.
        task_id: Stable join key across steps. -1 if missing.
    """
    fully = float(_row_fully_solved(sub_scores))
    n_tc = len(sub_scores)
    return {
        "score": sum(sub_scores) / max(n_tc, 1),
        "sub_scores": list(sub_scores),
        "num_test_cases": n_tc,
        "num_programs_generated": 1,
        "num_programs_executed": int(num_programs_executed),
        "num_test_executions": int(num_test_executions),
        "first_solution_solves": fully,
        "any_solution_solves": fully,
        "task_id": _coerce_task_id(task_id),
    }
