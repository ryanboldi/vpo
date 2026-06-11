"""Reward entry point.

``compute_score`` is the function veRL calls per rollout
(``custom_reward_function.path=vpo/reward.py``, ``name=compute_score``). It
finds the task by ``data_source`` and picks a mode from the row:

  - ``extra_info["num_solutions"] > 1``  -> multi-solution (score matrix)
  - else ``ground_truth["weights"]`` set -> goal-conditioned (scalar = w · channels)
  - else                                 -> single solution

It returns a flat dict containing ``score`` (the reward), ``sub_scores``, and
one key per objective channel.
"""

from __future__ import annotations

import json

import vpo_tasks  # noqa: F401  — import side-effect populates vpo.task.REGISTRY
from vpo.metrics import build_multi_metrics, build_single_metrics
from vpo.task import REGISTRY


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    task = REGISTRY[data_source]
    ei = extra_info or {}
    # NB: do NOT `or -1` — a legitimate 0-based task_id of 0 is falsy and would
    # otherwise be silently rewritten to the -1 "missing" sentinel.
    task_id = ei.get("task_id", -1)
    if task_id is None:
        task_id = -1

    # Some pipelines stringify ground_truth (e.g. LCB) — tolerate it.
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    m = ei.get("num_solutions") or 1

    # ─── multi-solution (matrix) ─────────────────────────────────────────────
    if m > 1:
        if isinstance(ground_truth, dict) and "weights" in ground_truth:
            # A row carrying both modes would announce a weighted objective in
            # the prompt while being scored unweighted — refuse rather than
            # silently optimize a different objective than advertised.
            raise ValueError(
                f"data_source={data_source!r}: row has both "
                f"extra_info.num_solutions={m} and ground_truth['weights'] — "
                "combined multi-solution + goal-conditioned scoring is not "
                "supported; bake the dataset with one mode or the other."
            )
        rec = task.score_multi(solution_str, m, ground_truth, ei)
        out = build_multi_metrics(
            rec["sub_scores"],
            num_parsed=rec["num_parsed"],
            num_test_executions=rec.get("num_test_executions", 0),
            task_id=task_id,
        )
        # GDPO contract: the bare named per-objective channels exist as
        # top-level keys in every mode.
        out.update(task.multi_named_channels(rec["sub_scores"]))
        out.update(rec.get("extra", {}))
        if rec.get("scalar") is not None:
            out["score"] = rec["scalar"]
        return out

    # ─── single-solution (scalar / GDPO vector / goal-cond) ──────────────────
    rec = task.score_one(solution_str, ground_truth, ei)
    out = build_single_metrics(
        rec["sub_scores"],
        num_test_executions=rec.get("num_test_executions", 0),
        num_programs_executed=rec.get("num_programs_executed", 1),
        task_id=task_id,
    )
    out.update(rec["channels"])
    out.update(rec.get("extra", {}))

    if isinstance(ground_truth, dict) and "weights" in ground_truth:
        # Goal-conditioned (or random-w): scalarize against the named channels
        # in objective order. Generic — works for LCB's 16 padded tc_* too.
        weights = ground_truth["weights"]
        if len(weights) != len(task.objectives):
            raise ValueError(
                f"goal-cond weights length {len(weights)} != "
                f"{len(task.objectives)} objectives for data_source={data_source!r}"
            )
        out["score"] = sum(
            w * rec["channels"][k] for w, (k, _) in zip(weights, task.objectives)
        )
    elif rec.get("scalar") is not None:
        out["score"] = rec["scalar"]

    return out
