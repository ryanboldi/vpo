"""Task interface and registry.

Defines:
  - ``Task``: the base class a task subclasses. It declares ``name``,
    ``data_source``, ``objectives`` and a multi-solution prompt-rewrite spec,
    and implements scoring (``score_one`` / ``score_multi``), prompt parsing,
    dataset preprocessing (``preprocess``) and held-out eval (``evaluate``).
  - ``REGISTRY``: maps ``data_source`` -> ``Task``; ``register`` adds to it and
    ``resolve`` looks a task up by ``data_source`` or by ``name``.
  - ``SubRewrite`` / ``AppendRewrite``: the two ways to turn a single-solution
    prompt into a multi-solution one.
  - ``dedup_by_vector`` / ``pad_matrix``: small helpers for building multi
    score matrices.

``score_one`` returns a record with ``sub_scores`` (the score vector), per
objective ``channels``, a ``scalar`` (or ``None`` to use the default), and
``extra`` logging fields. ``score_multi`` returns ``sub_scores`` as a matrix
plus ``scalar``, ``num_parsed`` and ``extra``. The reward dispatcher
(:mod:`vpo.reward`) turns these records into the dict veRL reads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# data_source string -> Task. Populated at import time by tasks/__init__.py.
REGISTRY: dict[str, "Task"] = {}


def register(task: "Task") -> "Task":
    """Register a task under its ``data_source`` (and ``name`` alias)."""
    REGISTRY[task.data_source] = task
    return task


def resolve(key: str) -> "Task":
    """Look up a task by ``data_source`` or by CLI ``name`` alias."""
    if key in REGISTRY:
        return REGISTRY[key]
    for task in REGISTRY.values():
        if task.name == key:
            return task
    raise KeyError(
        f"Unknown task {key!r}. Known: "
        f"{sorted({t.name for t in REGISTRY.values()} | set(REGISTRY))}"
    )


# ---------------------------------------------------------------------------
# Multi-solution prompt-rewrite specs (instantiated per task in vpo_tasks/)
# ---------------------------------------------------------------------------


@dataclass
class SubRewrite:
    """Rewrite single→multi by *substituting* the output-format instruction.

    ``pattern`` matches the single-output instruction line in the user message;
    it is replaced with ``template`` (which may reference ``{m}``). ``extra_subs``
    applies follow-up regex fixes (e.g. "You have N steps." → "…per route.").
    """

    pattern: re.Pattern
    template: str
    extra_subs: list[tuple[re.Pattern, str]] = field(default_factory=list)

    def __call__(self, content: str, m: int) -> str:
        out, n_subs = self.pattern.subn(self.template.format(m=m), content)
        if n_subs == 0:
            # A silent no-op here would leave a single-answer prompt that the
            # reward then scores as an m-solution matrix padded with zeros —
            # systematically deflated scores with no error anywhere.
            raise ValueError(
                f"multi-solution rewrite pattern {self.pattern.pattern!r} did "
                "not match the prompt — the dataset's prompt template has "
                "drifted from the task's multi_rewrite spec."
            )
        for pat, repl in self.extra_subs:
            out = pat.sub(repl, out)
        return out


@dataclass
class AppendRewrite:
    """Rewrite single→multi by *appending* a "give m attempts" instruction.

    Used when the single-output format spec lives in the system prompt (or is
    otherwise unreachable by a regex on the last user turn), e.g. ToolRL.
    """

    template: str

    def __call__(self, content: str, m: int) -> str:
        return content + self.template.format(m=m)


# ---------------------------------------------------------------------------
# Small reusable scoring helpers (used by task files, not the engine)
# ---------------------------------------------------------------------------


def dedup_by_vector(matrix: list[list[float]]) -> list[list[float]]:
    """Drop duplicate score vectors, keeping first occurrence (stable)."""
    seen: set[tuple] = set()
    out: list[list[float]] = []
    for row in matrix:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def pad_matrix(matrix: list[list[float]], m: int, zero: list[float]) -> list[list[float]]:
    """Pad/truncate a score matrix to exactly ``m`` rows."""
    out = list(matrix)
    while len(out) < m:
        out.append(list(zero))
    return out[:m]


# ---------------------------------------------------------------------------
# The Task base class
# ---------------------------------------------------------------------------


class Task:
    """Base class every task subclasses. See module docstring for the contract.

    Subclasses set the class attributes (``name``, ``data_source``,
    ``objectives``, ``multi_rewrite``) and implement
    the scoring + CLI methods. ``score_one``/``score_multi`` return *records*
    (plain dicts) — the engine assembles the veRL reward dict from them.
    """

    name: str
    data_source: str
    #: Ordered (key, human-description) pairs. The keys are the GDPO/goal-cond
    #: named channels; the descriptions feed goal-conditioning prompt suffixes.
    objectives: list[tuple[str, str]] = []
    #: SubRewrite | AppendRewrite | None — single→multi prompt rewrite.
    multi_rewrite = None

    # ----- derived -----
    @property
    def objective_names(self) -> list[str]:
        return [k for k, _ in self.objectives]

    @property
    def num_objectives(self) -> int:
        return len(self.objectives)

    @property
    def zero_scores(self) -> list[float]:
        return [0.0] * self.num_objectives

    def named_channels(self, vector: list[float]) -> dict[str, float]:
        """Map a score vector onto its named objective channels (for GDPO)."""
        return {name: vector[i] for i, name in enumerate(self.objective_names)}

    def multi_named_channels(self, matrix: list[list[float]]) -> dict[str, float]:
        """Bare ``{name}`` channels for a multi-solution rollout (GDPO basis).

        Per-objective column sum, width-padding each row to ``num_objectives``
        (LCB rows hold only the real tests). Decomposes the default ``sum_all``
        scalar and reduces to ``named_channels(vector)`` at m=1.
        """
        k = self.num_objectives
        sums = [0.0] * k
        for row in matrix:
            padded_row = (list(row) + [0.0] * k)[:k]
            for j in range(k):
                sums[j] += padded_row[j]
        return {name: sums[j] for j, name in enumerate(self.objective_names)}

    def multi_channel_extras(self, padded: list[list[float]]) -> dict[str, float]:
        """``{name}_sum/_max/_first`` per objective over a padded multi matrix."""
        extras: dict[str, float] = {}
        for j, name in enumerate(self.objective_names):
            col = [row[j] for row in padded]
            extras[f"{name}_sum"] = sum(col)
            extras[f"{name}_max"] = max(col)
            extras[f"{name}_first"] = padded[0][j]
        return extras

    def weighted_scalar(self, vector: list[float]) -> float:
        """Default single-objective scalar = unweighted mean. Override per task."""
        return sum(vector) / max(len(vector), 1)

    def rewrite_multi_solution(self, content: str, m: int) -> str:
        if self.multi_rewrite is None:
            raise ValueError(
                f"Multi-solution rewrite not configured for task {self.name!r}"
            )
        return self.multi_rewrite(content, m)

    # ----- scoring (implemented by subclasses) -----
    def score_one(self, solution_str: str, ground_truth, extra_info: dict) -> dict:
        """Single-solution record:

        ``{"sub_scores": [...], "channels": {...}, "scalar": float|None,
           "extra": {...}, "num_test_executions": int, "num_programs_executed": int}``
        """
        raise NotImplementedError

    def score_multi(self, solution_str: str, m: int, ground_truth, extra_info: dict) -> dict:
        """Multi-solution record:

        ``{"sub_scores": [[...]], "scalar": float|None, "extra": {...},
           "num_parsed": int, "num_test_executions": int}``
        """
        raise NotImplementedError

    # ----- CLI hooks (implemented by subclasses) -----
    def preprocess(self, args) -> None:
        """Build train.parquet/test.parquet. ``args`` is an argparse Namespace."""
        raise NotImplementedError

    def add_preprocess_args(self, parser) -> None:
        """Register task-specific argparse flags for preprocessing."""
        raise NotImplementedError

    def evaluate(self, args) -> None:
        """Run held-out eval and write JSON (+ optional .npy)."""
        raise NotImplementedError

    def add_eval_args(self, parser) -> None:
        """Register task-specific argparse flags for eval."""
        raise NotImplementedError
