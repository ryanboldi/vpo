"""Dataset wrapper that rewrites prompts at load time.

``AugmentedDataset`` wraps veRL's ``RLHFDataset`` and, per row, optionally:
  - rewrites a single-solution prompt into a multi-solution one
    (``++data.num_solutions``, ``++data.multi_solution_domain``), and
  - appends Dirichlet-sampled objective weights to the prompt and writes them
    into ``ground_truth['weights']`` (``++data.goal_cond_domain``).

Also exposes ``format_weight_suffix`` and ``rewrite_multi_solution`` (used by
eval), and an ``augment_dataset`` CLI that writes weight-augmented parquets.
veRL loads this file by path (``++data.custom_cls.path=vpo/augment.py``,
``++data.custom_cls.name=AugmentedDataset``).
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
from torch.utils.data import Dataset as _TorchDataset

import vpo_tasks  # noqa: F401  — import side-effect populates vpo.task.REGISTRY
from vpo.task import resolve

WEIGHT_SUFFIX = """

Your objective weights (higher = more important to optimize):
{weight_lines}

Maximize your weighted score."""


def format_weight_suffix(weights, objectives) -> str:
    """Render a goal-conditioning weight block from (key, description) pairs."""
    lines = [f"- {desc}: {w:.2f}" for (_, desc), w in zip(objectives, weights)]
    return WEIGHT_SUFFIX.format(weight_lines="\n".join(lines))


def rewrite_multi_solution(content: str, domain: str, m: int) -> str:
    """Convert a single-solution prompt to multi-solution for ``domain``.

    Delegates to the task's configured rewrite spec (SubRewrite / AppendRewrite).
    """
    return resolve(domain).rewrite_multi_solution(content, m)


class AugmentedDataset(_TorchDataset):
    """Wraps RLHFDataset with on-the-fly prompt augmentation.

    Config keys (Hydra DictConfig):
        - ``num_solutions`` (int, default 0): if >1, rewrite for multi-solution.
        - ``multi_solution_domain`` (str): task whose rewrite spec to use.
        - ``goal_cond_domain`` (str): if set, sample Dirichlet weights and append.
          Mutually exclusive with ``num_solutions>1`` (the reward scores one
          mode per row).
        - ``random_reward_only`` (bool): inject weights into ground_truth WITHOUT
          appending the prompt suffix (the "random-w" baseline).
    """

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        from verl.utils.dataset.rl_dataset import RLHFDataset

        self._inner = RLHFDataset(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

        self._num_solutions = int(config.get("num_solutions", 0))
        self._domain = config.get("goal_cond_domain", None)
        self._multi_domain = config.get("multi_solution_domain", None) or self._domain
        self._random_reward_only = bool(config.get("random_reward_only", False))
        # Per-row weights are drawn from a Generator seeded by (run seed, row
        # index), so goal-conditioning is reproducible across runs AND identical
        # regardless of how many DataLoader workers shard the dataset. Falls back
        # to VPO_SEED then 0 if ``data.seed`` is unset.
        _seed = config.get("seed", None)
        if _seed is None:
            _seed = os.environ.get("VPO_SEED", 0)
        self._seed = int(_seed)

        if self._domain:
            try:
                self._objectives = resolve(self._domain).objectives
            except KeyError:
                raise ValueError(f"Unknown goal_cond_domain={self._domain!r}")
        else:
            self._objectives = []

        if self._num_solutions > 1:
            if self._domain:
                # The reward dispatcher refuses rows carrying both modes (the
                # weighted objective announced in the prompt would be scored
                # unweighted) — fail here, at construction, not per-row.
                raise ValueError(
                    "num_solutions>1 cannot be combined with goal_cond_domain: "
                    "multi-solution + goal-conditioned scoring is not supported."
                )
            if self._multi_domain is None:
                raise ValueError(
                    "num_solutions>1 requires multi_solution_domain to be set"
                )
            # Fail at construction — not lazily, per-row, deep inside a
            # DataLoader worker — if the chosen task has no rewrite spec.
            try:
                _multi_task = resolve(self._multi_domain)
            except KeyError:
                raise ValueError(
                    f"Unknown multi_solution_domain={self._multi_domain!r}"
                )
            if _multi_task.multi_rewrite is None:
                raise ValueError(
                    f"multi_solution_domain={self._multi_domain!r} has no "
                    "multi-solution rewrite configured; its multi-solution "
                    "prompts must be baked at preprocess time, not rewritten "
                    "at load time."
                )

        self._k = len(self._objectives)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, item):
        row = self._inner[item]
        prompt = row["raw_prompt"]
        content = prompt[-1]["content"]

        if self._num_solutions > 1:
            content = rewrite_multi_solution(content, self._multi_domain, self._num_solutions)
            ei = row.get("extra_info") or {}
            ei["num_solutions"] = self._num_solutions
            row["extra_info"] = ei

        if self._domain and self._k > 0:
            rng = np.random.default_rng([self._seed, int(item)])
            w = rng.dirichlet(np.ones(self._k)).tolist()
            rm = row.get("reward_model")
            gt = rm.get("ground_truth") if rm is not None else None
            if gt is None:
                # Without a ground_truth to carry them, the sampled weights
                # could never reach the reward — the prompt would promise
                # weighted scoring while compute_score routes single-mode.
                raise ValueError(
                    f"goal_cond_domain={self._domain!r}: row {item} has no "
                    "reward_model.ground_truth to carry the sampled weights."
                )
            if isinstance(gt, str):
                # Some pipelines stringify ground_truth (e.g. LCB); reward.py
                # json.loads it back, so inject through the same encoding.
                parsed = json.loads(gt)
                parsed["weights"] = w
                rm["ground_truth"] = json.dumps(parsed)
            else:
                gt["weights"] = w
            if not self._random_reward_only:
                content += format_weight_suffix(w, self._objectives)

        prompt[-1]["content"] = content
        return row

    def __getattr__(self, name):
        # Guard against infinite recursion when the instance is restored
        # without __init__ running (pickle / copy in DataLoader workers,
        # checkpoint resume): before ``_inner`` exists, delegating would
        # re-enter __getattr__('_inner') forever -> RecursionError.
        if name == "_inner" or "_inner" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._inner, name)


# Backwards-compatible alias
GoalCondDataset = AugmentedDataset


def augment_dataset(input_dir, output_dir, domain, seed=None):
    """CLI: load parquets, append goal-cond weights to every row, write out."""
    import datasets as hf_datasets

    objectives = resolve(domain).objectives
    k = len(objectives)
    rng = np.random.default_rng(seed)

    input_dir = os.path.expanduser(input_dir)
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for split in ("train", "test"):
        path = os.path.join(input_dir, f"{split}.parquet")
        if not os.path.exists(path):
            continue
        ds = hf_datasets.Dataset.from_parquet(path)
        all_weights = rng.dirichlet(np.ones(k), size=len(ds)).tolist()
        rows = []
        for i, row in enumerate(ds):
            row = copy.deepcopy(row)
            w = all_weights[i]
            row["prompt"][-1]["content"] += format_weight_suffix(w, objectives)
            row["reward_model"]["ground_truth"]["weights"] = w
            rows.append(row)
        hf_datasets.Dataset.from_list(rows).to_parquet(os.path.join(output_dir, f"{split}.parquet"))
        print(f"  {split}: {len(rows)} examples")

    print(f"Goal-conditioned {domain} dataset (seed={seed}) written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment a dataset with goal-conditioned weights.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    augment_dataset(args.input_dir, args.output_dir, args.domain, args.seed)
