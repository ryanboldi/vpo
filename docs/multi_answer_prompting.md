# Multi-answer prompting

VPO and Multi-RLVR train the model to emit **m solutions per response**, so the reward can score each separately and the advantage estimator can use the full (m, k) matrix. **You have to change the prompt** to ask for these m solutions in a parseable format — otherwise the multi-sol pipeline collapses to single-sol and the (m, k) matrix degenerates to (1, k).

This is the *only* user-facing change beyond the veRL patch. It is implemented as a wrapper around veRL's `RLHFDataset` that rewrites the prompt at `__getitem__` time.

## How it works

`vpo/augment.py::AugmentedDataset` wraps `RLHFDataset`. At `__getitem__`, when `num_solutions > 1` it looks the active task up in the registry (`vpo.task.resolve(domain)`) and calls `task.rewrite_multi_solution(content, m)`, which applies that task's **multi-solution rewrite spec** — a `SubRewrite` or `AppendRewrite` declared as `multi_rewrite` on the Task in `vpo_tasks/<task>.py`. The wrapper is a few lines:

```python
# vpo/augment.py
class AugmentedDataset(_TorchDataset):
    def __getitem__(self, item):
        row = self._inner[item]
        prompt = row["raw_prompt"]
        content = prompt[-1]["content"]
        if self._num_solutions > 1:
            content = rewrite_multi_solution(content, self._multi_domain, self._num_solutions)
            ei = row.get("extra_info") or {}
            ei["num_solutions"] = self._num_solutions   # what vpo/reward.py keys on
            row["extra_info"] = ei
        # (goal-cond path: also sample Dirichlet weights and append a suffix)
        prompt[-1]["content"] = content
        return row
```

The two rewrite kinds (both from `vpo.task`):

- **`SubRewrite`** *substitutes* the single-output format line in the last user message for the multi-solution template (and applies follow-up `extra_subs` — maze rewrites "You have N steps." → "You have N steps per route."). Used by maze, musique, eureqa. If the substitution pattern doesn't match the prompt (template drift), the rewrite **raises** rather than silently leaving a single-answer prompt that would be scored as m zero-padded solutions.
- **`AppendRewrite`** *appends* a "give m attempts" block to the prompt. Used when the single-output format spec lives in the **system prompt** (so a regex on the last user turn can't reach it) — e.g. `vpo_tasks/tool.py`.

There is no `MULTI_OUTPUT_TEMPLATE` dict and no central per-domain table anymore: each template lives next to its task in `vpo_tasks/<task>.py`.

`train.sh` already passes the right Hydra args for any multi-sol or goal-cond run; you don't need to set these by hand. For reference, they are:

```
++data.custom_cls.path=vpo/augment.py
++data.custom_cls.name=AugmentedDataset
++data.multi_solution_domain=<task>          # e.g. maze, musique, eureqa, ...
++data.num_solutions=3                        # m
++data.truncation=left                        # required — see note below
```

> **`truncation=left` is required.** The rewrite happens *after* `filter_overlong_prompts` runs, so some prompts exceed `max_prompt_length` at collate time. Left-truncation preserves the question + rewritten instruction at the end.

## Per-domain templates

The templates below are verbatim from each task's `multi_rewrite` spec in `vpo_tasks/<task>.py`. Paper uses **m=3** for maze, musique, tool, lcb and **m=5** for eureqa.

### maze — k=4

`vpo_tasks/maze.py::_MULTI_TEMPLATE` (a `SubRewrite` that replaces the `<answer>…</answer>` output line). `{m}` is filled at load time; the example below is the literal template with the `<route_i>` tags pre-rendered for m=3:

```
Reason briefly about the maze, then provide {m} genuinely different routes
from S to E. Each route is a sequence of UP/DOWN/LEFT/RIGHT moves
(space-separated). Wrap each route in numbered tags (<route_1>...</route_1>,
<route_2>...</route_2>, <route_3>...</route_3>). Inside each tag put ONLY
moves (no arrows, no coordinates, no prose); any reasoning goes outside the
tags. Each route must independently reach E within the per-route step budget
stated above (score is zero if it doesn't).
Format example (m=3):
  <route_1>RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN</route_1>
  <route_2>DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT</route_2>
  <route_3>RIGHT DOWN RIGHT DOWN RIGHT DOWN RIGHT DOWN</route_3>
```

### musique — k=5

`vpo_tasks/musique.py::_MULTI_TEMPLATE` (`SubRewrite`):

```
Give {m} different answers in <response_i>...</response_i> tags, each with
<support>indices</support> and <answer>answer</answer>.
```

### eureqa — k=5, m=5

`vpo_tasks/eureqa.py::_MULTI_TEMPLATE` (`SubRewrite`):

```
Provide {m} different reasoning chains, each wrapped in numbered tags
<response_1>...</response_1> through <response_{m}>...</response_{m}>. Each
chain must contain all 5 entity resolutions; replace each `...` with ONLY
the resolved canonical name (underscored Wikipedia-style, no prose, no
labels):
  <response_i>
    <entity_A>...</entity_A>   (resolves Person A — this is the answer)
    <entity_B>...</entity_B>   (resolves the second mask)
    <entity_C>...</entity_C>   (resolves the third mask)
    <entity_D>...</entity_D>   (resolves the fourth mask)
    <entity_E>...</entity_E>   (resolves the fifth mask)
  </response_i>
Closing tags are required for every response and every entity. The {m}
chains should be genuinely different attempts (e.g. different candidate
resolutions, not paraphrases). The answer to the question is whatever you
put in <entity_A> within each chain.
```

### tool — k=4

`vpo_tasks/tool.py::_MULTI_APPEND` (an `AppendRewrite`, because the single-output format lives in the system prompt):

```
---
ADDITIONAL INSTRUCTION (overrides the single-attempt output format above):
Provide {m} different attempts at the task. Wrap each attempt in numbered
outer tags <response_1>...</response_1> through <response_{m}>...</response_{m}>.
Inside each <response_i>, follow the original output format from the system
prompt (your <think>, optional <tool_call>, and inner <response> sections).
The {m} attempts should be genuinely different — different tool choices,
different argument values, or different reasoning — not paraphrases.
Closing tags are required on every outer attempt.
```

### lcb (LiveCodeBench) — k up to 16

lcb is the exception: its `multi_rewrite` is `None`. The multi-solution prompt is **baked into the prompt at build time** (`vpo_tasks/livecodebench.py::build_prompt`, driven by the `num_solutions` argument threaded through `process_example`), not rewritten at load time, and the multiple solutions are `---`-separated ```python``` code blocks rather than `<response_i>` tags. The functional-test variant is `_MULTI_SUFFIX_FUNCTIONAL`:

```
Your solutions should be Python functions with this signature:
```python
{signature}
```

Write {m} different solutions that solve this problem. Each solution
should use a different approach. Wrap each solution in a ```python``` code
block and separate them with '---'.
```

and the stdin variant is `_MULTI_SUFFIX_STDIN`:

```
Each solution should be a complete Python program that reads input from
stdin and writes output to stdout. Write {m} different solutions that solve
this problem, each using a different approach. Wrap each solution in a
```python``` code block and separate them with '---'.
```

(Because lcb's multi prompt is baked in, a multi-sol *training* run that tried to rewrite lcb at load time would hit `ValueError: Multi-solution rewrite not configured` — lcb multi is wired through preprocessing and eval, not `AugmentedDataset`.)

## Parsing

Each task's `score_multi` (in `vpo_tasks/<task>.py`) extracts the m blocks — `<response_i>(.*?)</response_i>` (musique, tool, eureqa via `parse_multi_responses` / `parse_response`), `<route_i>(.*?)</route_i>` (maze, via `parse_multi_routes`), or `---`-separated code blocks (lcb, via `parse_solutions`) — de-dupes structurally identical solutions (`dedup_by_vector`, `dedup_routes`, or `ast_dedup` for lcb), scores each one, and returns the (m, k) matrix. maze/musique/eureqa/tool pad with zero-rows (`pad_matrix`) to exactly m when fewer than m parseable solutions are emitted; lcb returns the de-duped matrix unpadded (matching the original lcb_multi reward).

For unparseable responses (no tags found, malformed) the scorer returns an all-zero (m, k) matrix and the rollout gets zero advantage. This is the intended failure mode — VPO does not need any explicit format-bonus.

## Output-length budget

Multi-sol needs more tokens than single-sol so the trailing solution block isn't truncated (a truncated row scores zero and the comparison stops being fair). Exact per-domain values are in [`reproducibility.md`](reproducibility.md) and baked into `train.sh`.
