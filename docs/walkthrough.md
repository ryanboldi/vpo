# How VPO works — a researcher's walkthrough

## 30-second mental model

VPO is a custom advantage estimator that replaces GRPO's `(reward - mean) / std` with a signal that rewards **diverse Pareto coverage under randomly sampled reward scalarizations** (weightings of the vector reward — not user-supplied preferences). The whole algorithm is one function, `vpo_advantage(sub_scores, uid)`, and the rest of the repo is plumbing to feed it an `(m, k)` reward matrix.

The repo is split into a **task-agnostic engine** (`vpo/`) and a **per-task plugin** (`vpo_tasks/<name>.py`). Nothing in `vpo/` is task-specific: a single reward dispatcher (`vpo/reward.py`) and a single dataset wrapper (`vpo/augment.py`) serve every task by looking the task up in a registry and calling into it.

```
              ┌──────────────────────────────────────────┐
              │  YOUR TASK PLUGIN                        │
              │  vpo_tasks/<task>.py (subclass of Task)     │
              │  score_multi() returns (m, k) matrix     │
              └────────────────┬─────────────────────────┘
                               │  (registered in vpo.task.REGISTRY)
                               ▼
   ┌──────────────────────┐    ┌────────────────────────────┐
   │  PROMPT REWRITE      │    │  REWARD DISPATCHER         │
   │  vpo/augment.py      │ ──►│  vpo/reward.py             │
   │  AugmentedDataset    │    │  compute_score(...)        │
   │  "give m answers     │    │  routes by data_source,    │
   │   in <route_i>"      │    │  infers single/multi/      │
   │  (spec lives on the  │    │  goal-cond from the row,   │
   │   Task)              │    │  returns sub_scores: (m,k) │
   └──────────────────────┘    └─────────────┬──────────────┘
                                             │
                               ┌─────────────▼──────────────┐
                               │  PATCHED veRL DISPATCH     │
                               │  verl/.../ray_trainer.py   │
                               │  reads sub_scores, calls   │
                               │  vpo_advantage(...)        │
                               │  (vpo/utils/vpo.py)        │
                               └─────────────┬──────────────┘
                                             │
                                             ▼
                                    advantages → standard PPO update
```

VPO doesn't change the optimizer, KL term, or rollout engine — only the advantage path. That's why the patch is 3 files.

## What happens during one training step

Tracing `train_vpo.sh TASK=maze`.

### 1. Dataset emits a prompt

veRL's `RLHFDataset` reads `train.parquet`. Each maze row:

```python
{"prompt": [{"role": "user", "content": "Navigate a 9x9 maze from S to E ..."}],
 "reward_model": {"ground_truth": {"grid_text": "...", "start": [0,0], ...}}}
```

But we wrapped it: `++data.custom_cls.path=vpo/augment.py`, `++data.custom_cls.name=AugmentedDataset`. At `__getitem__`, our `AugmentedDataset` (`vpo/augment.py`) intercepts and **rewrites the user-message content**. It looks the task up by `++data.multi_solution_domain=maze`, fetches that Task's multi-solution rewrite spec, and applies it. For maze the spec is a `SubRewrite` (declared on `MazeTask.multi_rewrite` in `vpo_tasks/maze.py`) that substitutes the single-output instruction line for the maze multi-route template:

```
Reason briefly about the maze, then provide 3 genuinely different routes from
S to E. ... Wrap each route in numbered tags (<route_1>...</route_1>,
<route_2>...</route_2>, <route_3>...</route_3>).
```

The wrapper also sets `extra_info["num_solutions"] = 3` on the row, which is what the reward dispatcher later keys on. The model now expects to emit 3 routes in one response.

### 2. Rollout (vLLM)

veRL calls vLLM `n=8` times per prompt (`rollout.n`). Each rollout is **one long response containing 3 routes** in `<route_i>` tags.

### 3. Reward → `(m, k)` matrix

veRL calls `compute_score(...)` once per rollout. The single dispatcher in `vpo/reward.py` looks the task up by `data_source`, sees `extra_info["num_solutions"] > 1`, and routes to the task's `score_multi`. For maze that's `MazeTask.score_multi` in `vpo_tasks/maze.py`:

```python
parsed = parse_multi_routes(solution_str, m)           # extract 3 strings from <route_i> tags
unique = dedup_routes(parsed)                           # drop empty + duplicate move-lists
unique_scores = [score_route(r, ground_truth) for r in unique]
# Each row is [completion, gold, diamond, avoid_lava] — k=4 objectives.
padded = pad_matrix(unique_scores, m, ZERO_SCORES)      # (m, k) padded with zeros
return {"sub_scores": padded, "scalar": ..., "num_parsed": ..., "extra": {...}}
```

`vpo/reward.py` turns that record into the flat dict veRL reads (via `build_multi_metrics`), with `sub_scores` as the load-bearing key. veRL stashes it in `data.non_tensor_batch["sub_scores"]`.

### 4. Advantage computation (patched veRL → `vpo_advantage`)

`verl/verl/trainer/ppo/ray_trainer.py:221`:

```python
elif adv_estimator == AdvantageEstimator.VPO:
    from vpo.utils.vpo import vpo_advantage
    sub_scores = data.non_tensor_batch["sub_scores"]   # list of (m, k) matrices, len=B
    index      = data.non_tensor_batch["uid"]          # prompt-group ids, len=B
    advantages, vpo_diag = vpo_advantage(sub_scores, index)
    advantages = advantages.unsqueeze(-1).to(data.batch["response_mask"].device)
    advantages = advantages * data.batch["response_mask"]
    data.batch["advantages"] = advantages
    data.batch["returns"]    = advantages
```

The entire veRL change: pull `sub_scores`, hand them to `vpo_advantage`, broadcast over the response mask.

### 5. The actual VPO step

`vpo_advantage` (`vpo/utils/vpo.py`), for each prompt group:

```python
# For each rollout r with its m solutions, compute own-pool best-of-m under random w:
for r in group:
    samples   = sample_dirichlet(N=100, alpha=1, k=4)   # 100 weight vectors on simplex
    per_w_max = max over s in r of (w · s.scores)        # for each w
    raw_score[r] = mean over w of per_w_max              # ≈ E_w[best-of-m]

# z-norm within the prompt group (just like GRPO):
advantages[r] = (raw_score[r] - mean(raw)) / (std(raw) + ε)
```

The Dirichlet sampler can be naive iid or Sobol QMC (`VPO_SAMPLER=sobol`); `α` is tunable (`VPO_ALPHA=0.5`). Both env vars, no code change.

### 6. Standard PPO update

Nothing else is custom. veRL's vanilla PPO loop consumes `data.batch["advantages"]` and runs the usual clipped objective + KL term + Adam step.

## To apply VPO to a new domain, you add one file

The whole task lives in `vpo_tasks/<name>.py` as a subclass of `vpo.task.Task`. Nothing in `vpo/` changes — the dispatcher and the dataset wrapper find your task through the registry. Concretely:

### 1. Declare the objectives

The `objectives` class attribute is an ordered list of `(key, human_description)` pairs. The keys become the named per-objective reward channels GDPO/goal-cond consume; the descriptions feed the goal-conditioning weight suffix. From `vpo_tasks/maze.py`:

```python
class MazeTask(Task):
    name = "maze"
    data_source = "maze"
    objectives = [
        ("completion", "Reaching the exit (E)"),
        ("gold", "Collecting Gold (G) tiles"),
        ("diamond", "Collecting Diamond (D) tiles"),
        ("avoid_lava", "Avoiding Lava (L) tiles"),
    ]
```

### 2. Declare the multi-solution prompt rewrite

Set `multi_rewrite` to a `SubRewrite` or `AppendRewrite` (both from `vpo.task`). `SubRewrite` *substitutes* the single-output format line in the user message (use when the format spec is in the last user turn); `AppendRewrite` *appends* a "give m attempts" block (use when the format spec lives in the system prompt — e.g. `vpo_tasks/tool.py`). For maze:

```python
multi_rewrite = SubRewrite(
    pattern=_SINGLE_OUTPUT_PATTERN,   # matches the single <answer>…</answer> line
    template=_MULTI_TEMPLATE,         # the "give {m} routes in <route_i> tags" block
    extra_subs=[_STEPS_PER_ROUTE],    # "You have N steps." → "…steps per route."
)
```

`AugmentedDataset` calls `task.rewrite_multi_solution(content, m)` at load time, which dispatches to this spec. There is no separate per-domain template dict anymore — the template text lives next to the task.

### 3. Implement scoring

Implement `score_one` (single solution → record) and, for multi-sol methods, `score_multi` (m solutions → record). Both return plain-dict *records*; `vpo/reward.py` assembles the veRL dict.

`score_one` returns `sub_scores` (the k-vector), `channels` (the named-objective dict — use the `self.named_channels(vector)` helper), `scalar` (the logged reward, or `None`), and `extra`. From `vpo_tasks/maze.py`:

```python
def score_one(self, solution_str, ground_truth, extra_info):
    sub = score_route(solution_str, ground_truth)        # k-vector
    return {
        "sub_scores": sub,
        "channels": self.named_channels(sub),
        "scalar": self.weighted_scalar(sub),
        "extra": {},
        "num_test_executions": 0,
        "num_programs_executed": 1 if extract_moves(solution_str) else 0,
    }
```

`score_multi` returns `sub_scores` as the `(m, k)` matrix (pad with `pad_matrix(...)`), plus `scalar`, `num_parsed`, and `extra`:

```python
def score_multi(self, solution_str, m, ground_truth, extra_info):
    parsed = parse_multi_routes(solution_str, m)
    unique = dedup_routes(parsed)
    padded = pad_matrix([score_route(r, ground_truth) for r in unique], m, ZERO_SCORES)
    return {"sub_scores": padded, "scalar": ..., "num_parsed": ..., "extra": ...}
```

`sub_scores` is the load-bearing key for VPO. `scalar` is what veRL logs as the reward; VPO doesn't use it for the gradient (GRPO/GDPO/goal-cond do).

### 4. Implement `preprocess` and `evaluate`

`preprocess(self, args)` writes `train.parquet` + `test.parquet` with `prompt`, `data_source`, `reward_model.ground_truth`, `extra_info` columns in veRL's schema (`add_preprocess_args` registers the CLI flags). `evaluate(self, args)` runs held-out eval and writes JSON (+ optional `.npy`); `add_eval_args` registers its flags. See `MazeTask.preprocess` / `MazeTask.evaluate` in `vpo_tasks/maze.py` for the pattern. The `data/preprocess_maze.py` and `eval/eval_maze.py` entry points are thin shims that just call `vpo_tasks.maze.TASK.preprocess` / `.evaluate`.

### 5. Register it and add the import

End the file with `TASK = register(MyTask())` (`register` keys the task into `vpo.task.REGISTRY` by `data_source`), then add one import line to `vpo_tasks/__init__.py`:

```python
from . import my_task  # noqa: F401
```

That's it. **No changes to `vpo/utils/vpo.py`, `vpo/reward.py`, `vpo/augment.py`, or the veRL patch.** `train.sh` already derives GDPO's per-objective reward keys from `resolve(TASK).objectives`, so even that needs no per-task edit (you only add a `case` arm for token budgets and the data dir). VPO is `k`-agnostic — it inspects each prompt group's matrix width, so even variable `k` per prompt works (which is what lcb does — variable number of test cases, padded to 16 `tc_*` channels).

## Things that bite

A few non-obvious failure modes:

- **`truncation=left` is mandatory** for multi-sol/goal-cond runs. The prompt rewrite happens at `__getitem__`, after `filter_overlong_prompts`. Without left-truncation, prompts that grew past `max_prompt_length` crash at collate. `train.sh` sets this for you whenever a multi-sol or goal-cond domain is active.

- **`response_length` must fit m solutions comfortably.** If the trailing solution block gets truncated, that row scores zero and the `(m, k)` matrix is silently degraded. See `docs/reproducibility.md` for per-task budgets.

- **The single-sol VPO fallback** (`VPO_SINGLE`) wraps each flat k-vector as `[[list(r)]]` — a 1×k matrix — in the veRL dispatch. Single-sol = degenerate multi-sol with m=1, no special-casing in `vpo_advantage`.

- **Reward must return `"sub_scores"` regardless of method.** `vpo/reward.py` always emits it (via `build_single_metrics` / `build_multi_metrics`), and the patched dispatcher always reads it. The dispatcher also **always emits the named per-objective channel keys** — GDPO and goal-cond depend on them.

- **`AugmentedDataset` is only loaded when `data.custom_cls.path` is set.** Plain GRPO/GDPO/MaxRL use veRL's stock `RLHFDataset` and never see the multi-answer rewrite. `train.sh` sets `custom_cls` conditionally — only if `MULTI_SOL_DOM` or `GOAL_COND_DOM` is non-empty (i.e. `METHOD` is multi-sol or goal-cond).

- **Pareto / diversity metrics are computed in the task's `score_multi`** (maze calls the shared `diversity` / `pareto_count` / `feature_spread` helpers from `vpo/utils/eval_metrics.py`), not in `vpo_advantage`. The advantage estimator's own diagnostics (`vpo/pool_size_mean`, `vpo/own_pool_expected_max_mean`, …) are different.

## Parameters

All env-var-controlled tunables.

### Algorithm

| Env var | Default | Effect |
|---|---|---|
| `VPO_SAMPLER` | `naive` | `naive` (iid Dirichlet MC) or `sobol`/`sobol_qmc`/`qmc` (quasi-MC, ~2–6× lower advantage variance) |
| `VPO_ALPHA` | `1.0` | Dirichlet concentration. `α<1` sharpens to corners (per-objective extremists). `α>1` softens to centroid. |

### Top-level launcher (`train.sh`)

| Env var | Default | Values |
|---|---|---|
| `METHOD` | `grpo` | `grpo` \| `gdpo` \| `maxrl` \| `multi_rlvr` \| **`vpo`** \| `goal_cond` |
| `TASK` | `maze` | `maze` \| `musique` \| `eureqa` \| `tool` \| `lcb` |
| `MODEL` | `Qwen/Qwen3-4B` | any HF model id |
| `EPOCHS` | `50` | int |
| `N_GPUS` | `4` | int |
| `SEED` | `0` | int — paper uses 0, 1, 2 |
| `TAG` | jobid/timestamp | freeform experiment tag |

### Per-task data dir overrides

| `MAZE_DATA` | `MUSIQUE_DATA` | `EUREQA_DATA` | `TOOL_DATA` | `LCB_DATA` |
|---|---|---|---|---|
| all default to `$HOME/data/<task>/`. | | | | |

### Cluster hygiene (auto-set by `train.sh`)

| Env var | What it controls |
|---|---|
| `RAY_TMPDIR` | per-job Ray socket dir |
| `VERL_ZMQ_DIR` | ZMQ IPC dir for actor↔rollout weight xfer |
| `HF_DATASETS_CACHE` | per-job datasets cache (concurrent jobs corrupt the shared one) |
| `MASTER_PORT` | TCPStore rendezvous port (randomized 29500-39499) |
| `WANDB_RUN_ID` / `WANDB_RESUME` / `WANDB_NAME` | W&B run identity (stable across preemption) |

### Hydra overrides

Any string after `KEY=VAL` env vars is forwarded to veRL verbatim, e.g.:

```bash
bash train_vpo.sh TASK=maze ++trainer.test_freq=10 ++actor_rollout_ref.actor.optim.lr=5e-7
```

Useful keys baked into `train.sh` you might want to override:

| Hydra key | Default in train.sh |
|---|---|
| `actor_rollout_ref.actor.optim.lr` | `1e-6` |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.50` |
| `actor_rollout_ref.rollout.val_kwargs.n` | `3` |
| `actor_rollout_ref.rollout.val_kwargs.temperature` | `1.0` |
| `trainer.save_freq` / `trainer.test_freq` | `10` / `50` |

## Where to look if X breaks

| Symptom | First place to check |
|---|---|
| `KeyError: 'sub_scores'` in ray_trainer | the task's `score_one`/`score_multi` isn't returning the right record schema |
| Advantage is always 0 | all rollouts in a group have identical `sub_scores` → z-norm denom → ε. Check parsing/dedup in `score_multi`. |
| `torch.cat` shape mismatch at collate | prompt rewrite produced over-long prompts; add `++data.truncation=left` |
| Multi-sol scores identical to single-sol | `multi_rewrite` not declared on your Task, or its `SubRewrite` regex doesn't match the prompt → no rewrite happens |
| `ValueError: Multi-solution rewrite not configured` | the task's `multi_rewrite` is `None` (e.g. lcb, whose multi prompt is baked at preprocess time) but a multi-sol run tried to rewrite at load time |
| Enum collision / aliasing at startup | new enum value collides with an existing one — pick a unique string |

`vpo/utils/vpo.py` is the only file you really need to read to understand the method. Everything else is engine ergonomics (`vpo/reward.py`, `vpo/augment.py`) and per-task plugins (`vpo_tasks/<name>.py`).
