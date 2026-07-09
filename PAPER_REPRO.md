# Reproducibility — VPO

Everything needed to reproduce every (method × task) experiment in the paper
with the code in this repository. Hyperparameters were recorded from the Hydra
config dumps of the paper's production training jobs (snapshot 2026-05-05) and
cross-checked against `train.sh`, which reproduces them; values that
`train.sh` sets automatically are marked *(derived)*.

For every task we list:
- dataset (source, splits, prompt-length budget)
- reward vector (every dimension: meaning, range, direction, gating)
- scalar aggregation rule (the GRPO `score`)
- the paper-run configuration and the launcher command that reproduces it

Result tables and figures live in the paper, not here — this form is about
*configs*.

---

## 1. Methods

"Method" is the `METHOD=` value passed to `train.sh`; it maps 1-1 to veRL's
`algorithm.adv_estimator` except `goal_cond`, which keeps the `grpo` estimator
and adds prompt-side weight injection.

| Method (CLI) | Advantage estimator | Reward shape per rollout | Solutions per rollout | Notes |
|---|---|---|---|---|
| **`vpo` (ours)** | `vpo` | (m, k) reward matrix | m (multi-sol) | Own-pool `E_w[max_s w·r_s]`; no peer coupling. `vpo/utils/vpo.py::vpo_advantage`. Single-solution datasets use the `vpo_single` fallback (wraps the flat k-vector as a 1×k matrix). |
| `multi_rlvr` | `multi_rlvr` | (m, k) → scalar | m (multi-sol) | Same multi-answer prompt rewrite as VPO, but the set is scored with the same fixed GRPO scalar (the per-solution scalar summed over the m solutions) — no stochastic scalarization. |
| `gdpo` | `gdpo` | per-objective scalars (`gdpo_reward_keys`) | 1 | Per-dimension GRPO advantage, averaged across keys. `train.sh` derives `++algorithm.gdpo_reward_keys=[...]` from the task registry *(derived)*. |
| `grpo` | `grpo` | scalar (`score`) | 1 | Standard GRPO. Scalar = the task-defined weighted mean (per-task rules in §3). |
| `maxrl` | `maxrl` | scalar | 1 | GRPO with the std normalizer replaced by the group mean — `(r − mean)/(mean + ε)` (Tajwar & Zeng et al., arXiv:2602.02710). Implemented in the vendored veRL (`core_algos.py`). |
| `goal_cond` | `grpo` (+ `AugmentedDataset` weight injection) | scalar w·r | 1 | Per-row Dirichlet(1) weights are appended to the prompt and written into `ground_truth["weights"]`; the reward dispatcher scalarises against the named objective channels. The `random_reward_only` flag gives the random-w baseline (weights in the reward, not the prompt). |

The paper also reports two scalar search-aware baselines that the launcher
covers without a dedicated `METHOD` name:
- **Max-at-k** [Bagirov et al., 2025] — directly optimizes a best/max@k
  objective. Runnable as `bash train.sh METHOD=max_at_k TASK=<task> …`
  (forwarded to the vendored `max_at_k` advantage estimator; `k` defaults to
  `max(2, n//2)`, override with `++algorithm.max_at_k_k=<k>`).
- **Random-Weighting GRPO** — single-answer, but the scalarization weights are
  resampled `w ∼ Dir(α)` during training (weights in the reward, not the
  prompt). Runnable as the `goal_cond` plumbing with
  `++data.random_reward_only=True`.

**VPO weights.** For each group, scalarization weights are sampled i.i.d. from
a Dirichlet, `w ∼ Dir(α)` (`np.random.Generator.dirichlet`). `VPO_ALPHA` sets
the concentration; the paper uses **α = 1** (uniform on the simplex) for every
domain.

**Weight draws per group.** `n_selections=100` (the `vpo_advantage` default).

**Seeding.** `train.sh` exports `VPO_SEED=$SEED`, which seeds the process-wide
weight-sampling RNG (see §6).

---

## 2. Common training stack (constant across runs unless noted)

| Component | Setting |
|---|---|
| Framework | veRL, vendored at `verl/` with 3 patched files: `verl/verl/trainer/ppo/core_algos.py`, `verl/verl/trainer/ppo/ray_trainer.py`, `verl/verl/experimental/agent_loop/agent_loop.py`. Upstream pin in `REQUIRED_VERL.txt`. |
| Training engine | FSDP (`strategy=fsdp`), gradient checkpointing on, `use_torch_compile=True`, ref-model param offload, `clip_grad=1.0`. |
| Optimizer | AdamW, β=(0.9, 0.999), weight_decay=0.01, **lr = 1e-6, constant schedule** (no warmup). |
| Loss | PPO-clip, `clip_ratio=0.2` (low/high both 0.2), `clip_ratio_c=3.0`, `loss_agg_mode=token-mean`, `ppo_epochs=1`, `entropy_coeff=0`. |
| KL | `use_kl_loss=True`, `kl_loss_type=low_var_kl`, **kl_loss_coef = 1e-3**; `algorithm.use_kl_in_reward=False` (no in-reward KL penalty). |
| Rollout engine | vLLM 0.12.0 (pins torch 2.9.0), `tensor_model_parallel_size=1`, `gpu_memory_utilization=0.50`; **temperature 1.0, top-p 1.0** at training time. |
| In-loop validation | `val_kwargs.n=3, temperature=1.0, do_sample=True` every `test_freq=50` steps — for m=3 multi-sol methods this is one inference vs. three single-sol samples, the natural fairness cap. |
| Save cadence | `save_freq=10`, final FSDP checkpoint auto-merged to `actor/huggingface_merged/` for eval *(derived — `train.sh` runs `verl.model_merger` after training)*. |
| Hardware | 1 node, 4 × H100 80 GB, job-exclusive. |
| Logging | W&B project `vpo_<task>`, run name `<method>_<task>_seed<seed>_<tag>`; stable run id so preempted jobs re-attach *(derived)*. |
| Filtering / truncation | `data.filter_overlong_prompts=True`. Any `AugmentedDataset` run (multi-sol or goal-cond) adds `data.truncation=left` — the load-time rewrite can push past `max_prompt_length`, handled by the vendored left-truncation patch in `agent_loop.py`. |
| Chat template | `enable_thinking=False` everywhere — `train.sh` passes `++data.apply_chat_template_kwargs.enable_thinking=False`; eval scripts do the same. |

### Batch sizing *(derived — `train.sh`)*

| Configuration | TRAIN_BATCH | ROLLOUT_N | MINI_BATCH | MICRO_BATCH (per-GPU) |
|---|---|---|---|---|
| Qwen ≤ 4B (Maze, MuSiQue, ToolRL) | 128 | 8 | 64 | 8 |
| Qwen 7B/8B (EUREQA) | 64 | 8 | 32 | 2 |

Constraints: `mini_batch` divides `train_batch × rollout.n`;
`micro_batch × n_gpus` divides `mini_batch`. NB: the launcher's model-size
detection only matches `-7B` names — **the 8B EUREQA runs pass their batch
overrides explicitly** (shown in §3.3).

### AugmentedDataset (`vpo/augment.py`)

Wraps veRL's `RLHFDataset` and rewrites prompts at `__getitem__` time (no
separate preprocessing). Hydra flags *(set automatically by `train.sh`)*:

- `++data.multi_solution_domain=<task>` + `++data.num_solutions=m` — replaces
  the task's single-output instruction with one asking for m solutions in
  numbered tags (`<route_i>`, `<response_i>`, …; specs in `vpo_tasks/<task>.py`).
- `++data.goal_cond_domain=<task>` — appends a Dir(1,…,1)-sampled weight
  vector to the prompt and writes it into `ground_truth["weights"]`; the
  reward returns `score = w · channels`.

The two modes are **mutually exclusive** (enforced at construction; the reward
scores one mode per row). Activated via `++data.custom_cls.path=vpo/augment.py
++data.custom_cls.name=AugmentedDataset`.

---

## 3. Tasks (4)

Reward-vector conventions:
- "Range" is the closed interval each component falls in. Every dim in every
  task is non-negative, higher = better.
- No task gates components on each other, **except** maze's reach-the-exit
  gate (noted below).
- "Scalar (GRPO)" is the explicit `score` formula used by scalar methods.
  VPO/GDPO consume the full vector.
- "Empty parse" is the score when the response cannot be parsed.

---

### 3.1 Maze — designed-conflict grid navigation

**Source.** Procedurally generated, fully self-contained —
`data/preprocess_maze.py` (no external dataset). Defaults: train=1000,
test=100, generation seeds 42 / 4242 (`--train_size/--test_size/--*_seed` to
change).

**Geometry.** 9×9 Prim's maze with 18–28 extra
cycles opened. S and E sit on opposite corners of a random diagonal; the two
perpendicular corners hold a **gold zone** and a **diamond zone** (3–5 items
each, radius-2); the center cell is a **bonus** tile — advertised in the prompt
as a score multiplier but a **distractor with no reward effect**; 3–5 lava
tiles in the interior. The step budget is `max(path-via-gold,
path-via-diamond) + 7` and instances are filtered so that a safe path exists
within budget, one corner detour fits, but **visiting both
corners never fits** — the objectives conflict by construction.

**Reward vector (k = 4)** — `vpo_tasks/maze.py`. Linear item/safety fractions
`r = (1, g, d, ℓ)`:

| Dim | Name | Range | Direction | Definition |
|---|---|---|---|---|
| 0 | `completion` | {0, 1} | ↑ | Trajectory ends on the exit cell E (within budget). |
| 1 | `gold` | [0, 1] | ↑ | `#gold collected / #gold`. |
| 2 | `diamond` | [0, 1] | ↑ | `#diamond collected / #diamond`. |
| 3 | `avoid_lava` | [0, 1] | ↑ | `1 − #lava stepped / #lava`. |

**Gating:** the whole vector is zero unless the trajectory reaches E; items
only count if collected before E.

**Scalar (GRPO)** = mean of the 4 dims (uniform weights).

**Empty parse** = all zeros.

**Paper-run configuration.**

| Knob | Value |
|---|---|
| Model | `Qwen/Qwen3-4B` |
| `max_prompt_length` / `max_response_length` | 512/512 (single-sol); 768/1024 (multi-sol); 768/512 (goal-cond) *(derived)* |
| `num_solutions` (multi-sol) | 3, wrapped in `<route_i>...</route_i>` |
| Batch sizing | default preset (128 / 8 / 64 / 8) |

```bash
bash train.sh METHOD=vpo TASK=maze MODEL=Qwen/Qwen3-4B EPOCHS=2 SEED=0
```

---

### 3.2 MuSiQue — multi-hop QA

**Source.** MuSiQue-Ans via HuggingFace `dgslibisey/MuSiQue`;
`data/preprocess_musique.py` uses the full train/validation splits
(≈19.9k train / ≈2.4k validation; `--max_train/--max_test` to subsample).

**Eval split (important — the released default is *not* stratified).** The paper
reports best@k on a **300-question hop-stratified** held-out split. The released
`eval/eval_musique.py --num-examples 300`, however, takes the **first 300 rows of
`test.parquet`** (the validation split in its original HuggingFace order) — which
is skewed toward 2-hop questions, not stratified. To reproduce the paper's split,
rebuild `test.parquet` so the 300 questions are balanced across hop counts before
running the eval. Each row carries its hop count in `extra_info.num_hops` (also
encoded in the MuSiQue `id` prefix: `2hop__…`, `3hop__…`, `4hop__…`):

```python
import datasets

ds = datasets.Dataset.from_parquet("~/data/musique/test.parquet")

PER_HOP, SEED = 100, 42          # 100 each of 2/3/4-hop -> 300 total
parts = []
for h in (2, 3, 4):
    bucket = ds.filter(lambda r: r["extra_info"]["num_hops"] == h)
    bucket = bucket.shuffle(seed=SEED).select(range(min(PER_HOP, len(bucket))))
    parts.append(bucket)

datasets.concatenate_datasets(parts).shuffle(seed=SEED).to_parquet(
    "~/data/musique/test.parquet")   # overwrite so the eval loads the stratified set
```

Then run `eval/eval_musique.py --num-examples 300` as usual; the `select(range(300))`
now returns the full stratified set. (Stratification will be folded into
`preprocess_musique.py` so it becomes the default.)

**Reward vector (k = 5)** — `vpo_tasks/musique.py`:

| Dim | Name | Range | Direction | Definition |
|---|---|---|---|---|
| 0–3 | `hop_1..hop_4` | {0, 1} | ↑ | `<support>` cites the gold paragraph for reasoning step i. Hops beyond the question's hop count default to 1.0 (a 2-hop question trivially satisfies hops 3–4). |
| 4 | `answer_f1` | [0, 1] | ↑ | Word-level F1 between `<answer>` and gold (best over aliases), SQuAD-style normalization (lowercase, drop articles, drop punctuation). |

Citations are truncated to the first 4 distinct paragraph indices (MAX_HOPS=4)
to prevent the dump-all-indices exploit.

**Scalar (GRPO)** = `(Σ hop_i + 3·answer_f1) / 7` — answer_f1 weight 3, each
hop weight 1.

**Empty parse:** a response with **no parseable `<answer>`** scores all zeros
(citing support without answering earns nothing — consistent across
single-sol training, multi-sol training, and eval).

**Paper-run configuration.**

| Knob | Value |
|---|---|
| Model | `Qwen/Qwen3-1.7B`, 1 epoch |
| `max_prompt_length` / `max_response_length` | 4096/512 (single-sol); 5120/1024 (multi-sol); 5120/512 (goal-cond) *(derived; prompt tokens median ≈2500, p90 ≈3400 — the budget filters ≲1.2% of rows)* |
| `num_solutions` | 3, in `<response_i>` tags |

```bash
bash train.sh METHOD=vpo TASK=musique MODEL=Qwen/Qwen3-1.7B EPOCHS=1 SEED=0
```

---

### 3.3 EUREQA — multi-hop entity back-chaining

**Source.** `vincentleebang/EUREQA` (HuggingFace);
`data/preprocess_eureqa.py`. **The paper uses `--mode mixed_split
--mixed_split_seed 0`** (note: the CLI default is `normal_to_hard` — pass the
flag): `questions_hard_5` (1363 rows) is shuffled with seed 0 and split in
half; train = all of `questions_normal_5` (428) + the first 681 hard rows =
**1109**; test = the remaining **682** hard rows. The test set is therefore a
held-out random half of `hard_5` (in-distribution w.r.t. the harder half of
training), not an easy→hard generalization split.

**Reward vector (k = 5)** — `vpo_tasks/eureqa.py`:

| Dim | Name | Range | Direction | Definition |
|---|---|---|---|---|
| 0 | `entity_A` | {0, 1} | ↑ | EM (after Wikipedia normalization) on the answered entity — the question's answer. |
| 1–4 | `entity_B..entity_E` | {0, 1} | ↑ | EM on each masked intermediate chain entity. |

Entity F is shown verbatim in the narrative and not scored. EM normalization:
`_`→space, lowercase, strip articles, drop punctuation, collapse whitespace —
`Robert_Rodriguez` ≡ `the robert rodriguez!`.

**Scalar (GRPO)** = mean of the 5 dims.

**Empty parse** = all zeros.

**Paper-run configuration.**

| Knob | Value |
|---|---|
| Model | `Qwen/Qwen3-8B`, **6 epochs** (≈18 steps/epoch at batch 64 → ~108 steps) |
| `max_prompt_length` / `max_response_length` | 1024/512 (single-sol); 1024/2048 (multi-sol) *(derived)* |
| `num_solutions` | 3 chains, in `<response_i>` tags |
| Batch sizing | 64 / 8 / 32 / 2 (passed explicitly — the launcher's auto-preset only matches `-7B` names, and EUREQA uses Qwen3-8B) |

```bash
bash train.sh METHOD=vpo TASK=eureqa MODEL=Qwen/Qwen3-8B EPOCHS=6 SEED=0 \
    ++data.train_batch_size=64 \
    ++actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
```

---

### 3.4 Tool — function calling (ToolRL)

**Source.** `qiancheng0/ToolRL` rlla_4k; `data/preprocess_tool.py`. Splits:
train=3920, test=80.

**Reward vector (k = 4)** — `vpo_tasks/tool.py`:

| Dim | Name | Range | Direction | Definition |
|---|---|---|---|---|
| 0 | `format` | {0, 1} | ↑ | Structure well-formed: `<think>` present and the `<tool_call>` section matches gold's structure (calls present iff gold has calls). |
| 1 | `tool_name` | [0, 1] | ↑ | Multiset F1 between predicted vs gold tool-call names. |
| 2 | `arg_key` | [0, 1] | ↑ | Mean set-F1 over parameter keys across **aligned** calls (greedy match by name, then key overlap). |
| 3 | `arg_value` | [0, 1] | ↑ | Mean per-key value match across aligned calls (token F1 for free text; EM for numerics/bools). Unmatched calls contribute 0. |

**Scalar (GRPO)** = mean of the 4 dims.

**Empty parse** = all zeros.

**Paper-run configuration.**

| Knob | Value |
|---|---|
| Model | `Qwen/Qwen3-1.7B`, **20 epochs** (≈31 steps/epoch → ~620 steps) |
| `max_prompt_length` / `max_response_length` | 4096/1024 (single-sol); 4352/2048 (multi-sol); 4352/1024 (goal-cond) — the 4352 prompt budget leaves room for the rewrite/weight suffix *(derived)* |
| `num_solutions` | 3, in `<response_i>` tags (`AppendRewrite` — the format spec lives in the system message) |

```bash
bash train.sh METHOD=vpo TASK=tool MODEL=Qwen/Qwen3-1.7B EPOCHS=20 SEED=0
```

---

## 4. Reward summary

| Task | k | Component ranges | Gating | GRPO scalar |
|---|---|---|---|---|
| Maze | 4 | {0,1} × [0,1]³ | zero vector unless E reached | mean |
| MuSiQue | 5 | {0,1}⁴ × [0,1] | zero vector if no `<answer>` | (Σ hops + 3·answer_f1) / 7 |
| EUREQA | 5 | {0,1}⁵ | none | mean |
| Tool | 4 | {0,1} × [0,1]³ | none | mean |

---

## 5. Where to look in the code

- Reward dispatcher (single / multi / goal-cond inferred per row): `vpo/reward.py`.
- Per-task objectives + scoring + prompts + preprocess + eval: `vpo_tasks/<task>.py`.
- Advantage estimator (own-pool VPO, Dirichlet weight sampler): `vpo/utils/vpo.py`; the `gdpo`/`maxrl`/`max_at_k` estimators live in the vendored `verl/verl/trainer/ppo/core_algos.py`.
- Multi-solution + goal-cond prompt rewriting: `vpo/augment.py`.
- Launcher: `train.sh` (unified; `METHOD=` selects the algorithm).
- Preprocessing: `data/preprocess_<task>.py`. Held-out eval: `eval/eval_<task>.py`.
- Shared, unit-tested eval metrics: `vpo/utils/eval_metrics.py` (`tests/`).

---

## 6. Random seeds, evaluation sampling, and determinism

`SEED` (default 0) flows three ways from `train.sh`:
1. `data.seed=$SEED` — veRL data ordering.
2. `VPO_SEED=$SEED` — seeds the process-persistent RNG behind the VPO weight
   sampler (`vpo/utils/vpo.py`), so the scalarization stream is reproducible.
3. Goal-cond weight draws are keyed by `(seed, row index)` — the same prompt
   sees the same weights every epoch and regardless of DataLoader sharding.

**Training.** The paper reports **one training run per (method × task) cell**
— there is no multi-seed training average. Train-time vLLM sampling is
non-deterministic (`seed=None`), so a re-run differs slightly even at fixed
`SEED`.

**Final-evaluation sampling.** Reported best@k comes from a separate
post-training eval pass with stochastic decoding (so the pool is
non-degenerate): `top_k = −1` everywhere, with per-domain settings applied
uniformly across methods — Maze and MuSiQue at **temperature 0.7, top-p 1.0**;
EUREQA and ToolRL at **temperature 0.7, top-p 0.95**. Because the EUREQA and
ToolRL held-out sets are small, their best@k is **averaged over 4 evaluation
seeds**. The held-out eval scripts take an explicit `--seed` and are
deterministic given the merged checkpoint.
