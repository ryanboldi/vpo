# Reproducibility

**The full per-experiment form lives at [`PAPER_REPRO.md`](../PAPER_REPRO.md)**
— per-task datasets and splits, every reward dimension, scalar rules, the
common training stack, paper-run configurations, and the exact launcher
command for each (method × task). This page covers the part PAPER_REPRO
doesn't: the vendored veRL and how to rebuild it.

## Software stack

| Component | Pin |
|---|---|
| Python | ≥ 3.11 |
| torch | 2.9.0 (pinned by vLLM) |
| vLLM | 0.12.0 (newer breaks this veRL) |
| veRL | vendored at `verl/` — upstream `volcengine/verl` @ `d5c5daa84290b445f071fb812a1731a7325f350f` (0.8.0.dev, main HEAD as of 2026-04-03) + the 3-file VPO patch |

The pin and re-derivation instructions are in [`REQUIRED_VERL.txt`](../REQUIRED_VERL.txt).

## The 3-file veRL patch

The vendored copy already has the patch applied; the README section
["VPO is a 3-file patch to veRL"](../README.md#vpo-is-a-3-file-patch-to-verl)
documents each change. The touched files:

1. `verl/verl/trainer/ppo/core_algos.py` — `VPO` / `VPO_SINGLE` / `MAXRL`
   advantage-estimator enum values (+ the MaxRL estimator inline).
2. `verl/verl/trainer/ppo/ray_trainer.py` — dispatch: the two branches that
   feed `sub_scores` + `uid` into `vpo.utils.vpo.vpo_advantage` and route the
   diagnostics dict (`meta_info["_vpo_diag"]`) to wandb.
3. `verl/verl/experimental/agent_loop/agent_loop.py` — left-truncate prompts
   that `AugmentedDataset`'s load-time rewrite pushed past
   `max_prompt_length`.

To port the patch to a newer veRL: check out the upstream SHA above, diff the
three files against `verl/`, and re-apply the hunks to your target version
(re-test the left-truncation hunk in particular — the agent-loop code moves
between veRL releases).

## Rebuilding from scratch

```bash
uv venv .venv && source .venv/bin/activate
cd verl && pip install -e . && cd ..   # the vendored, patched veRL
pip install -e .                        # vpo + vpo_tasks
python data/preprocess_<task>.py ...    # per-task flags in PAPER_REPRO §3
bash train.sh METHOD=<method> TASK=<task> ...  # exact commands in PAPER_REPRO §3
```

## Seeds and aggregation

`SEED` drives veRL data ordering (`data.seed`), the VPO scalarization sampler
(`VPO_SEED`), and the goal-conditioning weight draws (keyed per `(seed, row)`)
— see PAPER_REPRO §6. Train-time vLLM sampling is non-deterministic, so the
paper reports mean ± stderr over `SEED=0,1,2` for every (method × task).

## Release tag ↔ arXiv

The repository state accompanying arXiv v1 is the initial release tag
(`v1.0.0`); subsequent code changes that affect any reported number will be
noted in the release notes of later tags.
