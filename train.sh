#!/bin/bash
# vpo: unified launcher.
#
# Usage:
#   METHOD=vpo TASK=maze bash train.sh
#   bash train.sh METHOD=vpo TASK=maze MODEL=Qwen/Qwen3-4B EPOCHS=50
#
# Env knobs (all optional, sensible defaults):
#   METHOD   grpo | gdpo | maxrl | multi_rlvr | vpo | goal_cond
#            (also accepts vpo_single — the m=1 VPO estimator, used for lcb)
#   TASK     maze | musique | eureqa | tool | lcb
#   MODEL    HF model id  (default Qwen/Qwen3-4B)
#   EPOCHS   total epochs  (default 50)
#   N_GPUS   gpus per node  (default 4)
#   TAG      experiment tag  (default <jobid|timestamp>)
#   SEED     seed  (default 0)
#
# Override any Hydra arg by appending it after the env vars:
#   bash train.sh METHOD=vpo TASK=maze ++trainer.test_freq=10

set -euo pipefail

# Run from repo root regardless of caller's cwd.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

# Ensure the vendored, patched veRL and the vpo package win on the import path,
# even if a different veRL happens to be pip-installed in the active env. Points
# at the same patched code `cd verl && pip install -e .` would install.
export PYTHONPATH="$PWD/verl:$PWD${PYTHONPATH:+:$PYTHONPATH}"

# ─── Parse positional KEY=VALUE env vars ─────────────────────────────────────
VERL_OVERRIDES=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" =~ ^[A-Z_][A-Z0-9_]*= ]]; then
        export "$1"; shift
    else
        VERL_OVERRIDES+=("$1"); shift
    fi
done

# ─── Defaults ────────────────────────────────────────────────────────────────
METHOD="${METHOD:-grpo}"
TASK="${TASK:-maze}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
EPOCHS="${EPOCHS:-50}"
N_GPUS="${N_GPUS:-4}"
SEED="${SEED:-0}"
# Seed the VPO weight sampler (vpo/utils/vpo.py) and goal-cond weight draws
# (vpo/augment.py) so advantages/prompts are reproducible for a given run.
export VPO_SEED="$SEED"
JOB_ID="${SLURM_JOB_ID:-${LSB_JOBID:-$$}}"
TAG="${TAG:-${JOB_ID}}"
EXPERIMENT="${METHOD}_${TASK}_seed${SEED}_${TAG}"

# Method-name → veRL adv_estimator (goal_cond IS scalar GRPO with weights in prompt).
ADV_ESTIMATOR="$METHOD"
[[ "$METHOD" == "goal_cond" ]] && ADV_ESTIMATOR="grpo"

# ─── Cluster hygiene ─────────────────────────────────────────────────────────
# Per-job Ray + ZMQ + HF datasets dirs so concurrent jobs on the same node
# don't collide on /tmp/ray/sockets/ (AF_UNIX 107-byte path limit), TCPStore
# port 29500, or cache-<hash>.arrow rewrites in the datasets cache.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER}-${JOB_ID}}"
export VERL_ZMQ_DIR="${VERL_ZMQ_DIR:-$RAY_TMPDIR}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 10000))}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf-ds-${USER}-${JOB_ID}}"
mkdir -p "$RAY_TMPDIR" "$HF_DATASETS_CACHE" logs

# Stable W&B run id so a preempted/requeued job re-attaches to the same run.
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(printf '%s' "$EXPERIMENT" | md5sum | cut -c1-32)}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_NAME="${WANDB_NAME:-$EXPERIMENT}"

# ─── Per-task config ─────────────────────────────────────────────────────────
# One reward dispatcher serves every (task × method): vpo/reward.py infers
# single / multi / goal-cond mode from the row itself (see vpo/reward.py), so
# there is no per-(task,method) reward-file table to maintain here.
REWARD_FN="vpo/reward.py"
is_multi() { [[ "$1" == "multi_rlvr" || "$1" == "vpo" ]]; }

MULTI_SOL_DOM=""; GOAL_COND_DOM=""; NUM_SOLUTIONS=""

case "$TASK" in
    maze)
        DATA_DIR="${MAZE_DATA:-$HOME/data/maze}"
        if [[ "$METHOD" == "goal_cond" ]]; then
            PROMPT_LEN=768; RESPONSE_LEN=512; GOAL_COND_DOM="maze"
        elif is_multi "$METHOD"; then
            PROMPT_LEN=768; RESPONSE_LEN=1024; MULTI_SOL_DOM="maze"; NUM_SOLUTIONS=3
        else
            PROMPT_LEN=512; RESPONSE_LEN=512
        fi ;;
    musique)
        DATA_DIR="${MUSIQUE_DATA:-$HOME/data/musique}"
        if [[ "$METHOD" == "goal_cond" ]]; then
            PROMPT_LEN=5120; RESPONSE_LEN=512; GOAL_COND_DOM="musique"
        elif is_multi "$METHOD"; then
            PROMPT_LEN=5120; RESPONSE_LEN=1024; MULTI_SOL_DOM="musique"; NUM_SOLUTIONS=3
        else
            PROMPT_LEN=4096; RESPONSE_LEN=512
        fi ;;
    eureqa)
        DATA_DIR="${EUREQA_DATA:-$HOME/data/eureqa}"
        if [[ "$METHOD" == "goal_cond" ]]; then
            PROMPT_LEN=1024; RESPONSE_LEN=512; GOAL_COND_DOM="eureqa"
        elif is_multi "$METHOD"; then
            PROMPT_LEN=1024; RESPONSE_LEN=2048; MULTI_SOL_DOM="eureqa"; NUM_SOLUTIONS=5
        else
            PROMPT_LEN=1024; RESPONSE_LEN=512
        fi ;;
    tool)
        DATA_DIR="${TOOL_DATA:-$HOME/data/toolrl}"
        if [[ "$METHOD" == "goal_cond" ]]; then
            PROMPT_LEN=4352; RESPONSE_LEN=1024; GOAL_COND_DOM="tool"
        elif is_multi "$METHOD"; then
            PROMPT_LEN=4352; RESPONSE_LEN=2048; MULTI_SOL_DOM="tool"; NUM_SOLUTIONS=3
        else
            PROMPT_LEN=4096; RESPONSE_LEN=1024
        fi ;;
    lcb)
        DATA_DIR="${LCB_DATA:-$HOME/data/lcb}"
        if is_multi "$METHOD"; then
            # lcb has no runtime multi-solution rewrite (its multi prompts are
            # baked at preprocess time), so AugmentedDataset can't serve it.
            # Fail clearly here instead of crashing deep in the data loader.
            echo "ERROR: lcb multi-solution training (METHOD=$METHOD) is not wired." >&2
            echo "       lcb has no load-time multi-solution rewrite; bake the multi" >&2
            echo "       prompts at preprocess time and train on that dataset, or use a" >&2
            echo "       single-solution method (grpo | gdpo | maxrl | vpo_single)." >&2
            exit 1
        elif [[ "$METHOD" == "goal_cond" ]]; then
            # Without GOAL_COND_DOM, AugmentedDataset never runs, no weights are
            # injected, and the run silently trains plain GRPO under a
            # goal_cond_lcb label. Fail loudly until lcb goal-cond is wired.
            echo "ERROR: lcb goal-conditioned training is not wired (no GOAL_COND_DOM" >&2
            echo "       for lcb); the run would silently train vanilla GRPO." >&2
            exit 1
        else
            PROMPT_LEN=1280; RESPONSE_LEN=2048
        fi ;;
    *)
        echo "Unknown TASK '$TASK' (use: maze | musique | eureqa | tool | lcb)" >&2
        exit 1 ;;
esac

# ─── Batch sizing ────────────────────────────────────────────────────────────
# Defaults assume a 4×H100 80GB exclusive node. N_GPUS=1 switches to a preset
# sized for a single 24-32 GB card with a ≤1B model (constraints: mini_batch
# divides train_batch×rollout_n; micro_batch×n_gpus divides mini_batch). For
# bigger models on one GPU, shrink ++data.train_batch_size / rollout.n further.
if [[ "$N_GPUS" == "1" ]]; then
    TRAIN_BATCH=32; ROLLOUT_N=4; MINI_BATCH=16; MICRO_BATCH=2
else
    case "$MODEL" in
        *Qwen*-7B*|*qwen*-7b*|*Llama*-7b*) TRAIN_BATCH=64;  ROLLOUT_N=8; MINI_BATCH=32; MICRO_BATCH=2 ;;
        *)                                  TRAIN_BATCH=128; ROLLOUT_N=8; MINI_BATCH=64; MICRO_BATCH=8 ;;
    esac
fi

# ─── AugmentedDataset plumbing (multi-sol and/or goal-cond) ──────────────────
EXTRA=()
if [[ -n "$MULTI_SOL_DOM" || -n "$GOAL_COND_DOM" ]]; then
    EXTRA+=(
        "++data.custom_cls.path=vpo/augment.py"
        "++data.custom_cls.name=AugmentedDataset"
        "++data.truncation=left"
    )
    [[ -n "$MULTI_SOL_DOM" ]] && EXTRA+=("++data.multi_solution_domain=$MULTI_SOL_DOM"
                                         "++data.num_solutions=$NUM_SOLUTIONS")
    [[ -n "$GOAL_COND_DOM" ]] && EXTRA+=("++data.goal_cond_domain=$GOAL_COND_DOM")
fi
# GDPO needs the per-objective reward channels as named keys; derive them from
# the task registry so changing a task's objectives never requires editing this.
if [[ "$METHOD" == "gdpo" ]]; then
    GDPO_KEYS=$(python3 -c "import vpo_tasks; from vpo.task import resolve; print('['+','.join(k for k,_ in resolve('$TASK').objectives)+']')")
    EXTRA+=("++algorithm.gdpo_reward_keys=$GDPO_KEYS")
fi

# ─── Startup banner ──────────────────────────────────────────────────────────
echo "════════════════════════════════════════════"
echo "  $METHOD on $TASK   (seed=$SEED)"
echo "  model:      $MODEL"
echo "  data:       $DATA_DIR"
echo "  reward:     $REWARD_FN"
echo "  adv:        $ADV_ESTIMATOR"
echo "  prompt/resp:$PROMPT_LEN / $RESPONSE_LEN tokens"
[[ -n "$NUM_SOLUTIONS" ]] && echo "  m:          $NUM_SOLUTIONS"
echo "  epochs:     $EPOCHS"
echo "  gpus:       $N_GPUS"
echo "  tag:        $TAG    (wandb run id: $WANDB_RUN_ID)"
echo "════════════════════════════════════════════"

# ─── Train ───────────────────────────────────────────────────────────────────
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="$ADV_ESTIMATOR" \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.max_prompt_length="$PROMPT_LEN" \
    data.max_response_length="$RESPONSE_LEN" \
    data.filter_overlong_prompts=True \
    ++data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    data.seed="$SEED" \
    actor_rollout_ref.rollout.val_kwargs.n=3 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$REWARD_FN" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name="vpo_${TASK}" \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=50 \
    trainer.total_epochs="$EPOCHS" \
    "${EXTRA[@]}" \
    "${VERL_OVERRIDES[@]}" \
    2>&1 | tee "logs/${EXPERIMENT}.log"

# ─── Merge final FSDP checkpoint → HF format for held-out eval ───────────────
# veRL saves sharded FSDP checkpoints; the eval/ scripts want a standard HF dir.
# Merge the latest global_step into actor/huggingface_merged/. Non-fatal: a
# failure here never loses the trained checkpoint (eval can load the shards too).
CKPT_ROOT="checkpoints/vpo_${TASK}/${EXPERIMENT}"
LAST_STEP="$(ls -d "$CKPT_ROOT"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1 || true)"
if [[ -n "$LAST_STEP" && -d "$LAST_STEP/actor" ]]; then
    echo "Merging $LAST_STEP/actor → huggingface_merged/ ..."
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$LAST_STEP/actor" \
        --target_dir "$LAST_STEP/actor/huggingface_merged" \
        || echo "WARNING: model merge failed; eval can still load the raw FSDP checkpoint."
else
    echo "No checkpoint found under $CKPT_ROOT (save_freq=-1?); skipping HF merge."
fi
