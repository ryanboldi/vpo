"""ToolRL task: function calling.

The reward is a 4-dim vector ``[format, tool_name, arg_key, arg_value]``, each
in [0, 1]: ``format`` is binary (well-formed structure); ``tool_name`` is
multiset F1 over tool names; ``arg_key`` is set-F1 of argument keys; and
``arg_value`` matches argument values, averaged over aligned calls.

Contains the structured-response and tool-call parsers, the scoring helpers,
the multi-attempt prompt rewrite, dataset preprocessing (``preprocess``), and
the held-out eval (``evaluate``).
"""

from __future__ import annotations

import json
import re

from vpo.task import AppendRewrite, Task, dedup_by_vector, pad_matrix, register
from vpo.utils.parse_solutions import extract_numbered_tags
from vpo.utils.text_metrics import multiset_f1, normalize_text, set_f1, token_f1

NUM_OBJECTIVES = 4
SUB_OBJECTIVE_NAMES = ["format", "tool_name", "arg_key", "arg_value"]
ZERO_SCORES = [0.0] * NUM_OBJECTIVES

_THINK_PAT = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_PAT = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_RESPONSE_PAT = re.compile(r"<response>(.*?)</response>", re.DOTALL)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_json_object_stream(text: str) -> list[dict]:
    out = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        if text[i] != "{":
            break
        depth, in_str, escape, start = 0, False, False, i
        while i < n:
            c = text[i]
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = not in_str
            elif not in_str:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
            i += 1
        if depth != 0:
            break
        try:
            obj = json.loads(text[start:i])
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            out.append(obj)
    return out


def parse_tool_calls(tool_call_block: str) -> list[dict]:
    if not tool_call_block or not tool_call_block.strip():
        return []
    text = tool_call_block.strip()
    if text.startswith("["):
        # Models commonly emit a JSON array of calls instead of the prompted
        # one-object-per-line stream; accept it rather than scoring a
        # semantically correct call list as the zero vector. (Gold answers
        # are never array-framed.)
        try:
            arr = json.loads(text)
        except json.JSONDecodeError:
            arr = []
        objs = [o for o in arr if isinstance(o, dict)] if isinstance(arr, list) else []
    else:
        objs = _parse_json_object_stream(text)
    calls = []
    for obj in objs:
        name = obj.get("name", "")
        params = obj.get("parameters", obj.get("arguments", {}))
        if not isinstance(params, dict):
            params = {}
        calls.append({"name": str(name), "parameters": params})
    return calls


def parse_multi_responses(text: str, m: int) -> list[str]:
    """Extract m <response_i>...</response_i> blocks (i=1..m)."""
    return extract_numbered_tags(text, m)


def parse_structured_response(text: str) -> dict:
    if not isinstance(text, str):
        text = str(text or "")
    think_m = _THINK_PAT.search(text)
    tc_m = _TOOL_CALL_PAT.search(text)
    resp_m = _RESPONSE_PAT.search(text)
    tool_calls = []
    tc_body_nonempty = False
    if tc_m:
        body = tc_m.group(1).strip()
        tc_body_nonempty = bool(body)
        tool_calls = parse_tool_calls(body)
    return {
        "think": think_m.group(1).strip() if think_m else "",
        "tool_calls": tool_calls,
        "response": resp_m.group(1).strip() if resp_m else "",
        "has_think": bool(think_m),
        "has_tool_call_tag": bool(tc_m),
        "tool_call_body_nonempty": tc_body_nonempty,
        "has_response": bool(resp_m),
    }


def parse_ground_truth_string(gt: str) -> dict:
    p = parse_structured_response(gt)
    return {"tool_calls": p["tool_calls"], "response_text": p["response"], "think_text": p["think"]}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


# Shared implementations (vpo/utils/text_metrics.py); aliased so the scoring
# code below keeps its original vocabulary.
_normalize = normalize_text
_multiset_f1 = multiset_f1
_set_f1 = set_f1
_token_f1 = token_f1


def _value_match(pred, gold) -> float:
    if isinstance(gold, bool) or isinstance(pred, bool):
        return float(bool(gold) == bool(pred))
    if isinstance(gold, (int, float)):
        if isinstance(pred, (int, float)):
            return float(gold == pred)
        try:
            return float(abs(float(str(pred)) - float(gold)) < 1e-9)
        except (ValueError, TypeError):
            return float(_normalize(pred) == _normalize(gold))
    if isinstance(gold, list):
        pred_l = pred if isinstance(pred, list) else [pred]
        return _multiset_f1([_normalize(x) for x in pred_l], [_normalize(x) for x in gold])
    if isinstance(gold, dict):
        if not isinstance(pred, dict):
            return 0.0
        return _set_f1(set(pred.keys()), set(gold.keys()))
    if isinstance(gold, str):
        if not isinstance(pred, str):
            return float(_normalize(pred) == _normalize(gold))
        return _token_f1(pred, gold)
    return float(_normalize(pred) == _normalize(gold))


def _align_calls(pred, gold):
    used = [False] * len(pred)
    pairs = []
    for g in gold:
        gname = _normalize(g.get("name", ""))
        gkeys = set((g.get("parameters") or {}).keys())
        best_j, best_score = -1, -1
        for j, p in enumerate(pred):
            if used[j]:
                continue
            pname = _normalize(p.get("name", ""))
            pkeys = set((p.get("parameters") or {}).keys())
            score = (100 if (pname and pname == gname) else 0) + len(pkeys & gkeys)
            if score > best_score:
                best_score, best_j = score, j
        if best_j >= 0:
            used[best_j] = True
            pairs.append((g, pred[best_j]))
        else:
            pairs.append((g, None))
    for j, p in enumerate(pred):
        if not used[j]:
            pairs.append((None, p))
    return pairs


def _score_calls(pred, gold):
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    name_f1 = _multiset_f1(
        [_normalize(c.get("name", "")) for c in pred],
        [_normalize(c.get("name", "")) for c in gold],
    )
    pairs = _align_calls(pred, gold)
    key_scores, val_scores = [], []
    for g, p in pairs:
        if g is None or p is None:
            key_scores.append(0.0)
            val_scores.append(0.0)
            continue
        gp = g.get("parameters") or {}
        pp = p.get("parameters") or {}
        gk, pk = set(gp.keys()), set(pp.keys())
        key_scores.append(_set_f1(pk, gk))
        # arg_value measures how correct the *values* are for the keys that
        # should be present (the gold keys). Extra/missing keys are already
        # penalized by the arg_key channel; dividing by the union here would
        # double-count that error and make this channel unreachable at 1.0
        # whenever the prediction has any extra key — coupling two objectives
        # that VPO/GDPO treat as independent.
        if not gk:
            val_scores.append(1.0)
        else:
            total = sum(_value_match(pp[k], gp[k]) for k in gk if k in pp)
            val_scores.append(total / len(gk))
    n = max(len(pairs), 1)
    return name_f1, sum(key_scores) / n, sum(val_scores) / n


def _gold_tool_calls(gold: dict) -> list[dict]:
    calls = gold.get("tool_calls")
    if calls is not None:
        return calls
    raw = gold.get("tool_calls_json")
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def score_response(pred_text: str, gold: dict) -> list[float]:
    """4-dim score vector vs gold (parsed ground-truth dict)."""
    parsed = parse_structured_response(pred_text)
    gold_calls = _gold_tool_calls(gold)
    fmt = 0.0
    if parsed["has_think"]:
        if gold_calls:
            if parsed["has_tool_call_tag"] and parsed["tool_calls"]:
                fmt = 1.0
        else:
            if not parsed["tool_call_body_nonempty"]:
                fmt = 1.0
    tn, ak, av = _score_calls(parsed["tool_calls"], gold_calls)
    return [fmt, tn, ak, av]


# Qwen3's chat template with enable_thinking=False pre-fills the assistant turn
# with `<think>\n\n</think>\n\n`, stripped from the generation. Prepend it at
# eval to match training-time scoring.
_QWEN3_NO_THINK_PREFIX = "<think>\n\n</think>\n\n"

_MULTI_APPEND = (
    "\n\n---\n"
    "ADDITIONAL INSTRUCTION (overrides the single-attempt output format above): "
    "Provide {m} different attempts at the task. Wrap each attempt in numbered "
    "outer tags <response_1>...</response_1> through <response_{m}>...</response_{m}>. "
    "Inside each <response_i>, follow the original output format from the system "
    "prompt (your <think>, optional <tool_call>, and inner <response> sections). "
    "The {m} attempts should be genuinely different — different tool choices, "
    "different argument values, or different reasoning — not paraphrases. "
    "Closing tags are required on every outer attempt."
)


# ---------------------------------------------------------------------------
# The Task
# ---------------------------------------------------------------------------


class ToolTask(Task):
    name = "tool"
    data_source = "tool"
    objectives = [
        ("format", "Producing a well-formed structured response"),
        ("tool_name", "Calling the right tools"),
        ("arg_key", "Filling in the right argument keys"),
        ("arg_value", "Using the right argument values"),
    ]
    multi_rewrite = AppendRewrite(template=_MULTI_APPEND)

    def weighted_scalar(self, vector):
        if not vector:
            return 0.0
        return sum(vector) / len(vector)

    def score_one(self, solution_str, ground_truth, extra_info):
        sub = score_response(solution_str, ground_truth)
        return {
            "sub_scores": sub,
            "channels": self.named_channels(sub),
            "scalar": self.weighted_scalar(sub),
            "extra": {"num_gold_tool_calls": ground_truth.get("num_tool_calls", 0)},
            "num_test_executions": 0,
            "num_programs_executed": 1,
        }

    def score_multi(self, solution_str, m, ground_truth, extra_info):
        blocks = parse_multi_responses(solution_str, m)
        num_parsed = sum(1 for b in blocks if b.strip())
        matrix = [
            score_response(b, ground_truth) if b.strip() else list(ZERO_SCORES)
            for b in blocks
        ]
        padded = pad_matrix(dedup_by_vector(matrix), m, ZERO_SCORES)
        extra = self.multi_channel_extras(padded)
        extra["num_gold_tool_calls"] = ground_truth.get("num_tool_calls", 0)
        return {
            "sub_scores": padded,
            "scalar": sum(self.weighted_scalar(row) for row in padded),
            "num_parsed": num_parsed,
            "num_test_executions": 0,
            "extra": extra,
        }

    # ----- preprocess -----
    def add_preprocess_args(self, p):
        import os

        p.add_argument("--local_save_dir", default=os.path.expanduser("~/data/toolrl"))
        p.add_argument("--source_dir", default=None,
                       help="Local dir with train.parquet + test.parquet "
                            "(default: download from GitHub).")

    def preprocess(self, args):
        import os
        import urllib.request

        import datasets as hf_datasets

        url_base = "https://raw.githubusercontent.com/qiancheng0/ToolRL/main/dataset/rlla_4k"
        save_dir = os.path.expanduser(args.local_save_dir)
        os.makedirs(save_dir, exist_ok=True)

        if args.source_dir:
            src = os.path.expanduser(args.source_dir)
        else:
            src = "/tmp/toolrl_dl"
            os.makedirs(src, exist_ok=True)
            for fn in ("train.parquet", "test.parquet"):
                dst = os.path.join(src, fn)
                if not os.path.exists(dst):
                    print(f"Downloading {fn}...")
                    urllib.request.urlretrieve(f"{url_base}/{fn}", dst)

        for split in ("train", "test"):
            ds = hf_datasets.Dataset.from_parquet(os.path.join(src, f"{split}.parquet"))
            rows = [_transform_row(row, i, split) for i, row in enumerate(ds)]
            out = os.path.join(save_dir, f"{split}.parquet")
            hf_datasets.Dataset.from_list(rows).to_parquet(out)
            print(f"  {split}: {len(rows)} rows -> {out}")
        print(f"\nSaved to {save_dir}")

    # ----- eval -----
    def add_eval_args(self, p):
        from vpo.eval_harness import add_sampling_eval_args, add_vllm_eval_args

        p.add_argument("--method", required=True,
                       choices=["grpo", "gdpo", "maxrl", "multi_rlvr", "vpo",
                                "goal_cond", "max_at_k"])
        p.add_argument("--data-dir", default="~/data/toolrl")
        p.add_argument("--output", default="results/eval_tool.json")
        p.add_argument("--num-prompts", type=int, default=80)
        add_sampling_eval_args(p, n_chains=50, num_solutions=1,
                               max_tokens=1024, temperature=0.7, seed=42)
        add_vllm_eval_args(p, max_model_len=6144)

    def evaluate(self, args):
        import json
        import math
        import os

        import datasets
        import numpy as np

        from vllm import SamplingParams
        from vpo.eval_harness import EvalHarness
        from vpo.utils.eval_metrics import (
            best_of_k_curve,
            compute_pool_metrics,
            solve_at_k_curve,
        )

        N_OBJ = NUM_OBJECTIVES
        is_multi = args.method in ("multi_rlvr", "vpo")
        n, m = args.n_chains, args.num_solutions
        if not is_multi and m != 1:
            print(f"WARNING: method={args.method} is single-response; forcing num_solutions=1")
            m = 1

        data_dir = os.path.expanduser(args.data_dir)
        ds = datasets.Dataset.from_parquet(os.path.join(data_dir, "test.parquet"))
        if args.num_prompts:
            ds = ds.select(range(min(args.num_prompts, len(ds))))
        print(f"Loaded {len(ds)} test prompts; pool n={n} × m={m} = {n * m}")

        harness = EvalHarness(args.model, max_model_len=args.max_model_len,
                              gpu_mem=args.gpu_mem, tp=args.tp, dtype="auto",
                              enforce_eager=True)

        formatted = []
        for row in ds:
            content = row["prompt"][-1]["content"]
            prompt = list(row["prompt"])
            if is_multi:
                # Rewrite only the last (user) turn — the system message holds
                # the tool definitions and output-format spec; dropping it
                # would handicap multi methods relative to the baselines.
                prompt[-1] = {**prompt[-1], "content": self.rewrite_multi_solution(content, m)}
            formatted.append(harness.apply_chat_template(prompt, enable_thinking=args.enable_thinking))

        params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                                top_p=0.95, n=n, seed=args.seed)
        outputs = harness.llm.generate(formatted, params)

        def score_text(text, gold):
            if not isinstance(text, str) or not text.strip():
                return list(ZERO_SCORES)
            if is_multi:
                # Inner <response_i> blocks are scored raw at training time
                # (each block must carry its own <think>); prepending the
                # chat-template prefill here would grant the format channel
                # for free and inflate eval relative to the trained reward.
                return score_response(text, gold)
            return score_response(_QWEN3_NO_THINK_PREFIX + text, gold)

        num_prompts = len(ds)
        tensor = np.zeros((num_prompts, n, m, N_OBJ))
        raw_grids = []
        for i in range(num_prompts):
            grid = [["" for _ in range(m)] for _ in range(n)]
            if is_multi:
                for j in range(n):
                    parsed = parse_multi_responses(outputs[i].outputs[j].text, m)
                    for k in range(m):
                        grid[j][k] = parsed[k]
            else:
                for j in range(n):
                    grid[j][0] = outputs[i].outputs[j].text
            raw_grids.append(grid)

        for i, row in enumerate(ds):
            gold = row["reward_model"]["ground_truth"]
            for j in range(n):
                for k in range(m):
                    tensor[i, j, k] = score_text(raw_grids[i][j][k], gold)

        pool_size = n * m
        pool_metrics_per_prompt, best_at_k, pass_at_k = [], [], []
        for i in range(num_prompts):
            pool_vectors, rids, positions = [], [], []
            for j in range(n):
                for k in range(m):
                    pool_vectors.append(tensor[i, j, k].tolist())
                    rids.append(j)
                    positions.append(k)
            pool_metrics_per_prompt.append(
                compute_pool_metrics(pool_vectors, rollout_ids=rids, positions=positions, rng_seed=42))
            sums = np.asarray([sum(v) for v in pool_vectors])
            full = np.asarray([all(s >= 1.0 for s in v) for v in pool_vectors], dtype=float)
            perm = np.random.default_rng(42 + i).permutation(pool_size)  # shared by both curves
            best_at_k.append([float(x) for x in best_of_k_curve(sums, perm)])
            pass_at_k.append([float(x) for x in solve_at_k_curve(full, perm)])

        pool_summary = {}
        if pool_metrics_per_prompt:
            for key in pool_metrics_per_prompt[0]:
                vals = [pm[key] for pm in pool_metrics_per_prompt
                        if not (isinstance(pm[key], float) and math.isnan(pm[key]))]
                if vals:
                    pool_summary[f"pool/{key}/mean"] = float(np.mean(vals))
                    pool_summary[f"pool/{key}/std"] = float(np.std(vals))

        best_arr = np.asarray(best_at_k)
        pass_arr = np.asarray(pass_at_k)
        all_flat = tensor.reshape(-1, N_OBJ)
        summary = {}
        for j, name in enumerate(SUB_OBJECTIVE_NAMES):
            summary[f"{name}/mean"] = float(all_flat[:, j].mean())
            summary[f"{name}/max"] = float(all_flat[:, j].max())
        summary.update(pool_summary)
        summary["sum_vec/mean"] = float(all_flat.sum(axis=1).mean())
        summary["sum_vec/max"] = float(all_flat.sum(axis=1).max())
        for k_landmark in (1, 5, 10, 25, 50, pool_size):
            if k_landmark <= pool_size:
                summary[f"best_at_k/k={k_landmark}/mean"] = float(best_arr[:, k_landmark - 1].mean())
                summary[f"pass_at_k/k={k_landmark}/mean"] = float(pass_arr[:, k_landmark - 1].mean())

        print("\n=== Summary ===")
        for k_, v in summary.items():
            print(f"  {k_}: {v:.4f}")

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        result = {
            "model": args.model, "method": args.method, "n_chains": n,
            "num_solutions": m, "pool_size": pool_size, "num_prompts": num_prompts,
            "summary": summary,
            "best_at_k_curve": best_arr.mean(axis=0).tolist(),
            "pass_at_k_curve": pass_arr.mean(axis=0).tolist(),
            "tensor_shape": [num_prompts, n, m, N_OBJ],
            "feature_names": list(SUB_OBJECTIVE_NAMES),
        }
        npy_path = args.output.replace(".json", "_tensor.npy")
        np.save(npy_path, tensor)
        result["tensor_path"] = npy_path
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results to {args.output}")


def _transform_row(row, idx, split):
    raw_gt = row["reward_model"]["ground_truth"]
    parsed_gt = parse_ground_truth_string(raw_gt)
    return {
        "data_source": "tool",
        "prompt": row["prompt"],
        "ability": "tool_use",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "tool_calls_json": json.dumps(parsed_gt["tool_calls"]),
                "response_text": parsed_gt["response_text"],
                "think_text": parsed_gt["think_text"],
                "raw_gt": raw_gt,
                "num_tool_calls": len(parsed_gt["tool_calls"]),
            },
        },
        "extra_info": {
            "split": split, "index": idx, "task_id": idx, "num_test_cases": 4,
            "source_idx": (row.get("extra_info") or {}).get("index", idx),
        },
    }


TASK = register(ToolTask())
