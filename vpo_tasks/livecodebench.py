"""LiveCodeBench task: run generated Python against test cases.

The reward is the per-test pass/fail vector (1.0 per passing test). For the
GDPO/goal-cond channels this is exposed as a fixed-width ``tc_0..tc_15`` vector
(zero-padded); ``sub_scores`` keeps the real per-problem test count.

Contains the test-execution scorer (via :mod:`vpo.utils.sandbox`), the HF
dataset loader with date-based train/test split, prompt construction, dataset
preprocessing (``preprocess``), and the held-out eval (``evaluate``) reporting
pass@k.
"""

from __future__ import annotations

import base64
import json
import pickle
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from vpo.task import Task, register
from vpo.utils.ast_dedup import ast_dedup
from vpo.utils.eval_metrics import pass_at_k
from vpo.utils.parse_solutions import extract_code_block, parse_solutions
from vpo.utils.sandbox import safe_execute, safe_execute_stdin

# Match the GDPO channel width and preprocess default — tc_0..tc_{MAX-1}.
MAX_TEST_CASES = 16
LCB_TIME_LIMIT = 6  # per-test wall-clock budget (seconds)

LCB_TRAIN_CUTOFF = datetime(2025, 2, 1)
LCB_V6_END = datetime(2025, 5, 1)


# ---------------------------------------------------------------------------
# Code execution + per-solution scoring
# ---------------------------------------------------------------------------


def _build_functional_test_wrapper(fn_name: str, input_str: str, output_str: str) -> str:
    return (
        "import json as _json\n"
        f"_input = {input_str!r}\n"
        f"_expected = _json.loads({output_str!r})\n"
        "_args = [_json.loads(_l) for _l in _input.split('\\n') if _l.strip()]\n"
        f"_got = {fn_name}(*_args)\n"
        "assert _got == _expected, (_got, _expected)\n"
    )


# Concurrent submissions per solution. Matches the sandbox pool width (4
# worker processes): serial round-trips leave 3 of the 4 workers idle and
# make a timeout-heavy solution cost len(tests) × LCB_TIME_LIMIT wall-clock.
_TEST_SUBMIT_WORKERS = 4


def _score_one_solution(code, inputs, outputs, testtype, fn_name) -> list[float]:
    """Run `code` against every test; return per-test 0/1 list.

    Tests are submitted to the sandbox pool concurrently (results keep input
    order). Per-test outcomes are unchanged: an infinite-looping test is cut
    by the worker-side SIGALRM without disturbing its siblings.
    """
    if not code:
        return [0.0] * len(inputs)

    if testtype == "functional":
        def run(pair):
            inp, out = pair
            wrapper = _build_functional_test_wrapper(fn_name, inp, out)
            return 1.0 if safe_execute(code, wrapper, timeout=LCB_TIME_LIMIT) else 0.0
    else:  # stdin
        def run(pair):
            inp, out = pair
            return 1.0 if safe_execute_stdin(code, inp, out, timeout=LCB_TIME_LIMIT) else 0.0

    pairs = list(zip(inputs, outputs))
    if len(pairs) <= 1:
        return [run(p) for p in pairs]
    with ThreadPoolExecutor(max_workers=min(len(pairs), _TEST_SUBMIT_WORKERS)) as tp:
        return list(tp.map(run, pairs))


def _gt_fields(ground_truth):
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    return (ground_truth["inputs"], ground_truth["outputs"],
            ground_truth["testtype"], ground_truth.get("fn_name", ""))


# ---------------------------------------------------------------------------
# The Task
# ---------------------------------------------------------------------------


class LiveCodeBenchTask(Task):
    name = "lcb"
    data_source = "livecodebench"
    objectives = [(f"tc_{i}", f"Test case {i + 1}") for i in range(MAX_TEST_CASES)]
    multi_rewrite = None  # multi-solution prompt is baked at preprocess time

    def score_one(self, solution_str, ground_truth, extra_info):
        inputs, outputs, testtype, fn_name = _gt_fields(ground_truth)
        code = extract_code_block(solution_str)
        scores = _score_one_solution(code, inputs, outputs, testtype, fn_name)

        # GDPO/goal-cond channels: pad to a fixed 16-wide tc_* vector.
        padded = (scores + [0.0] * MAX_TEST_CASES)[:MAX_TEST_CASES]
        channels = {f"tc_{i}": padded[i] for i in range(MAX_TEST_CASES)}
        return {
            "sub_scores": scores,          # real per-test vector (VPO basis)
            "channels": channels,          # padded tc_* (GDPO/goal-cond basis)
            "scalar": None,                # keep build_single's mean over real tests
            "extra": {},
            "num_test_executions": len(inputs) if code else 0,
            "num_programs_executed": 1 if code else 0,
        }

    def score_multi(self, solution_str, m, ground_truth, extra_info):
        inputs, outputs, testtype, fn_name = _gt_fields(ground_truth)
        solutions = parse_solutions(solution_str, m)
        num_parsed = sum(1 for s in solutions if s)
        unique = ast_dedup(solutions)
        # NB: not padded to m — matches the original lcb_multi reward.
        score_matrix = [
            _score_one_solution(extract_code_block(sol), inputs, outputs, testtype, fn_name)
            for sol in unique
        ]
        return {
            "sub_scores": score_matrix,
            "scalar": None,  # keep build_multi's active-variant scalar (sum_all)
            "num_parsed": num_parsed,
            "num_test_executions": len(unique) * len(inputs),
            "extra": {},
        }

    # ----- preprocess -----
    def add_preprocess_args(self, p):
        p.add_argument("--local_save_dir", default="~/data/lcb")
        p.add_argument("--max_tests_per_problem", type=int, default=MAX_TEST_CASES,
                       help=f"Keep at most this many smallest tests per problem "
                            f"(default: {MAX_TEST_CASES}; matches the tc_* channel width).")
        p.add_argument("--max_bytes_per_test", type=int, default=8192,
                       help="Drop tests whose input+output exceeds this many bytes.")

    def preprocess(self, args):
        import os

        save_dir = os.path.expanduser(args.local_save_dir)
        os.makedirs(save_dir, exist_ok=True)
        for split in ("train", "test"):
            print(f"\nLoading LCB v6 {split} split...")
            ds = load_lcb_split(split)
            n_raw = len(ds)
            print(f"  raw problems: {n_raw}")
            ds = ds.map(
                function=_make_map_fn(split, args.max_tests_per_problem, args.max_bytes_per_test),
                with_indices=True,
                remove_columns=ds.column_names,
            )
            ds = ds.filter(lambda ex: ex["data_source"] == "livecodebench")
            print(f"  kept problems: {len(ds)} (dropped {n_raw - len(ds)} with no usable tests)")
            out_path = os.path.join(save_dir, f"{split}.parquet")
            ds.to_parquet(out_path)
            print(f"  wrote {out_path}")

    # ----- eval -----
    def add_eval_args(self, p):
        p.add_argument("--method",
                       # no goal_cond: lcb goal-conditioned training is not wired
                       # (train.sh refuses it), so there is nothing to evaluate.
                       choices=["grpo", "gdpo", "maxrl", "multi_rlvr", "vpo"],
                       default="grpo")
        p.add_argument("--output", default=None)
        p.add_argument("--n-chains", type=int, default=10)
        p.add_argument("--num-solutions", type=int, default=3)
        p.add_argument("--max-tokens", type=int, default=None)
        p.add_argument("--temperature", type=float, default=0.8)
        p.add_argument("--max-problems", type=int, default=None)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--dataset-path", default=None,
                       help="Preprocessed LCB parquet dir. Defaults based on --method.")

    def evaluate(self, args):
        import json
        import os

        import numpy as np
        import pyarrow.parquet as pq

        from vllm import SamplingParams
        from vpo.eval_harness import EvalHarness

        multi = args.method in ("multi_rlvr", "vpo")
        m = args.num_solutions if multi else 1
        if args.max_tokens is None:
            args.max_tokens = 3072 if multi else 2048
        n_chains = args.n_chains if multi else args.n_chains * args.num_solutions
        n_total_sols = n_chains * m

        if args.dataset_path is None:
            home = os.environ.get("HOME", os.path.expanduser("~"))
            args.dataset_path = os.path.join(home, "data", "lcb_multi" if multi else "lcb")

        test_pq = os.path.join(args.dataset_path, "test.parquet")
        print(f"Loading test set: {test_pq}")
        rows = pq.read_table(test_pq).to_pylist()
        if args.max_problems:
            rows = rows[: args.max_problems]

        prompts, problems = [], []
        for row in rows:
            msgs = row["prompt"]
            prompts.append(msgs[0]["content"] if isinstance(msgs, list) and msgs else str(msgs))
            gt = row["reward_model"]["ground_truth"]
            if isinstance(gt, str):
                gt = json.loads(gt)
            problems.append({"task_id": row.get("extra_info", {}).get("task_id", ""),
                             "ground_truth": gt})

        print(f"\n=== Eval config ===\n  method={args.method} multi={multi} m={m} "
              f"n_chains={n_chains} n_total={n_total_sols} problems={len(prompts)}")

        harness = EvalHarness(args.model, max_model_len=8192, gpu_mem=0.55,
                              dtype="bfloat16", trust_remote_code=True,
                              seed=args.seed)
        formatted = [harness.render(p) for p in prompts]
        params = SamplingParams(temperature=args.temperature, top_p=0.95,
                                max_tokens=args.max_tokens, n=n_chains,
                                seed=args.seed)
        outputs = harness.llm.generate(formatted, params)
        chain_outputs = [[c.text for c in o.outputs] for o in outputs]

        print(f"\nScoring {len(prompts)} problems × {n_chains} chains × {m} sols...")
        per_problem = []
        for i, (prob, chains) in enumerate(zip(problems, chain_outputs)):
            chain_results = [self._score_chain(ct, prob["ground_truth"], multi, m) for ct in chains]
            per_problem.append({"task_id": prob["task_id"], "chain_results": chain_results})
            if (i + 1) % 20 == 0 or i == len(problems) - 1:
                print(f"  scored {i + 1}/{len(problems)}")

        # Two metric families, both labelled by their true k:
        #   pass@{k}_sol       — among k of the n_total_sols sampled solutions,
        #                        does at least one solve? (the headline metric;
        #                        n_total_sols is held equal across methods, so
        #                        these are directly comparable method-to-method)
        #   pass@{k}_chain_any — among k of the n_chains generations, does at
        #                        least one chain contain a solving solution?
        K_VALUES = [1, 5, 10]
        metrics = {}
        for k in K_VALUES:
            metrics[f"pass@{k}_sol"] = []
            metrics[f"pass@{k}_chain_any"] = []
        for pp in per_problem:
            results = pp["chain_results"]
            c_any = sum(1 for chain in results if any(chain))
            c_total = sum(sum(chain) for chain in results)
            for k in K_VALUES:
                metrics[f"pass@{k}_sol"].append(pass_at_k(n_total_sols, c_total, k))
                metrics[f"pass@{k}_chain_any"].append(pass_at_k(n_chains, c_any, k))
        summary_metrics = {key: float(np.mean(v)) for key, v in metrics.items()}

        print("\n" + "═" * 60)
        print(f"  EVAL SUMMARY: {args.method}  (LiveCodeBench v6)")
        print("═" * 60)
        for key, v in summary_metrics.items():
            print(f"  {key:30s} = {v:.4f}")

        out = {
            "method": args.method, "model": args.model, "benchmark": "livecodebench_v6",
            "n_problems": len(per_problem), "m": m, "n_chains": n_chains,
            "n_total_sols": n_total_sols, "temperature": args.temperature,
            "max_tokens": args.max_tokens, "metrics": summary_metrics,
            "pass@1": summary_metrics["pass@1_sol"],
            "pass@5": summary_metrics["pass@5_sol"],
            "pass@10": summary_metrics["pass@10_sol"],
            "per_problem": [
                {"task_id": pp["task_id"],
                 "n_correct_first": sum(1 for c in pp["chain_results"] if c[0]),
                 "n_correct_any": sum(1 for c in pp["chain_results"] if any(c)),
                 "n_correct_sols": sum(sum(c) for c in pp["chain_results"]),
                 "n_total_sols": n_total_sols, "n_rollouts": n_chains}
                for pp in per_problem
            ],
        }
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Wrote {args.output}")

    def _score_chain(self, chain_text, ground_truth, multi, m) -> list[bool]:
        if multi:
            codes = _extract_all_code_blocks(chain_text)[:m]
            while len(codes) < m:
                codes.append("")
        else:
            codes = [chain_text]
        out = []
        for code in codes:
            if not code.strip():
                out.append(False)
            else:
                wrapped = code if "```python" in code else f"```python\n{code}\n```"
                out.append(self._solution_passes(wrapped, ground_truth))
        return out

    def _solution_passes(self, solution_str, ground_truth) -> bool:
        inputs, outputs, testtype, fn_name = _gt_fields(ground_truth)
        scores = _score_one_solution(extract_code_block(solution_str), inputs, outputs, testtype, fn_name)
        return bool(scores) and all(s >= 1.0 for s in scores)


# ---------------------------------------------------------------------------
# Dataset utilities (HF load + temporal split + per-example processing)
# ---------------------------------------------------------------------------


def decode_private_tests(encoded_blob: str) -> list[dict]:
    decoded = base64.b64decode(encoded_blob)
    return json.loads(pickle.loads(zlib.decompress(decoded)))


def filter_and_cap_tests(tests, max_tests, max_bytes):
    sized = []
    for t in tests:
        total = len(t.get("input", "")) + len(t.get("output", ""))
        if total <= max_bytes:
            sized.append((total, t))
    sized.sort(key=lambda x: x[0])
    return [t for _, t in sized[:max_tests]]


def strip_self(text: str) -> str:
    return text.replace("(self, ", "(").replace("(self)", "()")


def parse_starter_signature(starter_code: str):
    if "def " not in starter_code:
        return None
    after_def = starter_code.split("def ", 1)[1]
    sig_line = after_def.split("\n", 1)[0].strip()
    return "def " + strip_self(sig_line)


def load_lcb_split(split: str):
    import datasets

    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    ds = datasets.load_dataset(
        "livecodebench/code_generation_lite", split="test", revision="refs/pr/6")
    if split == "train":
        return ds.filter(lambda ex: ex["contest_date"] < LCB_TRAIN_CUTOFF)
    return ds.filter(lambda ex: LCB_TRAIN_CUTOFF <= ex["contest_date"] < LCB_V6_END)


_SINGLE_SUFFIX_FUNCTIONAL = (
    "\n\nYour solution should be a Python function with this signature:\n"
    "```python\n{signature}\n```\n\n"
    "Wrap your solution in a ```python``` code block."
)
_SINGLE_SUFFIX_STDIN = (
    "\n\nWrite a complete Python program that reads input from stdin and "
    "writes output to stdout. Wrap your solution in a ```python``` code block."
)
_MULTI_SUFFIX_FUNCTIONAL = (
    "\n\nYour solutions should be Python functions with this signature:\n"
    "```python\n{signature}\n```\n\n"
    "Write {m} different solutions that solve this problem. Each solution "
    "should use a different approach. Wrap each solution in a ```python``` "
    "code block and separate them with '---'."
)
_MULTI_SUFFIX_STDIN = (
    "\n\nEach solution should be a complete Python program that reads input "
    "from stdin and writes output to stdout. Write {m} different solutions "
    "that solve this problem, each using a different approach. Wrap each "
    "solution in a ```python``` code block and separate them with '---'."
)


def build_prompt(question_content, starter_code, testtype, num_solutions=1):
    body = strip_self(question_content)
    if testtype == "functional":
        signature = parse_starter_signature(starter_code) or "def solve(...):"
        tail = (_SINGLE_SUFFIX_FUNCTIONAL.format(signature=signature) if num_solutions <= 1
                else _MULTI_SUFFIX_FUNCTIONAL.format(signature=signature, m=num_solutions))
    else:
        tail = _SINGLE_SUFFIX_STDIN if num_solutions <= 1 else _MULTI_SUFFIX_STDIN.format(m=num_solutions)
    return body + tail


def process_example(example, split, idx, num_solutions, max_tests, max_bytes):
    raw_tests = decode_private_tests(example["private_test_cases"])
    if not raw_tests:
        return None
    testtype = raw_tests[0]["testtype"]
    kept = filter_and_cap_tests(raw_tests, max_tests, max_bytes)
    if not kept:
        return None
    meta = json.loads(example["metadata"]) if example["metadata"].strip() else {}
    fn_name = meta.get("func_name", "") if testtype == "functional" else ""
    prompt_text = build_prompt(example["question_content"], example["starter_code"],
                               testtype, num_solutions=num_solutions)
    extra_info = {
        "split": split, "index": idx, "task_id": example["question_id"],
        "platform": example["platform"], "difficulty": example["difficulty"],
        "num_test_cases": len(kept), "num_test_cases_original": len(raw_tests),
        "testtype": testtype,
    }
    if num_solutions > 1:
        extra_info["num_solutions"] = num_solutions
    return {
        "data_source": "livecodebench",
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "testtype": testtype, "fn_name": fn_name,
                "inputs": [t["input"] for t in kept],
                "outputs": [t["output"] for t in kept],
            },
        },
        "extra_info": extra_info,
    }


def _make_map_fn(split, max_tests, max_bytes):
    def process_fn(example, idx):
        row = process_example(example, split=split, idx=idx, num_solutions=1,
                              max_tests=max_tests, max_bytes=max_bytes)
        if row is None:
            return {"data_source": "", "prompt": [], "ability": "",
                    "reward_model": {}, "extra_info": {}}
        return row
    return process_fn


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------


def _extract_all_code_blocks(text: str) -> list[str]:
    blocks = re.findall(r"```python\s*\n?(.*?)```", text, re.DOTALL)
    if blocks:
        return [b.strip() for b in blocks]
    parts = [p.strip() for p in text.split("---") if p.strip()]
    if len(parts) > 1:
        return [extract_code_block(p) for p in parts]
    return [extract_code_block(text)]


TASK = register(LiveCodeBenchTask())
