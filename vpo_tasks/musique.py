"""MuSiQue task: multi-hop question answering.

The reward is a 5-dim vector ``[hop_1, hop_2, hop_3, hop_4, answer_f1]``: the
first four are per-hop supporting-paragraph recall, the last is word-level F1
of the answer.

Contains answer normalization with F1/EM, the single- and multi-response
parsers, the prompt templates, dataset preprocessing (``preprocess``), and the
held-out eval (``evaluate``), which reports E_w[max_s w·r_s] over Dirichlet
weights and a majority-vote-vs-k curve.
"""

from __future__ import annotations

import re
from collections import Counter

from vpo.task import SubRewrite, Task, dedup_by_vector, pad_matrix, register
from vpo.utils.eval_metrics import (
    dirichlet_weights,
    expected_max_weighted,
    mean_weighted,
)
from vpo.utils.parse_solutions import extract_numbered_tags
from vpo.utils.text_metrics import exact_match, normalize_text, token_f1

MAX_HOPS = 4
NUM_OBJECTIVES = MAX_HOPS + 1  # per-hop evidence + answer_f1
SUB_OBJECTIVE_NAMES = ["hop_1", "hop_2", "hop_3", "hop_4", "answer_f1"]
ZERO_SCORES = [0.0] * NUM_OBJECTIVES

_SUPPORT_RE = re.compile(r"<support>(.*?)</support>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


# ---------------------------------------------------------------------------
# Answer normalization + metrics
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    return normalize_text(text, strip_articles=True)


def compute_f1(pred: str, gold: str) -> float:
    return token_f1(pred, gold, strip_articles=True)


def compute_em(pred: str, gold: str) -> float:
    return exact_match(pred, gold, strip_articles=True)


def best_score(pred: str, gold: str, aliases: list[str], metric_fn) -> float:
    scores = [metric_fn(pred, gold)]
    for alias in aliases:
        scores.append(metric_fn(pred, alias))
    return max(scores)


def parse_response(text: str) -> tuple[list[int], str]:
    """Extract <support> indices and <answer> text from a response."""
    support_match = _SUPPORT_RE.search(text)
    answer_match = _ANSWER_RE.search(text)
    cited = []
    if support_match:
        for tok in support_match.group(1).split(","):
            tok = tok.strip()
            if tok.isdigit():
                cited.append(int(tok))
    answer = answer_match.group(1).strip() if answer_match else ""
    return cited, answer


def parse_multi_responses(text: str, m: int) -> list[tuple[list[int], str]]:
    """Parse m responses from <response_i>...</response_i> tags."""
    return [parse_response(b) for b in extract_numbered_tags(text, m)]


def score_response(cited_indices, pred_answer, gold_support_ordered, gold_answer, answer_aliases):
    """5-dim reward vector: [hop_1..hop_4, answer_f1]."""
    truncated, seen = [], set()
    for idx in cited_indices:
        if idx in seen:
            continue
        seen.add(idx)
        truncated.append(idx)
        if len(truncated) >= MAX_HOPS:
            break
    cited_set = set(truncated)

    hop_scores = []
    for i in range(MAX_HOPS):
        if i < len(gold_support_ordered):
            hop_scores.append(1.0 if gold_support_ordered[i] in cited_set else 0.0)
        else:
            hop_scores.append(1.0)

    f1 = best_score(pred_answer, gold_answer, answer_aliases, compute_f1) if pred_answer else 0.0
    return hop_scores + [f1]


def support_recall(cited, gold):
    if not gold:
        return 1.0
    cited_set = set(cited)
    return sum(1 for g in gold if g in cited_set) / len(gold)


def support_precision(cited, gold):
    if not cited:
        return 0.0
    gold_set = set(gold)
    return sum(1 for c in cited if c in gold_set) / len(cited)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
Read the paragraphs below and answer the multi-hop question. Identify which paragraphs support your answer.

Paragraphs:
{paragraphs_text}

Question: {question}

First reason about which paragraphs are relevant, then output your supporting paragraph indices and answer.
<support>comma-separated paragraph indices (e.g., 3, 7, 12)</support>
<answer>your answer</answer>"""

_SINGLE_OUTPUT_PATTERN = re.compile(
    r"First reason about which paragraphs are relevant, then output your "
    r"supporting paragraph indices and answer\.\n"
    r"<support>comma-separated paragraph indices \(e\.g\., 3, 7, 12\)</support>\n"
    r"<answer>your answer</answer>",
)

_MULTI_TEMPLATE = (
    "Give {m} different answers in <response_i>...</response_i> tags, "
    "each with <support>indices</support> and <answer>answer</answer>."
)


# ---------------------------------------------------------------------------
# The Task
# ---------------------------------------------------------------------------


class MusiqueTask(Task):
    name = "musique"
    data_source = "musique"
    objectives = [
        ("hop_1", "Finding evidence for reasoning step 1"),
        ("hop_2", "Finding evidence for reasoning step 2"),
        ("hop_3", "Finding evidence for reasoning step 3"),
        ("hop_4", "Finding evidence for reasoning step 4"),
        ("answer_f1", "Answer word-level accuracy"),
    ]
    multi_rewrite = SubRewrite(
        pattern=_SINGLE_OUTPUT_PATTERN,
        template=_MULTI_TEMPLATE,
    )

    def weighted_scalar(self, vector):
        """answer_f1 has weight 3, each hop weight 1."""
        if not vector:
            return 0.0
        hop_scores = vector[:-1]
        answer_f1 = vector[-1]
        return (sum(hop_scores) + 3 * answer_f1) / (len(hop_scores) + 3)

    def score_one(self, solution_str, ground_truth, extra_info):
        gold_support = ground_truth["supporting_indices_ordered"]
        gold_answer = ground_truth["answer"]
        aliases = ground_truth.get("answer_aliases", [])
        cited, pred_answer = parse_response(solution_str)
        # No answer → zero vector, matching score_multi and eval. Granting hop
        # credit for a cite-only response would let training reward a
        # degenerate never-answer policy that scores 0 everywhere else.
        if not pred_answer:
            sub = list(ZERO_SCORES)
        else:
            sub = score_response(cited, pred_answer, gold_support, gold_answer, aliases)
        return {
            "sub_scores": sub,
            "channels": self.named_channels(sub),
            "scalar": self.weighted_scalar(sub),
            "extra": {
                "support_recall": support_recall(cited, gold_support),
                "support_precision": support_precision(cited, gold_support),
                "num_cited": len(cited),
                "num_hops": ground_truth.get("num_hops", len(gold_support)),
                "parsed_answer": bool(pred_answer),
                "parsed_support": bool(cited),
            },
            "num_test_executions": 0,
            "num_programs_executed": 1 if pred_answer else 0,
        }

    def score_multi(self, solution_str, m, ground_truth, extra_info):
        gold_support = ground_truth["supporting_indices_ordered"]
        gold_answer = ground_truth["answer"]
        aliases = ground_truth.get("answer_aliases", [])
        responses = parse_multi_responses(solution_str, m)
        num_parsed = sum(1 for _, ans in responses if ans)

        matrix = [
            score_response(cited, ans, gold_support, gold_answer, aliases) if ans else list(ZERO_SCORES)
            for cited, ans in responses
        ]
        padded = pad_matrix(dedup_by_vector(matrix), m, ZERO_SCORES)

        extra = self.multi_channel_extras(padded)
        extra["num_hops"] = ground_truth.get("num_hops", len(gold_support))
        return {
            "sub_scores": padded,
            "scalar": sum(self.weighted_scalar(row) for row in padded),
            "num_parsed": num_parsed,
            "num_test_executions": 0,
            "extra": extra,
        }

    # ----- preprocess -----
    def add_preprocess_args(self, p):
        p.add_argument("--local_save_dir", default="~/data/musique")
        p.add_argument("--max_train", type=int, default=-1)
        p.add_argument("--max_test", type=int, default=-1)

    def preprocess(self, args):
        import os

        import datasets as hf_datasets

        print("Loading MuSiQue from HuggingFace...")
        ds = hf_datasets.load_dataset("dgslibisey/MuSiQue")
        train_ds, val_ds = ds["train"], ds["validation"]
        if args.max_train > 0:
            train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))
        if args.max_test > 0:
            val_ds = val_ds.select(range(min(args.max_test, len(val_ds))))

        train_rows = [_build_row(ex, "train", i) for i, ex in enumerate(train_ds)]
        test_rows = [_build_row(ex, "test", i) for i, ex in enumerate(val_ds)]

        save_dir = os.path.expanduser(args.local_save_dir)
        os.makedirs(save_dir, exist_ok=True)
        hf_datasets.Dataset.from_list(train_rows).to_parquet(os.path.join(save_dir, "train.parquet"))
        hf_datasets.Dataset.from_list(test_rows).to_parquet(os.path.join(save_dir, "test.parquet"))
        print(f"Saved {len(train_rows)} train, {len(test_rows)} test to {save_dir}")

    # ----- eval -----
    def add_eval_args(self, p):
        from vpo.eval_harness import add_sampling_eval_args, add_vllm_eval_args

        # required (not defaulted) like every other task — the reported method
        # label must never come from a silently-applied default.
        p.add_argument("--method", required=True,
                       choices=["grpo", "gdpo", "maxrl", "multi_rlvr", "vpo",
                                "goal_cond", "goal_cond_oracle"])
        p.add_argument("--data-dir", default="~/data/musique")
        p.add_argument("--output", default="results/eval_musique.json")
        p.add_argument("--num-examples", type=int, default=300)
        p.add_argument("--n-weights", type=int, default=100)
        add_sampling_eval_args(p, n_chains=10, num_solutions=3,
                               max_tokens=1024, temperature=0.7, seed=42)
        add_vllm_eval_args(p, max_model_len=6400)
        p.add_argument("--mv-k-values", default="1,2,3,5,10,15,20,30")
        p.add_argument("--mv-n-subsets", type=int, default=50)

    def evaluate(self, args):
        import copy
        import json
        import os

        import datasets
        import numpy as np

        from vllm import SamplingParams
        from vpo.augment import format_weight_suffix
        from vpo.eval_harness import EvalHarness

        is_multi = args.method in ("multi_rlvr", "vpo")
        is_goal_cond = args.method == "goal_cond"
        is_goal_cond_oracle = args.method == "goal_cond_oracle"
        m, n = args.num_solutions, args.n_chains

        n_obj = NUM_OBJECTIVES
        weights = dirichlet_weights(n_obj, args.n_weights, args.seed)

        data_dir = os.path.expanduser(args.data_dir)
        ds = datasets.Dataset.from_parquet(os.path.join(data_dir, "test.parquet"))
        if args.num_examples:
            ds = ds.select(range(min(args.num_examples, len(ds))))
        num_ex = len(ds)
        print(f"Loaded {num_ex} examples from {data_dir}")

        harness = EvalHarness(args.model, max_model_len=args.max_model_len,
                              gpu_mem=args.gpu_mem, tp=args.tp, seed=args.seed)
        tmpl_kwargs = {"enable_thinking": args.enable_thinking}

        def render(prompt_msgs):
            return harness.apply_chat_template(prompt_msgs, **tmpl_kwargs)

        if is_goal_cond or is_goal_cond_oracle:
            objectives = self.objectives
            k_inj = len(objectives)
            if is_goal_cond_oracle:
                assert n <= args.n_weights, (
                    f"goal_cond_oracle: n_chains ({n}) must be <= n_weights ({args.n_weights})")
                injected_weights = np.tile(weights[:n], (num_ex, 1))
            else:
                rng = np.random.default_rng(args.seed)
                injected_weights = rng.dirichlet(np.ones(k_inj), size=num_ex * n)
            params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, n=m)
            formatted = []
            for i, row in enumerate(ds):
                for j in range(n):
                    w = injected_weights[i * n + j].tolist()
                    pc = copy.deepcopy(list(row["prompt"]))
                    pc[-1]["content"] = pc[-1]["content"] + format_weight_suffix(w, objectives)
                    formatted.append(render(pc))
            outputs = harness.llm.generate(formatted, params)
        elif is_multi:
            params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, n=n)
            formatted = []
            for row in ds:
                pc = copy.deepcopy(list(row["prompt"]))
                pc[-1]["content"] = self.rewrite_multi_solution(pc[-1]["content"], m)
                formatted.append(render(pc))
            outputs = harness.llm.generate(formatted, params)
        else:
            params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, n=n * m)
            formatted = [render(row["prompt"]) for row in ds]
            outputs = harness.llm.generate(formatted, params)

        tensor = np.zeros((num_ex, n, m, n_obj), dtype=np.float64)
        answers_per_example = []
        per_example = []
        for i, row in enumerate(ds):
            gt = row["reward_model"]["ground_truth"]
            ex_answers = []
            if is_multi:
                for j in range(n):
                    rows, anss = _score_response_full(outputs[i].outputs[j].text, gt, m)
                    tensor[i, j] = rows
                    ex_answers.extend(anss)
            elif is_goal_cond or is_goal_cond_oracle:
                for j in range(n):
                    for k in range(m):
                        rows, anss = _score_response_full(outputs[i * n + j].outputs[k].text, gt, 1)
                        tensor[i, j, k] = rows[0]
                        ex_answers.append(anss[0])
            else:
                for j in range(n):
                    for k in range(m):
                        rows, anss = _score_response_full(outputs[i].outputs[j * m + k].text, gt, 1)
                        tensor[i, j, k] = rows[0]
                        ex_answers.append(anss[0])

            answers_per_example.append(ex_answers)
            mv_answer = _majority_vote_answer(ex_answers)
            gold_answer = gt["answer"]
            aliases = gt.get("answer_aliases", []) or []
            mv_f1 = best_score(mv_answer, gold_answer, aliases, compute_f1) if mv_answer else 0.0
            mv_em = best_score(mv_answer, gold_answer, aliases, compute_em) if mv_answer else 0.0
            flat = tensor[i].reshape(-1, n_obj)
            per_example.append({
                "num_hops": gt.get("num_hops", len(gt["supporting_indices_ordered"])),
                "answer_f1_max": float(flat[:, -1].max()),
                "answer_f1_mean": float(flat[:, -1].mean()),
                "hop_recall_mean": float(flat[:, :-1].mean()),
                "weighted_scalar_max": float(max(self.weighted_scalar(list(v)) for v in flat)),
                "mv_answer": mv_answer, "mv_answer_f1": mv_f1, "mv_answer_em": mv_em,
            })

        pool_em, chain_em, pool_mean = [], [], []
        for i in range(num_ex):
            pool = tensor[i].reshape(-1, n_obj)
            pool_em.append(expected_max_weighted(pool, weights))
            pool_mean.append(mean_weighted(pool, weights))
            chain_em.append(float(np.mean([expected_max_weighted(tensor[i, j], weights) for j in range(n)])))

        all_flat = tensor.reshape(-1, n_obj)
        mv_f1s = np.array([pe["mv_answer_f1"] for pe in per_example])
        mv_ems = np.array([pe["mv_answer_em"] for pe in per_example])
        summary = {
            "Ew_max_pool": float(np.mean(pool_em)),
            "Ew_max_pool_se": float(np.std(pool_em) / np.sqrt(max(1, num_ex))),
            "Ew_max_chain_mean": float(np.mean(chain_em)),
            "Ew_mean_pool": float(np.mean(pool_mean)),
            "majority_vote/answer_f1": float(mv_f1s.mean()),
            "majority_vote/answer_f1_se": float(mv_f1s.std() / np.sqrt(max(1, num_ex))),
            "majority_vote/answer_em": float(mv_ems.mean()),
            "majority_vote/answer_em_se": float(mv_ems.std() / np.sqrt(max(1, num_ex))),
            "majority_vote/n_votes": int(n * m),
        }
        for j, name in enumerate(SUB_OBJECTIVE_NAMES):
            summary[f"{name}/mean"] = float(all_flat[:, j].mean())
            summary[f"{name}/max_per_example_mean"] = float(
                np.mean([tensor[i, :, :, j].max() for i in range(num_ex)]))

        k_values = [int(s) for s in args.mv_k_values.split(",") if s.strip()]
        ground_truths = [row["reward_model"]["ground_truth"] for row in ds]
        mv_curve = _compute_mv_curve(answers_per_example, ground_truths, k_values,
                                     n_subsets=args.mv_n_subsets, seed=args.seed)

        print("\n=== Summary ===")
        for key, val in summary.items():
            print(f"  {key}: {val:.4f}")

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        result = {
            "model": args.model, "method": args.method, "n_chains": n,
            "num_solutions": m, "num_examples": num_ex, "n_weights": args.n_weights,
            "seed": args.seed, "enable_thinking": args.enable_thinking,
            "temperature": args.temperature, "summary": summary, "mv_curve": mv_curve,
            "mv_k_values": k_values, "mv_n_subsets": args.mv_n_subsets,
            "per_example": per_example, "tensor_shape": list(tensor.shape),
            "feature_names": list(SUB_OBJECTIVE_NAMES),
            "answers_per_example": answers_per_example,
        }
        npy_path = args.output.replace(".json", "_tensor.npy")
        np.save(npy_path, tensor)
        result["tensor_path"] = npy_path
        weights_path = args.output.replace(".json", "_weights.npy")
        np.save(weights_path, weights)
        result["weights_path"] = weights_path
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results to {args.output}")


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------


def _score_one(cited, ans, gt):
    if not ans:
        return list(ZERO_SCORES)
    return score_response(cited, ans, gt["supporting_indices_ordered"],
                          gt["answer"], gt.get("answer_aliases", []))


def _score_response_full(response, ground_truth, num_solutions):
    if num_solutions > 1:
        parsed = parse_multi_responses(response, num_solutions)
    else:
        parsed = [parse_response(response)]
    rows = [_score_one(cited, ans, ground_truth) for cited, ans in parsed]
    answers = [ans for _, ans in parsed]
    while len(rows) < num_solutions:
        rows.append(list(ZERO_SCORES))
        answers.append("")
    return rows[:num_solutions], answers[:num_solutions]


def _majority_vote_answer(answers):
    counts = Counter()
    surface = {}
    for ans in answers:
        if not ans:
            continue
        key = normalize_answer(ans)
        if not key:
            continue
        counts[key] += 1
        surface.setdefault(key, ans)
    if not counts:
        return ""
    best_key, _ = counts.most_common(1)[0]
    return surface[best_key]


def _compute_mv_curve(answers_per_example, ground_truths, k_values, n_subsets=50, seed=42):
    import numpy as np

    rng = np.random.default_rng(seed)
    K_total = len(answers_per_example[0]) if answers_per_example else 0
    n_ex = len(answers_per_example)
    curve = []
    for k in sorted(set(int(kk) for kk in k_values)):
        if k <= 0 or k > K_total:
            continue
        f1_per_ex = np.zeros(n_ex)
        em_per_ex = np.zeros(n_ex)
        for i, anss in enumerate(answers_per_example):
            gt = ground_truths[i]
            gold = gt["answer"]
            aliases = gt.get("answer_aliases", []) or []
            if k >= K_total:
                subsets = [list(range(K_total))]
            else:
                subsets = [rng.choice(K_total, size=k, replace=False).tolist() for _ in range(n_subsets)]
            f1s, ems = [], []
            for idx in subsets:
                mv = _majority_vote_answer([anss[j] for j in idx])
                if mv:
                    f1s.append(best_score(mv, gold, aliases, compute_f1))
                    ems.append(best_score(mv, gold, aliases, compute_em))
                else:
                    f1s.append(0.0)
                    ems.append(0.0)
            f1_per_ex[i] = float(np.mean(f1s))
            em_per_ex[i] = float(np.mean(ems))
        curve.append({
            "k": k, "mv_f1": float(f1_per_ex.mean()),
            "mv_f1_se": float(f1_per_ex.std() / np.sqrt(max(1, n_ex))),
            "mv_em": float(em_per_ex.mean()),
            "mv_em_se": float(em_per_ex.std() / np.sqrt(max(1, n_ex))),
            "n_subsets": 1 if k >= K_total else n_subsets,
        })
    return curve


# ---------------------------------------------------------------------------
# Preprocess row builder
# ---------------------------------------------------------------------------


def _format_paragraphs(paragraphs):
    return "\n".join(
        f"[{p['idx']}] (Title: {p['title']}) {p['paragraph_text']}" for p in paragraphs
    )


def _build_row(example, split, index):
    content = PROMPT_TEMPLATE.format(
        paragraphs_text=_format_paragraphs(example["paragraphs"]),
        question=example["question"],
    )
    decomp = example["question_decomposition"]
    supporting_ordered = [step["paragraph_support_idx"] for step in decomp]
    num_hops = len(decomp)
    return {
        "data_source": "musique",
        "prompt": [{"role": "user", "content": content}],
        "ability": "reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answer": example["answer"],
                "answer_aliases": example.get("answer_aliases", []),
                "supporting_indices_ordered": supporting_ordered,
                "num_hops": num_hops,
            },
        },
        "extra_info": {
            "split": split, "index": index, "task_id": index,
            "num_test_cases": 5, "musique_id": example["id"], "num_hops": num_hops,
        },
    }


TASK = register(MusiqueTask())
