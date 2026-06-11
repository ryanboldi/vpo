"""EUREQA task: multi-hop entity back-chaining.

The reward is a 5-dim exact-match vector over masked entities A..E. entity_A is
the question's answer; B..E are intermediate chain resolutions.

Contains entity normalization and EM scoring, the single- and multi-chain
response parsers, the prompt templates, dataset preprocessing (``preprocess``),
and the held-out eval (``evaluate``).
"""

from __future__ import annotations

import re

from vpo.task import SubRewrite, Task, dedup_by_vector, pad_matrix, register
from vpo.utils.eval_metrics import best_of_k_curve, diversity, solve_at_k_curve
from vpo.utils.parse_solutions import extract_numbered_tags
from vpo.utils.text_metrics import normalize_text

NUM_OBJECTIVES = 5
SUB_OBJECTIVE_NAMES = ["entity_A", "entity_B", "entity_C", "entity_D", "entity_E"]
ZERO_SCORES = [0.0] * NUM_OBJECTIVES

_TAG_LETTERS = ["A", "B", "C", "D", "E"]
_TAG_PATTERNS = [
    re.compile(rf"<entity_{ltr}>(.*?)</entity_{ltr}>", re.DOTALL | re.IGNORECASE)
    for ltr in _TAG_LETTERS
]


# ---------------------------------------------------------------------------
# Normalization + parsing + scoring
# ---------------------------------------------------------------------------


def normalize_entity(text: str) -> str:
    """Normalize a Wikipedia-style entity name for EM comparison."""
    return normalize_text(text, strip_articles=True, underscores_to_spaces=True)


def entity_em(pred: str, gold: str) -> float:
    """Exact match after normalization."""
    if not pred or not gold:
        return 0.0
    return float(normalize_entity(pred) == normalize_entity(gold))


def parse_response(text: str) -> list[str]:
    """Extract <entity_A>..<entity_E> contents from one chain (5 strings)."""
    out = []
    for pat in _TAG_PATTERNS:
        m = pat.search(text)
        out.append(m.group(1).strip() if m else "")
    return out


def parse_multi_responses(text: str, m: int) -> list[list[str]]:
    """Parse m chains from <response_i>...</response_i> tags (each 5 strings)."""
    return [parse_response(b) for b in extract_numbered_tags(text, m)]


def score_chain(pred_entities: list[str], gold_entities: list[str]) -> list[float]:
    """5-dim EM vector for one chain vs gold entities."""
    out = []
    for j in range(NUM_OBJECTIVES):
        pred = pred_entities[j] if j < len(pred_entities) else ""
        gold = gold_entities[j] if j < len(gold_entities) else ""
        out.append(entity_em(pred, gold))
    return out


def num_resolved(pred_entities: list[str]) -> int:
    """How many entity slots were non-empty (the model attempted)."""
    return sum(1 for s in pred_entities if s.strip())


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You will be given a multi-hop reasoning question over a small narrative. The narrative contains masked entities (Person A, Country B, etc.) connected by relations. The leaf entity is revealed in the final sentence; you must back-chain to identify the rest.

Narrative:
{narrative}

The masked entities you must resolve (in order of introduction in the narrative):
{mask_list}

For each masked entity above, output its underlying canonical name. Use underscored Wikipedia-style names exactly as they appear in the source (e.g. `Robert_Rodriguez`, `32nd_European_Film_Awards`, `From_Dusk_till_Dawn:_The_Series`).

Replace each `...` below with the resolved entity (NOTHING else inside the tags — just the canonical name):
<entity_A>...</entity_A>   (resolves {mask_a})
<entity_B>...</entity_B>   (resolves {mask_b})
<entity_C>...</entity_C>   (resolves {mask_c})
<entity_D>...</entity_D>   (resolves {mask_d})
<entity_E>...</entity_E>   (resolves {mask_e})

The answer to the question is whatever you put in <entity_A>."""

_SINGLE_OUTPUT_PATTERN = re.compile(
    r"Replace each `\.\.\.` below with the resolved entity[^\n]*\n"
    r"<entity_A>\.\.\.</entity_A>[^\n]*\n"
    r"<entity_B>\.\.\.</entity_B>[^\n]*\n"
    r"<entity_C>\.\.\.</entity_C>[^\n]*\n"
    r"<entity_D>\.\.\.</entity_D>[^\n]*\n"
    r"<entity_E>\.\.\.</entity_E>[^\n]*\n\n"
    r"The answer to the question is whatever you put in <entity_A>\.",
    re.DOTALL,
)

_MULTI_TEMPLATE = (
    "Provide {m} different reasoning chains, each wrapped in numbered "
    "tags <response_1>...</response_1> through <response_{m}>...</response_{m}>. "
    "Each chain must contain all 5 entity resolutions; replace each `...` "
    "with ONLY the resolved canonical name (underscored Wikipedia-style, "
    "no prose, no labels):\n"
    "  <response_i>\n"
    "    <entity_A>...</entity_A>   (resolves Person A — this is the answer)\n"
    "    <entity_B>...</entity_B>   (resolves the second mask)\n"
    "    <entity_C>...</entity_C>   (resolves the third mask)\n"
    "    <entity_D>...</entity_D>   (resolves the fourth mask)\n"
    "    <entity_E>...</entity_E>   (resolves the fifth mask)\n"
    "  </response_i>\n"
    "Closing tags are required for every response and every entity. The {m} "
    "chains should be genuinely different attempts (e.g. different "
    "candidate resolutions, not paraphrases). The answer to the question "
    "is whatever you put in <entity_A> within each chain."
)


# ---------------------------------------------------------------------------
# The Task
# ---------------------------------------------------------------------------


class EureqaTask(Task):
    name = "eureqa"
    data_source = "eureqa"
    objectives = [
        ("entity_A", "Resolving the answered entity (Person A)"),
        ("entity_B", "Resolving masked entity B"),
        ("entity_C", "Resolving masked entity C"),
        ("entity_D", "Resolving masked entity D"),
        ("entity_E", "Resolving masked entity E"),
    ]
    multi_rewrite = SubRewrite(
        pattern=_SINGLE_OUTPUT_PATTERN,
        template=_MULTI_TEMPLATE,
    )

    def weighted_scalar(self, vector):
        if not vector:
            return 0.0
        return sum(vector) / len(vector)

    def score_one(self, solution_str, ground_truth, extra_info):
        gold_entities = ground_truth["chain_entities"]
        pred_entities = parse_response(solution_str)
        n_resolved = num_resolved(pred_entities)
        sub = list(ZERO_SCORES) if n_resolved == 0 else score_chain(pred_entities, gold_entities)
        return {
            "sub_scores": sub,
            "channels": self.named_channels(sub),
            "scalar": self.weighted_scalar(sub),
            "extra": {
                "num_resolved": n_resolved,
                "answer_em": sub[0],
                "num_hops": ground_truth.get("num_hops", 5),
            },
            "num_test_executions": 0,
            "num_programs_executed": 1 if n_resolved else 0,
        }

    def score_multi(self, solution_str, m, ground_truth, extra_info):
        gold_entities = ground_truth["chain_entities"]
        chains = parse_multi_responses(solution_str, m)
        num_parsed = sum(1 for c in chains if num_resolved(c) > 0)

        matrix = [
            score_chain(chain, gold_entities) if num_resolved(chain) > 0 else list(ZERO_SCORES)
            for chain in chains
        ]
        padded = pad_matrix(dedup_by_vector(matrix), m, ZERO_SCORES)

        extra = self.multi_channel_extras(padded)
        extra["num_hops"] = ground_truth.get("num_hops", 5)
        return {
            "sub_scores": padded,
            "scalar": sum(self.weighted_scalar(row) for row in padded),
            "num_parsed": num_parsed,
            "num_test_executions": 0,
            "extra": extra,
        }

    # ----- preprocess -----
    def add_preprocess_args(self, p):
        p.add_argument("--local_save_dir", default="~/data/eureqa")
        p.add_argument("--mode", choices=["normal_to_hard", "mixed_split"],
                       default="normal_to_hard")
        p.add_argument("--train_split", default="questions_normal_5")
        p.add_argument("--test_split", default="questions_hard_5")
        p.add_argument("--mixed_split_seed", type=int, default=0)
        p.add_argument("--max_train", type=int, default=-1)
        p.add_argument("--max_test", type=int, default=-1)

    def preprocess(self, args):
        import os

        import datasets as hf_datasets

        print("Loading EUREQA from HuggingFace...")
        ds = hf_datasets.load_dataset("vincentleebang/EUREQA")

        if args.mode == "normal_to_hard":
            train_ds = ds[args.train_split]
            test_ds = ds[args.test_split]
            train_label, test_label = args.train_split, args.test_split
        else:
            normal = ds["questions_normal_5"]
            hard = ds["questions_hard_5"].shuffle(seed=args.mixed_split_seed)
            half = len(hard) // 2
            train_ds = hf_datasets.concatenate_datasets([normal, hard.select(range(half))])
            test_ds = hard.select(range(half, len(hard)))
            train_label = f"normal_5 + hard_5[:{half}] (seed={args.mixed_split_seed})"
            test_label = f"hard_5[{half}:] (seed={args.mixed_split_seed})"

        if args.max_train > 0:
            train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))
        if args.max_test > 0:
            test_ds = test_ds.select(range(min(args.max_test, len(test_ds))))

        train_rows = [_build_row(ex, "train", i) for i, ex in enumerate(train_ds)]
        test_rows = [_build_row(ex, "test", i) for i, ex in enumerate(test_ds)]

        save_dir = os.path.expanduser(args.local_save_dir)
        os.makedirs(save_dir, exist_ok=True)
        hf_datasets.Dataset.from_list(train_rows).to_parquet(os.path.join(save_dir, "train.parquet"))
        hf_datasets.Dataset.from_list(test_rows).to_parquet(os.path.join(save_dir, "test.parquet"))
        print(f"Saved {len(train_rows)} train ({train_label}), "
              f"{len(test_rows)} test ({test_label}) to {save_dir}")

    # ----- eval -----
    def add_eval_args(self, p):
        from vpo.eval_harness import add_sampling_eval_args, add_vllm_eval_args

        p.add_argument("--method", required=True,
                       choices=["grpo", "gdpo", "maxrl", "multi_rlvr", "vpo", "goal_cond"])
        p.add_argument("--data-dir", default="~/data/eureqa")
        p.add_argument("--output", default="results/eval_eureqa.json")
        p.add_argument("--num-prompts", type=int, default=None)
        add_sampling_eval_args(p, n_chains=30, num_solutions=1,
                               max_tokens=512, temperature=0.7, seed=42)
        add_vllm_eval_args(p, max_model_len=4096)

    def evaluate(self, args):
        import json
        import os

        import datasets as hf_datasets
        import numpy as np

        from vllm import SamplingParams
        from vpo.eval_harness import EvalHarness

        is_multi = args.method in ("multi_rlvr", "vpo")
        n, m = args.n_chains, args.num_solutions
        if not is_multi and m != 1:
            print(f"WARNING: method={args.method} is single-chain; forcing num_solutions=1")
            m = 1

        data_dir = os.path.expanduser(args.data_dir)
        ds = hf_datasets.Dataset.from_parquet(os.path.join(data_dir, "test.parquet"))
        if args.num_prompts:
            ds = ds.select(range(min(args.num_prompts, len(ds))))
        print(f"Loaded {len(ds)} test prompts; pool n={n} × m={m} = {n * m}")

        harness = EvalHarness(args.model, max_model_len=args.max_model_len,
                              gpu_mem=args.gpu_mem, tp=args.tp, seed=args.seed)

        formatted = []
        for row in ds:
            content = row["prompt"][-1]["content"]
            if is_multi:
                content = self.rewrite_multi_solution(content, m)
            formatted.append(harness.render(content, enable_thinking=args.enable_thinking))

        params = SamplingParams(temperature=args.temperature, top_p=1.0,
                                max_tokens=args.max_tokens, n=n, seed=args.seed)
        outputs = harness.llm.generate(formatted, params)

        num_prompts = len(ds)
        tensor = np.zeros((num_prompts, n, m, NUM_OBJECTIVES))
        for i, row in enumerate(ds):
            gt = row["reward_model"]["ground_truth"]
            if isinstance(gt, str):
                gt = json.loads(gt)
            gold = gt.get("chain_entities", []) or [""] * NUM_OBJECTIVES
            for j in range(n):
                text = outputs[i].outputs[j].text
                chains = parse_multi_responses(text, m) if is_multi else [parse_response(text)]
                for k in range(m):
                    pred = chains[k] if k < len(chains) else [""] * NUM_OBJECTIVES
                    tensor[i, j, k] = list(ZERO_SCORES) if num_resolved(pred) == 0 else score_chain(pred, gold)

        per_component = tensor.mean(axis=(0, 1, 2))
        flat = tensor.reshape(num_prompts, n * m, NUM_OBJECTIVES)
        means = flat.mean(axis=-1)
        full = (flat == 1.0).all(axis=-1).astype(float)
        rng = np.random.default_rng(args.seed)
        K_total = means.shape[1]
        best = np.zeros((num_prompts, K_total))
        passk = np.zeros((num_prompts, K_total))
        for p in range(num_prompts):
            perm = rng.permutation(K_total)  # one ordering shared by both curves
            best[p] = best_of_k_curve(means[p], perm)
            passk[p] = solve_at_k_curve(full[p], perm)
        best_curve, pass_curve = best.mean(axis=0), passk.mean(axis=0)

        diversity_5d = float(np.mean([diversity(flat[p]) for p in range(num_prompts)]))

        summary = {
            "model": args.model, "method": args.method, "n_chains": n,
            "num_solutions": m, "pool_size": n * m, "num_prompts": num_prompts,
            "temperature": args.temperature, "max_tokens": args.max_tokens,
            "enable_thinking": args.enable_thinking, "seed": args.seed,
            "feature_names": SUB_OBJECTIVE_NAMES,
            "per_component_mean": per_component.tolist(),
            "diversity_pairwise_l1_5d": diversity_5d,
            "best_at_k_curve": best_curve.tolist(),
            "pass_at_k_curve": pass_curve.tolist(),
        }
        for k in [1, 2, 3, 5, 10, 15, 20, 25, 30]:
            if k <= K_total:
                summary[f"best_at_k/k={k}/mean"] = float(best_curve[k - 1])
                summary[f"pass_at_k/k={k}/mean"] = float(pass_curve[k - 1])

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({**summary, "tensor_shape": list(tensor.shape)}, f, indent=2)
        np.save(args.output.replace(".json", "_tensor.npy"), tensor)
        print(json.dumps(summary, indent=2))


def _build_row(example: dict, split: str, index: int) -> dict:
    narrative = "\n".join(example["question"])
    masks = example["entity_masks"]
    entities = example["entities"]
    if len(masks) < 6 or len(entities) < 6:
        raise ValueError(
            f"Expected 5-hop question with 6 masks/entities, got "
            f"{len(masks)} masks / {len(entities)} entities at idx={index}"
        )
    content = PROMPT_TEMPLATE.format(
        narrative=narrative,
        mask_list="\n".join(f"  - {m}" for m in masks[:5]),
        mask_a=masks[0], mask_b=masks[1], mask_c=masks[2],
        mask_d=masks[3], mask_e=masks[4],
    )
    return {
        "data_source": "eureqa",
        "prompt": [{"role": "user", "content": content}],
        "ability": "reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answer": entities[0],
                "chain_entities": entities[:5],
                "entity_masks": masks[:5],
                "num_hops": 5,
            },
        },
        "extra_info": {
            "split": split, "index": index, "task_id": index,
            "num_test_cases": 5, "eureqa_idx": example["idx"], "num_hops": 5,
        },
    }


TASK = register(EureqaTask())
