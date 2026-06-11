"""vLLM helper used by task ``evaluate()`` methods.

``EvalHarness`` loads a vLLM model and its tokenizer and provides:
  - ``render`` / ``apply_chat_template`` — turn prompts into model input text
  - ``generate`` — batched sampling
  - ``load_test`` — read a ``test.parquet`` as a column dict

Module-level helpers shared by the eval CLIs:
  - ``bootstrap_eval_process`` — process init (spawn before vLLM/CUDA)
  - ``add_sampling_eval_args`` / ``add_vllm_eval_args`` — the argparse flags
    every task's ``add_eval_args`` shares, parameterized by per-task defaults
"""

from __future__ import annotations

import os


def bootstrap_eval_process() -> None:
    """Process init for the eval CLI shims. Call before any vLLM import.

    vLLM workers re-initialize CUDA; under the Linux default ``fork`` start
    method that crashes, so force ``spawn`` for both this process and vLLM's
    own worker pool.
    """
    import multiprocessing

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass


def add_sampling_eval_args(
    parser,
    *,
    n_chains: int,
    num_solutions: int | None,
    max_tokens: int | None,
    temperature: float,
    seed: int,
) -> None:
    """Register the sampling flags every task's eval shares.

    Defaults are per-task (passed by the caller), so consolidating here
    changes no CLI behavior — it only keeps the flag *names* from drifting.
    """
    parser.add_argument("--n-chains", type=int, default=n_chains,
                        help="rollouts per prompt")
    parser.add_argument("--num-solutions", type=int, default=num_solutions,
                        help="m solutions per rollout (multi-sol methods)")
    parser.add_argument("--max-tokens", type=int, default=max_tokens)
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--seed", type=int, default=seed)


def add_vllm_eval_args(
    parser,
    *,
    max_model_len: int,
    tp: int = 1,
    gpu_mem: float = 0.85,
) -> None:
    """Register the vLLM-sizing flags shared by tasks that expose them."""
    parser.add_argument("--max-model-len", type=int, default=max_model_len)
    parser.add_argument("--tp", type=int, default=tp)
    parser.add_argument("--gpu-mem", type=float, default=gpu_mem)
    parser.add_argument("--enable-thinking", action="store_true")


class EvalHarness:
    """Holds a loaded vLLM model + tokenizer and the common eval primitives."""

    def __init__(
        self,
        model: str,
        *,
        max_model_len: int,
        gpu_mem: float = 0.85,
        tp: int = 1,
        dtype: str = "bfloat16",
        seed: int = 0,
        enforce_eager: bool = False,
        trust_remote_code: bool = False,
    ):
        # Imported lazily so `--help` and import-checks work without vLLM/CUDA.
        from vllm import LLM

        self.model = model
        llm_kwargs = dict(
            model=model,
            dtype=dtype,
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_model_len,
            tensor_parallel_size=tp,
            seed=seed,
        )
        if enforce_eager:
            llm_kwargs["enforce_eager"] = True
        if trust_remote_code:
            llm_kwargs["trust_remote_code"] = True
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def apply_chat_template(self, messages: list[dict], **chat_template_kwargs) -> str:
        """Render a full chat-message list, tolerating tokenizers that don't
        accept ``enable_thinking`` (drops it and retries)."""
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **chat_template_kwargs,
            )
        except TypeError:
            chat_template_kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **chat_template_kwargs,
            )

    def render(self, content: str, **chat_template_kwargs) -> str:
        """Apply the chat template to a single user-message string."""
        return self.apply_chat_template(
            [{"role": "user", "content": content}], **chat_template_kwargs,
        )

    def generate(
        self,
        prompts: list[str],
        *,
        n: int,
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        seed: int = 0,
    ):
        """Batched sampling; returns vLLM RequestOutputs (one per prompt)."""
        from vllm import SamplingParams

        sp = SamplingParams(
            temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, n=n, seed=seed,
        )
        return self.llm.generate(prompts, sp)

    @staticmethod
    def load_test(data_dir: str) -> dict:
        """Read ``<data_dir>/test.parquet`` as a column dict."""
        import pyarrow.parquet as pq

        path = os.path.join(os.path.expanduser(data_dir), "test.parquet")
        return pq.read_table(path).to_pydict()
