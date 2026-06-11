"""Thin CLI shim — the LiveCodeBench eval lives in tasks/livecodebench.py.

Usage: python eval/eval_lcb.py --model <hf_path> --method vpo --n-chains 10 --num-solutions 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vpo.eval_harness import bootstrap_eval_process  # noqa: E402

# Must run before vLLM (imported lazily inside TASK.evaluate) loads CUDA.
bootstrap_eval_process()

from vpo_tasks.livecodebench import TASK  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF-format merged checkpoint or HF model id")
    TASK.add_eval_args(p)
    TASK.evaluate(p.parse_args())


if __name__ == "__main__":
    main()
