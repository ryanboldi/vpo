"""Thin CLI shim — the MuSiQue eval lives in tasks/musique.py (TASK.evaluate).

Usage: python eval/eval_musique.py --model <hf_ckpt> --method vpo --num-examples 300
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vpo.eval_harness import bootstrap_eval_process  # noqa: E402

# Must run before vLLM (imported lazily inside TASK.evaluate) loads CUDA.
bootstrap_eval_process()

from vpo_tasks.musique import TASK  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF-format merged checkpoint or HF model id")
    TASK.add_eval_args(p)
    TASK.evaluate(p.parse_args())


if __name__ == "__main__":
    main()
