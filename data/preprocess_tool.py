"""Thin CLI shim — the ToolRL preprocessor lives in tasks/tool.py.

Usage: python data/preprocess_tool.py --local_save_dir ~/data/toolrl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vpo_tasks.tool import TASK  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    TASK.add_preprocess_args(p)
    TASK.preprocess(p.parse_args())


if __name__ == "__main__":
    main()
