"""Thin CLI shim — the maze dataset generator lives in tasks/maze.py.

Usage: python data/preprocess_maze.py --local_save_dir ~/data/maze
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vpo_tasks.maze import TASK  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    TASK.add_preprocess_args(p)
    TASK.preprocess(p.parse_args())


if __name__ == "__main__":
    main()
