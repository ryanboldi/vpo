"""Thin CLI shim — the MuSiQue preprocessor lives in tasks/musique.py.

Usage: python data/preprocess_musique.py --local_save_dir ~/data/musique
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vpo_tasks.musique import TASK  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    TASK.add_preprocess_args(p)
    TASK.preprocess(p.parse_args())


if __name__ == "__main__":
    main()
