"""Shared text normalization and F1/EM metrics.

One implementation of the SQuAD-style normalize → token-F1 / exact-match
pipeline used by the QA-flavored tasks (musique answers, eureqa entities,
tool argument values). Task differences are flags, not copies:

  - ``strip_articles``: drop a/an/the (musique, eureqa)
  - ``underscores_to_spaces``: Wikipedia-style entity names (eureqa)

Tasks keep thin named wrappers (``normalize_answer``, ``normalize_entity``,
``_normalize``) so their scoring code reads in domain vocabulary, but the
math lives — and is tested — here.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(
    text,
    *,
    strip_articles: bool = False,
    underscores_to_spaces: bool = False,
) -> str:
    """Lowercase, strip punctuation, collapse whitespace; optional extras."""
    if text is None:
        return ""
    text = str(text)
    if underscores_to_spaces:
        text = text.replace("_", " ")
    text = text.lower()
    if strip_articles:
        text = _ARTICLE_RE.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    return " ".join(text.split())


def multiset_f1(pred_items, gold_items) -> float:
    """F1 over two multisets (1.0 when both are empty)."""
    if not pred_items and not gold_items:
        return 1.0
    if not pred_items or not gold_items:
        return 0.0
    pc, gc = Counter(pred_items), Counter(gold_items)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    p = overlap / sum(pc.values())
    r = overlap / sum(gc.values())
    return 2 * p * r / (p + r)


def set_f1(pred_set, gold_set) -> float:
    """F1 over two sets (1.0 when both are empty)."""
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    if overlap == 0:
        return 0.0
    p = overlap / len(pred_set)
    r = overlap / len(gold_set)
    return 2 * p * r / (p + r)


def token_f1(pred, gold, **norm_kwargs) -> float:
    """Word-level F1 over normalized tokens (1.0 when both sides are empty)."""
    return multiset_f1(
        normalize_text(pred, **norm_kwargs).split(),
        normalize_text(gold, **norm_kwargs).split(),
    )


def exact_match(pred, gold, **norm_kwargs) -> float:
    """1.0 iff the two strings normalize identically."""
    return float(
        normalize_text(pred, **norm_kwargs) == normalize_text(gold, **norm_kwargs)
    )
