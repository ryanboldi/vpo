"""Tests for the shared text normalization + F1/EM metrics.

These guard the single implementation in ``vpo.utils.text_metrics`` that
musique (answers), eureqa (entities) and tool (argument values) all wrap —
including equivalence with the per-task copies they replaced.

Run directly (``python tests/test_text_metrics.py``) or under pytest.
"""

import re
import string
from collections import Counter

from vpo.utils.text_metrics import (
    exact_match,
    multiset_f1,
    normalize_text,
    set_f1,
    token_f1,
)


# ─── normalize_text ──────────────────────────────────────────────────────────


def test_normalize_basic():
    assert normalize_text('  The  Quick, Brown Fox! ') == 'the quick brown fox'
    assert normalize_text(None) == ''
    assert normalize_text(42) == '42'


def test_normalize_flags():
    assert normalize_text('The Fox', strip_articles=True) == 'fox'
    assert normalize_text('A_b_c', underscores_to_spaces=True) == 'a b c'
    assert (
        normalize_text('The_Lord_of_the_Rings!', strip_articles=True,
                       underscores_to_spaces=True)
        == 'lord of rings'
    )


def _legacy_musique_normalize(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def test_normalize_matches_legacy_musique():
    for s in ("An Apple a Day", "the   end.", "O'Brien & sons", ""):
        assert normalize_text(s, strip_articles=True) == _legacy_musique_normalize(s)


def _legacy_tool_normalize(s):
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def test_normalize_matches_legacy_tool():
    for s in (None, 3.5, "A, B!", " the end ", ""):
        assert normalize_text(s) == _legacy_tool_normalize(s)


# ─── F1 / EM ─────────────────────────────────────────────────────────────────


def test_token_f1_known():
    assert token_f1('', '') == 1.0
    assert token_f1('a word', '') == 0.0
    assert token_f1('', 'a word') == 0.0
    assert token_f1('quick fox', 'quick brown fox') == 2 * 1.0 * (2 / 3) / (1.0 + 2 / 3)


def _legacy_musique_f1(pred, gold):
    pred_tokens = _legacy_musique_normalize(pred).split()
    gold_tokens = _legacy_musique_normalize(gold).split()
    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def test_token_f1_matches_legacy_musique():
    cases = [
        ('the quick fox', 'a quick brown fox'),
        ('', ''), ('x', ''), ('', 'x'),
        ('exact match', 'exact match'),
        ('no overlap at all', 'completely different words'),
    ]
    for pred, gold in cases:
        assert abs(
            token_f1(pred, gold, strip_articles=True) - _legacy_musique_f1(pred, gold)
        ) < 1e-12


def test_multiset_and_set_f1():
    assert multiset_f1([], []) == 1.0
    assert multiset_f1(['x'], []) == 0.0
    assert multiset_f1(['x', 'x'], ['x']) == 2 * 0.5 * 1.0 / 1.5
    assert set_f1(set(), set()) == 1.0
    assert set_f1({'a'}, {'a', 'b'}) == 2 * 1.0 * 0.5 / 1.5


def test_exact_match():
    assert exact_match('The Fox.', 'fox', strip_articles=True) == 1.0
    assert exact_match('fox', 'foxes') == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} text-metric tests passed.")
