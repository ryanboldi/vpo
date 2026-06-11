"""AST-based deduplication of Python code solutions."""

import ast

from vpo.utils.parse_solutions import extract_code_block


def ast_dedup(solutions: list[str]) -> list[str]:
    """Keep only structurally unique solutions via AST comparison.

    Falls back to normalized string comparison for unparseable code.

    Args:
        solutions: List of solution strings (may contain markdown code blocks).

    Returns:
        Deduplicated list, preserving order. Always returns at least 1 element.
    """
    seen: list[str] = []
    unique: list[str] = []

    for sol in solutions:
        if not sol:
            continue
        code = extract_code_block(sol)
        try:
            fingerprint = ast.dump(ast.parse(code))
        except SyntaxError:
            # Fallback: normalize whitespace and case
            fingerprint = " ".join(code.split()).lower()
        except RecursionError:
            # Pathologically deep AST (e.g. thousands of nested ops). Fall
            # back to the same normalized-string fingerprint we use for
            # unparseable code. Without this catch, the worker dies and
            # the entire training step hangs.
            fingerprint = " ".join(code.split()).lower()

        if fingerprint not in seen:
            seen.append(fingerprint)
            unique.append(sol)

    return unique if unique else solutions[:1]
