"""Parse multiple solutions from a single model generation."""

import re


def extract_code_block(text: str) -> str:
    """Extract code from a ```python ... ``` block.

    Handles truncated responses (missing closing ```) by falling back to
    open-ended capture.
    """
    # Complete python block
    match = re.search(r"```python\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Truncated python block (no closing fence — vLLM hit max_tokens)
    match = re.search(r"```python\s*\n?(.+)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Generic code block
    match = re.search(r"```\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_numbered_tags(text, m: int, tag: str = "response") -> list[str]:
    """Extract the bodies of m numbered ``<tag_i>...</tag_i>`` blocks (i=1..m).

    The shared kernel of every task's multi-solution response parser.
    Returns exactly m strings; a missing block is ``""``. Non-string input
    (e.g. a non-text generation) yields all-empty.
    """
    if not isinstance(text, str):
        return [""] * m
    out = []
    for i in range(1, m + 1):
        match = re.search(rf"<{tag}_{i}>(.*?)</{tag}_{i}>", text, re.DOTALL)
        out.append(match.group(1) if match else "")
    return out


def parse_solutions(response_str: str, m: int, delimiter: str = "---") -> list[str]:
    """Split a single generation into up to m sub-solutions.

    Tries delimiter-based splitting first, then falls back to extracting
    all ```python blocks.

    Args:
        response_str: The full model response containing m solutions.
        m: Expected number of solutions.
        delimiter: Separator between solutions.

    Returns:
        List of exactly m solution strings (padded with "" if fewer found).
    """
    # Strategy 1: explicit ```python blocks are the most reliable multi-solution
    # signal. Prefer them when at least two are present.
    blocks = re.findall(r"```python\s*\n?(.*?)```", response_str, re.DOTALL)
    if len(blocks) >= 2:
        parts = [b.strip() for b in blocks]
    else:
        # Strategy 2: split on the delimiter the prompt tells models to use, but
        # only when it occupies its OWN line (a separator rule). A stray "---"
        # *inside* a solution — an inline comment, a markdown rule mid-prose —
        # must not shred that single solution into garbage fragments.
        # `\r?` so a CRLF delimiter line ("---\r\n") still matches — MULTILINE
        # `$` sits before the \n, leaving the \r that this consumes.
        sep = re.compile(rf"^[ \t]*{re.escape(delimiter)}[ \t]*\r?$", re.MULTILINE)
        parts = [p.strip() for p in sep.split(response_str) if p.strip()]
        if len(parts) <= 1 and len(blocks) == 1:
            parts = [blocks[0].strip()]

    # Pad to exactly m
    while len(parts) < m:
        parts.append("")

    return parts[:m]
