#!/usr/bin/env python3
"""Fix backslash-underscore inside math blocks.

Bug
---
MathJax 3 inside `\text{...}` renders `\_` literally (backslash + underscore
visible) instead of just `_`. Example:

    \text{frame\_length}    →  rendered as "frame\_length" (with visible backslash)

In `\text{...}` mode, underscore `_` is already a literal character (no
subscript interpretation), so the escape is unnecessary. Replacing `\_`
with `_` produces correct rendering across MathJax/KaTeX.

Strategy
--------
Operate ONLY inside math regions:
- Display math: $$ ... $$
- Inline math:  $ ... $   (skip dollar-pairs to avoid currency false-positives)
- TeX-style:    \( ... \) and \[ ... \]

Within those regions, replace `\_` with `_`.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"

# Line-based state machine: toggle "inside display math" on any line whose
# stripped content is exactly "$$" (handles both top-level and indented blocks
# inside admonitions where the line has leading whitespace).
DISPLAY_DELIM = "$$"

# Inline math regex (single-line, not double-dollar)
INLINE_RE = re.compile(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)")


def fix_math_region(text: str) -> str:
    """Replace backslash-underscore with underscore within math content."""
    return text.replace("\\_", "_")


def fix_inline_in_line(line: str) -> tuple[str, int]:
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        body = m.group(1)
        fixed = fix_math_region(body)
        count += body.count("\\_") - fixed.count("\\_")
        return f"${fixed}$"

    return INLINE_RE.sub(repl, line), count


def fix_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    out: list[str] = []
    inside_display = False
    total = 0

    for line in lines:
        stripped = line.strip()
        if stripped == DISPLAY_DELIM:
            inside_display = not inside_display
            out.append(line)
            continue

        if inside_display:
            before = line.count("\\_")
            fixed = fix_math_region(line)
            after = fixed.count("\\_")
            total += before - after
            out.append(fixed)
        else:
            fixed_line, n = fix_inline_in_line(line)
            total += n
            out.append(fixed_line)

    if total > 0:
        path.write_text("\n".join(out), encoding="utf-8")
    return total


def main() -> int:
    total = 0
    changed = 0
    for md in sorted(DOCS.rglob("*.md")):
        n = fix_file(md)
        if n:
            changed += 1
            total += n
            print(f"  {md.relative_to(ROOT)}: fixed {n} occurrences")
    print(f"\nTotal: {total} occurrences across {changed} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
