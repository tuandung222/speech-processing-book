#!/usr/bin/env python3
"""Re-escape underscores inside \\text{...} for KaTeX.

Context
-------
Earlier (when using MathJax + MkDocs), we replaced \\_ with _ inside math because
MathJax was rendering \\_ as literal backslash-underscore. Now that we use
KaTeX (mdbook-katex preprocessor), KaTeX needs \\_ for literal underscore
inside \\text{...} — otherwise it interprets _ as subscript and errors:

    ParseError: Expected 'EOF', got '_' at position N: \\text{frame_length}

Strategy
--------
Within `\\text{...}` groups (inside both inline $...$ and display $$...$$),
replace `_` with `\\_`. Use line-based state machine for display blocks.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


# Inside a \text{...} group, replace _ with \_.
# Use a non-greedy match for the brace content, but since \text can be nested
# we keep it simple by matching balanced first-level braces only.
TEXT_GROUP_RE = re.compile(r"\\text\{([^{}]*)\}")


def fix_math_region(text: str) -> str:
    """Within math content, escape _ → \\_ inside \\text{} groups."""
    def repl(m: re.Match) -> str:
        inner = m.group(1)
        # Only escape underscores not already preceded by backslash
        escaped = re.sub(r"(?<!\\)_", r"\\_", inner)
        return r"\text{" + escaped + r"}"
    return TEXT_GROUP_RE.sub(repl, text)


INLINE_RE = re.compile(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)")


def fix_inline_in_line(line: str) -> tuple[str, int]:
    count = 0
    def repl(m: re.Match) -> str:
        nonlocal count
        body = m.group(1)
        fixed = fix_math_region(body)
        if fixed != body:
            count += 1
        return f"${fixed}$"
    return INLINE_RE.sub(repl, line), count


def fix_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    out: list[str] = []
    inside_display = False
    total = 0

    for line in lines:
        if line.strip() == "$$":
            inside_display = not inside_display
            out.append(line)
            continue

        if inside_display:
            fixed = fix_math_region(line)
            if fixed != line:
                total += 1
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
    for md in sorted(SRC.rglob("*.md")):
        n = fix_file(md)
        if n:
            changed += 1
            total += n
            print(f"  {md.relative_to(ROOT)}: {n} lines fixed")
    print(f"\nTotal: {total} lines across {changed} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
