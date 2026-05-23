#!/usr/bin/env python3
"""Fix math block delimiters so pymdownx.arithmatex parses correctly.

Problem
-------
Source `.md` files have patterns like:

    $$
    formula here
    $$ <a id="eq-xxx"></a>

pymdownx.arithmatex requires `$$` to be on its own line for display-math
recognition. The trailing `<a id="...">` on the same line breaks parsing,
so the block is left as raw `$$ ... $$` in HTML, and MathJax (configured
for `\[ ... \]` only) does not render it.

Fix
---
Move the anchor BEFORE the math block:

    <a id="eq-xxx"></a>

    $$
    formula here
    $$

This preserves cross-reference targets while letting arithmatex parse.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"


# Match a display math block whose closing $$ has a trailing anchor:
#   $$
#   ...content...
#   $$ <a id="eq-xxx"></a>
#
# Capture: (content), (anchor_id)
BLOCK_RE = re.compile(
    r"(?ms)^\$\$\n(.*?)\n\$\$[ \t]+(<a id=\"([\w:-]+)\"></a>)[ \t]*$",
)


def fix_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    new, n = BLOCK_RE.subn(
        lambda m: f'<a id="{m.group(3)}"></a>\n\n$$\n{m.group(1)}\n$$',
        text,
    )
    if n:
        path.write_text(new, encoding="utf-8")
    return n


def main() -> int:
    total_blocks = 0
    files_changed = 0
    for md in sorted(DOCS.rglob("*.md")):
        n = fix_file(md)
        if n:
            files_changed += 1
            total_blocks += n
            print(f"  {md.relative_to(ROOT)}: fixed {n} blocks")
    print(f"\nTotal: {total_blocks} math blocks fixed across {files_changed} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
