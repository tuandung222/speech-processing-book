#!/usr/bin/env python3
"""Fix admonition body indentation in docs/ — specifically when a `$$` math
block has its closing delimiter de-indented (a bug from earlier migration).

Logic
-----
Walk the file. When we encounter `!!! type ...`, we enter admonition mode.
We collect lines until we see a NON-INDENTED non-blank line (which ends the
admonition). However, if a `$$` line appears WITHOUT indent but we're inside
an admonition AND we previously saw an indented `$$` opening, we treat that
trailing `$$` as the closing and re-indent it.

Also handles the case where the line AFTER the closing `$$` is indented
(meaning admonition body continues).
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"

ADMON_RE = re.compile(r'^!!! \w+(?:\s+"[^"]*")?\s*$')


def fix_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    fixes = 0

    while i < n:
        line = lines[i]
        if not ADMON_RE.match(line):
            out.append(line)
            i += 1
            continue

        # Found admonition. Emit header line.
        out.append(line)
        i += 1

        # Detect indent of FIRST body line
        admon_indent = None
        body_buffer: list[str] = []

        # Scan body until we hit a non-indented non-blank line
        while i < n:
            ln = lines[i]
            if ln == "":
                # Blank line — keep, check if next line continues
                body_buffer.append(ln)
                i += 1
                continue
            if ln.startswith("    ") or ln.startswith("\t"):
                if admon_indent is None:
                    admon_indent = "    " if ln.startswith("    ") else "\t"
                body_buffer.append(ln)
                i += 1
            elif ln.strip() == "$$" and admon_indent is not None:
                # Un-indented $$ inside an admonition body — re-indent it.
                body_buffer.append(admon_indent + "$$")
                fixes += 1
                i += 1
            else:
                # Non-indented non-blank, non-$$. Admonition ended.
                break

        # Strip trailing blank lines from body_buffer (they're not body)
        while body_buffer and body_buffer[-1] == "":
            i_back = len(body_buffer) - 1
            out_before_buf_len = 0  # unused
            body_buffer.pop()

        # Emit body
        out.extend(body_buffer)
        # Re-add the trailing blanks we stripped (they belong to outer scope)
        # Actually we want blank between body and next section, but we already
        # consumed them above. Add single blank to separate.
        out.append("")

    if fixes > 0:
        path.write_text("\n".join(out), encoding="utf-8")
    return fixes


def main() -> int:
    total = 0
    changed = 0
    for md in sorted(DOCS.rglob("*.md")):
        n = fix_file(md)
        if n:
            changed += 1
            total += n
            print(f"  {md.relative_to(ROOT)}: {n} de-indented $$ re-indented")
    print(f"\nTotal: {total} fixes across {changed} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
