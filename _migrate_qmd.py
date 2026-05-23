#!/usr/bin/env python3
"""Convert Quarto .qmd files to MkDocs Material .md files.

Transformations applied:
- Strip YAML frontmatter (keep title as H1 if not present).
- ::: {.callout-note title="Foo"} body :::  →  !!! note "Foo"\n    body
- @key citations (auto-cite) → [^key] footnote refs; reference defs appended at end.
- Cross-refs @fig-xxx / @tbl-xxx / @eq-xxx → linked text "[Hình X.Y](#fig-xxx)" simplified.
- {#fig-xxx} / {#tbl-xxx} attribute id → <a id="fig-xxx"></a> placed above.
- Code fences with ```{python} → ```python.
- Misc Quarto-only artifacts removed.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
SRC_FILES: list[tuple[Path, Path]] = []  # (src .qmd, dst .md)


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
TITLE_RE = re.compile(r'^title:\s*"?(.+?)"?\s*$', re.MULTILINE)

# Callout pattern: ::: {.callout-TYPE [title="X"]} ... :::
# Supports both:
#   ::: {.callout-note title="Foo"}
#   ::: {.callout-note}
CALLOUT_RE = re.compile(
    r":::\s*\{\.callout-(note|tip|warning|important|caution|info)"
    r"(?:\s+title=\"([^\"]+)\")?"
    r"[^}]*\}\s*\n"
    r"(.*?)"
    r":::",
    re.DOTALL,
)

# Generic Quarto block (e.g. ::: {.column-margin}, ::: {.panel-tabset}, etc.)
QUARTO_DIV_RE = re.compile(
    r":::\s*\{[^}]+\}\s*\n(.*?):::",
    re.DOTALL,
)

# Code fences with curly braces: ```{python}, ```{r}, ```{python .echo}
CODE_LANG_BRACE_RE = re.compile(r"```\{(\w+)(?:[^}]*)\}")

# Citation: [@key1; @key2; @key3] or [@key]
CITE_BLOCK_RE = re.compile(r"\[((?:[-\w@; ]+))\]")
CITE_SINGLE_RE = re.compile(r"@([A-Za-z][\w-]+)")  # standalone @key

# Cross-ref: @fig-xxx, @tbl-xxx, @eq-xxx, @sec-xxx
CROSSREF_RE = re.compile(r"@(fig|tbl|eq|sec)-([\w-]+)")

# Attribute id: {#fig-xxx} or {#tbl-xxx .class}
ATTR_ID_RE = re.compile(r"\{#((?:fig|tbl|eq|sec)-[\w-]+)(?:[^}]*)\}")

# Quarto figure with caption like ![Caption](path){#fig-x}
FIG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)\{#(fig-[\w-]+)[^}]*\}")


CALLOUT_TYPE_MAP = {
    "note": "note",
    "tip": "tip",
    "warning": "warning",
    "important": "danger",   # Material: danger for important
    "caution": "warning",
    "info": "info",
}


def strip_frontmatter(text: str) -> tuple[str, str | None]:
    """Return (body without frontmatter, title if present)."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return text, None
    fm = m.group(1)
    title_m = TITLE_RE.search(fm)
    title = title_m.group(1).strip() if title_m else None
    return text[m.end():], title


def transform_callouts(text: str) -> str:
    def repl(m: re.Match) -> str:
        ctype = m.group(1)
        title = m.group(2)
        body = m.group(3).strip()
        mat = CALLOUT_TYPE_MAP.get(ctype, "note")
        head = f'!!! {mat} "{title}"' if title else f"!!! {mat}"
        indented = "\n".join("    " + ln if ln else "" for ln in body.split("\n"))
        return f"{head}\n{indented}\n"
    return CALLOUT_RE.sub(repl, text)


def transform_generic_quarto_divs(text: str) -> str:
    """Convert remaining ::: {...} blocks (not callouts) to plain content."""
    def repl(m: re.Match) -> str:
        body = m.group(1).strip()
        return body + "\n"
    return QUARTO_DIV_RE.sub(repl, text)


def transform_code_fences(text: str) -> str:
    return CODE_LANG_BRACE_RE.sub(r"```\1", text)


def transform_figures(text: str) -> str:
    """Convert ![cap](path){#fig-x} → <figure>...</figure> with id."""
    def repl(m: re.Match) -> str:
        cap = m.group(1)
        path = m.group(2)
        fid = m.group(3)
        return (
            f'<figure markdown id="{fid}">\n'
            f"  ![{cap}]({path})\n"
            f"  <figcaption>{cap}</figcaption>\n"
            f"</figure>"
        )
    return FIG_RE.sub(repl, text)


def transform_attr_ids(text: str) -> str:
    """Convert {#xxx-xxx} (orphan attr ids on lines) to anchor tags."""
    def repl(m: re.Match) -> str:
        target = m.group(1)
        return f'<a id="{target}"></a>'
    return ATTR_ID_RE.sub(repl, text)


def transform_crossrefs(text: str) -> str:
    """@fig-foo → [Hình](#fig-foo). @tbl-foo → [Bảng](#tbl-foo). @eq-foo → [Phương trình](#eq-foo)."""
    label_map = {"fig": "Hình", "tbl": "Bảng", "eq": "Phương trình", "sec": "Mục"}
    def repl(m: re.Match) -> str:
        kind = m.group(1)
        slug = m.group(2)
        label = label_map[kind]
        return f"[{label}](#{kind}-{slug})"
    return CROSSREF_RE.sub(repl, text)


def collect_citations(text: str) -> list[str]:
    """Collect all citation keys used in the text."""
    keys: list[str] = []
    # @key in brackets [@key1; @key2]
    for m in CITE_BLOCK_RE.finditer(text):
        inner = m.group(1)
        for sub in CITE_SINGLE_RE.finditer(inner):
            keys.append(sub.group(1))
    # standalone @key (outside brackets) — handled elsewhere as crossref already
    return list(dict.fromkeys(keys))  # dedupe preserve order


def transform_citations(text: str, bib_keys: dict[str, str]) -> str:
    """Convert [@key1; @key2] → [^key1] [^key2] footnote refs.
    bib_keys: {key: "Author, Title, Year"} for footnote definitions.
    """
    def repl_block(m: re.Match) -> str:
        inner = m.group(1)
        keys = [s.group(1) for s in CITE_SINGLE_RE.finditer(inner)]
        if not keys:
            return m.group(0)  # not a citation block
        return " ".join(f"[^{k}]" for k in keys)
    return CITE_BLOCK_RE.sub(repl_block, text)


def load_bib(bib_path: Path) -> dict[str, str]:
    """Parse references.bib loosely to extract key → short description."""
    if not bib_path.exists():
        return {}
    text = bib_path.read_text(encoding="utf-8")
    entries: dict[str, str] = {}
    # @article{key, author = {...}, title = {...}, year = {...}, ...}
    entry_re = re.compile(r"@\w+\{([\w-]+),\s*(.*?)\n\}", re.DOTALL)
    field_re = re.compile(r"(\w+)\s*=\s*[{\"](.+?)[}\"]\s*[,\n]", re.DOTALL)
    for m in entry_re.finditer(text):
        key = m.group(1)
        fields = dict(field_re.findall(m.group(2)))
        author = fields.get("author", "").replace("\n", " ").strip()
        title = fields.get("title", "").replace("\n", " ").strip()
        year = fields.get("year", "").strip()
        journal = fields.get("journal", fields.get("booktitle", "")).strip()
        bits = [b for b in [author, f'"{title}"' if title else "", journal, year] if b]
        entries[key] = ", ".join(bits)
    return entries


def append_footnote_defs(text: str, used_keys: list[str], bib: dict[str, str]) -> str:
    if not used_keys:
        return text
    seen = set()
    lines = ["", "", "---", "", "<!-- References (auto-generated from .bib) -->"]
    for key in used_keys:
        if key in seen:
            continue
        seen.add(key)
        desc = bib.get(key, f"Reference: {key}")
        lines.append(f"[^{key}]: {desc}")
    return text + "\n" + "\n".join(lines) + "\n"


def convert_file(src: Path, dst: Path, bib: dict[str, str]) -> None:
    text = src.read_text(encoding="utf-8")

    body, fm_title = strip_frontmatter(text)

    # Order matters: figures first (preserve {#fig-x}), then attr ids, then crossrefs.
    body = transform_figures(body)
    body = transform_attr_ids(body)
    body = transform_callouts(body)
    body = transform_generic_quarto_divs(body)
    body = transform_code_fences(body)
    body = transform_crossrefs(body)

    cite_keys = collect_citations(body)
    body = transform_citations(body, bib)
    body = append_footnote_defs(body, cite_keys, bib)

    # Ensure H1 title present
    if fm_title and not body.lstrip().startswith("# "):
        body = f"# {fm_title}\n\n" + body.lstrip("\n")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(body, encoding="utf-8")
    print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}  ({len(cite_keys)} cites)")


def main() -> int:
    bib = load_bib(ROOT / "references.bib")
    print(f"Loaded {len(bib)} bib entries.")

    # Map .qmd to docs/.md, preserving folder structure under chapters/
    sources = []
    # Index
    sources.append((ROOT / "index.qmd", ROOT / "docs" / "index.md"))
    # All chapters
    for qmd in sorted((ROOT / "chapters").rglob("*.qmd")):
        rel = qmd.relative_to(ROOT)
        # Strip the leading "chapters/" segment so docs structure mirrors parts
        rel_no_chapters = Path(*rel.parts[1:])  # e.g. part1/01-nlp-to-speech.qmd
        sources.append((qmd, ROOT / "docs" / rel_no_chapters.with_suffix(".md")))
    # References page if exists
    if (ROOT / "references.qmd").exists():
        sources.append((ROOT / "references.qmd", ROOT / "docs" / "references.md"))

    for src, dst in sources:
        if not src.exists():
            print(f"  SKIP missing: {src.relative_to(ROOT)}")
            continue
        convert_file(src, dst, bib)

    print(f"\nConverted {len(sources)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
