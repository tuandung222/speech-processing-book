#!/usr/bin/env python3
"""Convert MkDocs Material .md files (in docs/) to mdBook .md files (in src/).

Transformations
---------------
1. Admonitions:
       !!! note "Title"
           body line 1
           body line 2
   ->  > **📝 Title — Note**
       >
       > body line 1
       > body line 2

2. Figure HTML blocks (<figure markdown ...>) — keep as-is (mdBook passes HTML).
3. Math `$...$` and `$$...$$` — keep as-is (mdbook-katex preprocesses).
4. Footnotes `[^key]` and definitions — keep as-is (pulldown-cmark supports).
5. Anchors `<a id="...">` — keep as-is.
6. Code fences — keep as-is.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
SRC_DOCS = ROOT / "docs"
DST_SRC = ROOT / "src"


# Type → (emoji, vietnamese label)
ADMON_LABELS: dict[str, tuple[str, str]] = {
    "note":    ("📝", "Ghi chú"),
    "tip":     ("💡", "Mẹo"),
    "warning": ("⚠️", "Cảnh báo"),
    "danger":  ("🚨", "Quan trọng"),
    "info":    ("ℹ️", "Thông tin"),
    "success": ("✅", "Lưu ý"),
    "abstract":("📋", "Tóm tắt"),
    "example": ("🔍", "Ví dụ"),
    "quote":   ("💬", "Trích dẫn"),
}


def transform_admonitions(text: str) -> str:
    """Convert MkDocs `!!! type "title"` admonitions to blockquote style."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    admon_re = re.compile(r'^!!! (\w+)(?:\s+"([^"]*)")?\s*$')

    while i < n:
        line = lines[i]
        m = admon_re.match(line)
        if not m:
            out.append(line)
            i += 1
            continue

        # Found admonition opening
        atype = m.group(1).lower()
        title = m.group(2) or ""
        emoji, default_label = ADMON_LABELS.get(atype, ("📝", "Ghi chú"))

        # Build header for the blockquote
        if title:
            header = f"> **{emoji} {title}**"
        else:
            header = f"> **{emoji} {default_label}**"

        # Collect indented body lines (4-space indent in MkDocs)
        body: list[str] = []
        i += 1
        while i < n:
            ln = lines[i]
            if ln == "":
                # blank inside admonition — keep but check if next line continues
                # If next non-empty line is still indented, keep blank as part of body
                j = i + 1
                while j < n and lines[j] == "":
                    j += 1
                if j < n and (lines[j].startswith("    ") or lines[j].startswith("\t")):
                    body.append("")
                    i += 1
                    continue
                else:
                    break  # blank ends admonition
            if ln.startswith("    "):
                body.append(ln[4:])
                i += 1
            elif ln.startswith("\t"):
                body.append(ln[1:])
                i += 1
            else:
                break

        # Emit blockquote
        out.append(header)
        out.append(">")
        for bln in body:
            if bln == "":
                out.append(">")
            else:
                out.append(f"> {bln}")
        out.append("")  # separator
        # i already at the next non-admonition line

    return "\n".join(out)


def transform_figures(text: str) -> str:
    """Simplify <figure markdown id="..."> to plain image with caption.

    mdBook passes HTML but `markdown` attribute is MkDocs-specific. Convert to
    standard markdown image with caption as a paragraph.
    """
    fig_re = re.compile(
        r'<figure\s+markdown\s+id="([^"]+)">\s*\n'
        r'\s*!\[([^\]]*)\]\(([^)]+)\)\s*\n'
        r'\s*<figcaption>([^<]*)</figcaption>\s*\n'
        r'</figure>',
        re.MULTILINE,
    )

    def repl(m: re.Match) -> str:
        fid = m.group(1)
        alt = m.group(2)
        path = m.group(3)
        cap = m.group(4)
        # mdBook supports HTML, so we keep figure semantics but drop `markdown` attr
        return (
            f'<figure id="{fid}">\n'
            f'  <img src="{path}" alt="{alt}" />\n'
            f'  <figcaption><strong>Hình:</strong> {cap}</figcaption>\n'
            f'</figure>'
        )

    return fig_re.sub(repl, text)


def fix_image_paths(text: str, src_md: Path) -> str:
    """No path fix needed if we mirror docs/ structure into src/."""
    return text


def convert_file(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    text = transform_admonitions(text)
    text = transform_figures(text)
    text = fix_image_paths(text, src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def copy_assets(src_dir: Path, dst_dir: Path) -> int:
    """Copy non-.md files (images, PDFs, etc.) preserving structure."""
    count = 0
    for f in src_dir.rglob("*"):
        if f.is_file() and f.suffix != ".md":
            rel = f.relative_to(src_dir)
            target = dst_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            count += 1
    return count


def generate_summary() -> str:
    """Generate src/SUMMARY.md from book structure."""
    return """# Summary

[Trang chủ](./index.md)

# Phần I — Nền tảng

- [Chương 1: Từ NLP đến Speech](./part1/01-nlp-to-speech.md)
- [Chương 2: Audio Signal Fundamentals](./part1/02-audio-fundamentals.md)
- [Chương 3: Speech Representations](./part1/03-speech-representations.md)

# Phần II — Nhận dạng Giọng nói (ASR)

- [Chương 4: ASR Foundations](./part2/04-asr-foundations.md)
- [Chương 5: Modern ASR Architectures](./part2/05-modern-asr.md)
- [Chương 6: Whisper & Canary](./part2/06-whisper-canary.md)
- [Chương 7: Streaming ASR](./part2/07-streaming-asr.md)

# Phần III — Tổng hợp Giọng nói (TTS)

- [Chương 8: TTS Foundations](./part3/08-tts-foundations.md)
- [Chương 9: End-to-End TTS](./part3/09-end-to-end-tts.md)
- [Chương 10: Audio Codecs](./part3/10-audio-codecs.md)

# Phần IV — Speech LLMs & Multimodal

- [Chương 11: Speech LLMs](./part4/11-speech-llms.md)
- [Chương 12: Multimodal Omni](./part4/12-multimodal-omni.md)
- [Chương 13: Full-Duplex Dialogue](./part4/13-full-duplex.md)

# Phần V — Understanding & Translation

- [Chương 14: Speech Classification](./part5/14-speech-classification.md)
- [Chương 15: Speech Translation](./part5/15-speech-translation.md)

# Phần VI — Tiếng Việt

- [Chương 16: Vietnamese Speech](./part6/16-vietnamese-speech.md)
- [Chương 17: Vietnamese Datasets](./part6/17-vietnamese-datasets.md)

# Phần VII — Tools & Production

- [Chương 18: Training Frameworks](./part7/18-training-frameworks.md)
- [Chương 19: Inference Engines](./part7/19-inference-engines.md)
- [Chương 20: Production Systems](./part7/20-production-systems.md)

# Phụ lục

- [A. Notation Reference](./appendices/appendix-a-notation.md)
- [B. Proofs & Derivations](./appendices/appendix-b-proofs.md)
- [C. NLP↔Speech Mapping](./appendices/appendix-c-mapping.md)
- [D. Code Snippets](./appendices/appendix-d-code.md)
- [E. Vietnamese Resources](./appendices/appendix-e-vietnamese-resources.md)
- [F. Tool Comparison](./appendices/appendix-f-tool-comparison.md)

[Tài liệu tham khảo](./references.md)
"""


def main() -> int:
    # Clean target
    if DST_SRC.exists():
        shutil.rmtree(DST_SRC)
    DST_SRC.mkdir()

    # Convert all .md files
    converted = 0
    for md in sorted(SRC_DOCS.rglob("*.md")):
        rel = md.relative_to(SRC_DOCS)
        dst = DST_SRC / rel
        convert_file(md, dst)
        converted += 1

    # Copy assets (figure PNGs, etc.)
    assets = copy_assets(SRC_DOCS, DST_SRC)

    # Generate SUMMARY
    (DST_SRC / "SUMMARY.md").write_text(generate_summary(), encoding="utf-8")

    print(f"Converted {converted} .md files")
    print(f"Copied {assets} assets")
    print(f"Generated SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
