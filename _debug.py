"""Verify no \\_ remains anywhere in docs/."""
from pathlib import Path

DOCS = Path(__file__).resolve().parent / "docs"
count = 0
for md in sorted(DOCS.rglob("*.md")):
    text = md.read_text(encoding="utf-8")
    for i, line in enumerate(text.split("\n"), 1):
        if "\\_" in line:
            print(f"{md.relative_to(DOCS.parent)}:{i}: {line.strip()[:120]}")
            count += 1
print(f"\nRemaining backslash-underscore: {count}")

