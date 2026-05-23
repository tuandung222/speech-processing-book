"""Trace state machine on src/part1/02-audio-fundamentals.md."""
from pathlib import Path

path = Path(__file__).resolve().parent / "src/part1/02-audio-fundamentals.md"
inside = False
for i, line in enumerate(path.read_text().split("\n"), 1):
    if line.strip() == "$$":
        inside = not inside
        print(f"L{i:3d}: $$ -> inside={inside}")
    if "frame_length" in line:
        print(f"L{i:3d}: HAS frame_length, inside={inside}")
    if i > 160:
        break

