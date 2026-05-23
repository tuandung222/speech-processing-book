#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
while IFS= read -r f; do
  dest="docs/${f#chapters/}"
  mkdir -p "$(dirname "$dest")"
  cp "$f" "$dest"
done < <(find chapters -name "fig-*.png")
echo "Done. Total in docs:"
find docs -name "fig-*.png" | wc -l
