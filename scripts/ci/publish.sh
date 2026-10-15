#!/usr/bin/env bash
# Copies the readable results (this week's calls, paper trades, per-company reports
# and charts) into results/ and commits them so they can be read on GitHub.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -d research ] || { echo "No results yet."; exit 0; }
mkdir -p results
for f in SUMMARY.md PAPER_TRADING.md paper_trades.csv; do
  [ -f "research/$f" ] && cp "research/$f" results/
done
if [ -d research/_universe ]; then
  mkdir -p results/_universe
  find research/_universe -maxdepth 1 -type f \( -name '*.json' -o -name '*.png' -o -name '*.csv' \) \
    -exec cp {} results/_universe/ \;
fi
for d in research/*/; do
  t=$(basename "$d")
  [ "$t" = "_universe" ] && continue
  mkdir -p "results/$t"
  for f in README.md prediction.json quarterly.csv evaluation.json; do
    [ -f "$d$f" ] && cp "$d$f" "results/$t/"
  done
  if [ -d "${d}charts" ]; then mkdir -p "results/$t/charts"; find "${d}charts" -name '*.png' -exec cp {} "results/$t/charts/" \;; fi
done
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add results
if git diff --cached --quiet; then echo "No changes to publish."; exit 0; fi
git commit -q -m "Weekly results $(date -u +%F)"
for _ in 1 2 3; do
  git pull -q --rebase && git push -q && { echo "Published results/."; exit 0; }
  sleep 10
done
echo "::error::Could not push results/"; exit 1
