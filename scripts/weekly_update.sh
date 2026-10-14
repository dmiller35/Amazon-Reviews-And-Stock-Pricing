#!/usr/bin/env bash
# Weekly refresh: new Apify reviews -> score new reviews -> re-evaluate -> reports
# -> dated snapshots in research/<TICKER>/history/. Safe to re-run: reviews.csv and
# scores.csv are append-only and de-duplicated.
#
# Cron example (Mondays 06:17):
#   17 6 * * 1  cd /path/to/Amazon-Reviews-And-Stock-Pricing && scripts/weekly_update.sh >> research/update.log 2>&1
#
# Needs ANTHROPIC_API_KEY, APIFY_TOKEN and SEC_USER_AGENT in the environment
# or in the repo's .env file (created by scripts/setup.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/reviewtone ] && PATH="$PWD/.venv/bin:$PATH"   # created by scripts/setup.sh
# reviewtone reads .env itself.
echo "== reviewtone weekly update $(date -u +%Y-%m-%dT%H:%M:%SZ)"
reviewtone update --max-reviews "${MAX_REVIEWS:-30}"
