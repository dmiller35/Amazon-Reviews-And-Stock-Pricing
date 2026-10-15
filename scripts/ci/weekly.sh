#!/usr/bin/env bash
# The scheduled pipeline (GitHub Actions). Every step is resumable, so a run cut
# off by the 6-hour job limit continues where it stopped on the next day's run.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
status=0
step() {
  local title=$1; shift
  echo "::group::$title"
  "$@" || { status=1; echo "::warning::$title did not finish (it resumes on the next run)"; }
  echo "::endgroup::"
}
step "History 2018-Sep 2023 (skips what is done)" reviewtone backfill --all
step "Live products (companies not checked yet)" reviewtone check-live --all --missing
step "Live reviews, Claude scoring, predictions, portfolio, paper trades" reviewtone update
mkdir -p results
printf '{"date": "%s", "status": "%s"}\n' "$(date -u +%F)" "$([ $status -eq 0 ] && echo complete || echo partial)" \
  > results/last_run.json
exit $status
