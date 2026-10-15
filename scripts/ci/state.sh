#!/usr/bin/env bash
# Keeps the pipeline's working data (research/: reviews, scores, products, history)
# between scheduled runs as assets of the GitHub release "pipeline-state". It is
# too large to commit; results/ (the readable part) is committed by publish.sh.
# Usage: state.sh restore | save. Needs GH_TOKEN with contents: write.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=pipeline-state
stamps() {  # saved snapshot names, oldest first; none (not an error) before the first save
  { gh release view "$tag" --json assets -q '.assets[].name' 2>/dev/null || true; } \
    | sed -n 's/^\(state-[0-9]*\)\.tgz\.part.*/\1/p' | sort -u
}
case "${1:-}" in
  restore)
    latest=$(stamps | tail -1)
    if [ -z "$latest" ]; then echo "No saved working data yet: this is the first run."; exit 0; fi
    rm -rf /tmp/state && mkdir -p /tmp/state
    gh release download "$tag" -p "$latest.tgz.part*" -D /tmp/state
    cat /tmp/state/"$latest".tgz.part* | tar xzf -
    echo "Restored $latest ($(du -sh research | cut -f1) in research/)."
    ;;
  save)
    [ -d research ] || { echo "Nothing to save."; exit 0; }
    stamp=state-$(date -u +%Y%m%d%H%M)
    rm -rf /tmp/save && mkdir -p /tmp/save
    paths=(research); [ -d data/cache ] && paths+=(data/cache)
    tar czf - "${paths[@]}" | split -b 1900m -d - "/tmp/save/$stamp.tgz.part"   # release assets max 2 GB each
    gh release view "$tag" >/dev/null 2>&1 || gh release create "$tag" --prerelease \
      --title "Pipeline working data (automatic)" \
      --notes "Saved by scripts/ci/state.sh after each scheduled run. Do not edit or delete."
    gh release upload "$tag" /tmp/save/"$stamp".tgz.part*
    echo "Saved $stamp ($(du -ch /tmp/save/* | tail -1 | cut -f1))."
    # Keep the two newest snapshots; older ones are deleted only after this upload succeeded.
    for old in $(stamps | head -n -2); do
      for a in $(gh release view "$tag" --json assets -q '.assets[].name' | grep "^$old"); do
        gh release delete-asset "$tag" "$a" -y
      done
    done
    ;;
  *) echo "usage: $0 restore|save" >&2; exit 2 ;;
esac
