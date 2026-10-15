#!/usr/bin/env bash
# Fails early, with instructions, when a repository secret is missing.
missing=()
for k in ANTHROPIC_API_KEY APIFY_TOKEN SEC_USER_AGENT; do [ -n "${!k:-}" ] || missing+=("$k"); done
if [ ${#missing[@]} -gt 0 ]; then
  echo "::error::Missing repository secrets: ${missing[*]}. Add them under the repository's Settings > Secrets and variables > Actions > New repository secret."
  exit 1
fi
echo "Secrets present."
