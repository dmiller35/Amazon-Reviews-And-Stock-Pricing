#!/usr/bin/env bash
# One-time setup on your own computer (macOS / Linux; on Windows use WSL).
#   git clone https://github.com/dmiller35/Amazon-Reviews-And-Stock-Pricing.git
#   cd Amazon-Reviews-And-Stock-Pricing && git checkout claude/focused-pascal-8sss7m
#   bash scripts/setup.sh
# Creates .venv, installs the package, stores your keys in .env (readable only by
# you, ignored by git), runs the tests, and checks every data source.
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$*"; }

PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    PY="$cand"; break
  fi
done
[ -n "$PY" ] || { echo "Python 3.10+ is required: https://www.python.org/downloads/"; exit 1; }

# pandas/scipy/scikit-learn need memory to install, and backfill needs more to run.
mem_mb=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
swap_mb=$(awk '/SwapTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$mem_mb" -gt 0 ] && [ $((mem_mb + swap_mb)) -lt 3500 ]; then
  echo "WARNING: only ${mem_mb} MB RAM + ${swap_mb} MB swap. Installs can crash ('Aborted') and"
  echo "backfill needs more. On WSL, create C:\\Users\\<you>\\.wslconfig with"
  echo "  [wsl2]"; echo "  memory=8GB"; echo "  swap=8GB"
  echo "then run 'wsl --shutdown' in PowerShell and reopen Ubuntu."
fi

# Python 3.14 is very new: Ubuntu 26.04's 3.14.4 crashed with interpreter memory
# corruption ("SystemError: null argument to internal routine", then a segfault)
# during backfill. Use a uv-managed Python 3.13 instead when only 3.14+ exists.
if "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 14))'; then
  say "Found $($PY --version); installing a separate Python 3.13 for this project with uv"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  if [ -x .venv/bin/python ] && ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info >= (3, 14))'; then
    echo "Removing the old Python 3.14 environment (.venv)"
    rm -rf .venv
  fi
  [ -d .venv ] || uv venv --quiet --python 3.13 --seed .venv
fi

say "Installing into .venv with $(.venv/bin/python --version 2>/dev/null || $PY --version)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --progress-bar off --upgrade pip | tail -1
if .venv/bin/python -m pip install --progress-bar off -e '.[dev]' > .venv/install.log 2>&1; then
  grep -E "^Successfully installed" .venv/install.log || echo "All packages already installed."
else
  tail -15 .venv/install.log
  echo "Install failed (full log: .venv/install.log). 'Aborted' or 'Killed' means the machine ran out"
  echo "of memory: see the warning above, then re-run: bash scripts/setup.sh"
  exit 1
fi

if [ ! -f .env ]; then
  say "API keys (input is hidden; saved to .env)"
  read -r -s -p "Anthropic API key (console.anthropic.com): " ANTHROPIC_API_KEY; echo
  read -r -s -p "Apify API token (console.apify.com > Settings > API & Integrations): " APIFY_TOKEN; echo
  read -r -p "Your name and email for SEC requests (e.g. Jane Doe jane@example.com): " SEC_USER_AGENT
  umask 077
  printf 'ANTHROPIC_API_KEY="%s"\nAPIFY_TOKEN="%s"\nSEC_USER_AGENT="%s"\n' "$ANTHROPIC_API_KEY" "$APIFY_TOKEN" "$SEC_USER_AGENT" > .env
else
  say "Using existing .env"
fi
# Load .env without executing it (values may contain spaces).
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in *=*) ;; *) continue ;; esac
  key="${line%%=*}"; key="${key#export }"; val="${line#*=}"
  case "$key" in ''|\#*) continue ;; esac
  val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
  export "$key=$val"
done < .env

say "Running tests"
.venv/bin/python -m pytest -q

say "Checking data sources"
ok=1
check() { if "$@" >/dev/null 2>&1; then echo "  ok    $label"; else echo "  FAIL  $label"; ok=0; fi; }
label="Anthropic API key";   check .venv/bin/python -c 'import anthropic; anthropic.Anthropic().models.list(limit=1)'
label="Apify token";         check curl -sf "https://api.apify.com/v2/users/me?token=${APIFY_TOKEN}"
label="SEC EDGAR";           check curl -sf -A "$SEC_USER_AGENT" https://www.sec.gov/files/company_tickers.json
label="Prices (Yahoo/Stooq)"; check sh -c 'curl -sf -A "Mozilla/5.0" "https://query1.finance.yahoo.com/v8/finance/chart/LOGI?range=5d" || curl -sf "https://stooq.com/q/d/l/?s=logi.us&i=d"'
label="Fama-French factors"; check curl -sfI https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip
label="McAuley review data"; check curl -sfI https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz

say "Verifying companies against SEC filings"
.venv/bin/reviewtone verify --all | grep -E '"(ticker|sec_name|fiscal_year_end_matches|error)"' || true

if [ "$ok" = 1 ]; then
  say "Setup complete. Activate with:  source .venv/bin/activate"
  echo "Next: see 'Running it' in README.md, starting with the historical backfill."
else
  say "Setup finished with failures above. Fix the key or network access, then re-run: bash scripts/setup.sh"
  exit 1
fi
