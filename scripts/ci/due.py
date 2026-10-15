"""Prints run=true|false for GitHub Actions: is a pipeline run due?

Due when no run has completed in the last 6 days (results/last_run.json, committed by
the previous run), or the last run was cut short (status "partial").
"""

import json
from datetime import date, timedelta
from pathlib import Path

try:
    info = json.loads(Path("results/last_run.json").read_text())
    done = info.get("status") == "complete" and date.fromisoformat(info["date"]) > date.today() - timedelta(days=6)
except (OSError, ValueError, KeyError):
    done = False
print(f"run={'false' if done else 'true'}")
