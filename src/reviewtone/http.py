"""HTTP access with a disk cache and an append-only access log.

Every request (including failures) is recorded in the ticker's access_log.jsonl
so the research folder documents exactly what was reachable, when, and what
limits were hit.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

CACHE_DIR = Path(os.environ.get("REVIEWTONE_CACHE", Path(__file__).resolve().parents[2] / "data" / "cache"))


class SourceUnavailable(RuntimeError):
    """A data source could not be reached (network policy, auth, or outage)."""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AccessLog:
    def __init__(self, path: Path | None):
        self.path = path

    def record(self, **event: Any) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"ts": utcnow(), **event}, default=str) + "\n")


def _user_agent(url: str) -> str:
    # SEC requires a descriptive User-Agent with a contact address
    # (https://www.sec.gov/os/accessing-edgar-data). It is read from the
    # environment so no personal address is ever committed.
    if "sec.gov" in url:
        ua = os.environ.get("SEC_USER_AGENT")
        if not ua:
            raise SourceUnavailable("Set SEC_USER_AGENT to 'Your Name your@email' before calling SEC EDGAR.")
        return ua
    return "reviewtone-research/0.1 (+https://github.com)"


def get(
    url: str,
    *,
    log: AccessLog | None = None,
    params: dict[str, Any] | None = None,
    cache_hours: float = 24.0,
    as_json: bool = True,
    timeout: float = 60.0,
    retries: int = 3,
) -> Any:
    """GET with caching and logging. Raises SourceUnavailable on failure."""
    key = hashlib.sha256((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()[:32]
    cache_file = CACHE_DIR / key
    if cache_hours > 0 and cache_file.exists() and time.time() - cache_file.stat().st_mtime < cache_hours * 3600:
        body = cache_file.read_bytes()
        return json.loads(body) if as_json else body

    last_error = ""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": _user_agent(url)})
        except requests.exceptions.ProxyError:
            last_error = f"blocked by network proxy/policy ({urlsplit(url).netloc} not allowed)"
            break  # policy denials do not succeed on retry
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
        else:
            if log:
                log.record(source="http", url=url, status=resp.status_code, bytes=len(resp.content))
            if resp.status_code == 200:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(resp.content)
                return resp.json() if as_json else resp.content
            last_error = f"HTTP {resp.status_code}"
            if resp.status_code in (400, 401, 403, 404):
                break
        time.sleep(2 ** attempt)
    if log:
        log.record(source="http", url=url, status="error", error=last_error)
    raise SourceUnavailable(f"{url}: {last_error}")
