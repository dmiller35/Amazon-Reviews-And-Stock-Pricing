"""Historical Amazon reviews from the McAuley Lab Amazon Reviews 2023 dataset.

Jin (2022) used the McAuley Lab's earlier Amazon release. The 2023 release
(reviews through September 2023, https://amazon-reviews-2023.github.io/) gives
the multi-year history a live scraper cannot, so it is the backtest source.

Two streaming passes over JSONL(.gz) files, local paths or URLs:
1. meta_<Category>.jsonl.gz -> products whose `store` matches a brand storefront
2. <Category>.jsonl.gz      -> reviews whose parent_asin is in that product set
"""

from __future__ import annotations

import gzip
import io
import json
import time
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
import urllib3

from ..config import Company
from ..http import AccessLog, SourceUnavailable, utcnow

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"


def category_urls(category: str) -> tuple[str, str]:
    return (f"{BASE_URL}/meta_categories/meta_{category}.jsonl.gz",
            f"{BASE_URL}/review_categories/{category}.jsonl.gz")


class ResumableDownload(io.RawIOBase):
    """A read-only HTTP body that reconnects where it left off.

    The category files are several GB and take an hour or more to stream; a
    dropped connection resumes with an HTTP Range request instead of
    restarting (or failing) the backfill.
    """

    def __init__(self, url: str, log: AccessLog | None = None, retries: int = 20, timeout: float = 120):
        self.url, self.log, self.retries, self.timeout = url, log, retries, timeout
        self.pos = 0
        self.resp = self._open()

    def _open(self) -> requests.Response:
        headers = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        for attempt in range(self.retries):
            try:
                resp = requests.get(self.url, stream=True, timeout=self.timeout, headers=headers)
            except requests.RequestException as e:
                err = str(e)
            else:
                if resp.status_code in (200, 206):
                    if self.pos and resp.status_code != 206:
                        raise SourceUnavailable(f"{self.url}: server cannot resume (no Range support)")
                    if self.log:
                        self.log.record(source="mcauley", url=self.url, status=resp.status_code, offset=self.pos)
                    return resp
                err = f"HTTP {resp.status_code}"
                if resp.status_code in (403, 404):
                    break
            time.sleep(min(60, 2 ** attempt))
        if self.log:
            self.log.record(source="mcauley", url=self.url, status="error", error=err, offset=self.pos)
        raise SourceUnavailable(f"{self.url}: {err}")

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        for _ in range(self.retries):
            try:
                n = self.resp.raw.readinto(buf)
                self.pos += n
                return n
            except (requests.RequestException, urllib3.exceptions.HTTPError, ConnectionError, OSError):
                self.resp.close()
                self.resp = self._open()  # resume at self.pos
        raise SourceUnavailable(f"{self.url}: too many dropped connections")

    def close(self) -> None:
        self.resp.close()
        super().close()


def iter_jsonl(location: str | Path, log: AccessLog | None = None) -> Iterator[dict]:
    loc = str(location)
    if loc.startswith("http"):
        raw = io.BufferedReader(ResumableDownload(loc, log), buffer_size=1 << 20)
        stream = gzip.GzipFile(fileobj=raw) if loc.endswith(".gz") else raw
        lines = io.TextIOWrapper(stream, encoding="utf-8")
    else:
        lines = gzip.open(loc, "rt", encoding="utf-8") if loc.endswith(".gz") else open(loc, encoding="utf-8")
    with lines:
        for line in lines:
            if line.strip():
                yield json.loads(line)


def match_brand(store: str | None, company: Company) -> str | None:
    if not store:
        return None
    s = store.strip().casefold()
    for b in company.brands:
        if s in {n.casefold() for n in b.amazon_store_names}:
            return b.name
    return None


def products_from_meta(meta_location: str | Path, company: Company, log: AccessLog | None = None) -> pd.DataFrame:
    rows = []
    for item in iter_jsonl(meta_location, log):
        brand = match_brand(item.get("store"), company)
        if brand is None:
            continue
        rows.append({
            "ticker": company.ticker,
            "brand": brand,
            "asin": item.get("parent_asin"),
            "parent_asin": item.get("parent_asin"),
            "title": item.get("title", ""),
            "store": item.get("store"),
            "category": item.get("main_category", ""),
            "rating_number": item.get("rating_number"),
            "url": f"https://www.amazon.com/dp/{item.get('parent_asin')}",
            "official_store": True,
            "source": f"mcauley2023:{Path(str(meta_location)).name}",
            "discovered_at": utcnow(),
        })
    return pd.DataFrame(rows)


def reviews_for_products(review_location: str | Path, products: pd.DataFrame, company: Company,
                         log: AccessLog | None = None) -> pd.DataFrame:
    brand_of = dict(zip(products["parent_asin"], products["brand"]))
    fetched_at = utcnow()
    rows = []
    for r in iter_jsonl(review_location, log):
        parent = r.get("parent_asin")
        if parent not in brand_of:
            continue
        date = pd.Timestamp(r["timestamp"], unit="ms").strftime("%Y-%m-%d")
        rows.append({
            "review_id": None,  # dataset has no review id; a content hash is assigned downstream
            "ticker": company.ticker,
            "brand": brand_of[parent],
            "asin": r.get("asin"),
            "parent_asin": parent,
            "review_date": date,
            "rating": r.get("rating"),
            "title": r.get("title", ""),
            "text": r.get("text", ""),
            "verified": r.get("verified_purchase"),
            "variant": "",
            "source": f"mcauley2023:{Path(str(review_location)).name}",
            "source_url": f"https://www.amazon.com/dp/{r.get('asin')}",
            "fetched_at": fetched_at,
        })
    return pd.DataFrame(rows)


def match_any(store: str | None, companies: list[Company]) -> tuple[str, str] | None:
    for c in companies:
        b = match_brand(store, c)
        if b:
            return c.ticker, b
    return None


def scan_category(category: str, companies: list[Company], log: AccessLog | None = None,
                  meta_location: str | None = None, review_location: str | None = None):
    """Stream one category's metadata and reviews once for several companies.

    Returns (products DataFrame with a `ticker` column, generator of review rows).
    """
    meta_url, review_url = category_urls(category)
    rows, owner = [], {}
    for item in iter_jsonl(meta_location or meta_url, log):
        hit = match_any(item.get("store"), companies)
        if not hit or not item.get("parent_asin"):
            continue
        ticker, brand = hit
        owner[item["parent_asin"]] = (ticker, brand)
        rows.append({"ticker": ticker, "brand": brand, "asin": item["parent_asin"],
                     "parent_asin": item["parent_asin"], "title": item.get("title", ""), "store": item.get("store"),
                     "category": item.get("main_category") or category, "rating_number": item.get("rating_number"),
                     "url": f"https://www.amazon.com/dp/{item['parent_asin']}", "official_store": True,
                     "source": f"mcauley2023:{category}", "discovered_at": utcnow()})

    def reviews():
        fetched_at = utcnow()
        for r in iter_jsonl(review_location or review_url, log):
            hit = owner.get(r.get("parent_asin"))
            if hit is None:
                continue
            yield {"review_id": None, "ticker": hit[0], "brand": hit[1], "asin": r.get("asin"),
                   "parent_asin": r.get("parent_asin"),
                   "review_date": pd.Timestamp(r["timestamp"], unit="ms").strftime("%Y-%m-%d"),
                   "rating": r.get("rating"), "title": r.get("title", ""), "text": r.get("text", ""),
                   "verified": r.get("verified_purchase"), "variant": "", "source": f"mcauley2023:{category}",
                   "source_url": f"https://www.amazon.com/dp/{r.get('asin')}", "fetched_at": fetched_at}

    return pd.DataFrame(rows), reviews()
