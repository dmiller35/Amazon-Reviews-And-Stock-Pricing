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
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests

from ..config import Company
from ..http import AccessLog, SourceUnavailable, utcnow

BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw"


def category_urls(category: str) -> tuple[str, str]:
    return (f"{BASE_URL}/meta_categories/meta_{category}.jsonl.gz",
            f"{BASE_URL}/review_categories/{category}.jsonl.gz")


def iter_jsonl(location: str | Path, log: AccessLog | None = None) -> Iterator[dict]:
    loc = str(location)
    if loc.startswith("http"):
        try:
            resp = requests.get(loc, stream=True, timeout=120)
        except requests.RequestException as e:
            if log:
                log.record(source="mcauley", url=loc, status="error", error=str(e))
            raise SourceUnavailable(f"{loc}: {e}") from e
        if log:
            log.record(source="mcauley", url=loc, status=resp.status_code)
        if resp.status_code != 200:
            raise SourceUnavailable(f"{loc}: HTTP {resp.status_code}")
        resp.raw.decode_content = True
        raw = resp.raw
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
