"""Amazon reviews via one Apify actor (step 3 of the research prompt).

Actor input/output schemas differ between Apify actors and change over time, so
the actor ID and input template are configurable and the output is mapped with
a tolerant key list. Everything the run returns that could limit history
(per-product caps, oldest date reached, errors) goes to the access log, because
Amazon only exposes a limited number of review pages per product and a scraper
rarely reaches 12 full quarters back.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Iterable

import pandas as pd
import requests

from ..http import AccessLog, SourceUnavailable, utcnow

API = "https://api.apify.com/v2"
DEFAULT_ACTOR = os.environ.get("APIFY_REVIEWS_ACTOR", "junglee~amazon-reviews-scraper")

FIELD_CANDIDATES = {
    "review_id": ["reviewId", "id", "review_id"],
    "asin": ["productAsin", "asin", "originalAsin", "product_asin"],
    "parent_asin": ["parentAsin", "parent_asin"],
    "rating": ["ratingScore", "rating", "stars", "reviewRating"],
    "title": ["reviewTitle", "title", "headline"],
    "text": ["reviewDescription", "text", "reviewText", "body", "content"],
    "review_date": ["date", "reviewDate", "reviewedAt", "review_date"],
    "verified": ["isVerified", "verified", "verifiedPurchase", "isVerifiedPurchase"],
    "variant": ["variant", "variantAttributes", "productVariant", "variation"],
    "source_url": ["reviewUrl", "url", "review_url"],
    "country": ["reviewedIn", "country", "countryCode"],
}

_RATING_RE = re.compile(r"(\d(?:\.\d)?)")
_DATE_RE = re.compile(r"on\s+(.+)$")


def _first(item: dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in item and item[k] not in (None, ""):
            return item[k]
    return None


def parse_review_date(value: Any) -> str | None:
    """Accepts ISO dates, epoch ms, and 'Reviewed in the United States on September 22, 2026'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return pd.Timestamp(value, unit="ms" if value > 1e11 else "s").strftime("%Y-%m-%d")
    s = str(value).strip()
    m = _DATE_RE.search(s)
    if m:
        s = m.group(1)
    for fmt in ("%B %d, %Y", "%d %B %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    return None if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def parse_rating(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _RATING_RE.search(str(value))  # "4.0 out of 5 stars"
    return float(m.group(1)) if m else None


def normalize_items(items: list[dict[str, Any]], *, ticker: str, brand: str, fetched_at: str) -> tuple[pd.DataFrame, list[str]]:
    """Map raw actor items to review rows. Returns (rows, unmapped field names)."""
    rows, seen_keys = [], set()
    known = {k for keys in FIELD_CANDIDATES.values() for k in keys}
    for it in items:
        seen_keys.update(it.keys())
        country = _first(it, FIELD_CANDIDATES["country"])
        if country and not re.search(r"united states|^us$", str(country), re.I):
            continue  # Amazon US only
        variant = _first(it, FIELD_CANDIDATES["variant"])
        rows.append({
            "review_id": _first(it, FIELD_CANDIDATES["review_id"]),
            "ticker": ticker,
            "brand": brand,
            "asin": _first(it, FIELD_CANDIDATES["asin"]),
            "parent_asin": _first(it, FIELD_CANDIDATES["parent_asin"]),
            "review_date": parse_review_date(_first(it, FIELD_CANDIDATES["review_date"])),
            "rating": parse_rating(_first(it, FIELD_CANDIDATES["rating"])),
            "title": _first(it, FIELD_CANDIDATES["title"]) or "",
            "text": _first(it, FIELD_CANDIDATES["text"]) or "",
            "verified": _first(it, FIELD_CANDIDATES["verified"]),
            "variant": variant if isinstance(variant, str) else (str(variant) if variant else ""),
            "source": "apify",
            "source_url": _first(it, FIELD_CANDIDATES["source_url"]),
            "fetched_at": fetched_at,
        })
    return pd.DataFrame(rows), sorted(seen_keys - known)


class ApifyReviews:
    def __init__(self, token: str | None = None, actor: str = DEFAULT_ACTOR, log: AccessLog | None = None):
        self.token = token or os.environ.get("APIFY_TOKEN")
        self.actor = actor.replace("/", "~")
        self.log = log or AccessLog(None)

    def build_input(self, asins: list[str], max_reviews: int) -> dict[str, Any]:
        return {
            "productUrls": [{"url": f"https://www.amazon.com/dp/{a}"} for a in asins],
            "maxReviews": max_reviews,
            "sort": "recent",
            "filterByRatings": ["allStars"],
            "reviewsUseProductVariantFilter": False,
            "scrapeProductDetails": False,
            "proxyCountry": "US",
        }

    def fetch(self, asins: list[str], *, ticker: str, brand: str, max_reviews: int = 500,
              timeout_s: int = 900) -> pd.DataFrame:
        if not self.token:
            raise SourceUnavailable("APIFY_TOKEN is not set; cannot run the Apify review scraper.")
        url = f"{API}/acts/{self.actor}/run-sync-get-dataset-items"
        payload = self.build_input(asins, max_reviews)
        try:
            resp = requests.post(url, params={"token": self.token, "timeout": timeout_s}, json=payload,
                                 timeout=timeout_s + 60)
        except requests.RequestException as e:
            self.log.record(source="apify", actor=self.actor, asins=asins, status="error", error=str(e))
            raise SourceUnavailable(f"Apify unreachable: {e}") from e
        self.log.record(source="apify", actor=self.actor, asins=asins, status=resp.status_code,
                        max_reviews=max_reviews)
        if resp.status_code not in (200, 201):
            raise SourceUnavailable(f"Apify actor {self.actor} returned HTTP {resp.status_code}: {resp.text[:300]}")
        items = resp.json()
        df, unmapped = normalize_items(items, ticker=ticker, brand=brand, fetched_at=utcnow())
        self._log_limits(df, asins, max_reviews, unmapped)
        return df

    def _log_limits(self, df: pd.DataFrame, asins: list[str], max_reviews: int, unmapped: list[str]) -> None:
        per_asin = df.groupby("asin")["review_date"].agg(["count", "min", "max"]) if len(df) else pd.DataFrame()
        for asin in asins:
            if asin in per_asin.index:
                r = per_asin.loc[asin]
                self.log.record(source="apify", kind="coverage", asin=asin, reviews=int(r["count"]),
                                oldest=r["min"], newest=r["max"], hit_cap=bool(r["count"] >= max_reviews))
            else:
                self.log.record(source="apify", kind="coverage", asin=asin, reviews=0)
        if unmapped:
            self.log.record(source="apify", kind="unmapped_fields", fields=unmapped)
