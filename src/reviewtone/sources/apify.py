"""Amazon reviews via an Apify actor (step 3 of the research prompt).

Default: axesso_data/amazon-reviews-scraper (no Amazon login, star filters,
newest-first, pay per review). junglee/amazon-reviews-scraper is also
supported; choose with APIFY_REVIEWS_ACTOR. Each actor has an input profile
below, and output is mapped with a tolerant key list. Everything the run returns that could limit history
(per-product caps, oldest date reached, errors) goes to the access log, because
Amazon only exposes a limited number of review pages per product and a scraper
rarely reaches 12 full quarters back.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable

import pandas as pd
import requests

from ..http import AccessLog, SourceUnavailable, utcnow

API = "https://api.apify.com/v2"
DEFAULT_ACTOR = os.environ.get("APIFY_REVIEWS_ACTOR", "axesso_data~amazon-reviews-scraper")

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
        raw_date = str(_first(it, FIELD_CANDIDATES["review_date"]) or "")
        written_in = re.search(r"Reviewed in (.+?) on ", raw_date)
        if (country and not re.search(r"united states|^us$", str(country), re.I)) or \
                (written_in and written_in.group(1).strip().casefold() != "the united states"):
            continue  # Amazon US reviews only
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


STARS = {1: "one_star", 2: "two_star", 3: "three_star", 4: "four_star", 5: "five_star"}


@dataclass
class Profile:
    """How to drive one Apify actor.

    by_star: request each star rating separately. Amazon shows at most ~10
    pages (~100 reviews) per filter view, so five star-filtered requests reach
    about five times deeper than one all-stars request, and the negative
    reviews the signal depends on can't be crowded out by positive ones.
    """
    build: Callable[[str, int], Any]
    cap: Callable[[int], int]  # reviews per request (per star when by_star) at which history is truncated
    by_star: bool


def _axesso(asin: str, max_reviews: int) -> dict[str, Any]:
    pages = max(1, min(10, -(-max_reviews // 10)))
    return {"input": [{"asin": asin, "domainCode": "com", "sortBy": "recent", "maxPages": pages,
                       "filterByStar": star, "reviewerType": "all_reviews", "formatType": "current_format",
                       "mediaType": "all_contents"} for star in STARS.values()]}


def _junglee(asin: str, max_reviews: int) -> dict[str, Any]:
    return {"productUrls": [{"url": f"https://www.amazon.com/dp/{asin}"}], "maxReviews": max_reviews,
            "sort": "recent", "filterByRatings": ["allStars"], "scrapeProductDetails": False, "proxyCountry": "US"}


PROFILES = {
    "axesso_data~amazon-reviews-scraper": Profile(_axesso, lambda m: max(1, min(10, -(-m // 10))) * 10, True),
    "junglee~amazon-reviews-scraper": Profile(_junglee, lambda m: m, False),
}


@dataclass
class FetchResult:
    reviews: pd.DataFrame
    hit_cap: bool
    complete_from: str | None  # reviews are complete from this date on; None = full history returned


def completeness(df: pd.DataFrame, cap: int, by_star: bool) -> tuple[bool, str | None]:
    """Find where truncation starts. With star filters, each star's list is
    complete back to its own oldest review unless it hit the cap; the set as a
    whole is complete from the latest of those truncation points."""
    if df.empty:
        return False, None
    groups = df.groupby(df["rating"].round()) if by_star else [(None, df)]
    cut = [g["review_date"].min() for _, g in groups if len(g) >= cap]
    return bool(cut), max(cut) if cut else None


class ApifyReviews:
    def __init__(self, token: str | None = None, actor: str = DEFAULT_ACTOR, log: AccessLog | None = None):
        self.token = token or os.environ.get("APIFY_TOKEN")
        self.actor = actor.replace("/", "~")
        if self.actor not in PROFILES:
            raise ValueError(f"No input profile for Apify actor {self.actor!r}. Supported: {', '.join(PROFILES)}. "
                             "Add one to PROFILES in sources/apify.py.")
        self.profile = PROFILES[self.actor]
        self.log = log or AccessLog(None)

    def fetch(self, asin: str, *, ticker: str, brand: str, max_reviews: int = 100,
              timeout_s: int = 900) -> FetchResult:
        """Newest reviews of one product. max_reviews applies per star filter for star-filtered actors."""
        if not self.token:
            raise SourceUnavailable("APIFY_TOKEN is not set; cannot run the Apify review scraper.")
        url = f"{API}/acts/{self.actor}/run-sync-get-dataset-items"
        try:
            resp = requests.post(url, params={"token": self.token, "timeout": timeout_s},
                                 json=self.profile.build(asin, max_reviews), timeout=timeout_s + 60)
        except requests.RequestException as e:
            self.log.record(source="apify", actor=self.actor, asin=asin, status="error", error=str(e))
            raise SourceUnavailable(f"Apify unreachable: {e}") from e
        self.log.record(source="apify", actor=self.actor, asin=asin, status=resp.status_code, max_reviews=max_reviews)
        if resp.status_code not in (200, 201):
            raise SourceUnavailable(f"Apify actor {self.actor} returned HTTP {resp.status_code}: {resp.text[:300]}")
        df, unmapped = normalize_items(resp.json(), ticker=ticker, brand=brand, fetched_at=utcnow())
        if len(df):
            df["asin"] = df["asin"].fillna(asin)
        hit_cap, complete_from = completeness(df, self.profile.cap(max_reviews), self.profile.by_star)
        self.log.record(source="apify", kind="coverage", asin=asin, reviews=len(df),
                        oldest=df["review_date"].min() if len(df) else None, hit_cap=hit_cap,
                        complete_from=complete_from)
        if unmapped:
            self.log.record(source="apify", kind="unmapped_fields", fields=unmapped)
        return FetchResult(df, hit_cap, complete_from)
