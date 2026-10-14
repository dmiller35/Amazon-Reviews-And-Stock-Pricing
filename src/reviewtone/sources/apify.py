"""Amazon reviews via an Apify actor (step 3 of the research prompt).

The actor and its input format live in config/apify.yaml, so switching to a
cheaper or better actor needs no code change. Output is mapped with a tolerant
key list; unknown fields are logged. Everything the run returns that could limit history
(per-product caps, oldest date reached, errors) goes to the access log, because
Amazon only exposes a limited number of review pages per product and a scraper
rarely reaches 12 full quarters back.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import requests
import yaml

from ..http import AccessLog, SourceUnavailable, utcnow

API = "https://api.apify.com/v2"

FIELD_CANDIDATES = {
    "review_id": ["reviewId", "id", "review_id", "reviewID"],
    "asin": ["productAsin", "asin", "originalAsin", "product_asin"],
    "parent_asin": ["parentAsin", "parent_asin"],
    "rating": ["ratingScore", "rating", "stars", "reviewRating"],
    "title": ["reviewTitle", "title", "headline"],
    "text": ["reviewDescription", "text", "reviewText", "body", "content", "review", "reviewBody"],
    "review_date": ["date", "reviewDate", "reviewedAt", "review_date", "published_date", "publishedDate"],
    "verified": ["isVerified", "verified", "verifiedPurchase", "isVerifiedPurchase"],
    "variant": ["variant", "variantAttributes", "productVariant", "variation"],
    "source_url": ["reviewUrl", "url", "review_url", "sourceUrl", "source_url"],
    "origin": ["reviewOrigin", "review_origin", "origin"],
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
            # some actors mix feeds (e.g. Amazon reviews served via Woot); keep the origin visible
            "source": f"apify:{_first(it, FIELD_CANDIDATES['origin'])}" if _first(it, FIELD_CANDIDATES["origin"])
            else "apify",
            "source_url": _first(it, FIELD_CANDIDATES["source_url"]),
            "fetched_at": fetched_at,
        })
    return pd.DataFrame(rows), sorted(seen_keys - known)


STARS = {1: "one_star", 2: "two_star", 3: "three_star", 4: "four_star", 5: "five_star"}
CONFIG = Path(__file__).resolve().parents[3] / "config" / "apify.yaml"


@dataclass
class Profile:
    """How to drive one Apify actor.

    payloads(asin, max) -> one input per actor run.
    cap(max): reviews per star list (by_star) or per run at which history is
    truncated. With by_star, each star rating is requested separately: Amazon
    shows at most ~100 reviews per filter view, so this reaches about five
    times deeper and the low-star reviews the signal needs can't be crowded out.
    """
    payloads: Callable[[str, int], list[dict[str, Any]]]
    cap: Callable[[int], int]
    by_star: bool


def _pages(m: int) -> int:
    return max(1, min(10, -(-m // 10)))


def _axesso(asin: str, max_reviews: int) -> list[dict[str, Any]]:
    return [{"input": [{"asin": asin, "domainCode": "com", "sortBy": "recent", "maxPages": _pages(max_reviews),
                        "filterByStar": star, "reviewerType": "all_reviews", "formatType": "current_format",
                        "mediaType": "all_contents"} for star in STARS.values()]}]


def _junglee(asin: str, max_reviews: int) -> list[dict[str, Any]]:
    return [{"productUrls": [{"url": f"https://www.amazon.com/dp/{asin}"}], "maxReviews": max_reviews,
             "sort": "recent", "filterByRatings": ["allStars"], "scrapeProductDetails": False, "proxyCountry": "US"}]


BUILTIN = {
    "axesso_data~amazon-reviews-scraper": Profile(_axesso, lambda m: _pages(m) * 10, True),
    "junglee~amazon-reviews-scraper": Profile(_junglee, lambda m: m, False),
}


def fill(template: Any, values: dict[str, Any]) -> Any:
    """Substitute {asin}/{max}/{star}; a value that is only a placeholder keeps its type."""
    if isinstance(template, dict):
        return {k: fill(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [fill(v, values) for v in template]
    if isinstance(template, str):
        for key, val in values.items():
            if template == f"{{{key}}}":
                return val
        for key, val in values.items():
            template = template.replace(f"{{{key}}}", str(val))
    return template


def template_profile(cfg: dict[str, Any]) -> Profile:
    tmpl, by_star = cfg["input"], bool(cfg.get("by_star"))
    stars = cfg.get("star_values", list(STARS.values()))

    def payloads(asin: str, m: int) -> list[dict[str, Any]]:
        if by_star:
            return [fill(tmpl, {"asin": asin, "max": m, "star": s}) for s in stars]
        return [fill(tmpl, {"asin": asin, "max": m})]

    cap = cfg.get("cap", "{max}")
    return Profile(payloads, lambda m: int(fill(cap, {"max": m})), by_star)


def resolve_profile(actor: str | None = None, config: Path = CONFIG) -> tuple[str, Profile]:
    cfg = yaml.safe_load(config.read_text()) if config.exists() else {}
    name = (actor or os.environ.get("APIFY_REVIEWS_ACTOR") or cfg.get("actor") or "").replace("/", "~")
    if not name:
        raise ValueError(f"No Apify actor configured; set `actor` in {config}.")
    if cfg.get("input") and cfg.get("actor", "").replace("/", "~") == name:
        return name, template_profile(cfg)
    if name in BUILTIN:
        return name, BUILTIN[name]
    raise ValueError(f"No input format for Apify actor {name!r}: add its `input` template to {config}.")


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
    def __init__(self, token: str | None = None, actor: str | None = None, log: AccessLog | None = None,
                 config: Path = CONFIG):
        self.token = token or os.environ.get("APIFY_TOKEN")
        self.actor, self.profile = resolve_profile(actor, config)
        self.log = log or AccessLog(None)

    def fetch(self, asin: str, *, ticker: str, brand: str, max_reviews: int = 100,
              timeout_s: int = 900) -> FetchResult:
        """Newest reviews of one product. max_reviews applies per star filter for star-filtered actors."""
        if not self.token:
            raise SourceUnavailable("APIFY_TOKEN is not set; cannot run the Apify review scraper.")
        url = f"{API}/acts/{self.actor}/run-sync-get-dataset-items"
        items: list[dict[str, Any]] = []
        for payload in self.profile.payloads(asin, max_reviews):
            try:
                resp = requests.post(url, params={"token": self.token, "timeout": timeout_s}, json=payload,
                                     timeout=timeout_s + 60)
            except requests.RequestException as e:
                self.log.record(source="apify", actor=self.actor, asin=asin, status="error", error=str(e))
                raise SourceUnavailable(f"Apify unreachable: {e}") from e
            self.log.record(source="apify", actor=self.actor, asin=asin, status=resp.status_code,
                            max_reviews=max_reviews)
            if resp.status_code not in (200, 201):
                raise SourceUnavailable(f"Apify actor {self.actor} returned HTTP {resp.status_code}: {resp.text[:300]}")
            items.extend(resp.json())
        df, unmapped = normalize_items(items, ticker=ticker, brand=brand, fetched_at=utcnow())
        if len(df):
            df["asin"] = df["asin"].fillna(asin)
        hit_cap, complete_from = completeness(df, self.profile.cap(max_reviews), self.profile.by_star)
        self.log.record(source="apify", kind="coverage", asin=asin, reviews=len(df),
                        oldest=df["review_date"].min() if len(df) else None, hit_cap=hit_cap,
                        complete_from=complete_from)
        if unmapped:
            self.log.record(source="apify", kind="unmapped_fields", fields=unmapped)
        return FetchResult(df, hit_cap, complete_from)
