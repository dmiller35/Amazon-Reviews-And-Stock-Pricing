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
import time
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
    "verified": ["isVerified", "verified", "verifiedPurchase", "isVerifiedPurchase", "verified_purchase"],
    "variant": ["variant", "variantAttributes", "productVariant", "variation"],
    "source_url": ["reviewUrl", "url", "review_url", "sourceUrl", "source_url"],
    "origin": ["reviewOrigin", "review_origin", "origin", "source"],
    "country": ["reviewedIn", "country", "countryCode", "review_origin_country"],
}

# Fields actors return that are understood but not stored.
IGNORED_FIELDS = {"amazon_review_id", "id_type", "marketplace", "published_date_raw", "date_status", "author",
                  "helpful_votes", "images", "videos", "product_url", "scraped_at", "schema_version", "vine_review",
                  "data", "helpful_count", "image_url", "vine"}

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
    known = {k for keys in FIELD_CANDIDATES.values() for k in keys} | IGNORED_FIELDS
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
    from ..reviews import REVIEW_COLUMNS
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS), sorted(seen_keys - known)


STARS = {1: "one_star", 2: "two_star", 3: "three_star", 4: "four_star", 5: "five_star"}
CONFIG = Path(__file__).resolve().parents[3] / "config" / "apify.yaml"


@dataclass
class Profile:
    """How to drive one Apify actor.

    payloads(asin, max, since) -> one input per actor run. since: only reviews on
    or after this date (None = no limit); actors without a date filter ignore it.
    cap(max): reviews per star list (by_star) or per run at which history is
    truncated. With by_star, each star rating is requested separately: Amazon
    shows at most ~100 reviews per filter view, so this reaches about five
    times deeper and the low-star reviews the signal needs can't be crowded out.
    """
    payloads: Callable[[str, int, str | None], list[dict[str, Any]]]
    cap: Callable[[int], int]
    by_star: bool


def _pages(m: int) -> int:
    return max(1, min(10, -(-m // 10)))


def _axesso(asin: str, max_reviews: int, since: str | None = None) -> list[dict[str, Any]]:
    return [{"input": [{"asin": asin, "domainCode": "com", "sortBy": "recent", "maxPages": _pages(max_reviews),
                        "filterByStar": star, "reviewerType": "all_reviews", "formatType": "current_format",
                        "mediaType": "all_contents"} for star in STARS.values()]}]


def _junglee(asin: str, max_reviews: int, since: str | None = None) -> list[dict[str, Any]]:
    return [{"productUrls": [{"url": f"https://www.amazon.com/dp/{asin}"}], "maxReviews": max_reviews,
             "sort": "recent", "filterByRatings": ["allStars"], "scrapeProductDetails": False, "proxyCountry": "US"}]


BUILTIN = {
    "axesso_data~amazon-reviews-scraper": Profile(_axesso, lambda m: _pages(m) * 10, True),
    "junglee~amazon-reviews-scraper": Profile(_junglee, lambda m: m, False),
}


_DROP = object()


def fill(template: Any, values: dict[str, Any]) -> Any:
    """Substitute {asin}/{max}/{star}/{since}; a value that is only a placeholder
    keeps its type, and a key whose placeholder value is None is dropped."""
    if isinstance(template, dict):
        out = {k: fill(v, values) for k, v in template.items()}
        return {k: v for k, v in out.items() if v is not _DROP}
    if isinstance(template, list):
        return [fill(v, values) for v in template]
    if isinstance(template, str):
        for key, val in values.items():
            if template == f"{{{key}}}":
                return _DROP if val is None else val
        for key, val in values.items():
            template = template.replace(f"{{{key}}}", "" if val is None else str(val))
    return template


def template_profile(cfg: dict[str, Any]) -> Profile:
    tmpl = cfg["input"]
    split = bool(cfg.get("split_by_star", cfg.get("by_star")))
    by_star = bool(cfg.get("by_star", split))
    stars = cfg.get("star_values", [1, 2, 3, 4, 5])
    limit = int(cfg.get("max_per_request", 10**9))

    def payloads(asin: str, m: int | dict[int, int], since: str | None = None) -> list[dict[str, Any]]:
        # m: one limit for every request, or {star (1-5): limit} when split by star
        per = (lambda i: min(m.get(i, max(m.values())), limit)) if isinstance(m, dict) else (lambda i: min(m, limit))
        if split:
            return [fill(tmpl, {"asin": asin, "max": per(i + 1), "since": since, "star": s})
                    for i, s in enumerate(stars)]
        return [fill(tmpl, {"asin": asin, "max": per(0), "since": since})]

    cap = cfg.get("cap", "{max}")

    def cap_for(m: int | dict[int, int]) -> int | dict[int, int]:
        if isinstance(m, dict):
            return {k: int(fill(cap, {"max": min(v, limit)})) for k, v in m.items()}
        return int(fill(cap, {"max": min(m, limit)}))

    return Profile(payloads, cap_for, by_star)


def load_config(config: Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(config.read_text()) if config.exists() else {}


def resolve_profile(actor: str | None = None, config: Path = CONFIG) -> tuple[str, Profile]:
    cfg = load_config(config)
    name = (actor or os.environ.get("APIFY_REVIEWS_ACTOR") or cfg.get("actor") or "").replace("/", "~")
    if not name:
        raise ValueError(f"No Apify actor configured; set `actor` in {config}.")
    if cfg.get("input") and cfg.get("actor", "").replace("/", "~") == name:
        return name, template_profile(cfg)
    if name in BUILTIN:
        return name, BUILTIN[name]
    raise ValueError(f"No input format for Apify actor {name!r}: add its `input` template to {config}.")


class ProductUnavailable(SourceUnavailable):
    """The actor could not read this product (delisted, merged or invalid ASIN)."""


@dataclass
class FetchResult:
    reviews: pd.DataFrame
    hit_cap: bool
    complete_from: str | None  # reviews are complete from this date on; None = full history returned


def completeness(df: pd.DataFrame, cap: int | dict[int, int], by_star: bool) -> tuple[bool, str | None]:
    """Find where truncation starts. With star filters, each star's list is
    complete back to its own oldest review unless it hit the cap; the set as a
    whole is complete from the latest of those truncation points."""
    if df.empty:
        return False, None
    groups = df.groupby(df["rating"].round()) if by_star else [(None, df)]
    limit = (lambda star: cap.get(int(star), 10**9)) if isinstance(cap, dict) else (lambda star: cap)
    cut = [g["review_date"].min() for star, g in groups if len(g) >= limit(star)]
    return bool(cut), max(cut) if cut else None


class ApifyReviews:
    def __init__(self, token: str | None = None, actor: str | None = None, log: AccessLog | None = None,
                 config: Path = CONFIG):
        self.token = token or os.environ.get("APIFY_TOKEN")
        self.actor, self.profile = resolve_profile(actor, config)
        self.log = log or AccessLog(None)

    def _call(self, method: str, path: str, *, retries: int = 5, **kw) -> requests.Response:
        """One short API call, retried on dropped connections and server errors."""
        err = ""
        for attempt in range(retries):
            try:
                resp = requests.request(method, f"{API}{path}", params={"token": self.token, **kw.pop("params", {})},
                                        timeout=90, **kw)
            except requests.RequestException as e:
                err = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code < 500 and resp.status_code != 429:
                    return resp
                err = f"HTTP {resp.status_code}"
            time.sleep(min(60, 3 * 2 ** attempt))
        raise SourceUnavailable(f"Apify unreachable after {retries} tries: {err}")

    def run(self, payload: dict[str, Any], *, asin: str = "", timeout_s: int = 900) -> list[dict[str, Any]]:
        """Start an actor run, wait for it with short polls, return its dataset items."""
        for attempt in range(8):
            resp = self._call("POST", f"/acts/{self.actor}/runs", json=payload)
            if resp.status_code != 402:
                break
            # 402 = too many concurrent runs for the plan's memory (wait for one to
            # finish), or the monthly credit is used up (waiting won't help).
            time.sleep(min(120, 15 * (attempt + 1)))
        if resp.status_code == 402:
            raise SourceUnavailable("Apify unreachable: HTTP 402 persisted. Either the account's monthly credit is "
                                    "used up (console.apify.com/billing) or too many runs were started at once "
                                    f"for the plan's memory limit. Details: {resp.text[:200]}")
        if resp.status_code not in (200, 201):
            raise SourceUnavailable(f"Apify actor {self.actor} returned HTTP {resp.status_code}: {resp.text[:300]}")
        run = resp.json()["data"]
        deadline = time.time() + timeout_s
        while run["status"] in ("READY", "RUNNING") and time.time() < deadline:
            run = self._call("GET", f"/actor-runs/{run['id']}", params={"waitForFinish": 60}).json()["data"]
        self.log.record(source="apify", actor=self.actor, asin=asin, run=run["id"], status=run["status"])
        if run["status"] != "SUCCEEDED":
            # Readable products succeed even with zero reviews; a failed run means
            # the actor could not read the product (delisted, merged or invalid).
            raise ProductUnavailable(f"{asin}: actor run {run['status']} (product delisted or unreadable?)")
        return self._call("GET", f"/datasets/{run['defaultDatasetId']}/items", params={"clean": "true"}).json()

    def newest_review(self, asin: str) -> str | None:
        """Date of the product's newest review, from a few-review request (cheap).

        Uses the last configured star filter (5 stars: the most frequent) so it
        costs at most 3 reviews.
        """
        items = self.run(self.profile.payloads(asin, 3, None)[-1], asin=asin)
        df, _ = normalize_items(items, ticker="", brand="", fetched_at="")
        return df["review_date"].max() if len(df) else None

    def fetch(self, asin: str, *, ticker: str, brand: str, max_reviews: int | dict[int, int] = 100,
              since: str | None = None, timeout_s: int = 900, back_to: str | None = None,
              top_up: int | None = None) -> FetchResult:
        """Newest reviews of one product. max_reviews applies per star filter for star-filtered actors.

        back_to/top_up (star-split actors): if a star's list fills its request and still
        doesn't reach back to `back_to` (the newest review already stored), that star is
        asked again with `top_up` reviews, so a busier week than expected doesn't open a gap.
        """
        if not self.token:
            raise SourceUnavailable("APIFY_TOKEN is not set; cannot run the Apify review scraper.")
        payloads = self.profile.payloads(asin, max_reviews, since)
        batches = [self.run(payload, asin=asin, timeout_s=timeout_s) for payload in payloads]
        split = self.profile.by_star and len(payloads) == len(STARS)
        sizes = self.profile.cap(max_reviews)
        size = (lambda star: sizes.get(star, max(sizes.values()))) if isinstance(sizes, dict) else (lambda star: sizes)
        if split:
            sizes = {star: size(star) for star in range(1, len(STARS) + 1)}
            for i, batch in enumerate(batches):
                star = i + 1
                if not (back_to and top_up and len(batch) >= sizes[star] and sizes[star] < top_up):
                    continue
                d, _ = normalize_items(batch, ticker=ticker, brand=brand, fetched_at="")
                if len(d) and d["review_date"].min() <= back_to:
                    continue  # reached what is stored: complete
                bigger = {**sizes, star: top_up}
                batches[i] = self.run(self.profile.payloads(asin, bigger, since)[i], asin=asin, timeout_s=timeout_s)
                sizes = {**sizes, star: self.profile.cap(bigger)[star]}
                self.log.record(source="apify", kind="top_up", asin=asin, star=star, size=top_up)
        items = [x for batch in batches for x in batch]
        df, unmapped = normalize_items(items, ticker=ticker, brand=brand, fetched_at=utcnow())
        if len(df):
            df["asin"] = df["asin"].fillna(asin)
        if split:
            # Truncation is judged on the raw count per star: reviews from other
            # marketplaces are dropped after, and must not make a cut list look complete.
            cut = []
            for i, batch in enumerate(batches):
                if len(batch) >= sizes[i + 1]:
                    d, _ = normalize_items(batch, ticker=ticker, brand=brand, fetched_at="")
                    cut.append(d["review_date"].min() if len(d) else utcnow()[:10])
            hit_cap, complete_from = bool(cut), (max(cut) if cut else None)
        else:
            hit_cap, complete_from = completeness(df, self.profile.cap(max_reviews), self.profile.by_star)
        self.log.record(source="apify", kind="coverage", asin=asin, reviews=len(df),
                        oldest=df["review_date"].min() if len(df) else None, hit_cap=hit_cap,
                        complete_from=complete_from)
        if unmapped:
            self.log.record(source="apify", kind="unmapped_fields", fields=unmapped)
        return FetchResult(df, hit_cap, complete_from)


def preflight(client: ApifyReviews, as_of: Any, config: Path = CONFIG) -> str:
    """Fail fast if the actor returns nothing recent for a known-active product.

    Some actors end "successfully" with zero results when the account's plan
    doesn't allow them; without this check every product would look dead.
    """
    cfg = load_config(config)
    asin = cfg.get("check_asin")
    if not asin:
        return "skipped"
    window = int(cfg.get("live_window_days", 120))
    latest = client.newest_review(asin)
    cutoff = (pd.Timestamp(as_of) - pd.Timedelta(days=window)).strftime("%Y-%m-%d")
    if not latest or latest < cutoff:
        raise SourceUnavailable(
            f"Preflight failed: actor {client.actor} returned no recent reviews for check_asin {asin} "
            f"(newest: {latest}). The actor may need a paid Apify plan, its input format may have changed, "
            "or check_asin is no longer sold. See config/apify.yaml.")
    return latest
