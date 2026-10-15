"""Cost and timing estimates from a company's own data (no API calls, free).

For each company's tracked products, review rates per star come from the last
year of the historical dataset. From them:

- first live fetch: sized by pipeline.first_fetch_sizes (bridge to the history
  when possible, else only the current quarter); charged per review returned;
- weekly updates: request sizes follow Run._request_sizes;
- coverage: if a product's newest `cap` reviews of some star don't reach back to
  the end of the historical data, it has a gap and only rejoins the tone
  measure once five gap-free quarters have accumulated (matched products);
- first live call: the first quarter whose matched products have at least
  min_reviews_per_quarter reviews.

Rates can drift since 2023 (one product checked live in testing matched its
2022-23 rate closely), so treat the numbers as estimates.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from . import reviews as rv
from .sources.apify import load_config
from .tone.claude import PRICES, SYSTEM

STARS = (1, 2, 3, 4, 5)


def company(run, cap: int = 100, label_sample: int = 1000) -> dict:
    """Estimates for one company (a pipeline.Run)."""
    cfg = load_config()
    price = float(cfg.get("price_per_review", 0.0018))
    min_reviews = int(run.study.aggregation["min_reviews_per_quarter"])
    tracked = run.tracked_products()
    status = run._status()
    dataset_end = status.get("backfill_mcauley", {}).get("dataset_end")
    out: dict = {"ticker": run.company.ticker, "tracked_products": int(len(tracked)),
                 "live_checked": "check_live" in status}
    if not dataset_end or tracked.empty:
        out["note"] = "run `reviewtone backfill` (and `check-live`) first"
        return out
    end = pd.Timestamp(dataset_end)
    days_since = max(0, (pd.Timestamp(run.as_of) - end).days)
    reviews = rv.load(run.dir / "reviews.csv")
    hist = reviews[reviews["source"].astype(str).str.startswith("mcauley")].copy()
    hist["d"] = pd.to_datetime(hist["review_date"])
    hist["star"] = pd.to_numeric(hist["rating"], errors="coerce").round()
    live_done = set()
    cov_path = run.dir / "coverage.csv"
    if cov_path.exists():
        live_done = set(pd.read_csv(cov_path, dtype={"asin": str})["asin"])

    from .pipeline import first_fetch_sizes, history_rates

    qstart = run.cal.quarter(run.as_of).start
    first_reviews = 0.0
    update = {"weekly": 0.0, "four_weekly": 0.0}
    per_quarter = 0.0
    products = []
    for p in tracked.itertuples():
        mine = hist[(hist["parent_asin"] == p.asin) | (hist["asin"] == p.asin)]
        if mine.empty:
            continue
        rate = history_rates(mine)
        available = {s: float((mine["star"] == s).sum()) + rate[s] * days_since for s in STARS}
        sizes, bridges = first_fetch_sizes(mine, run.as_of, qstart, cap)
        if p.asin not in live_done:
            first_reviews += sum(min(sizes[s] if isinstance(sizes, dict) else sizes, available[s]) for s in STARS)
        for days, key in ((7, "weekly"), (28, "four_weekly")):  # sizes as Run._request_sizes
            req = {s: min(cap, max(2, math.ceil(rate[s] * (days + 2) * 1.5) + 2)) for s in STARS}
            update[key] += sum(min(req[s], available[s]) for s in STARS)
        per_quarter += sum(rate.values()) * 91
        products.append({"asin": p.asin, "per_quarter": sum(rate.values()) * 91, "gap": not bridges})

    out.update({
        "reviews_per_quarter": round(per_quarter),
        "first_fetch_reviews": round(first_reviews), "first_fetch_usd": round(first_reviews * price, 2),
        "monthly_usd_weekly_updates": round(update["weekly"] * price * 52 / 12, 2),
        "monthly_usd_four_weekly_updates": round(update["four_weekly"] * price * 13 / 12, 2),
        "products_with_gap": sum(p["gap"] for p in products),
        "claude_label_usd": round(_claude_cost(hist, run.study.tone, label_sample), 3),
        "first_live_call": _first_call(run, products, min_reviews),
    })
    return out


def _claude_cost(hist: pd.DataFrame, tone: dict, n: int) -> float:
    """Haiku cost of labelling n reviews: ~3.5 characters per token, ~55 output tokens per
    review plus ~250 tokens of low-effort thinking per request of 25 (stand-in measurement)."""
    price = PRICES.get(tone.get("claude_model", ""), (0.10, 0.50))
    sample = hist.head(2000)
    chars = (sample["title"].astype(str).str.len() + sample["text"].astype(str).str.len()).mean() + 60 \
        if len(sample) else 300
    batch = int(tone.get("reviews_per_request", 25))
    requests = math.ceil(n / batch)
    tokens_in = requests * (len(SYSTEM) + 1200) / 3.5 + n * chars / 3.5  # + ~1200 chars of output schema
    tokens_out = n * 55 + requests * 250
    return (tokens_in * price[0] + tokens_out * price[1]) / 1e6


def _first_call(run, products: list[dict], min_reviews: int) -> str:
    """First quarter whose gap-free products carry enough reviews for a call."""
    cal = run.cal
    today = run.as_of
    total = sum(p["per_quarter"] for p in products)
    if total < min_reviews:
        return (f"unlikely: tracked products get ~{total:.0f} reviews a quarter, a call needs {min_reviews}; "
                "consider more top_products before the first run")
    last_done = cal.quarter(today).shift(-1)
    clean_from = cal.quarter(today)  # the first fetch reaches back to this quarter's start
    for k in range(0, 12):
        q = last_done.shift(k)
        ready = sum(p["per_quarter"] for p in products
                    if not p["gap"] or q.shift(-4).index >= clean_from.index)
        if ready >= min_reviews:
            when = "now (after the first live fetch)" if q.end < today else f"after {q.end.isoformat()}"
            return f"{q.label} ({when})"
    return "not within three years at current review rates"


def table(rows: list[dict]) -> str:
    head = ("Ticker", "Live", "Reviews/qtr", "First fetch", "Per month (weekly / 4-weekly updates)",
            "Gap products", "Claude labels", "First live call")
    lines = [" | ".join(head), " | ".join("---" for _ in head)]
    for r in rows:
        if "note" in r:
            lines.append(f"{r['ticker']} | {r['note']}")
            continue
        lines.append(" | ".join([
            r["ticker"], f"{r['tracked_products']}{'' if r['live_checked'] else ' (unchecked)'}",
            str(r["reviews_per_quarter"]), f"${r['first_fetch_usd']:.2f} ({r['first_fetch_reviews']})",
            f"${r['monthly_usd_weekly_updates']:.2f} / ${r['monthly_usd_four_weekly_updates']:.2f}",
            str(r["products_with_gap"]), f"${r['claude_label_usd']:.3f}",
            r["first_live_call"]]))
    ok = [r for r in rows if "note" not in r]
    if ok:
        lines.append("")
        lines.append(f"Totals on Apify: first live fetch ${sum(r['first_fetch_usd'] for r in ok):.2f} once, then about "
                     f"${sum(r['monthly_usd_weekly_updates'] for r in ok):.2f} a month with weekly updates or "
                     f"${sum(r['monthly_usd_four_weekly_updates'] for r in ok):.2f} with updates every 4 weeks. "
                     f"Claude labels: ${sum(r['claude_label_usd'] for r in ok):.2f} once (Anthropic).")
    return "\n".join(lines)


def as_of_note(day: date) -> str:
    return f"Estimates as of {day.isoformat()} from each company's 2022-23 review rates (no API calls made)."
