"""Cost and timing estimates from a company's own data (no API calls, free).

For each company's tracked products, review rates per star come from the last
year of the historical dataset. From them:

- first live fetch: sized by pipeline.first_fetch_plan (back to the history, back
  five quarters, or only this quarter: the deepest that is affordable); charged per
  review returned; only the star ratings in the study's `aggregation.stars`;
- updates (weekly or every 4 weeks): request sizes follow Run._request_sizes; a
  product with more new reviews of one star between updates than one request
  holds (100) is cut off every time and never enters the tone measure;
- coverage: products already fetched use their recorded gaps (coverage.csv);
  others get the first-fetch plan. A product with a gap rejoins the tone
  measure once five complete quarters have accumulated (matched products);
- first live call: the first quarter whose matched products have at least
  min_reviews_per_quarter reviews.

Products already followed live use their live review rates; the rest use
2022-23 rates, which can be far off today (in testing one product matched its
2022-23 rate closely and another had fallen by two thirds), so treat the
numbers as estimates and the `run` report as the truth.
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
    from .pipeline import first_fetch_plan, history_rates, live_rates

    cfg = load_config()
    price = float(cfg.get("price_per_review", 0.0018))
    min_reviews = int(run.study.aggregation["min_reviews_per_quarter"])
    status = run._status()
    checked = "check_live" in status
    path = run.dir / "products.csv"
    products = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame()
    if len(products) and not checked:
        # Not checked yet: skip what the free history filter already knows is retired (in memory only).
        products = products.copy()
        if "live" not in products:
            products["live"] = pd.NA
        products.loc[run.inactive_in_history(products), "live"] = "False"
    tracked = (run.tracked_products(products, include_unchecked=not checked) if len(products) else products)
    dataset_end = status.get("backfill_mcauley", {}).get("dataset_end")
    out: dict = {"ticker": run.company.ticker, "tracked_products": int(len(tracked)), "live_checked": checked}
    if not dataset_end or tracked.empty:
        out["note"] = "run `reviewtone backfill` (and `check-live`) first"
        return out
    end = pd.Timestamp(dataset_end)
    days_since = max(0, (pd.Timestamp(run.as_of) - end).days)
    reviews = rv.load(run.dir / "reviews.csv")
    hist = reviews[reviews["source"].astype(str).str.startswith("mcauley")].copy()
    hist["d"] = pd.to_datetime(hist["review_date"])
    hist["star"] = pd.to_numeric(hist["rating"], errors="coerce").round()
    cov_path = run.dir / "coverage.csv"
    cov = pd.read_csv(cov_path, dtype={"asin": str}) if cov_path.exists() else pd.DataFrame(columns=["asin"])
    live_done = set(cov["asin"])
    gap_end = (cov.dropna(subset=["gap_end"]).groupby("asin")["gap_end"].max()
               if "gap_end" in cov and len(cov) else pd.Series(dtype=str))

    cal = run.cal
    stars = run.study.stars
    this_q = cal.quarter(run.as_of)
    days_into_q = (pd.Timestamp(run.as_of) - pd.Timestamp(this_q.start)).days + 1
    first_reviews = 0.0
    update = {"weekly": 0.0, "four_weekly": 0.0}
    per_quarter = 0.0
    no_history = 0
    products_out = []
    for p in tracked.itertuples():
        mine = hist[(hist["parent_asin"] == p.asin) | (hist["asin"] == p.asin)]
        if mine.empty:
            # No history (e.g. a seed ASIN): upper bound for its first fetch, lower bound for
            # updates; it can't be placed on the timeline.
            no_history += 1
            if p.asin not in live_done:
                first_reviews += len(stars) * cap
            update["weekly"] += 2 * len(stars)
            update["four_weekly"] += 2 * len(stars)
            continue
        rate = history_rates(mine)
        live = reviews[((reviews["parent_asin"] == p.asin) | (reviews["asin"] == p.asin))
                       & reviews["source"].astype(str).str.startswith("apify")]
        if p.asin in live_done and len(live):
            rate = live_rates(live)  # already followed live: today's rate beats the 2022-23 one
        available = {s: float((mine["star"] == s).sum()) + rate[s] * days_since for s in STARS}
        if p.asin in live_done:
            gap = p.asin in gap_end.index
            clean_from = cal.quarter(date.fromisoformat(str(gap_end[p.asin]))).shift(1) if gap else None
        else:
            sizes, _, reach = first_fetch_plan(mine, run.as_of, cal, cap, stars)
            first_reviews += sum(min(sizes[s], available[s]) for s in stars)
            gap = reach != "history"
            if reach == "five_quarters":
                clean_from = this_q.shift(-5)
            else:
                # A busy star's capped list may not reach back to this quarter's start.
                late = any(rate[s] > 0 and cap / rate[s] < days_into_q for s in stars)
                clean_from = this_q.shift(1) if late else this_q
        for days, key in ((7, "weekly"), (28, "four_weekly")):  # sizes as Run._request_sizes
            req = {s: min(cap, max(2, math.ceil(rate[s] * (days + 2) * 1.5) + 2)) for s in stars}
            update[key] += sum(min(req[s], available[s]) for s in stars)
        q_reviews = sum(rate[s] for s in stars) * 91
        per_quarter += q_reviews
        products_out.append({
            "asin": p.asin, "per_quarter": q_reviews, "gap": gap, "clean_from": clean_from,
            # more new reviews of one star between updates than one request can hold: cut off every time
            "cut_off_weekly": any(rate[s] * 9 >= cap for s in stars),
            "cut_off_four_weekly": any(rate[s] * 30 >= cap for s in stars)})

    out.update({
        "reviews_per_quarter": round(per_quarter),
        "first_fetch_reviews": round(first_reviews), "first_fetch_usd": round(first_reviews * price, 2),
        "monthly_usd_weekly_updates": round(update["weekly"] * price * 52 / 12, 2),
        "monthly_usd_four_weekly_updates": round(update["four_weekly"] * price * 13 / 12, 2),
        "products_with_gap": sum(p["gap"] for p in products_out),
        "products_without_history": no_history,
        "cut_off_by_four_weekly": sum(p["cut_off_four_weekly"] for p in products_out),
        "claude_label_usd": round(_claude_cost(hist, run.study.tone, label_sample), 3),
        "first_live_call": _first_call(run, products_out, min_reviews, "weekly"),
        "first_live_call_four_weekly": _first_call(run, products_out, min_reviews, "four_weekly"),
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


def _first_call(run, products: list[dict], min_reviews: int, cadence: str) -> str:
    """First quarter whose matched products (complete in it and its 4-quarter baseline,
    and not cut off at this update cadence) carry enough reviews for a call."""
    if not products:
        return "unknown: no tracked product has history"
    usable = [p for p in products if not p[f"cut_off_{cadence}"]]
    total = sum(p["per_quarter"] for p in usable)
    if total < min_reviews:
        return (f"unlikely: ~{total:.0f} usable reviews a quarter, a call needs {min_reviews}; "
                "consider more top_products before the first run, or weekly updates")
    today = run.as_of
    last_done = run.cal.quarter(today).shift(-1)
    for k in range(0, 12):
        q = last_done.shift(k)
        ready = sum(p["per_quarter"] for p in usable
                    if not p["gap"] or q.shift(-4).index >= p["clean_from"].index)
        if ready >= min_reviews:
            when = "now (after the first live fetch)" if q.end < today else f"after {q.end.isoformat()}"
            return f"{q.label} ({when})"
    return "not within three years at current review rates"


def table(rows: list[dict]) -> str:
    head = ("Ticker", "Live", "Reviews/qtr", "First fetch", "Per month (weekly / 4-weekly updates)",
            "Gap products", "Claude labels", "First live call (weekly updates)", "Notes")
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
            r["first_live_call"], _notes(r)]))
    ok = [r for r in rows if "note" not in r]
    if ok:
        lines.append("")
        lines.append(f"Totals on Apify: first live fetch ${sum(r['first_fetch_usd'] for r in ok):.2f} once, then about "
                     f"${sum(r['monthly_usd_weekly_updates'] for r in ok):.2f} a month with weekly updates or "
                     f"${sum(r['monthly_usd_four_weekly_updates'] for r in ok):.2f} with updates every 4 weeks. "
                     f"Claude labels: ${sum(r['claude_label_usd'] for r in ok):.2f} once (Anthropic).")
    return "\n".join(lines)


def _notes(r: dict) -> str:
    notes = []
    if not r["live_checked"]:
        notes.append("not checked live yet: rough")
    if r["cut_off_by_four_weekly"]:
        notes.append(f"{r['cut_off_by_four_weekly']} product(s) too busy for 4-weekly updates "
                     f"(4-weekly first call: {r['first_live_call_four_weekly']})")
    if r["products_without_history"]:
        notes.append(f"{r['products_without_history']} without history: first fetch counted at up to 100 "
                     "reviews per star each, not on the timeline")
    return "; ".join(notes) or "-"


def as_of_note(day: date) -> str:
    return f"Estimates as of {day.isoformat()} from each company's 2022-23 review rates (no API calls made)."
