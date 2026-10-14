"""Firm-quarter review measures (step 5; Jin 2022 Appendix A).

AbnormalTone_q      = mean tone in q  - mean of the four prior quarterly means
AbnormalRating_q    = mean rating in q - mean of the four prior quarterly means
AbnormalReviewCnt_q = (reviews in q - mean of prior four counts) / 100
StdDevTone_q        = std of tone in q / mean tone in q (paper's scaling)
StdDevRating_q      = std of rating in q / mean rating in q
Negative/Positive_AbnormalTone split AbnormalTone by sign (Tables 9-11).

Tone is on the paper's [-1, 1] scale. Reviews flagged seller/shipping-only are
counted but excluded from tone.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .fiscal import FiscalCalendar

TAG_COLUMNS = ["defect", "quality_decline", "durability", "price_value"]


def attach_scores(reviews: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    cols = ["review_id", "tone", "tags", "seller_shipping_only"]
    return reviews.merge(scores[cols], on="review_id", how="left")


def quarterly_measures(scored: pd.DataFrame, cal: FiscalCalendar, *, min_reviews: int,
                       as_of: date | None = None, coverage_start: date | None = None) -> pd.DataFrame:
    """One row per fiscal quarter between the first review and `as_of`.

    coverage_start: earliest date from which the review history is believed
    complete (e.g. a scraper capped per product). Quarters starting before it
    are flagged `partial_coverage`.
    """
    if scored.empty:
        return pd.DataFrame()
    df = scored.copy()
    df["review_date"] = pd.to_datetime(df["review_date"])
    as_of = as_of or df["review_date"].max().date()
    fq = [cal.quarter(d) for d in df["review_date"]]
    df["fq_index"] = [q.index for q in fq]
    df["seller_shipping_only"] = df["seller_shipping_only"].fillna(False).astype(bool)
    opinion = df[~df["seller_shipping_only"] & df["tone"].notna()]

    tags = opinion["tags"].fillna("").astype(str)
    for t in TAG_COLUMNS:
        opinion = opinion.assign(**{f"tag_{t}": tags.str.contains(t, regex=False)})

    g = opinion.groupby("fq_index")
    out = pd.DataFrame({
        "n_reviews_all": df.groupby("fq_index").size(),
        "n_seller_shipping_only": df.groupby("fq_index")["seller_shipping_only"].sum(),
        "n_reviews": g.size(),
        "mean_tone": g["tone"].mean(),
        "sd_tone": g["tone"].std(),
        "neg_share": g["tone"].apply(lambda s: float((s < 0).mean())),
        "mean_rating": g["rating"].mean(),
        "sd_rating": g["rating"].std(),
        **{f"share_{t}": g[f"tag_{t}"].mean() for t in TAG_COLUMNS},
    })

    first = cal.quarter(df["review_date"].min())
    last = cal.quarter(as_of)
    idx = range(first.index, last.index + 1)
    out = out.reindex(idx)
    out.index.name = "fq_index"
    out[["n_reviews_all", "n_reviews", "n_seller_shipping_only"]] = (
        out[["n_reviews_all", "n_reviews", "n_seller_shipping_only"]].fillna(0).astype(int))

    quarters = [first.shift(i - first.index) for i in idx]
    out.insert(0, "fiscal_quarter", [q.label for q in quarters])
    out.insert(1, "quarter_start", [q.start.isoformat() for q in quarters])
    out.insert(2, "quarter_end", [q.end.isoformat() for q in quarters])
    out["fiscal_q"] = [q.q for q in quarters]

    with np.errstate(divide="ignore", invalid="ignore"):
        out["StdDevTone"] = np.where(out["mean_tone"].abs() >= 0.05, out["sd_tone"] / out["mean_tone"], np.nan)
        out["StdDevRating"] = out["sd_rating"] / out["mean_rating"]

    prior = lambda col: out[col].shift(1).rolling(4, min_periods=4).mean()  # noqa: E731
    out["prior4_tone"] = prior("mean_tone")
    out["AbnormalTone"] = out["mean_tone"] - out["prior4_tone"]
    out["AbnormalRating"] = out["mean_rating"] - prior("mean_rating")
    out["AbnormalReviewCnt"] = (out["n_reviews"] - prior("n_reviews")) / 100.0
    out["Negative_AbnormalTone"] = out["AbnormalTone"].where(out["AbnormalTone"] < 0)
    out["Positive_AbnormalTone"] = out["AbnormalTone"].where(out["AbnormalTone"] > 0)

    sparse = out["n_reviews"] < min_reviews
    out["sparse"] = sparse
    # A baseline built on a sparse quarter is unreliable too.
    out["sparse_baseline"] = sparse.astype(float).shift(1).rolling(4, min_periods=1).max().fillna(1).astype(bool)
    out["in_progress"] = [q.end > as_of for q in quarters]
    out["partial_coverage"] = [bool(coverage_start and q.start < coverage_start) for q in quarters]
    out["usable"] = (out["AbnormalTone"].notna() & ~out["sparse"] & ~out["sparse_baseline"]
                     & ~out["in_progress"] & ~out["partial_coverage"])
    return out.reset_index()


def coverage_start_from_log(coverage: pd.DataFrame) -> date | None:
    """Latest 'oldest review' among products whose fetch hit the per-product cap.

    If a product's reviews were truncated at the cap, history before its oldest
    fetched review is missing, so quarters before that date are incomplete.
    """
    if coverage is None or coverage.empty or "hit_cap" not in coverage:
        return None
    capped = coverage[coverage["hit_cap"].astype(str).str.lower() == "true"]
    if capped.empty:
        return None
    return pd.to_datetime(capped["oldest"]).max().date()


def history_status(quarterly: pd.DataFrame, target: int) -> dict[str, object]:
    complete = quarterly[~quarterly["sparse"] & ~quarterly["in_progress"] & ~quarterly["partial_coverage"]]
    return {
        "complete_quarters": int(len(complete)),
        "target_complete_quarters": target,
        "usable_signal_quarters": int(quarterly["usable"].sum()),
        "sufficient": bool(len(complete) >= target),
    }
