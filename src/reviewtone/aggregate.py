"""Firm-quarter review measures (step 5; Jin 2022 Appendix A).

AbnormalTone_q      = mean tone in q  - mean of the four prior quarterly means
AbnormalRating_q    = mean rating in q - mean of the four prior quarterly means
AbnormalReviewCnt_q = (reviews in q - mean of prior four counts) / 100
StdDevTone_q        = std of tone in q / mean tone in q (paper's scaling)
StdDevRating_q      = std of rating in q / mean rating in q
Negative/Positive_AbnormalTone split AbnormalTone by sign (Tables 9-11).

Tone is on the paper's [-1, 1] scale. Reviews flagged seller/shipping-only are
counted but excluded from tone.

Matched products: live scraping returns only each product's newest ~100
reviews per star, so a busy product's history can have a hole between the
historical dataset and the first live fetch. A quarter's abnormal measures
then compare it with its four prior quarters using only the products whose
reviews are complete in all five, so a missing product never looks like a
change in tone. With no holes (the historical backtest) every product is used.
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
                       as_of: date | None = None,
                       gaps: dict | list | None = None) -> pd.DataFrame:
    """One row per fiscal quarter between the first review and `as_of`.

    gaps: {product: [(start, end), ...]} windows in which that product's reviews
    are known to be incomplete (see coverage_gaps); start=None means "from the
    beginning" and the key None means every product (a plain list is treated
    that way). Reviews are matched to products by a `product` column (default:
    parent_asin, else asin). For each quarter, products with a hole in it or in
    its four-quarter baseline are left out of its abnormal measures (matched
    products, see the module docstring); `n_products_excluded` counts them and
    `partial_coverage` / `partial_baseline` say where the holes are.
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

    # Which products have a hole in which quarter.
    if isinstance(gaps, list):
        gaps = {None: gaps}
    holes: dict[int, set] = {i: set() for i in idx}
    everyone = object()  # marker: every product has a hole
    for key, windows in (gaps or {}).items():
        for i, q in zip(idx, quarters):
            if any((gs is None or gs < q.end) and ge > q.start for gs, ge in windows):
                holes[i].add(everyone if key is None else key)
    out["partial_coverage"] = [bool(holes[i]) for i in idx]
    out["partial_baseline"] = [any(holes.get(j) for j in range(i - 4, i)) for i in idx]

    # Matched-product overrides for quarters whose five-quarter window has holes.
    if any(holes.values()):
        if "product" in opinion:
            product = opinion["product"].astype(str)
        elif "parent_asin" in opinion:
            product = opinion["parent_asin"].where(opinion["parent_asin"].astype(str) != "", opinion["asin"]).astype(str)
        else:
            product = pd.Series("", index=opinion.index)  # only gaps for every product (key None) can apply
        cache: dict[frozenset, pd.DataFrame] = {}

        def stats(excluded: frozenset) -> pd.DataFrame:
            if excluded not in cache:
                keep = opinion if not excluded else opinion[~product.isin(excluded)]
                if everyone in excluded:
                    keep = opinion.iloc[0:0]
                gg = keep.groupby("fq_index")
                cache[excluded] = pd.DataFrame({"n": gg.size(), "tone": gg["tone"].mean(),
                                                "rating": gg["rating"].mean()}).reindex(idx)
            return cache[excluded]

        cols = ["n_reviews", "mean_tone", "mean_rating", "prior4_tone", "AbnormalTone", "AbnormalRating",
                "AbnormalReviewCnt", "sparse", "sparse_baseline"]
        override: dict[int, dict] = {}
        n_excluded = {}
        for i in idx:
            excluded = frozenset().union(*(holes.get(j, set()) for j in range(i - 4, i + 1)))
            if not excluded:
                continue
            n_excluded[i] = -1 if everyone in excluded else len(excluded)
            st = stats(excluded)
            n = st["n"].fillna(0)
            prior = [j for j in range(i - 4, i) if j >= first.index]
            tone_prior = st["tone"].reindex(prior)
            rating_prior = st["rating"].reindex(prior)
            full = len(prior) == 4
            prior_tone = tone_prior.mean() if full and tone_prior.notna().all() else np.nan
            prior_rating = rating_prior.mean() if full and rating_prior.notna().all() else np.nan
            override[i] = {
                "n_reviews": int(n[i]), "mean_tone": st["tone"][i], "mean_rating": st["rating"][i],
                "prior4_tone": prior_tone, "AbnormalTone": st["tone"][i] - prior_tone,
                "AbnormalRating": st["rating"][i] - prior_rating,
                "AbnormalReviewCnt": (n[i] - n.reindex(prior).mean()) / 100.0 if full else np.nan,
                "sparse": bool(n[i] < min_reviews),
                "sparse_baseline": bool(not prior or (n.reindex(prior) < min_reviews).any()),
            }
        for i, vals in override.items():
            for c in cols:
                out.loc[i, c] = vals[c]
        out["Negative_AbnormalTone"] = out["AbnormalTone"].where(out["AbnormalTone"] < 0)
        out["Positive_AbnormalTone"] = out["AbnormalTone"].where(out["AbnormalTone"] > 0)
        out["n_products_excluded"] = pd.Series(n_excluded, dtype=float).reindex(idx).fillna(0).astype(int)
    else:
        out["n_products_excluded"] = 0
    out["sparse"] = out["sparse"].astype(bool)
    out["sparse_baseline"] = out["sparse_baseline"].astype(bool)
    # -1 in n_products_excluded: no product is complete (e.g. after the historical
    # dataset ends and before any live fetch), so the quarter has no reviews to use.
    out["usable"] = (out["AbnormalTone"].notna() & ~out["sparse"] & ~out["sparse_baseline"]
                     & ~out["in_progress"])
    return out.reset_index()


def coverage_gaps(coverage: pd.DataFrame, products: list[str] | None = None,
                  dataset_end: str | None = None) -> dict[str, list[tuple[date | None, date]]]:
    """Per-product windows with reviews known to be missing.

    - holes recorded by capped scraper fetches (coverage.csv gap_start/gap_end);
    - after the last live fetch of a product: reviews since then are not collected yet;
    - after the historical dataset ends (dataset_end) for products never fetched live.
    """
    gaps: dict[str, list[tuple[date | None, date]]] = {}
    fetched: set[str] = set()
    if coverage is not None and len(coverage):
        if "gap_end" in coverage:
            for r in coverage.dropna(subset=["gap_end"]).itertuples():
                gaps.setdefault(str(r.asin), []).append(
                    (None if pd.isna(r.gap_start) else pd.Timestamp(r.gap_start).date(), pd.Timestamp(r.gap_end).date()))
        if "fetched_at" in coverage:
            last = pd.to_datetime(coverage["fetched_at"], errors="coerce", utc=True).groupby(coverage["asin"].astype(str)).max()
            for asin, at in last.dropna().items():
                gaps.setdefault(asin, []).append((at.tz_convert(None).date(), date.max))
        fetched = set(coverage["asin"].astype(str))
    if dataset_end:
        for asin in products or []:
            if asin not in fetched:
                gaps.setdefault(asin, []).append((date.fromisoformat(dataset_end), date.max))
    return gaps


def history_status(quarterly: pd.DataFrame, target: int) -> dict[str, object]:
    complete = quarterly[~quarterly["sparse"] & ~quarterly["in_progress"]]
    return {
        "complete_quarters": int(len(complete)),
        "target_complete_quarters": target,
        "usable_signal_quarters": int(quarterly["usable"].sum()),
        "sufficient": bool(len(complete) >= target),
    }
