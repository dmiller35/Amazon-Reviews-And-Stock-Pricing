"""Per-company research pipeline (prompt steps 1-7) and the universe portfolio.

Each step writes into research/<TICKER>/ and records anything that limited it
(unreachable source, thin data, missing history) in status.json, which the
report turns into the Limitations section. Steps degrade instead of inventing
data: no reviews -> no tone; no SEC access -> no outcomes; and so on.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from . import aggregate, catalog, charts, evaluate, reviews as rv
from .config import Company, Study
from .fiscal import CALENDAR_QUARTERS, FiscalCalendar
from .http import AccessLog, SourceUnavailable, utcnow
from .tone import score_reviews


@dataclass
class Run:
    study: Study
    company: Company
    as_of: date = field(default_factory=date.today)
    scorer: str | None = None

    @property
    def scorer_name(self) -> str:
        return self.scorer or self.study.tone["scorer"]

    @property
    def dir(self) -> Path:
        d = self.study.ticker_dir(self.company.ticker)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def log(self) -> AccessLog:
        return AccessLog(self.dir / "access_log.jsonl")

    @property
    def cal(self) -> FiscalCalendar:
        v = self._status().get("verification", {})
        month = self.company.fiscal_year_end_month
        if v.get("sec_fiscal_year_end"):
            from .fiscal import parse_sec_fiscal_year_end
            month = parse_sec_fiscal_year_end(v["sec_fiscal_year_end"]) or month
        return FiscalCalendar(month)

    def _status(self) -> dict:
        p = self.dir / "status.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def note(self, step: str, **info) -> None:
        status = self._status()
        status[step] = {**info, "updated": utcnow()}
        (self.dir / "status.json").write_text(json.dumps(status, indent=2, default=str))

    # Step 1 ---------------------------------------------------------------
    def verify(self) -> dict:
        from .fundamentals.sec import verify_company
        try:
            v = verify_company(self.company.ticker, self.company.fiscal_year_end_month, self.company.name, self.log)
        except (SourceUnavailable, KeyError) as e:
            v = {"error": str(e), "fiscal_year_end_matches": None,
                 "note": f"Unverified: using configured fiscal year end {self.company.fiscal_year_end}."}
        self.note("verification", **v)
        return v

    # Step 2 ---------------------------------------------------------------
    def build_catalog(self, meta_locations: list[str] | None = None) -> pd.DataFrame:
        from .sources import mcauley
        frames, errors = [], []
        for loc in meta_locations or []:
            try:
                frames.append(mcauley.products_from_meta(loc, self.company, self.log))
            except SourceUnavailable as e:
                errors.append(str(e))
        frames.append(catalog.load_seed(self.dir / "seed_asins.csv", self.company))
        existing = self.dir / "products.csv"
        if existing.exists():
            frames.append(pd.read_csv(existing, dtype=str))
        products, n_dupes = catalog.merge_products(*frames)
        products.to_csv(existing, index=False)
        self.note("catalog", products=len(products), duplicates_removed=n_dupes, errors=errors)
        return products

    # Step 3 ---------------------------------------------------------------
    def fetch_reviews(self, source: str, *, review_locations: list[str] | None = None,
                      max_reviews_per_product: int = 100, csv_path: str | None = None,
                      top_products: int | None = None) -> int:
        products = pd.read_csv(self.dir / "products.csv", dtype=str) if (self.dir / "products.csv").exists() \
            else pd.DataFrame(columns=catalog.PRODUCT_COLUMNS)
        frames, errors = [], []
        if source in ("mcauley", "apify") and products.empty:
            errors.append("products.csv is empty: run `reviewtone catalog --meta ...` or list ASINs in "
                          f"{self.dir / 'seed_asins.csv'} (columns: asin,brand) first")
        if source == "mcauley":
            from .sources import mcauley
            for loc in review_locations or []:
                try:
                    frames.append(mcauley.reviews_for_products(loc, products, self.company, self.log))
                except SourceUnavailable as e:
                    errors.append(str(e))
        elif source == "apify":
            from .sources.apify import ApifyReviews, ProductUnavailable, load_config, preflight
            client = ApifyReviews(log=self.log)
            want = top_products or self.study.raw.get("apify", {}).get("top_products", 20)
            try:
                preflight(client, self.as_of)
            except SourceUnavailable as e:
                errors.append(str(e))
                want = 0  # don't mark products dead because the scraper itself is broken
            window = int(load_config().get("live_window_days", 120))
            stored = rv.load(self.dir / "reviews.csv")
            live = 0
            # Walk down the ranking: products Amazon no longer sells are marked
            # dead and replaced by the next most-reviewed, so every tracked
            # product has both historical and current reviews.
            for p in self.tracked_products(products, top=len(products)).itertuples():
                if live >= want:
                    break
                mine = stored[(stored["asin"] == p.asin) | (stored["parent_asin"] == p.asin)]
                newest = mine["review_date"].max() if len(mine) else None
                since = (pd.Timestamp(newest) - pd.Timedelta(days=7)).strftime("%Y-%m-%d") if newest else None
                cutoff = (pd.Timestamp(self.as_of) - pd.Timedelta(days=window)).strftime("%Y-%m-%d")
                try:
                    if not (newest and newest >= cutoff):
                        # Cheap check before paying for a full fetch: is this
                        # listing still collecting reviews?
                        latest = client.newest_review(p.asin)
                        if not latest or latest < cutoff:
                            raise ProductUnavailable(f"{p.asin}: no reviews since {cutoff} (newest {latest}); "
                                                     "listing retired or reviews moved to a newer one")
                    res = client.fetch(p.asin, ticker=self.company.ticker, brand=p.brand,
                                       max_reviews=self._request_sizes(mine, max_reviews_per_product), since=since)
                except ProductUnavailable as e:
                    products.loc[products["asin"] == p.asin, ["live", "live_checked_at"]] = [False, utcnow()]
                    errors.append(str(e))
                    continue
                except SourceUnavailable as e:
                    errors.append(str(e))
                    if "APIFY_TOKEN" in str(e) or "unreachable" in str(e):
                        break
                    continue
                live += 1
                products.loc[products["asin"] == p.asin, ["live", "live_checked_at"]] = [True, utcnow()]
                frames.append(res.reviews.assign(parent_asin=res.reviews["parent_asin"].fillna(p.parent_asin)))
                self._record_coverage(p.asin, res.reviews, res.hit_cap, res.complete_from, newest)
            products.to_csv(self.dir / "products.csv", index=False)
        elif source == "csv":
            frames.append(pd.read_csv(csv_path).assign(ticker=self.company.ticker))
        else:
            raise ValueError(f"unknown review source {source!r}")

        new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=rv.REVIEW_COLUMNS)
        added = rv.append_new(self.dir / "reviews.csv", new) if len(new) else new
        self.note(f"fetch_{source}", fetched=len(new), added=len(added), errors=errors[:10],
                  n_errors=len(errors), **({"error": errors[0]} if errors and not len(new) else {}))
        return len(added)

    def _request_sizes(self, mine: pd.DataFrame, cap: int) -> int | dict[int, int]:
        """Per-star request sizes for a product already followed live.

        The scraper has no date filter, so each run pays for the newest N reviews
        per star even if they are already stored. Sizing N to the product's own
        recent review rate (1.5x the expected new reviews since the last fetch,
        plus a margin) keeps weekly cost close to the number of truly new reviews,
        while a list that still fills up is recorded as truncated, never hidden.
        """
        live = mine[mine["source"].astype(str).str.startswith("apify")]
        if live.empty:
            return cap  # first live fetch: go as deep as allowed
        dates = pd.to_datetime(live["review_date"])
        stars = pd.to_numeric(live["rating"], errors="coerce").round()
        last = pd.to_datetime(live["fetched_at"], errors="coerce", utc=True).max()
        last = last.tz_localize(None) if pd.notna(last) else dates.max()
        days = max(1, (pd.Timestamp(self.as_of) - last).days + 2)
        window = dates >= dates.max() - pd.Timedelta(days=180)
        sizes = {}
        for star in range(1, 6):
            d = dates[window & (stars == star)]
            # Rate over this star's own span: a list truncated at N reviews covers
            # a short span, so N / span is the true rate, not N / 180 days.
            rate = len(d) / max(2, (dates.max() - d.min()).days) if len(d) else 0.0
            sizes[star] = int(min(cap, max(5, -(-rate * days * 1.5 // 1) + 3)))
        return sizes

    def tracked_products(self, products: pd.DataFrame | None = None, top: int | None = None) -> pd.DataFrame:
        """The products followed live: hand-seeded ASINs, then the most-reviewed.

        Scraping is paid per review, so only these are scraped. Tone is also
        measured on exactly this set in every quarter, historical or live, so a
        change in AbnormalTone can't come from a change in which products are
        counted.
        """
        if products is None:
            path = self.dir / "products.csv"
            products = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=catalog.PRODUCT_COLUMNS)
        top = top or self.study.raw.get("apify", {}).get("top_products", 20)
        if "live" in products:
            products = products[products["live"].astype(str) != "False"]  # delisted products are skipped
        return (products.assign(_seed=products["source"].astype(str).str.startswith("manual"),
                                _n=pd.to_numeric(products["rating_number"], errors="coerce").fillna(0))
                .sort_values(["_seed", "_n"], ascending=False).head(top).drop(columns=["_seed", "_n"]))

    def _record_coverage(self, asin: str, df: pd.DataFrame, hit_cap: bool, complete_from: str | None,
                         stored_newest: str | None) -> None:
        """Log fetch depth and any hole it leaves in the product's history.

        A capped fetch is complete only from `complete_from` on. If that is still
        newer than the newest review already stored (from an earlier fetch or
        the historical dataset), reviews in between may be missing: that window
        is a gap. Overlapping fetches leave no gap.
        """
        gap = hit_cap and complete_from is not None and (stored_newest is None or complete_from > stored_newest)
        row = pd.DataFrame([{"asin": asin, "reviews": len(df), "oldest": df["review_date"].min() if len(df) else None,
                             "newest": df["review_date"].max() if len(df) else None, "hit_cap": hit_cap,
                             "complete_from": complete_from,
                             "gap_start": stored_newest if gap else None, "gap_end": complete_from if gap else None,
                             "fetched_at": utcnow()}])
        path = self.dir / "coverage.csv"
        row.to_csv(path, mode="a", header=not path.exists(), index=False)

    # Step 4 ---------------------------------------------------------------
    def score(self, scorer: str | None = None, sample: int | None = None) -> pd.DataFrame:
        reviews = rv.load(self.dir / "reviews.csv")
        if reviews.empty:
            raise RuntimeError(f"No reviews for {self.company.ticker}; run `reviewtone fetch` first.")
        name = scorer or self.scorer_name
        scores = score_reviews(self.study, self.dir, reviews, name, sample=sample)
        self.note("score", scorer=name, scored=len(scores), reviews=len(reviews))
        return scores

    def _scores(self) -> pd.DataFrame:
        from .tone import latest_valid, load_scores
        name = self.scorer_name
        file = "scores_naive_bayes.csv" if name == "naive_bayes" else "scores.csv"
        return latest_valid(load_scores(self.dir / file), name)

    # Step 5 ---------------------------------------------------------------
    def aggregate(self, cal: FiscalCalendar | None = None, write: bool = True) -> pd.DataFrame:
        reviews = rv.load(self.dir / "reviews.csv")
        scope = self.study.aggregation.get("product_scope", "tracked")
        tracked = self.tracked_products()
        if scope == "tracked" and len(tracked):
            ids = set(tracked["asin"]) | set(tracked["parent_asin"].dropna())
            reviews = reviews[reviews["asin"].isin(ids) | reviews["parent_asin"].isin(ids)]
        scored = aggregate.attach_scores(reviews, self._scores())
        cov_path = self.dir / "coverage.csv"
        coverage = pd.read_csv(cov_path) if cov_path.exists() else pd.DataFrame()
        gaps = aggregate.coverage_gaps(coverage)
        dataset_end = self._status().get("backfill_mcauley", {}).get("dataset_end")
        if dataset_end and coverage.empty:
            # Historical data stops here and no live fetch has filled in after it.
            gaps.append((date.fromisoformat(dataset_end), date.max))
        q = aggregate.quarterly_measures(scored, cal or self.cal,
                                         min_reviews=self.study.aggregation["min_reviews_per_quarter"],
                                         as_of=self.as_of, gaps=gaps)
        if write and len(q):
            q.to_csv(self.dir / "quarterly.csv", index=False)
            cdir = self.dir / "charts"
            cdir.mkdir(exist_ok=True)
            charts.tone_and_counts(q, self.company.ticker, cdir / "tone_and_counts.png")
            charts.negative_swing(q, self.company.ticker, self.study.rules["negative_swing_threshold"],
                                  cdir / "negative_swing.png")
            self.note("aggregate", **aggregate.history_status(q, self.study.aggregation["target_complete_quarters"]),
                      coverage_gaps=len(gaps), unscored_reviews=int(scored["tone"].isna().sum()))
        return q

    # Step 6 ---------------------------------------------------------------
    def fundamentals(self) -> pd.DataFrame | None:
        from .fundamentals.outcomes import build_fundamentals
        try:
            f = build_fundamentals(self.company.ticker, self.cal, self.log)
        except (SourceUnavailable, KeyError, ValueError) as e:
            self.note("fundamentals", error=str(e))
            return None
        f.to_csv(self.dir / "fundamentals.csv", index=False)
        self.note("fundamentals", quarters=int(f["revenue"].notna().sum()),
                  first=f["fiscal_quarter"].dropna().iloc[0], last=f["fiscal_quarter"].dropna().iloc[-1],
                  q4_derived=int(f["revenue_q4_derived"].fillna(False).astype(bool).sum()))
        return f

    def announcement_returns(self, fundamentals: pd.DataFrame, factors: pd.DataFrame | None) -> pd.DataFrame | None:
        from .market.data import daily_returns
        from .market.models import announcement_return
        if factors is None:
            return None
        try:
            ret = daily_returns(self.company.ticker, log=self.log)
        except SourceUnavailable as e:
            self.note("prices", error=str(e))
            return None
        m = self.study.market
        rows = []
        for r in fundamentals.dropna(subset=["announcement_date"]).itertuples():
            ev = announcement_return(ret, factors, r.announcement_date, tuple(m["estimation_window"]),
                                     tuple(m["event_window"]))
            if ev:
                rows.append({"fq_index": r.fq_index, "announcement_date": r.announcement_date,
                             "event_day": ev.event_day, "car": ev.car, "bhar": ev.bhar})
        out = pd.DataFrame(rows, columns=["fq_index", "announcement_date", "event_day", "car", "bhar"])
        out.to_csv(self.dir / "announcement_returns.csv", index=False)
        self.note("prices", announcements_with_returns=len(out), last_price=str(ret.index.max().date()))
        return out

    def evaluate(self, quarterly: pd.DataFrame, fundamentals: pd.DataFrame | None,
                 ann: pd.DataFrame | None) -> dict | None:
        if fundamentals is None or quarterly.empty:
            self.note("evaluation", skipped="needs both quarterly tone and SEC revenue")
            return None
        panel = evaluate.build_panel(quarterly, fundamentals, self.study.rules, ann)
        panel.insert(0, "ticker", self.company.ticker)
        results = evaluate.write_evaluation(self.dir, panel, self.study.evaluation["holdout_quarters"])
        held = evaluate.holdout(panel, self.study.evaluation["holdout_quarters"], 0)["signal_quarter"].tolist()
        charts.tone_vs_sales(panel, self.company.ticker, held, self.dir / "charts" / "tone_vs_sales.png")
        self.note("evaluation", **{k: v["summary"] for k, v in results.items()})
        return results

    # Step 7 ---------------------------------------------------------------
    def snapshot(self) -> Path:
        """Copy dated outputs into history/<date>/ so every weekly state is kept."""
        dest = self.dir / "history" / self.as_of.isoformat()
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("quarterly.csv", "prediction.json", "evaluation.json", "status.json"):
            if (self.dir / name).exists():
                shutil.copy2(self.dir / name, dest / name)
        return dest


def backfill(study: Study, tickers: list[str], categories: list[str] | None = None,
             chunk: int = 50_000) -> dict[str, dict]:
    """Historical reviews from Amazon Reviews 2023 (through Sep 2023).

    Each category file is streamed once for every company that sells in it;
    products go to products.csv and reviews to reviews.csv (append-only).
    """
    from .sources import mcauley

    companies = [study.company(t) for t in tickers]
    by_cat: dict[str, list[Company]] = {}
    for c in companies:
        for cat in categories or c.mcauley_categories:
            by_cat.setdefault(cat, []).append(c)
    runs = {c.ticker: Run(study, c) for c in companies}
    tmp = {t: r.dir / "_backfill.csv" for t, r in runs.items()}
    for p in tmp.values():
        p.unlink(missing_ok=True)
    summary = {t: {"products": 0, "reviews_streamed": 0, "categories": [], "errors": []} for t in runs}
    for cat, cos in by_cat.items():
        print(f"{cat}: scanning for {', '.join(c.ticker for c in cos)}", flush=True)
        try:
            products, rows = mcauley.scan_category(cat, cos, runs[cos[0].ticker].log)
            for t, prods in products.groupby("ticker") if len(products) else []:
                old = runs[t].dir / "products.csv"
                merged, _ = catalog.merge_products(prods, pd.read_csv(old, dtype=str) if old.exists() else None)
                merged.to_csv(old, index=False)
                summary[t]["products"] += len(prods)
            buf: list[dict] = []
            for row in rows:
                buf.append(row)
                if len(buf) >= chunk:
                    _flush(buf, tmp, summary)
            _flush(buf, tmp, summary)
            for c in cos:
                summary[c.ticker]["categories"].append(cat)
        except SourceUnavailable as e:
            for c in cos:
                summary[c.ticker]["errors"].append(f"{cat}: {e}")
    for t, run in runs.items():
        added = rv.append_new(run.dir / "reviews.csv", pd.read_csv(tmp[t], dtype=str)) if tmp[t].exists() else []
        tmp[t].unlink(missing_ok=True)
        summary[t]["reviews_added"] = len(added)
        stored = rv.load(run.dir / "reviews.csv")
        hist = stored[stored["source"].astype(str).str.startswith("mcauley")]
        summary[t]["dataset_end"] = hist["review_date"].max() if len(hist) else None
        run.note("backfill_mcauley", **summary[t])
    return summary


def _flush(buf: list[dict], tmp: dict[str, Path], summary: dict) -> None:
    if not buf:
        return
    df = pd.DataFrame(buf)
    for t, part in df.groupby("ticker"):
        part.to_csv(tmp[t], mode="a", header=not tmp[t].exists(), index=False)
        summary[t]["reviews_streamed"] += len(part)
    buf.clear()


def run_company(study: Study, ticker: str, *, source: str | None, factors: pd.DataFrame | None,
                meta: list[str] | None = None, review_files: list[str] | None = None, csv_path: str | None = None,
                max_reviews: int = 100, scorer: str | None = None, new_study: bool = False,
                as_of: date | None = None, top_products: int | None = None) -> dict:
    from .predict import prediction_card
    from .report import write_report

    run = Run(study, study.company(ticker), as_of=as_of or date.today(), scorer=scorer)
    run.verify()
    run.build_catalog(meta)
    if source:
        run.fetch_reviews(source, review_locations=review_files, max_reviews_per_product=max_reviews,
                          csv_path=csv_path, top_products=top_products)
    if rv.load(run.dir / "reviews.csv").empty:
        run.note("aggregate", error="no reviews collected")
        q = pd.DataFrame()
    else:
        try:
            run.score()
        except Exception as e:  # scorer unavailable: keep what was scored before
            run.note("score", error=f"{type(e).__name__}: {e}")
        q = run.aggregate()
    fund = run.fundamentals()
    ann = run.announcement_returns(fund, factors) if fund is not None else None
    results = None
    if len(q) and fund is not None:
        evaluate.check_preregistration(study, new_study)
        results = run.evaluate(q, fund, ann)
    card = prediction_card(ticker, q, fund, study.rules, results, run.as_of) if len(q) else {
        "ticker": ticker, "status": "no review data"}
    (run.dir / "prediction.json").write_text(json.dumps(card, indent=2, default=str))
    write_report(run, q, card, results)
    run.snapshot()
    return card


def run_universe(study: Study, factors: pd.DataFrame | None, as_of: date | None = None) -> dict:
    """Calendar-quarter long/short portfolio over every company with scored reviews."""
    from . import portfolio
    from .market.data import daily_returns

    out_dir = study.research_dir / "_universe"
    out_dir.mkdir(parents=True, exist_ok=True)
    measures, returns, missing = {}, {}, []
    for c in study.universe:
        run = Run(study, c, as_of=as_of or date.today())
        if rv.load(run.dir / "reviews.csv").empty:
            missing.append(f"{c.ticker}: no reviews")
            continue
        measures[c.ticker] = run.aggregate(cal=CALENDAR_QUARTERS, write=False)
        try:
            returns[c.ticker] = daily_returns(c.ticker, log=run.log)
        except SourceUnavailable as e:
            missing.append(f"{c.ticker}: prices unavailable ({e})")
    summary = {"as_of": (as_of or date.today()).isoformat(), "firms_with_reviews": sorted(measures),
               "missing": missing}
    if factors is None or not returns:
        summary["error"] = "needs Carhart factors and prices; see missing"
    else:
        evaluate.check_preregistration(study)
        pos, q, alphas, latest = portfolio.run(measures, returns, factors, study.portfolio)
        pos.to_csv(out_dir / "positions.csv", index=False)
        q.to_csv(out_dir / "strategy_returns.csv", index=False)
        if len(q):
            charts.portfolio_cumulative(q, out_dir / "long_short.png")
        summary["alphas"] = alphas
        if len(latest):
            k = int(latest["n_groups"].iloc[0])
            summary["latest_formation"] = {
                "formation_quarter": latest["fiscal_quarter"].iloc[0],
                "long": latest.loc[latest["group"] == k, "ticker"].tolist(),
                "short": latest.loc[latest["group"] == 1, "ticker"].tolist(),
                "groups": k,
            }
    panels = [pd.read_csv(study.ticker_dir(t) / "panel.csv") for t in measures
              if (study.ticker_dir(t) / "panel.csv").exists()]
    if panels:  # pooled, descriptive replication of Jin (2022) Eq. 1
        pooled = pd.concat(panels, ignore_index=True)
        for y in ("AbnormalSales", "SameQtrSalesGrowth"):
            table = evaluate.paper_regression(pooled, y)
            if table is not None:
                table.to_csv(out_dir / f"paper_eq1_{y}.csv")
                summary[f"paper_eq1_{y}_AbnormalTone"] = table.loc["AbnormalTone", ["coef", "t", "n"]].to_dict()
    (out_dir / "portfolio.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
