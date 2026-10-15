"""Offline end-to-end runs on synthetic data (code-path verification only)."""

import json
from datetime import date

import numpy as np
import pytest
import pandas as pd

from reviewtone import portfolio
from reviewtone.fiscal import CALENDAR_QUARTERS
from reviewtone.pipeline import run_company


def test_run_company_end_to_end(study, world, tmp_path, monkeypatch):
    reviews, facts, subs, quarters, shocks = world
    monkeypatch.setattr("reviewtone.fundamentals.sec.resolve_cik", lambda t, log=None: 1032975)
    monkeypatch.setattr("reviewtone.fundamentals.sec.fetch_companyfacts", lambda cik, log=None: facts)
    monkeypatch.setattr("reviewtone.fundamentals.sec.fetch_submissions", lambda cik, log=None: subs)
    csv = tmp_path / "synthetic_reviews.csv"
    reviews.to_csv(csv, index=False)
    as_of = quarters[-1].end

    card = run_company(study, "LOGI", source="csv", csv_path=str(csv), factors=None, scorer="lexicon", as_of=as_of)
    d = study.ticker_dir("LOGI")
    for name in ("reviews.csv", "scores.csv", "quarterly.csv", "fundamentals.csv", "panel.csv", "evaluation_h0.csv",
                 "prediction.json", "README.md", "rubric.md", "charts/negative_swing.png", "charts/tone_vs_sales.png"):
        assert (d / name).exists(), name
    assert (study.research_dir / "preregistration.json").exists()
    assert card["signal_quarter"] == quarters[-1].label

    status = json.loads((d / "status.json").read_text())
    assert status["verification"]["fiscal_year_end_matches"] is True

    ev = json.loads((d / "evaluation.json").read_text())["h0"]["summary"]
    assert ev["holdout_quarters"] == 8
    panel = pd.read_csv(d / "panel.csv")
    p0 = panel[(panel["horizon"] == 0) & panel["usable"]]
    shock_labels = {quarters[i].label for i in shocks}
    called = set(p0.loc[p0["call"] == "slowdown", "signal_quarter"])
    assert called & shock_labels, "negative swings planted in the synthetic data should be called"
    assert (p0.loc[p0["call"] == "slowdown", "sales_hit"] == True).mean() > 0.5  # noqa: E712

    # Re-running is idempotent: no duplicate reviews, and history snapshots accumulate.
    run_company(study, "LOGI", source="csv", csv_path=str(csv), factors=None, scorer="lexicon", as_of=as_of)
    assert len(pd.read_csv(d / "reviews.csv")) == len(reviews)
    assert (d / "history" / as_of.isoformat() / "prediction.json").exists()
    assert "Held-out evaluation" in (d / "README.md").read_text()


def test_universe_portfolio_long_short():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2021-01-01", "2024-12-31")
    fac = pd.DataFrame(rng.normal(0, 0.005, (len(idx), 4)), index=idx, columns=["mkt_rf", "smb", "hml", "umd"])
    fac["rf"] = 0.0001
    quarters = [CALENDAR_QUARTERS.quarter("2021-01-15").shift(i) for i in range(15)]
    measures, returns = {}, {}
    tickers = ["A", "B", "C", "D", "E", "F"]
    tone = {t: rng.normal(0, 0.1, len(quarters)) for t in tickers}
    for t in tickers:
        measures[t] = pd.DataFrame({"fq_index": [q.index for q in quarters],
                                    "fiscal_quarter": [q.label for q in quarters],
                                    "AbnormalTone": tone[t], "n_reviews": 100, "usable": True})
        daily = pd.Series(rng.normal(0, 0.004, len(idx)), index=idx)
        for k, q in enumerate(quarters[:-1]):  # next-quarter drift proportional to this quarter's tone
            nxt = quarters[k + 1]
            mask = (idx >= pd.Timestamp(nxt.start)) & (idx <= pd.Timestamp(nxt.end))
            daily[mask] += tone[t][k] * 0.004
        returns[t] = daily
    pos, q, alphas, latest = portfolio.run(measures, returns, fac, {"groups": 4, "min_firms_for_quartiles": 8,
                                                                    "weighting": ["review_count", "equal"]})
    assert int(latest["n_groups"].iloc[0]) == 2  # 6 firms -> halves
    assert q["equal_long_short"].mean() > 0
    assert alphas["equal"]["long_short"]["n_quarters"] == len(q)


def test_backfill_streams_category_once_for_several_companies(study, tmp_path, monkeypatch):
    import gzip

    from reviewtone.pipeline import backfill

    meta = [{"parent_asin": "P1", "store": "YETI", "title": "Tumbler"},
            {"parent_asin": "P2", "store": "hydro flask", "title": "Bottle"},
            {"parent_asin": "P3", "store": "Custom YETI", "title": "Not official"}]
    revs = [{"parent_asin": "P1", "asin": "C1", "timestamp": 1_600_000_000_000, "rating": 5.0, "title": "Great",
             "text": "Keeps ice for days", "verified_purchase": True},
            {"parent_asin": "P2", "asin": "P2", "timestamp": 1_650_000_000_000, "rating": 2.0, "title": "Leaks",
             "text": "Lid leaks after a month", "verified_purchase": True},
            {"parent_asin": "P3", "asin": "P3", "timestamp": 1_650_000_000_000, "rating": 4.0, "title": "x",
             "text": "y", "verified_purchase": False}]
    for name, rows in (("meta.jsonl.gz", meta), ("reviews.jsonl.gz", revs)):
        with gzip.open(tmp_path / name, "wt") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr("reviewtone.sources.mcauley.category_urls",
                        lambda c: (str(tmp_path / "meta.jsonl.gz"), str(tmp_path / "reviews.jsonl.gz")))

    out = backfill(study, ["YETI", "HELE"], categories=["Sports_and_Outdoors"], chunk=1)
    assert out["YETI"]["reviews_added"] == 1 and out["HELE"]["reviews_added"] == 1
    yeti = pd.read_csv(study.ticker_dir("YETI") / "reviews.csv")
    assert yeti.loc[0, "review_date"] == "2020-09-13" and yeti.loc[0, "parent_asin"] == "P1"
    assert pd.read_csv(study.ticker_dir("HELE") / "products.csv")["brand"].tolist() == ["Hydro Flask"]
    again = backfill(study, ["YETI", "HELE"], categories=["Sports_and_Outdoors"])  # idempotent
    assert again["YETI"]["reviews_added"] == 0 and again["HELE"]["reviews_added"] == 0
    assert len(pd.read_csv(study.ticker_dir("YETI") / "reviews.csv")) == 1


def test_backfill_resumes_after_interruption(study, tmp_path, monkeypatch):
    import gzip

    from reviewtone.pipeline import backfill

    def write(name, rows):
        with gzip.open(tmp_path / name, "wt") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")

    write("meta_A.jsonl.gz", [{"parent_asin": "P1", "store": "YETI", "title": "Tumbler"}])
    write("meta_B.jsonl.gz", [{"parent_asin": "P2", "store": "YETI", "title": "Cooler"}])
    write("A.jsonl.gz", [{"parent_asin": "P1", "asin": "P1", "timestamp": 1_600_000_000_000, "rating": 5.0,
                          "title": "Great", "text": "Keeps ice", "verified_purchase": True}])
    write("B.jsonl.gz", [{"parent_asin": "P2", "asin": "P2", "timestamp": 1_650_000_000_000, "rating": 2.0,
                          "title": "Leaks", "text": "Drain plug leaks", "verified_purchase": True}])
    scanned = []

    def urls(cat):
        scanned.append(cat)
        if cat == "B" and len(scanned) == 2:
            raise KeyboardInterrupt  # machine switched off mid-run
        return str(tmp_path / f"meta_{cat}.jsonl.gz"), str(tmp_path / f"{cat}.jsonl.gz")

    monkeypatch.setattr("reviewtone.sources.mcauley.category_urls", urls)
    with pytest.raises(KeyboardInterrupt):
        backfill(study, ["YETI"], categories=["A", "B"])
    assert len(pd.read_csv(study.ticker_dir("YETI") / "reviews.csv")) == 1  # A was saved

    out = backfill(study, ["YETI"], categories=["A", "B"])
    assert scanned == ["A", "B", "B"]  # A is not streamed again
    assert out["YETI"]["skipped"] == ["A"] and out["YETI"]["reviews_added"] == 1
    assert len(pd.read_csv(study.ticker_dir("YETI") / "reviews.csv")) == 2

    backfill(study, ["YETI"], categories=["A", "B"], restart=True)
    assert scanned[-2:] == ["A", "B"]
    assert len(pd.read_csv(study.ticker_dir("YETI") / "reviews.csv")) == 2


def test_tracked_products_scope_and_dataset_end(study):
    from reviewtone.pipeline import Run

    run = Run(study, study.company("YETI"), as_of=date(2024, 6, 30), scorer="lexicon")
    products = pd.DataFrame({"ticker": "YETI", "brand": "YETI", "asin": ["BIG", "SMALL", "SEED"],
                             "parent_asin": ["BIG", "SMALL", "SEED"], "rating_number": ["900", "5", ""],
                             "source": ["mcauley2023:x", "mcauley2023:x", "manual:seed_asins.csv"]})
    products.to_csv(run.dir / "products.csv", index=False)
    assert run.tracked_products(top=2)["asin"].tolist() == ["SEED", "BIG"]

    rows = []
    for k, d in enumerate(pd.date_range("2021-01-15", "2023-09-10", freq="7D")):
        for asin in ("BIG", "SMALL"):
            rows.append({"review_id": f"{asin}{k}", "asin": asin, "parent_asin": asin,
                         "review_date": d.strftime("%Y-%m-%d"), "rating": 5, "title": "Great", "text": "works",
                         "source": "mcauley2023:x"})
    from reviewtone import reviews as rv
    rv.append_new(run.dir / "reviews.csv", pd.DataFrame(rows))
    study.raw["apify"]["top_products"] = 2
    run.score()
    run.note("backfill_mcauley", dataset_end="2023-09-10")
    q = run.aggregate(write=False).set_index("fiscal_quarter")
    assert q.loc["FY22Q1", "n_reviews"] == 12           # one per week from BIG only; SMALL is not tracked
    assert q.loc["FY23Q3", "partial_coverage"]          # dataset ends mid-quarter
    assert not q.loc["FY23Q2", "partial_coverage"]
    assert not q["usable"].loc["FY23Q3":].any()


def test_check_live_marks_retired_products_before_the_study_locks(study, monkeypatch):
    from reviewtone.pipeline import Run
    from reviewtone.sources import apify

    run = Run(study, study.company("YETI"), as_of=date(2026, 10, 14))
    pd.DataFrame({"ticker": "YETI", "brand": "YETI", "asin": ["OLD", "NEW", "MID", "LOW"],
                  "parent_asin": ["OLD", "NEW", "MID", "LOW"], "rating_number": ["900", "800", "700", "10"],
                  "source": "mcauley2023:x", "live": "", "live_checked_at": ""}  # empty, as backfill writes them
                 ).to_csv(run.dir / "products.csv", index=False)
    newest = {"B07SGPJBZ6": "2026-10-10", "OLD": "2021-03-01", "NEW": "2026-10-01", "MID": "2026-09-20"}
    asked = []

    def fake(self, asin):
        asked.append(asin)
        return newest.get(asin)

    monkeypatch.setenv("APIFY_TOKEN", "t")
    monkeypatch.setattr(apify.ApifyReviews, "newest_review", fake)
    info = run.check_live(top=2)
    assert info["live"] == ["NEW", "MID"] and info["retired"] == ["OLD (newest 2021-03-01)"]
    assert "LOW" not in asked  # stops once enough live products are found
    assert run.tracked_products(top=2)["asin"].tolist() == ["NEW", "MID"]


def test_real_workflow_check_live_label_sample_then_naive_bayes(study, world, monkeypatch):
    """backfill output -> check-live -> Claude-labelled sample -> run with the default scorer."""
    from reviewtone import catalog, reviews as rv
    from reviewtone.pipeline import Run
    from reviewtone.sources import apify
    from reviewtone.tone.lexicon import LexiconToneScorer

    reviews, facts, subs, quarters, _ = world
    monkeypatch.setattr("reviewtone.fundamentals.sec.resolve_cik", lambda t, log=None: 1032975)
    monkeypatch.setattr("reviewtone.fundamentals.sec.fetch_companyfacts", lambda cik, log=None: facts)
    monkeypatch.setattr("reviewtone.fundamentals.sec.fetch_submissions", lambda cik, log=None: subs)
    as_of = quarters[-1].end
    run = Run(study, study.company("LOGI"), as_of=as_of)
    products, _ = catalog.merge_products(pd.DataFrame({
        "ticker": "LOGI", "brand": "Logitech", "asin": ["B000TEST01", "RETIRED"], "parent_asin": ["B000TEST01", "RETIRED"],
        "rating_number": [500, 900], "official_store": True, "source": "mcauley2023:Electronics"}))
    products.to_csv(run.dir / "products.csv", index=False)  # as backfill writes it: empty live columns
    rv.append_new(run.dir / "reviews.csv", reviews.assign(source="mcauley2023:Electronics"))

    monkeypatch.setenv("APIFY_TOKEN", "t")
    newest = {"B07SGPJBZ6": as_of.isoformat(), "B000TEST01": as_of.isoformat(), "RETIRED": "2019-01-01"}
    monkeypatch.setattr(apify.ApifyReviews, "newest_review", lambda self, asin: newest[asin])
    assert run.check_live()["live"] == ["B000TEST01"]

    class FakeClaude(LexiconToneScorer):
        name = "claude"

        def __init__(self, **kw):
            self.model = "fake"

    monkeypatch.setattr("reviewtone.tone.claude.ClaudeToneScorer", FakeClaude)
    assert len(run.score(scorer="claude", sample=400)) == 400

    card = run_company(study, "LOGI", source=None, factors=None, as_of=as_of)  # default scorer: naive_bayes
    d = study.ticker_dir("LOGI")
    assert len(pd.read_csv(d / "scores_naive_bayes.csv")) == len(reviews)
    assert (d / "quarterly.csv").exists() and (d / "evaluation.json").exists()
    assert card["signal_quarter"] == quarters[-1].label


def test_check_live_prefilters_caps_resumes_and_saves_on_failure(study, monkeypatch):
    from reviewtone import reviews as rv
    from reviewtone.http import SourceUnavailable
    from reviewtone.pipeline import Run
    from reviewtone.sources import apify

    run = Run(study, study.company("YETI"), as_of=date(2026, 10, 14))
    names = [f"P{i:02d}" for i in range(12)]
    pd.DataFrame({"ticker": "YETI", "brand": "YETI", "asin": names, "parent_asin": names,
                  "rating_number": [str(1000 - i) for i in range(12)], "source": "mcauley2023:x",
                  "live": "", "live_checked_at": ""}).to_csv(run.dir / "products.csv", index=False)
    # History ends 2023-09-30. P00 and P01 stopped getting reviews in 2020: free skip.
    last = {a: ("2020-05-01" if a in ("P00", "P01") else "2023-09-20") for a in names}
    rv.append_new(run.dir / "reviews.csv", pd.DataFrame(
        {"asin": names, "parent_asin": names, "ticker": "YETI", "brand": "YETI", "rating": 5.0,
         "review_date": [last[a] for a in names], "title": "t", "text": names, "source": "mcauley2023:x"}))
    run.note("backfill_mcauley", dataset_end="2023-09-30")
    live_now = {"B07SGPJBZ6", "P03", "P05", "P06"}
    asked = []

    def fake(self, asin):
        asked.append(asin)
        if asin == "P07":
            raise SourceUnavailable("Apify unreachable: HTTP 402 persisted")  # credit used up
        return "2026-10-10" if asin in live_now else "2022-01-01"

    monkeypatch.setenv("APIFY_TOKEN", "t")
    monkeypatch.setattr(apify.ApifyReviews, "newest_review", fake)
    with pytest.raises(SourceUnavailable):
        run.check_live(top=4, progress=False)
    saved = pd.read_csv(run.dir / "products.csv", dtype=str).set_index("asin")["live"]
    assert "P00" not in asked and "P01" not in asked and saved["P00"] == "False"  # skipped for free
    assert saved["P03"] == "True" and saved["P02"] == "False"  # checked before the failure: kept

    asked.clear()
    live_now.add("P07")
    monkeypatch.setattr(apify.ApifyReviews, "newest_review",
                        lambda self, asin: asked.append(asin) or ("2026-10-10" if asin in live_now else "2022-01-01"))
    info = run.check_live(top=4, progress=False)
    assert info["live"] == ["P03", "P05", "P06", "P07"]
    assert not {"P02", "P03", "P04", "P05", "P06"} & set(asked)  # rerun reuses saved checks
    assert run.tracked_products(top=10)["asin"].tolist() == ["P03", "P05", "P06", "P07"]  # unchecked never fill

    capped = run.check_live(top=10, progress=False)  # wants 10, but at most max_live_checks candidates
    study.raw["apify"]["max_live_checks"] = 3
    assert capped["checked"] <= 40
    assert run.check_live(top=10, progress=False)["checked"] == 3
