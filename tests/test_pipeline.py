"""Offline end-to-end runs on synthetic data (code-path verification only)."""

import json

import numpy as np
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
