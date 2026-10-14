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
