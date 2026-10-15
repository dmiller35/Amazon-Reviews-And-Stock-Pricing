"""Synthetic fixtures. They exercise the code paths only and are never results."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from reviewtone.config import load_study
from reviewtone.fiscal import FiscalCalendar

ROOT = Path(__file__).resolve().parents[1]

POS_TEXT = ["Great mouse, works perfectly", "Love it, reliable and comfortable", "Excellent, solid build"]
NEG_TEXT = ["Broke after a month, junk", "Defective, stopped working, returned it", "Terrible, worse than old model"]


@pytest.fixture
def study(tmp_path):
    cfg = yaml.safe_load((ROOT / "config" / "study.yaml").read_text())
    cfg["study"]["research_dir"] = str(tmp_path / "research")
    cfg["aggregation"]["min_reviews_per_quarter"] = 20
    cfg["aggregation"]["stars"] = [1, 2, 3, 4, 5]  # synthetic worlds use every star; 1-4 has its own tests
    shutil.copytree(ROOT / "config" / "companies", tmp_path / "config" / "companies")
    p = tmp_path / "config" / "study.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_study(p)


def synthetic_world(n_quarters: int = 28, seed: int = 1, cal: FiscalCalendar = FiscalCalendar(3),
                    first_fy: int = 2019):
    """Reviews whose negative share leads revenue growth, plus matching SEC JSON."""
    rng = np.random.default_rng(seed)
    first = cal.quarter(f"{first_fy - 1}-04-15")
    quarters = [first.shift(i) for i in range(n_quarters)]
    neg_share = 0.2 + 0.03 * rng.standard_normal(n_quarters)
    shocks = rng.choice(np.arange(8, n_quarters), size=5, replace=False)
    neg_share[shocks] += 0.25
    rows = []
    for fq, share in zip(quarters, neg_share):
        n = 60
        days = (fq.end - fq.start).days
        for i in range(n):
            neg = rng.random() < share
            title = rng.choice(NEG_TEXT if neg else POS_TEXT)
            d = pd.Timestamp(fq.start) + pd.Timedelta(days=int(rng.integers(0, days)))
            rows.append({"review_id": f"r{fq.index}_{i}", "asin": "B000TEST01", "parent_asin": "B000TEST01",
                         "brand": "Logitech", "review_date": d.strftime("%Y-%m-%d"),
                         "rating": 2 if neg else 5, "title": title, "text": f"{title}. review {fq.index}-{i}",
                         "source": "synthetic"})
    reviews = pd.DataFrame(rows)

    # Revenue: seasonal, trend growth, minus a hit in quarters with a negative swing.
    season = {1: 1.0, 2: 1.05, 3: 1.35, 4: 0.95}
    revenue, growth = {}, 0.05
    for k, fq in enumerate(quarters):
        g = growth - (0.08 if k in shocks else 0) + 0.005 * rng.standard_normal()
        base = 1e9 * season[fq.q]
        prev = revenue.get(fq.shift(-4).index)
        revenue[fq.index] = prev * (1 + g) if prev else base
    usd = []
    for fq in quarters:
        filed = (pd.Timestamp(fq.end) + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        if fq.q != 4:
            usd.append({"start": fq.start.isoformat(), "end": fq.end.isoformat(), "val": revenue[fq.index],
                        "form": "10-Q", "filed": filed, "fy": fq.fy, "fp": f"Q{fq.q}", "accn": f"a{fq.index}"})
        else:
            parts = [revenue.get(fq.shift(-k).index) for k in range(4)]
            if all(p is not None for p in parts):
                usd.append({"start": fq.shift(-3).start.isoformat(), "end": fq.end.isoformat(),
                            "val": sum(parts), "form": "10-K", "filed": filed, "fy": fq.fy, "fp": "FY",
                            "accn": f"k{fq.index}"})
        if fq.q != 4:  # a later filing restating the value; point-in-time logic must ignore it
            restated = (pd.Timestamp(filed) + pd.Timedelta(days=365)).strftime("%Y-%m-%d")
            usd.append({"start": fq.start.isoformat(), "end": fq.end.isoformat(), "val": revenue[fq.index] * 1.5,
                        "form": "10-Q", "filed": restated, "fy": fq.fy + 1, "fp": f"Q{fq.q}", "accn": f"r{fq.index}"})
    facts = {"facts": {"us-gaap": {"RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": usd}}}}}
    eight_k = [((pd.Timestamp(fq.end) + pd.Timedelta(days=35)).strftime("%Y-%m-%d"), f"8k{fq.index}")
               for fq in quarters]
    submissions = {"name": "LOGITECH INTERNATIONAL S.A.", "fiscalYearEnd": "0331",
                   "filings": {"recent": {"form": ["8-K"] * len(eight_k), "filingDate": [d for d, _ in eight_k],
                                          "items": ["2.02,9.01"] * len(eight_k),
                                          "accessionNumber": [a for _, a in eight_k]}}}
    return reviews, facts, submissions, quarters, sorted(shocks)


@pytest.fixture
def world():
    return synthetic_world()


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("reviewtone.http.CACHE_DIR", tmp_path / "cache")
    yield
    shutil.rmtree(tmp_path / "cache", ignore_errors=True)


@pytest.fixture(autouse=True)
def fresh_preflight(monkeypatch):
    from reviewtone.sources import apify
    monkeypatch.setattr(apify, "_PREFLIGHT_PASSED", {})
