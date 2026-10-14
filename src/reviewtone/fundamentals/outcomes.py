"""Quarterly fundamentals and the sales outcomes the tone signal is tested on.

AbnormalSales_q      = (sales_q - mean sales q-4..q-1) / mean sales q-4..q-1   (paper)
SameQtrSalesGrowth_q = sales_q / sales_q-4 - 1                                  (paper)
GrowthSurprise_q     = SameQtrSalesGrowth_q - mean growth q-4..q-1
    The single-company decision rule uses GrowthSurprise: it removes
    seasonality (as year-over-year growth does) and the company's recent growth
    trend (as AbnormalTone removes the recent tone baseline), so "below zero"
    means sales came in weaker than the company's own recent trajectory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fiscal import FiscalCalendar
from ..http import AccessLog
from . import sec


def build_fundamentals(ticker: str, cal: FiscalCalendar, log: AccessLog | None = None,
                       companyfacts: dict | None = None, submissions: dict | None = None) -> pd.DataFrame:
    if companyfacts is None or submissions is None:
        cik = sec.resolve_cik(ticker, log)
        companyfacts = companyfacts or sec.fetch_companyfacts(cik, log)
        submissions = submissions or sec.fetch_submissions(cik, log)

    rev = sec.flow_quarters(companyfacts, sec.FLOW_CONCEPTS["revenue"], cal)
    if rev.empty:
        raise ValueError(f"No quarterly revenue found in SEC XBRL for {ticker}")
    out = rev.rename(columns={"value": "revenue", "filed": "revenue_filed", "derived": "revenue_q4_derived",
                              "concept": "revenue_concept"})
    for name in ("net_income", "rnd", "eps_diluted"):
        f = sec.flow_quarters(companyfacts, sec.FLOW_CONCEPTS[name], cal)
        if len(f):
            out = out.merge(f[["fq_index", "value"]].rename(columns={"value": name}), on="fq_index", how="left")
        else:
            out[name] = np.nan
    a = sec.instant_quarters(companyfacts, sec.INSTANT_CONCEPTS["assets"], cal)
    out = out.merge(a.rename(columns={"value": "assets"}), on="fq_index", how="left")
    ann = sec.earnings_announcements(submissions, cal)
    out = out.merge(ann[["fq_index", "announcement_date"]], on="fq_index", how="left")
    return add_outcomes(out)


def add_outcomes(f: pd.DataFrame) -> pd.DataFrame:
    """Derived sales outcomes and paper controls. Requires consecutive fq_index."""
    f = f.sort_values("fq_index").set_index("fq_index")
    f = f.reindex(range(f.index.min(), f.index.max() + 1))  # expose gaps as NaN, never bridge them
    sales = f["revenue"]
    prior4 = sales.shift(1).rolling(4, min_periods=4).mean()
    f["AbnormalSales"] = (sales - prior4) / prior4
    f["SameQtrSalesGrowth"] = sales / sales.shift(4) - 1
    f["GrowthSurprise"] = f["SameQtrSalesGrowth"] - f["SameQtrSalesGrowth"].shift(1).rolling(4, min_periods=4).mean()
    f["LnRevenue"] = np.log1p(sales.clip(lower=0) / 1e6)
    f["ROA"] = f["net_income"] / f["assets"]
    f["Loss"] = (f["net_income"] < 0).astype(float).where(f["net_income"].notna())
    f["RD"] = (f["rnd"] / sales).fillna(0.0)
    return f.reset_index()
