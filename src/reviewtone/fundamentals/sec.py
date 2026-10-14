"""SEC EDGAR: fiscal calendar verification, quarterly revenue, announcement dates.

Values are taken "as first reported": when a period appears in several filings
(later 10-Qs repeat prior-year comparatives, restatements revise numbers) the
earliest filing wins, so a backtest only uses what was public at the time.
Fourth quarters are rarely tagged on their own, so Q4 = fiscal year - (Q1+Q2+Q3),
dated by the 10-K filing.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ..fiscal import FiscalCalendar
from ..http import AccessLog, get

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
]
FLOW_CONCEPTS = {
    "revenue": REVENUE_CONCEPTS,
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "rnd": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}
INSTANT_CONCEPTS = {"assets": ["Assets"]}
QUARTER_DAYS = (75, 105)
YEAR_DAYS = (350, 380)


def resolve_cik(ticker: str, log: AccessLog | None = None) -> int:
    data = get(TICKERS_URL, log=log, cache_hours=24 * 7)
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"])
    raise KeyError(f"{ticker} not found in SEC company_tickers.json")


def fetch_submissions(cik: int, log: AccessLog | None = None) -> dict:
    return get(SUBMISSIONS_URL.format(cik=cik), log=log)


def fetch_companyfacts(cik: int, log: AccessLog | None = None) -> dict:
    return get(FACTS_URL.format(cik=cik), log=log)


def _facts(companyfacts: dict, concept: str) -> list[dict]:
    node = companyfacts.get("facts", {}).get("us-gaap", {}).get(concept)
    if not node:
        return []
    units = node["units"]
    unit = "USD" if "USD" in units else ("USD/shares" if "USD/shares" in units else next(iter(units)))
    return units[unit]


def _first_reported(rows: list[dict], keys: tuple[str, ...]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["form"].str.match(r"^(10-Q|10-K|20-F|40-F)")]
    return df.sort_values("filed").drop_duplicates(list(keys), keep="first")


def flow_quarters(companyfacts: dict, concepts: list[str], cal: FiscalCalendar) -> pd.DataFrame:
    """Quarterly values of a duration concept, with derived Q4s."""
    frames = []
    for priority, concept in enumerate(concepts):
        df = _first_reported(_facts(companyfacts, concept), ("start", "end"))
        if df.empty:
            continue
        df["days"] = (pd.to_datetime(df["end"]) - pd.to_datetime(df["start"])).dt.days
        df["concept"], df["priority"] = concept, priority
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["fq_index", "fiscal_quarter", "period_end", "value", "filed", "derived", "concept"])
    allf = pd.concat(frames, ignore_index=True)

    def to_fq(end: str):
        return cal.quarter_for_period_end(end)

    q = allf[allf["days"].between(*QUARTER_DAYS)].copy()
    q["fq"] = q["end"].map(to_fq)
    q = q.dropna(subset=["fq"]).sort_values(["priority", "filed"])
    q["fq_index"] = q["fq"].map(lambda f: f.index)
    q = q.drop_duplicates("fq_index", keep="first")

    rows = {r.fq_index: {"fq_index": r.fq_index, "fiscal_quarter": r.fq.label, "period_end": r.end,
                         "value": float(r.val), "filed": r.filed, "derived": False, "concept": r.concept}
            for r in q.itertuples()}

    y = allf[allf["days"].between(*YEAR_DAYS)].copy()
    y["fq"] = y["end"].map(to_fq)
    y = y.dropna(subset=["fq"]).sort_values(["priority", "filed"])
    y = y[y["fq"].map(lambda f: f.q == 4)]
    y["fq_index"] = y["fq"].map(lambda f: f.index)
    for r in y.drop_duplicates("fq_index", keep="first").itertuples():
        if r.fq_index in rows:
            continue
        parts = [rows.get(r.fq_index - k) for k in (1, 2, 3)]
        if any(p is None for p in parts):
            continue
        rows[r.fq_index] = {"fq_index": r.fq_index, "fiscal_quarter": r.fq.label, "period_end": r.end,
                            "value": float(r.val) - sum(p["value"] for p in parts), "filed": r.filed,
                            "derived": True, "concept": r.concept}
    return pd.DataFrame(sorted(rows.values(), key=lambda d: d["fq_index"]))


def instant_quarters(companyfacts: dict, concepts: list[str], cal: FiscalCalendar) -> pd.DataFrame:
    for concept in concepts:
        df = _first_reported(_facts(companyfacts, concept), ("end",))
        if df.empty:
            continue
        df["fq"] = df["end"].map(lambda e: cal.quarter_for_period_end(e))
        df = df.dropna(subset=["fq"])
        df["fq_index"] = df["fq"].map(lambda f: f.index)
        df = df.drop_duplicates("fq_index", keep="first")
        return pd.DataFrame({"fq_index": df["fq_index"], "value": df["val"].astype(float)})
    return pd.DataFrame(columns=["fq_index", "value"])


def earnings_announcements(submissions: dict, cal: FiscalCalendar, max_lag_days: int = 100) -> pd.DataFrame:
    """8-K item 2.02 (Results of Operations) filings mapped to the quarter they report."""
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame(columns=["fq_index", "announcement_date", "accession"])
    f = pd.DataFrame({k: recent.get(k, []) for k in ("form", "filingDate", "items", "accessionNumber")})
    f = f[f["form"].isin(["8-K", "6-K"]) & f["items"].fillna("").str.contains("2.02", regex=False)]
    rows = []
    for r in f.itertuples():
        filed = date.fromisoformat(r.filingDate)
        fq = cal.quarter(filed - timedelta(days=1)).shift(-1)  # the quarter that just ended
        if 0 < (filed - fq.end).days <= max_lag_days:
            rows.append({"fq_index": fq.index, "announcement_date": r.filingDate, "accession": r.accessionNumber})
    out = pd.DataFrame(rows, columns=["fq_index", "announcement_date", "accession"])
    return out.sort_values("announcement_date").drop_duplicates("fq_index", keep="first")


def verify_company(ticker: str, expected_fy_end_month: int, expected_name: str, log: AccessLog | None = None) -> dict:
    """Step 1: confirm CIK, registrant name and fiscal year end against EDGAR."""
    from ..fiscal import parse_sec_fiscal_year_end

    cik = resolve_cik(ticker, log)
    sub = fetch_submissions(cik, log)
    sec_month = parse_sec_fiscal_year_end(sub.get("fiscalYearEnd"))
    return {
        "ticker": ticker,
        "cik": cik,
        "sec_name": sub.get("name"),
        "expected_name": expected_name,
        "sec_fiscal_year_end": sub.get("fiscalYearEnd"),
        "configured_fiscal_year_end_month": expected_fy_end_month,
        "fiscal_year_end_matches": sec_month == expected_fy_end_month,
        "sic_description": sub.get("sicDescription"),
        "exchanges": sub.get("exchanges"),
    }
