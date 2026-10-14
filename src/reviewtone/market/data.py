"""Daily prices (Yahoo Finance, Stooq fallback) and Ken French daily factors."""

from __future__ import annotations

import io
import time
import zipfile

import pandas as pd

from ..http import AccessLog, SourceUnavailable, get

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
STOOQ = "https://stooq.com/q/d/l/?s={ticker}.us&i=d"
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FF3_DAILY = "F-F_Research_Data_Factors_daily"
MOM_DAILY = "F-F_Momentum_Factor_daily"


def daily_returns(ticker: str, start: str = "2010-01-01", log: AccessLog | None = None) -> pd.Series:
    """Total daily returns from adjusted closes, indexed by date."""
    errors = []
    try:
        data = get(YAHOO.format(ticker=ticker), log=log, params={
            "period1": int(pd.Timestamp(start).timestamp()), "period2": int(time.time()),
            "interval": "1d", "events": "div,split"})
        res = data["chart"]["result"][0]
        idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
        px = pd.Series(res["indicators"]["adjclose"][0]["adjclose"], index=idx, dtype=float)
        return px.dropna().pct_change().dropna().rename(ticker)
    except (SourceUnavailable, KeyError, IndexError, TypeError) as e:
        errors.append(f"yahoo: {e}")
    try:
        raw = get(STOOQ.format(ticker=ticker.lower()), log=log, as_json=False)
        df = pd.read_csv(io.BytesIO(raw), parse_dates=["Date"]).set_index("Date")
        return df.loc[start:, "Close"].pct_change().dropna().rename(ticker)
    except (SourceUnavailable, KeyError, ValueError) as e:
        errors.append(f"stooq: {e}")
    raise SourceUnavailable(f"No price source reachable for {ticker}: {'; '.join(errors)}")


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French daily CSV (header lines, YYYYMMDD rows, trailing notes)."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip()[:8].isdigit()) - 1
    rows = []
    for ln in lines[start + 1:]:
        cells = [c.strip() for c in ln.split(",")]
        if not cells[0].isdigit() or len(cells[0]) != 8:
            break
        rows.append(cells)
    header = [h.strip().lower().replace("-", "_") for h in lines[start].split(",")]
    header[0] = "date"
    df = pd.DataFrame(rows, columns=header[: len(rows[0])])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float) / 100.0
    return df


def _french(name: str, log: AccessLog | None) -> pd.DataFrame:
    raw = get(FRENCH.format(name=name), log=log, as_json=False, cache_hours=24 * 7)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    return parse_french_csv(text)


def carhart_daily(log: AccessLog | None = None) -> pd.DataFrame:
    """Columns: mkt_rf, smb, hml, rf, umd (decimal daily returns)."""
    ff = _french(FF3_DAILY, log)
    mom = _french(MOM_DAILY, log)
    mom.columns = ["umd"]
    return ff.join(mom, how="inner")
