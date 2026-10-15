"""Paper trading: every live call is recorded before the earnings announcement and
scored after it, with fixed rules, so the track record can't be tidied up later.

Rules (set before any result, like the study's own rules):
- A call ("slowdown", or "acceleration" when enabled) opens a paper position the day
  it is first made, only if that quarter's revenue is not yet reported.
- slowdown = short, acceleration = long; $1,000 notional each, hedged against the
  market (SPY): the result is the stock's return minus SPY's.
- Entry: close of the first trading day after the call. Exit: close of the first
  trading day after the earnings announcement (the paper's [-1, +1] window ends there).
  No announcement within 150 days: closed at the last price as "expired".
- Each closed trade also records whether revenue growth came in below trend
  (GrowthSurprise < 0), the thing the call actually predicts.

research/paper_trades.csv is the ledger. Rows are added when calls are made and
their outcome columns filled in once; entries are never edited or removed.
research/PAPER_TRADING.md and research/SUMMARY.md are rebuilt from it each run.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd

from .config import Study
from .http import SourceUnavailable

NOTIONAL = 1000.0
MARKET = "SPY"
EXPIRE_DAYS = 150
COLUMNS = ["opened_at", "ticker", "signal_quarter", "call", "position", "abnormal_tone", "reviews",
           "notional", "entry_date", "expected_announcement", "announcement_date", "exit_date",
           "stock_return", "market_return", "excess_return", "pnl", "growth_surprise", "revenue_hit",
           "status"]


def ledger_path(study: Study):
    return study.research_dir / "paper_trades.csv"


def load(study: Study) -> pd.DataFrame:
    p = ledger_path(study)
    return pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else pd.DataFrame(columns=COLUMNS)


def update(study: Study, as_of: date, returns=None) -> pd.DataFrame:
    """Open positions for new calls, close positions whose announcement has passed.

    returns: optional function ticker -> daily return Series (tests); default Yahoo/Stooq.
    """
    from .market.data import daily_returns

    get_returns = returns or (lambda t: daily_returns(t))
    ledger = load(study)
    have = set(zip(ledger["ticker"], ledger["signal_quarter"]))
    new_rows = []
    for c in study.universe:
        p = study.ticker_dir(c.ticker) / "prediction.json"
        if not p.exists():
            continue
        card = json.loads(p.read_text())
        call = card.get("call")
        if call not in ("slowdown", "acceleration") or card.get("revenue_already_reported"):
            continue
        key = (c.ticker, str(card.get("signal_quarter")))
        if key in have:
            continue
        new_rows.append({"opened_at": as_of.isoformat(), "ticker": c.ticker, "signal_quarter": key[1], "call": call,
                         "position": "short" if call == "slowdown" else "long",
                         "abnormal_tone": card.get("AbnormalTone"), "reviews": card.get("reviews"),
                         "notional": NOTIONAL, "expected_announcement": card.get("expected_announcement", ""),
                         "status": "open"})
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows).astype(str)], ignore_index=True)

    market = None
    for i, row in ledger[ledger["status"] == "open"].iterrows():
        try:
            if market is None:
                market = get_returns(MARKET)
            stock = get_returns(row["ticker"])
        except SourceUnavailable:
            continue  # prices unreachable this run: try again next run
        _settle(study, ledger, i, row, stock, market, as_of)

    ledger = ledger.reindex(columns=COLUMNS).fillna("")
    ledger_path(study).parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(ledger_path(study), index=False)
    write_pages(study, ledger, as_of)
    return ledger


def _settle(study: Study, ledger: pd.DataFrame, i, row, stock: pd.Series, market: pd.Series, as_of: date) -> None:
    opened = pd.Timestamp(row["opened_at"])
    days = stock.index[stock.index > opened]
    if len(days) == 0:
        return  # no trading day after the call yet
    entry = days[0]
    ledger.loc[i, "entry_date"] = entry.date().isoformat()
    fpath = study.ticker_dir(row["ticker"]) / "fundamentals.csv"
    fund = pd.read_csv(fpath) if fpath.exists() else pd.DataFrame(columns=["fiscal_quarter"])
    q = fund[fund["fiscal_quarter"] == row["signal_quarter"]]
    ann = q["announcement_date"].dropna().iloc[0] if len(q) and q["announcement_date"].notna().any() else None
    if ann:
        ledger.loc[i, "announcement_date"] = str(ann)
        after = stock.index[stock.index > pd.Timestamp(ann)]
        if len(after) == 0:
            return  # announced, but no trading day after it yet
        exit_day, status = after[0], "closed"
    elif (as_of - opened.date()).days > EXPIRE_DAYS:
        exit_day, status = stock.index[-1], "expired"
    else:
        return
    # Returns from the day after entry through the exit day.
    s = float((1 + stock.loc[entry:exit_day].iloc[1:]).prod() - 1)
    m = float((1 + market.loc[entry:exit_day].iloc[1:]).prod() - 1)
    sign = -1.0 if row["position"] == "short" else 1.0
    gs = q["GrowthSurprise"].dropna() if len(q) and "GrowthSurprise" in q else pd.Series(dtype=float)
    ledger.loc[i, ["exit_date", "stock_return", "market_return", "excess_return", "pnl", "status"]] = [
        exit_day.date().isoformat(), f"{s:.4f}", f"{m:.4f}", f"{s - m:.4f}",
        f"{sign * float(row['notional']) * (s - m):.2f}", status]
    if len(gs):
        g = float(gs.iloc[0])
        ledger.loc[i, "growth_surprise"] = f"{g:.4f}"
        ledger.loc[i, "revenue_hit"] = str(g < 0 if row["position"] == "short" else g > 0)


def write_pages(study: Study, ledger: pd.DataFrame, as_of: date) -> None:
    """research/PAPER_TRADING.md (the record) and research/SUMMARY.md (this week's calls)."""
    closed = ledger[ledger["status"].isin(["closed", "expired"])]
    open_ = ledger[ledger["status"] == "open"]
    pnl = pd.to_numeric(closed["pnl"], errors="coerce")
    hits = closed["revenue_hit"].replace("", pd.NA).dropna()
    lines = ["# Paper trading record", "",
             f"Updated {as_of.isoformat()}. Rules are fixed in `src/reviewtone/paper.py`: each call is a "
             f"${NOTIONAL:,.0f} market-hedged paper position (stock minus {MARKET}), opened the day after the call "
             "and closed the day after the earnings announcement. No real money is involved.", "",
             "## Record", "",
             f"- Calls made: {len(ledger)} ({len(open_)} open, {len(closed)} closed)",
             f"- Closed: total paper P&L ${pnl.sum():,.2f}; won {int((pnl > 0).sum())} of {len(pnl)}"
             + (f"; average {pnl.mean():+,.2f} per trade" if len(pnl) else ""),
             f"- Revenue came in as called: {int((hits == 'True').sum())} of {len(hits)}", ""]
    if len(closed) < 10:
        lines += ["Too few closed trades to judge the signal yet: give it several earnings seasons.", ""]
    for title, df, cols in (
            ("Open positions", open_, ["opened_at", "ticker", "signal_quarter", "call", "position", "entry_date",
                                       "expected_announcement"]),
            ("Closed positions", closed, ["opened_at", "ticker", "signal_quarter", "position", "announcement_date",
                                          "excess_return", "pnl", "revenue_hit", "status"])):
        lines += [f"## {title}", ""]
        lines += (_table(df[cols]) if len(df) else ["None."]) + [""]
    lines += _portfolio_section(study, ledger, as_of)
    study.research_dir.mkdir(parents=True, exist_ok=True)
    (study.research_dir / "PAPER_TRADING.md").write_text("\n".join(lines) + "\n")

    rows = []
    for c in study.universe:
        p = study.ticker_dir(c.ticker) / "prediction.json"
        if not p.exists():
            rows.append({"ticker": c.ticker, "call": "not run yet"})
            continue
        card = json.loads(p.read_text())
        rows.append({"ticker": c.ticker, "quarter": card.get("signal_quarter", ""), "call": card.get("call", card.get("status", "")),
                     "AbnormalTone": card.get("AbnormalTone", ""), "reviews": card.get("reviews", ""),
                     "why no call": "; ".join(card.get("blockers", [])),
                     "revenue reported": card.get("revenue_already_reported", "")})
    summary = ["# This week's calls", "", f"Updated {as_of.isoformat()}. `slowdown` = review tone dropped clearly "
               "below its own recent baseline: expect revenue growth below trend. `no_call` = nothing to act on. "
               "Details: `research/<TICKER>/README.md`. Paper trading record: `PAPER_TRADING.md`.", ""]
    summary += _table(pd.DataFrame(rows).fillna(""))
    (study.research_dir / "SUMMARY.md").write_text("\n".join(summary) + "\n")


def _portfolio_section(study: Study, ledger: pd.DataFrame, as_of: date) -> list[str]:
    d = study.research_dir / "_universe"
    out = ["## Long/short portfolio (paper)", ""]
    pj = d / "portfolio.json"
    if not pj.exists():
        return out + ["Not formed yet.", ""]
    p = json.loads(pj.read_text())
    latest = p.get("latest_formation")
    if latest:
        out += [f"Latest formation ({latest['formation_quarter']}): long {', '.join(latest['long']) or '-'}; "
                f"short {', '.join(latest['short']) or '-'}.", ""]
    sr = d / "strategy_returns.csv"
    start = ledger["opened_at"].min() if len(ledger) else as_of.isoformat()
    if sr.exists():
        q = pd.read_csv(sr)
        if "holding_quarter" in q and "equal_long_short" in q:
            start_q = f"{pd.Timestamp(start).year}Q{(pd.Timestamp(start).month - 1) // 3 + 1}"
            live = q[q["holding_quarter"] >= start_q]
            out += [f"Quarters held since paper trading started ({start_q}): {len(live)}; equal-weight long-short "
                    f"return {live['equal_long_short'].sum():+.1%}." if len(live) else
                    "No holding quarter has finished since paper trading started.", ""]
    return out


def _table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    return [" | ".join(cols), " | ".join("---" for _ in cols)] + [
        " | ".join(str(v) for v in r) for r in df.itertuples(index=False)]


def last_run_due(path, today: date, every_days: int = 6) -> bool:
    """For the scheduler: is a run due (no complete run in the last `every_days`)?"""
    try:
        info = json.loads(path.read_text())
    except (OSError, ValueError):
        return True
    return info.get("status") != "complete" or date.fromisoformat(info["date"]) <= today - timedelta(days=every_days)
