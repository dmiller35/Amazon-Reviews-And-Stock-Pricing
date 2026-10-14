"""Calendar-time long/short portfolio across the universe (Jin 2022, Table 6).

At each calendar quarter end, firms with a usable AbnormalTone are sorted into
groups (quartiles, or halves for small universes). The strategy buys the top
group and sells the bottom group over the next calendar quarter, weighting by
review count or equally, and its quarterly returns are regressed on the
Carhart factors. The latest formation is the live "stock predictor" output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fiscal import FiscalQuarter
from .market.models import FACTORS, compound, portfolio_alpha

MIN_FIRMS = 4


def form_groups(signals: pd.DataFrame, groups: int, min_firms_for_groups: int) -> pd.DataFrame:
    """signals: ticker, AbnormalTone, n_reviews for one formation quarter."""
    s = signals.dropna(subset=["AbnormalTone"]).copy()
    if len(s) < MIN_FIRMS:
        return s.iloc[0:0].assign(group=pd.Series(dtype=int))
    k = groups if len(s) >= min_firms_for_groups else 2
    s["group"] = pd.qcut(s["AbnormalTone"].rank(method="first"), k, labels=False) + 1
    s["n_groups"] = k
    return s


def run(measures: dict[str, pd.DataFrame], returns: dict[str, pd.Series], factors: pd.DataFrame,
        cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Returns (positions, quarterly strategy returns, alphas, latest live sort)."""
    rows = []
    for t, m in measures.items():
        if m.empty:
            continue
        rows.append(m.loc[m["usable"], ["fq_index", "fiscal_quarter", "AbnormalTone", "n_reviews"]].assign(ticker=t))
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame()
    sig = pd.concat(rows, ignore_index=True)
    last_px = min(r.index.max() for r in returns.values()) if returns else pd.Timestamp.min

    positions, strat, latest = [], [], pd.DataFrame()
    for fq_index, grp in sig.groupby("fq_index"):
        sorted_ = form_groups(grp, cfg.get("groups", 4), cfg.get("min_firms_for_quartiles", 8))
        if sorted_.empty:
            continue
        latest = sorted_
        hold = _quarter_from_index(fq_index).shift(1)
        if pd.Timestamp(hold.end) > last_px:
            continue  # holding period not over yet
        start, end = hold.start.isoformat(), hold.end.isoformat()
        sorted_["hold_return"] = [compound(returns[t], start, end) if t in returns else np.nan
                                  for t in sorted_["ticker"]]
        sorted_["holding_quarter"] = f"{hold.end.year}Q{hold.q}"
        positions.append(sorted_)
        fq = compound(factors[FACTORS + ["rf"]], start, end)
        k = int(sorted_["n_groups"].iloc[0])
        rec = {"holding_quarter": f"{hold.end.year}Q{hold.q}", "rf": fq["rf"], **{f: fq[f] for f in FACTORS}}
        for scheme in cfg.get("weighting", ["review_count", "equal"]):
            legs = {}
            for name, g in (("long", k), ("short", 1)):
                leg = sorted_[sorted_["group"] == g].dropna(subset=["hold_return"])
                w = leg["n_reviews"].astype(float) if scheme == "review_count" else pd.Series(1.0, index=leg.index)
                legs[name] = float(np.average(leg["hold_return"], weights=w)) if len(leg) else np.nan
            rec[f"{scheme}_long"] = legs["long"]
            rec[f"{scheme}_short"] = legs["short"]
            rec[f"{scheme}_long_short"] = legs["long"] - legs["short"]
        strat.append(rec)

    pos = pd.concat(positions, ignore_index=True) if positions else pd.DataFrame()
    q = pd.DataFrame(strat).set_index("holding_quarter") if strat else pd.DataFrame()
    alphas = {}
    if len(q):
        for scheme in cfg.get("weighting", ["review_count", "equal"]):
            alphas[scheme] = {
                "long_short": portfolio_alpha(q[f"{scheme}_long_short"], q),
                "long": portfolio_alpha(q[f"{scheme}_long"] - q["rf"], q),
                "short": portfolio_alpha(q[f"{scheme}_short"] - q["rf"], q),
                "mean_long_short_return": float(q[f"{scheme}_long_short"].mean()),
            }
    return pos, q.reset_index() if len(q) else q, alphas, latest


def _quarter_from_index(idx: int) -> FiscalQuarter:
    return FiscalQuarter(idx // 4, idx % 4 + 1, 12)
