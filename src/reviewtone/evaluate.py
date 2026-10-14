"""Held-out evaluation with fixed, pre-registered rules (step 6).

Signal: AbnormalTone for fiscal quarter q, known when q ends.
Outcomes: GrowthSurprise of quarter q+h, first reported in a later SEC filing,
and the abnormal return around that quarter's earnings announcement.
Nothing is fitted: thresholds come from config/study.yaml and are locked by
research/preregistration.json before any result is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import Study
from .http import utcnow


class PreregistrationError(RuntimeError):
    pass


def check_preregistration(study: Study, new_study: bool = False) -> dict:
    """Lock the rules and every evaluated company before results are written.

    - Rules (thresholds, horizons, hold-out size, portfolio setup) never change
      within a study.
    - A registered company can't be edited or removed, so losers can't be
      dropped and storefront lists can't be tuned after seeing outcomes.
    - New companies may be added at any time; each records its registration
      time, which the report shows.
    """
    path = study.research_dir / "preregistration.json"
    now = utcnow()
    if path.exists() and new_study:
        old = json.loads(path.read_text())
        path.rename(path.with_name(f"preregistration.{old['study']}.{old['registered_at'][:10]}.json"))
    if not path.exists():
        reg = {"study": study.name, "registered_at": now, "rules_hash": study.rules_hash(), "rules": study.rules,
               "companies": {}}
    else:
        reg = json.loads(path.read_text())
        if reg["rules_hash"] != study.rules_hash():
            raise PreregistrationError(
                f"Rules in config/study.yaml changed after the study was registered on {reg['registered_at']}. "
                "Picking thresholds after seeing results is not allowed. Revert the change, or start a new study "
                "with --new-study (and a new study.name).")
        tickers = set(study.tickers)
        removed = sorted(set(reg["companies"]) - tickers)
        changed = sorted(t for t, c in reg["companies"].items()
                         if t in tickers and study.company(t).fingerprint() != c["fingerprint"])
        if removed or changed:
            raise PreregistrationError(
                "Registered companies cannot be removed or edited after results exist "
                f"(removed: {removed or 'none'}; edited: {changed or 'none'}). Restore their config files, "
                "or start a new study with --new-study.")
    for c in study.universe:
        reg["companies"].setdefault(c.ticker, {"fingerprint": c.fingerprint(), "registered_at": now})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2))
    return reg


def make_call(abnormal_tone: float, usable: bool, rules: dict) -> str:
    if not usable or pd.isna(abnormal_tone):
        return "no_call"
    if abnormal_tone <= rules["negative_swing_threshold"]:
        return "slowdown"
    if rules.get("call_positive_swings") and abnormal_tone >= rules["positive_swing_threshold"]:
        return "acceleration"
    return "no_call"


def build_panel(quarterly: pd.DataFrame, fundamentals: pd.DataFrame, rules: dict,
                announcement_returns: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (signal quarter, horizon) with call, outcome and hit."""
    f = fundamentals.set_index("fq_index")
    if announcement_returns is not None and len(announcement_returns):
        f = f.join(announcement_returns.set_index("fq_index")[["car", "bhar", "event_day"]])
    rows = []
    for r in quarterly.itertuples():
        call = make_call(r.AbnormalTone, r.usable, rules)
        for h in rules.get("horizons", [0]):
            target = r.fq_index + h
            o = f.loc[target] if target in f.index else None
            gs = o["GrowthSurprise"] if o is not None else np.nan
            car = o["car"] if o is not None and "car" in o else np.nan
            hit = None
            if call != "no_call" and pd.notna(gs):
                hit = bool(gs < 0) if call == "slowdown" else bool(gs > 0)
            rows.append({
                "fq_index": r.fq_index, "signal_quarter": r.fiscal_quarter, "horizon": h,
                "outcome_quarter": o["fiscal_quarter"] if o is not None else None,
                "n_reviews": r.n_reviews, "mean_tone": r.mean_tone, "AbnormalTone": r.AbnormalTone,
                "AbnormalRating": r.AbnormalRating, "AbnormalReviewCnt": r.AbnormalReviewCnt,
                "fiscal_q": r.fiscal_q, "usable": r.usable, "call": call,
                "GrowthSurprise": gs,
                "SameQtrSalesGrowth": o["SameQtrSalesGrowth"] if o is not None else np.nan,
                "AbnormalSales": o["AbnormalSales"] if o is not None else np.nan,
                "revenue": o["revenue"] if o is not None else np.nan,
                "revenue_filed": o["revenue_filed"] if o is not None else None,
                "announcement_date": o["announcement_date"] if o is not None else None,
                "announcement_car": car,
                "sales_hit": hit,
                "return_hit": (bool(car < 0) if call == "slowdown" else bool(car > 0))
                if call != "no_call" and pd.notna(car) else None,
            })
    return pd.DataFrame(rows)


def holdout(panel: pd.DataFrame, n_quarters: int, horizon: int = 0) -> pd.DataFrame:
    """The last n signal quarters that have both a usable signal and a reported outcome."""
    p = panel[(panel["horizon"] == horizon) & panel["usable"] & panel["GrowthSurprise"].notna()]
    return p.sort_values("fq_index").tail(n_quarters)


def summarize(h: pd.DataFrame) -> dict:
    calls = h[h["call"] != "no_call"]
    s = {
        "holdout_quarters": int(len(h)),
        "calls": int(len(calls)),
        "sales_hits": int((calls["sales_hit"] == True).sum()),  # noqa: E712
        "sales_misses": int((calls["sales_hit"] == False).sum()),  # noqa: E712
        "base_rate_slowdown": float((h["GrowthSurprise"] < 0).mean()) if len(h) else np.nan,
        "uncalled_slowdowns": int(((h["call"] == "no_call") & (h["GrowthSurprise"] < 0)).sum()),
    }
    s["sales_hit_rate"] = s["sales_hits"] / s["calls"] if s["calls"] else np.nan
    r = calls.dropna(subset=["return_hit"])
    s["return_calls"] = int(len(r))
    s["return_hit_rate"] = float(r["return_hit"].mean()) if len(r) else np.nan
    if len(h) < 8:
        s["warning"] = f"only {len(h)} held-out quarters available (8 sought); results are not meaningful"
    return s


def paper_regression(panel: pd.DataFrame, y: str = "AbnormalSales", horizon: int = 0) -> pd.DataFrame | None:
    """Pooled OLS in the spirit of Jin (2022) Eq. 1, descriptive only.

    y on AbnormalTone, AbnormalRating, AbnormalReviewCnt and fiscal-quarter
    dummies, with firm-clustered errors when several firms are pooled.
    """
    cols = ["AbnormalTone", "AbnormalRating", "AbnormalReviewCnt"]
    p = panel[(panel["horizon"] == horizon) & panel["usable"]].dropna(subset=[y] + cols)
    if len(p) < 20:
        return None
    x = p[cols].copy()
    if "fiscal_q" in p:
        x = x.join(pd.get_dummies(p["fiscal_q"], prefix="Q", drop_first=True, dtype=float))
    x = sm.add_constant(x)
    if "ticker" in p and p["ticker"].nunique() > 1:
        x = x.join(pd.get_dummies(p["ticker"], prefix="firm", drop_first=True, dtype=float))
        fit = sm.OLS(p[y].astype(float), x).fit(cov_type="cluster", cov_kwds={"groups": p["ticker"].astype("category").cat.codes})
    else:
        fit = sm.OLS(p[y].astype(float), x).fit(cov_type="HC1")
    return pd.DataFrame({"coef": fit.params, "t": fit.tvalues, "p": fit.pvalues}).assign(n=int(fit.nobs),
                                                                                       r2=fit.rsquared)


def write_evaluation(ticker_dir: Path, panel: pd.DataFrame, n_holdout: int) -> dict:
    panel.to_csv(ticker_dir / "panel.csv", index=False)
    results = {}
    for h in sorted(panel["horizon"].unique()):
        ho = holdout(panel, n_holdout, h)
        ho.to_csv(ticker_dir / f"evaluation_h{h}.csv", index=False)
        results[f"h{h}"] = {"summary": summarize(ho),
                            "misses": ho[ho["sales_hit"] == False]["signal_quarter"].tolist()}  # noqa: E712
    (ticker_dir / "evaluation.json").write_text(json.dumps(results, indent=2, default=float))
    return results
