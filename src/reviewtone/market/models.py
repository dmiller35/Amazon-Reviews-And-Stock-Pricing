"""Carhart four-factor models: announcement abnormal returns and portfolio alpha."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTORS = ["mkt_rf", "smb", "hml", "umd"]


@dataclass
class EventReturn:
    event_day: str
    car: float
    bhar: float
    n_estimation_days: int


def announcement_return(returns: pd.Series, factors: pd.DataFrame, announcement_date: str,
                        estimation: tuple[int, int] = (-250, -30), window: tuple[int, int] = (-1, 1),
                        min_estimation_days: int = 120) -> EventReturn | None:
    """CAR and BHAR around an earnings announcement (Jin 2022, Eq. 7).

    Day 0 is the first trading day on or after the announcement date. Expected
    returns come from a Carhart regression fitted over the estimation window.
    """
    data = pd.concat([returns.rename("ret"), factors], axis=1, join="inner").dropna()
    days = data.index
    pos = days.searchsorted(pd.Timestamp(announcement_date))
    if pos >= len(days) or pos + window[1] >= len(days) or pos + estimation[0] < 0:
        return None
    est = data.iloc[pos + estimation[0]: pos + estimation[1] + 1]
    if len(est) < min_estimation_days:
        return None
    model = sm.OLS(est["ret"] - est["rf"], sm.add_constant(est[FACTORS])).fit()
    ev = data.iloc[pos + window[0]: pos + window[1] + 1]
    expected = ev["rf"] + model.predict(sm.add_constant(ev[FACTORS], has_constant="add"))
    ar = ev["ret"] - expected
    bhar = float(np.prod(1 + ev["ret"]) - np.prod(1 + expected))
    return EventReturn(days[pos].strftime("%Y-%m-%d"), float(ar.sum()), bhar, len(est))


def compound(daily: pd.DataFrame | pd.Series, start: str, end: str):
    window = daily.loc[start:end]
    return (1 + window).prod() - 1


def portfolio_alpha(port_excess: pd.Series, quarterly_factors: pd.DataFrame) -> dict[str, float]:
    """Regress quarterly portfolio excess returns on quarterly Carhart factors (Eq. 3)."""
    df = pd.concat([port_excess.rename("y"), quarterly_factors[FACTORS]], axis=1, join="inner").dropna()
    n = len(df)
    if n < len(FACTORS) + 3:
        return {"n_quarters": n, "alpha": np.nan, "alpha_t": np.nan, "note": "too few quarters for a 4-factor fit"}
    fit = sm.OLS(df["y"], sm.add_constant(df[FACTORS])).fit(cov_type="HC1")
    out = {"n_quarters": n, "alpha": float(fit.params["const"]), "alpha_t": float(fit.tvalues["const"])}
    out.update({f"beta_{f}": float(fit.params[f]) for f in FACTORS})
    return out
