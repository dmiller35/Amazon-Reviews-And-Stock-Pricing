"""The live prediction card for one company."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .evaluate import make_call

CALL_TEXT = {
    "slowdown": "Negative tone swing: expect revenue growth below its recent trend and a weak earnings reaction.",
    "acceleration": "Positive tone swing: expect revenue growth above its recent trend.",
    "no_call": "No call: tone is within its normal range, or the data is too thin to judge.",
}


def prediction_card(ticker: str, quarterly: pd.DataFrame, fundamentals: pd.DataFrame | None, rules: dict,
                    evaluation: dict | None, as_of: date) -> dict:
    done = quarterly[~quarterly["in_progress"]]
    if done.empty:
        return {"ticker": ticker, "as_of": as_of.isoformat(), "status": "no completed quarter with reviews"}
    r = done.iloc[-1]
    call = make_call(r["AbnormalTone"], bool(r["usable"]), rules)
    excluded = int(r.get("n_products_excluded", 0) or 0)
    # Every tracked product has a gap (typically: no live reviews fetched yet after the historical data).
    nothing = excluded < 0 or (excluded > 0 and int(r["n_reviews"]) == 0)
    blockers = [name for name, flag in (
        ("no tracked product has complete reviews for this quarter and its baseline yet", nothing),
        ("sparse quarter", r["sparse"] and not nothing), ("sparse baseline", r["sparse_baseline"] and not nothing),
        ("fewer than 4 prior quarters", pd.isna(r["AbnormalTone"]) and not nothing)) if flag]
    card = {
        "ticker": ticker,
        "as_of": as_of.isoformat(),
        "signal_quarter": r["fiscal_quarter"],
        "quarter_end": r["quarter_end"],
        "reviews": int(r["n_reviews"]),
        "mean_tone": _f(r["mean_tone"]),
        "AbnormalTone": _f(r["AbnormalTone"]),
        "call": call,
        "call_text": CALL_TEXT[call],
        "blockers": blockers,
    }
    if excluded > 0 and not nothing:
        card["products_left_out"] = (f"{excluded} tracked product(s) have a gap in their reviews in this quarter "
                                     "or its baseline; the call compares the remaining products with their own "
                                     "four-quarter baseline")
    if fundamentals is not None and len(fundamentals):
        f = fundamentals.set_index("fq_index")
        reported = r["fq_index"] in f.index and pd.notna(f.loc[r["fq_index"], "revenue"])
        card["revenue_already_reported"] = bool(reported)
        if reported:
            card["outcome_growth_surprise"] = _f(f.loc[r["fq_index"], "GrowthSurprise"])
        else:
            prev = r["fq_index"] - 4
            if prev in f.index and pd.notna(f.loc[prev, "announcement_date"]):
                est = date.fromisoformat(f.loc[prev, "announcement_date"]) + timedelta(days=364)
                card["expected_announcement"] = f"around {est.isoformat()} (same quarter last year + 52 weeks)"
    in_prog = quarterly[quarterly["in_progress"]]
    if len(in_prog):
        p = in_prog.iloc[-1]
        card["current_quarter_so_far"] = {"quarter": p["fiscal_quarter"], "reviews": int(p["n_reviews"]),
                                          "mean_tone": _f(p["mean_tone"]), "AbnormalTone": _f(p["AbnormalTone"]),
                                          "note": "provisional; the quarter is not over"}
    if evaluation:
        card["track_record_h0"] = evaluation.get("h0", {}).get("summary")
    return card


def _f(v) -> float | None:
    return None if pd.isna(v) else round(float(v), 4)
