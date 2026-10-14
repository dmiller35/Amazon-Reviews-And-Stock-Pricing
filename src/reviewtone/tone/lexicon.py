"""Offline word-list baseline. For smoke tests and dry runs only.

It ignores negation and context, so it is not a research-grade measure; the
report labels any result produced with it accordingly.
"""

from __future__ import annotations

import re

import pandas as pd

from ..http import utcnow
from .rubric import RUBRIC_VERSION

POSITIVE = {
    "great", "excellent", "love", "perfect", "amazing", "awesome", "reliable", "comfortable", "best",
    "good", "works", "recommend", "solid", "happy", "fantastic", "smooth", "sturdy", "durable",
}
NEGATIVE = {
    "broke", "broken", "defective", "dead", "stopped", "terrible", "awful", "worst", "returned",
    "return", "junk", "garbage", "disappointed", "inferior", "fails", "failed", "flimsy", "poor",
    "useless", "waste", "doom", "cheap", "worse",
}
SHIPPING = {"package", "packaging", "delivery", "delivered", "shipping", "arrived", "seller", "box"}
_WORD = re.compile(r"[a-z']+")


def lexicon_tone(text: str) -> tuple[float, bool]:
    words = _WORD.findall(text.casefold())
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    ship = sum(w in SHIPPING for w in words)
    shipping_only = ship > 0 and pos + neg == 0
    if pos + neg == 0:
        return 0.0, shipping_only
    return (pos - neg) / (pos + neg), shipping_only


class LexiconToneScorer:
    name = "lexicon"

    def score(self, reviews: pd.DataFrame, progress: bool = False) -> pd.DataFrame:
        rows = []
        for r in reviews.itertuples():
            tone, ship_only = lexicon_tone(f"{r.title}. {r.text}")
            rows.append({"review_id": r.review_id, "tone_raw": round(tone * 2), "tone": tone, "tags": "",
                         "seller_shipping_only": ship_only, "reason": ""})
        out = pd.DataFrame(rows)
        out["scorer"], out["model"], out["rubric_version"], out["scored_at"] = (
            self.name, "word-list baseline", RUBRIC_VERSION, utcnow())
        return out
