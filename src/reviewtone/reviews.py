"""Review table schema, de-duplication and append-only storage."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

REVIEW_COLUMNS = [
    "review_id",
    "ticker",
    "brand",
    "asin",
    "parent_asin",
    "review_date",
    "rating",
    "title",
    "text",
    "verified",
    "variant",
    "source",
    "source_url",
    "fetched_at",
]


def stable_review_id(asin: str, review_date: str, title: str, text: str) -> str:
    blob = "\x1f".join([asin or "", review_date or "", (title or "").strip(), (text or "").strip()])
    return "h_" + hashlib.sha1(blob.encode()).hexdigest()[:16]


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().casefold()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a source frame into REVIEW_COLUMNS with clean types."""
    out = df.copy()
    for col in REVIEW_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[REVIEW_COLUMNS]
    out["review_date"] = pd.to_datetime(out["review_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    for col in ("title", "text"):
        out[col] = out[col].fillna("").astype(str)
    missing = out["review_id"].isna() | (out["review_id"].astype(str).str.len() == 0)
    out.loc[missing, "review_id"] = [
        stable_review_id(str(r.asin), str(r.review_date), r.title, r.text) for r in out[missing].itertuples()
    ]
    out["review_id"] = out["review_id"].astype(str)
    return out.dropna(subset=["review_date"])


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop repeated review IDs, then repeated review text.

    Jin (2022, fn. 10) removes duplicate reviews even when written by different
    accounts, to neutralise multi-account and plagiarised reviews. Variants of
    one product often share a review page, so the same review can also appear
    under several child ASINs.
    """
    n0 = len(df)
    df = df.drop_duplicates(subset="review_id", keep="first")
    n_id = n0 - len(df)
    body = df["title"].map(_norm_text) + "\x1f" + df["text"].map(_norm_text)
    has_body = df["text"].str.strip().str.len() > 20
    dup_text = body.duplicated(keep="first") & has_body
    df = df[~dup_text]
    return df.reset_index(drop=True), {"duplicate_ids": n_id, "duplicate_text": int(dup_text.sum())}


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    return pd.read_csv(path, dtype={"review_id": str, "asin": str, "parent_asin": str}, keep_default_na=False,
                       na_values={"rating": [""]})


def append_new(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Append reviews not already stored. Existing rows are never rewritten."""
    existing = load(path)
    new = normalize(new)
    combined, _ = deduplicate(pd.concat([existing, new], ignore_index=True))
    added = combined[~combined["review_id"].isin(set(existing["review_id"]))]
    path.parent.mkdir(parents=True, exist_ok=True)
    added.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)
    return added
