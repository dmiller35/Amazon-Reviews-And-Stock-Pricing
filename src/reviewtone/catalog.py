"""products.csv: discoverable ASINs, variants and URLs per company (step 2)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Company

PRODUCT_COLUMNS = [
    "ticker", "brand", "asin", "parent_asin", "title", "store", "category", "rating_number",
    "url", "official_store", "source", "discovered_at", "live", "live_checked_at",
]


def load_seed(path: Path, company: Company) -> pd.DataFrame:
    """Manual seed list: CSV with at least `asin` (optional brand, title, parent_asin)."""
    if not path.exists():
        return pd.DataFrame(columns=PRODUCT_COLUMNS)
    seed = pd.read_csv(path, dtype=str).fillna("")
    seed["ticker"] = company.ticker
    seed["brand"] = seed.get("brand", company.brands[0].name)
    seed["parent_asin"] = seed.get("parent_asin", seed["asin"]).where(lambda s: s != "", seed["asin"])
    seed["url"] = "https://www.amazon.com/dp/" + seed["asin"]
    seed["official_store"] = pd.NA  # unverified until a store field is observed
    seed["source"] = f"manual:{path.name}"
    return seed


def merge_products(*frames: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Union product lists; one row per ASIN, preferring rows with a verified store."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=PRODUCT_COLUMNS), 0
    df = pd.concat(frames, ignore_index=True)
    for col in PRODUCT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[PRODUCT_COLUMNS]
    n = len(df)
    df["_verified"] = df["official_store"].eq(True)
    df = df.sort_values("_verified", ascending=False).drop_duplicates("asin").drop(columns="_verified")
    return df.sort_values(["brand", "parent_asin", "asin"]).reset_index(drop=True), n - len(df)


def catalog_gaps(products: pd.DataFrame, reviews: pd.DataFrame, company: Company) -> list[str]:
    gaps = []
    for b in company.brands:
        n = int((products["brand"] == b.name).sum()) if len(products) else 0
        if n == 0:
            gaps.append(f"No products discovered for brand '{b.name}' (store names searched: "
                        f"{', '.join(b.amazon_store_names)}).")
    if len(products):
        reviewed = set(reviews["parent_asin"].dropna()) | set(reviews["asin"].dropna()) if len(reviews) else set()
        silent = products[~products["asin"].isin(reviewed) & ~products["parent_asin"].isin(reviewed)]
        if len(silent):
            gaps.append(f"{len(silent)} of {len(products)} catalogued products have no reviews in reviews.csv.")
        unverified = products["official_store"].ne(True).sum()
        if unverified:
            gaps.append(f"{unverified} products come from a manual seed list and their storefront was not verified.")
    gaps.append("Discovery only covers products whose Amazon `store` field matches the configured storefront "
                "names; third-party listings of the brand, bundles and delisted products without metadata are "
                "not included.")
    return gaps
