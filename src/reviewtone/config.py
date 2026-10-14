"""Study configuration: universe, scoring settings and pre-registered rules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "study.yaml"


@dataclass(frozen=True)
class Brand:
    name: str
    amazon_store_names: tuple[str, ...]


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    fiscal_year_end: str  # "MM-DD"
    brands: tuple[Brand, ...]
    source_file: str = ""
    mcauley_categories: tuple[str, ...] = ()  # data sourcing only; not part of the fingerprint

    def fingerprint(self) -> str:
        """Hash of the definition; registered companies may not change."""
        blob = json.dumps({"ticker": self.ticker, "fiscal_year_end": self.fiscal_year_end,
                           "brands": [[b.name, sorted(b.amazon_store_names)] for b in self.brands]}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def fiscal_year_end_month(self) -> int:
        return int(self.fiscal_year_end.split("-")[0])

    @property
    def store_names(self) -> set[str]:
        return {s.casefold() for b in self.brands for s in b.amazon_store_names}


@dataclass
class Study:
    raw: dict[str, Any]
    path: Path
    universe: list[Company] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.raw["study"]["name"]

    @property
    def primary_ticker(self) -> str:
        return self.raw["study"]["primary_ticker"]

    @property
    def research_dir(self) -> Path:
        d = Path(self.raw["study"].get("research_dir", "research"))
        return d if d.is_absolute() else self.path.parent.parent / d

    @property
    def tone(self) -> dict[str, Any]:
        return self.raw["tone"]

    @property
    def aggregation(self) -> dict[str, Any]:
        return self.raw["aggregation"]

    @property
    def rules(self) -> dict[str, Any]:
        return self.raw["rules"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def portfolio(self) -> dict[str, Any]:
        return self.raw["portfolio"]

    @property
    def market(self) -> dict[str, Any]:
        return self.raw["market"]

    def company(self, ticker: str) -> Company:
        for c in self.universe:
            if c.ticker.upper() == ticker.upper():
                return c
        raise KeyError(f"{ticker} is not in the registered universe ({', '.join(self.tickers)})")

    @property
    def tickers(self) -> list[str]:
        return [c.ticker for c in self.universe]

    def ticker_dir(self, ticker: str) -> Path:
        return self.research_dir / ticker.upper()

    @property
    def companies_dir(self) -> Path:
        return self.path.parent / self.raw.get("companies_dir", "companies")

    def rules_hash(self) -> str:
        """Hash of the decision rules, which must not change after results exist."""
        locked = {
            "rules": self.raw["rules"],
            "evaluation": self.raw["evaluation"],
            "portfolio": self.raw["portfolio"],
            "min_reviews_per_quarter": self.raw["aggregation"]["min_reviews_per_quarter"],
            "product_scope": self.raw["aggregation"].get("product_scope", "tracked"),
            "top_products": self.raw.get("apify", {}).get("top_products", 20),
        }
        blob = json.dumps(locked, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def company_from_dict(c: dict[str, Any], source_file: str = "") -> Company:
    return Company(
        ticker=str(c["ticker"]).upper(),
        name=c["name"],
        fiscal_year_end=str(c["fiscal_year_end"]),
        brands=tuple(
            Brand(name=b["name"], amazon_store_names=tuple(b.get("amazon_store_names") or [b["name"]]))
            for b in c["brands"]
        ),
        source_file=source_file,
        mcauley_categories=tuple(c.get("mcauley_categories") or ()),
    )


def load_study(path: str | Path | None = None) -> Study:
    p = (Path(path) if path else DEFAULT_CONFIG).resolve()
    raw = yaml.safe_load(p.read_text())
    entries = [(c, "study.yaml") for c in raw.get("universe", [])]
    cdir = p.parent / raw.get("companies_dir", "companies")
    if cdir.is_dir():
        entries += [(yaml.safe_load(f.read_text()), f.name) for f in sorted(cdir.glob("*.yaml"))]
    universe = [company_from_dict(c, src) for c, src in entries]
    dupes = {c.ticker for c in universe if sum(u.ticker == c.ticker for u in universe) > 1}
    if dupes:
        raise ValueError(f"companies defined more than once: {', '.join(sorted(dupes))}")
    return Study(raw=raw, path=p, universe=universe)


def write_company(study: Study, ticker: str, name: str, fiscal_year_end: str,
                  brands: dict[str, list[str]], categories: list[str] | None = None) -> Path:
    """Create config/companies/<TICKER>.yaml. Refuses to overwrite."""
    ticker = ticker.upper()
    if ticker in study.tickers:
        raise ValueError(f"{ticker} is already in the universe ({study.company(ticker).source_file})")
    month, day = (int(x) for x in fiscal_year_end.split("-"))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        raise ValueError("fiscal year end must be MM-DD, e.g. 03-31")
    doc = {"ticker": ticker, "name": name, "fiscal_year_end": f"{month:02d}-{day:02d}",
           "brands": [{"name": b, "amazon_store_names": stores or [b]} for b, stores in brands.items()],
           "mcauley_categories": list(categories or [])}
    study.companies_dir.mkdir(parents=True, exist_ok=True)
    path = study.companies_dir / f"{ticker}.yaml"
    path.write_text("# One file per company. Add more with `reviewtone add-company`.\n"
                    "# amazon_store_names: exact Amazon `store` names (matched case-insensitively).\n"
                    "# mcauley_categories: Amazon Reviews 2023 categories searched by `reviewtone backfill`.\n"
                    + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return path
