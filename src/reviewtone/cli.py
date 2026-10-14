"""Command line: `reviewtone <command>`. Run `reviewtone -h` for help."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from .config import load_study
from .http import SourceUnavailable


def _factors():
    from .market.data import carhart_daily
    try:
        return carhart_daily()
    except SourceUnavailable as e:
        print(f"! Carhart factors unavailable ({e}); abnormal returns and portfolio alpha are skipped.",
              file=sys.stderr)
        return None


def load_dotenv(path: Path) -> None:
    """Read KEY=value lines from .env into the environment (existing variables win)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _tickers(study, args) -> list[str]:
    if getattr(args, "all", False):
        return study.tickers
    return [(args.ticker or study.primary_ticker).upper()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reviewtone", description=__doc__)
    ap.add_argument("--config", help="study YAML (default: config/study.yaml)")
    ap.add_argument("--as-of", help="treat this date (YYYY-MM-DD) as today")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def company_args(p, allow_all=True):
        p.add_argument("--ticker", help="default: the study's primary ticker")
        if allow_all:
            p.add_argument("--all", action="store_true", help="every company in the universe")

    p = sub.add_parser("add-company", help="add a company to the universe (config/companies/<TICKER>.yaml)")
    p.add_argument("ticker")
    p.add_argument("--brand", action="append", required=True,
                   help='brand sold on Amazon, optionally with its storefront names: "Shark=Shark,SharkNinja". '
                        "Repeat for several brands.")
    p.add_argument("--name", help="registrant name (default: looked up on SEC EDGAR)")
    p.add_argument("--fye", help="fiscal year end MM-DD (default: looked up on SEC EDGAR)")
    p.add_argument("--asins", nargs="*", default=[], help="optional seed ASINs written to seed_asins.csv")
    p.add_argument("--categories", nargs="*", default=[],
                   help="Amazon Reviews 2023 categories for backfill, e.g. Electronics Home_and_Kitchen")

    p = sub.add_parser("backfill", help="historical reviews (2018-Sep 2023) from Amazon Reviews 2023")
    company_args(p)
    p.add_argument("--categories", nargs="*", help="override the categories in the company config")

    p = sub.add_parser("verify", help="step 1: check CIK, name and fiscal year end on SEC EDGAR")
    company_args(p)

    p = sub.add_parser("catalog", help="step 2: build products.csv")
    company_args(p, allow_all=False)
    p.add_argument("--meta", nargs="*", default=[], help="McAuley meta_<Category>.jsonl(.gz) paths or URLs")

    p = sub.add_parser("fetch", help="step 3: collect reviews into reviews.csv (append-only)")
    company_args(p, allow_all=False)
    p.add_argument("--source", choices=["apify", "mcauley", "csv"], required=True)
    p.add_argument("--reviews", nargs="*", default=[], help="McAuley <Category>.jsonl(.gz) paths or URLs")
    p.add_argument("--csv", help="reviews CSV for --source csv")
    p.add_argument("--top", type=int, help="Apify: scrape only the N most-reviewed products (default: config)")
    p.add_argument("--max-reviews", type=int, default=100,
                   help="Apify: newest reviews per product (per star rating for star-filtered actors)")

    p = sub.add_parser("score", help="step 4: score tone of unscored reviews")
    company_args(p, allow_all=False)
    p.add_argument("--scorer", choices=["claude", "naive_bayes", "lexicon"])
    p.add_argument("--sample", type=int, help="score only a random sample (e.g. to label data for naive_bayes)")

    p = sub.add_parser("run", help="steps 1-7 for one company (or --all) and write the report")
    company_args(p)
    p.add_argument("--source", choices=["apify", "mcauley", "csv"], help="also fetch new reviews first")
    p.add_argument("--meta", nargs="*", default=[])
    p.add_argument("--reviews", nargs="*", default=[])
    p.add_argument("--csv")
    p.add_argument("--max-reviews", type=int, default=100, help="Apify: newest reviews per product (per star for star-filtered actors)")
    p.add_argument("--scorer", choices=["claude", "naive_bayes", "lexicon"])
    p.add_argument("--top", type=int, help="Apify: scrape only the N most-reviewed products (default: config)")
    p.add_argument("--new-study", action="store_true", help="re-register the study after changing config")

    sub.add_parser("portfolio", help="universe long/short portfolio and its Carhart alpha")

    p = sub.add_parser("update", help="weekly job: fetch new Apify reviews, rescore, re-evaluate, snapshot")
    p.add_argument("--max-reviews", type=int, default=30,
                   help="newest reviews per product (per star for star-filtered actors); older ones are stored")

    args = ap.parse_args(argv)
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    study = load_study(args.config)
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    from .pipeline import Run, run_company, run_universe

    if args.cmd == "add-company":
        return _add_company(study, args)
    if args.cmd == "backfill":
        from .pipeline import backfill
        for t, info in backfill(study, _tickers(study, args), args.categories).items():
            print(f"{t}: {info['products']} products, {info['reviews_added']} reviews added"
                  + (f", errors: {info['errors']}" if info["errors"] else ""))
        return 0
    if args.cmd == "verify":
        for t in _tickers(study, args):
            print(json.dumps(Run(study, study.company(t), as_of=as_of).verify(), indent=2, default=str))
    elif args.cmd == "catalog":
        products = Run(study, study.company(_tickers(study, args)[0]), as_of=as_of).build_catalog(args.meta)
        print(f"{len(products)} products")
    elif args.cmd == "fetch":
        run = Run(study, study.company(_tickers(study, args)[0]), as_of=as_of)
        print(f"{run.fetch_reviews(args.source, review_locations=args.reviews, csv_path=args.csv, max_reviews_per_product=args.max_reviews, top_products=args.top)} new reviews")
    elif args.cmd == "score":
        run = Run(study, study.company(_tickers(study, args)[0]), as_of=as_of, scorer=args.scorer)
        print(f"{len(run.score(sample=args.sample))} reviews have scores")
    elif args.cmd == "run":
        factors = _factors()
        for t in _tickers(study, args):
            card = run_company(study, t, source=args.source, factors=factors, meta=args.meta,
                               review_files=args.reviews, csv_path=args.csv, max_reviews=args.max_reviews,
                               scorer=args.scorer, new_study=args.new_study, as_of=as_of, top_products=args.top)
            print(json.dumps(card, indent=2, default=str))
    elif args.cmd == "portfolio":
        print(json.dumps(run_universe(study, _factors(), as_of), indent=2, default=str))
    elif args.cmd == "update":
        factors = _factors()
        for t in study.tickers:
            card = run_company(study, t, source="apify", factors=factors, max_reviews=args.max_reviews, as_of=as_of)
            print(f"{t}: {card.get('call', card.get('status'))}")
        run_universe(study, factors, as_of)
    return 0


def _add_company(study, args) -> int:
    from .config import write_company

    name, fye = args.name, args.fye
    if not (name and fye):
        from .fundamentals.sec import fetch_submissions, resolve_cik
        try:
            sub = fetch_submissions(resolve_cik(args.ticker))
        except (SourceUnavailable, KeyError) as e:
            print(f"Could not look up {args.ticker} on SEC EDGAR ({e}). Pass --name and --fye MM-DD.",
                  file=sys.stderr)
            return 2
        name = name or sub["name"]
        fye = fye or f"{sub['fiscalYearEnd'][:2]}-{sub['fiscalYearEnd'][2:]}"
    brands = {}
    for spec in args.brand:
        brand, _, stores = spec.partition("=")
        brands[brand.strip()] = [s.strip() for s in stores.split(",") if s.strip()]
    path = write_company(study, args.ticker, name, fye, brands, args.categories)
    print(f"wrote {path}")
    if args.asins:
        seed = study.ticker_dir(args.ticker) / "seed_asins.csv"
        seed.parent.mkdir(parents=True, exist_ok=True)
        first = next(iter(brands))
        seed.write_text("asin,brand\n" + "".join(f"{a},{first}\n" for a in args.asins))
        print(f"wrote {seed}")
    print(f"Next: reviewtone run --ticker {args.ticker.upper()} --source apify   (or --source mcauley ...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
