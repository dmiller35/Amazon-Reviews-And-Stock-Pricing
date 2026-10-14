# Amazon review tone → revenue & stock predictor

`reviewtone` turns Amazon US product reviews into a quarterly signal that tries to
anticipate a company's revenue and its stock's reaction to earnings. It implements:

- **Jin, "More than Words: Can Tone of Consumer Product Reviews Help Predict Firms'
  Fundamentals?"** (SSRN 4291019). Across 18.8M Amazon reviews of 292 firms
  (1996–2014), the change in review tone (*AbnormalTone*) predicted quarterly sales,
  earnings surprises, earnings-announcement returns and risk. A long/short portfolio
  sorted on it earned about 0.6% a quarter of Carhart alpha. Negative swings carried
  the signal; positive swings did not.
- **The 7-step "Amazon review research" prompt** from the TikTok: one company, a
  reproducible research folder, fixed rules, eight held-out quarters, misses shown,
  weekly updates, and honest limits.

> Not investment advice. This is a research tool. A single company over a few
> years cannot confirm the paper; the reports say so in every output.

## How the predictor works

1. **Signal.** Every review is scored for tone from −2 to +2 by Claude (Haiku 5.5 by
   default; `claude_model` in `config/study.yaml`), using a
   fixed rubric (`src/reviewtone/tone/rubric.py`). The star rating is hidden from
   the scorer, so tone is measured separately from rating, as in the paper. Claude
   also tags reasons (defect, quality decline, durability, price…) and flags
   seller/shipping-only reviews, which are left out of the tone average. For each
   fiscal quarter:
   `AbnormalTone = mean tone this quarter − mean of the prior four quarterly means`
   (on the paper's −1..1 scale).
2. **Call.** At quarter end, if `AbnormalTone ≤ −0.05` and the data is usable
   (enough reviews, a full baseline, complete coverage), the call is **slowdown**:
   revenue growth below its recent trend and a weak earnings-day reaction. Other
   quarters get no call. The thresholds are in `config/study.yaml`.
3. **Outcome.** Revenue comes from SEC XBRL, using the value as first reported.
   The call is a hit if the *growth surprise* (year-over-year growth minus its
   prior four-quarter average) is negative. The stock outcome is the Carhart
   abnormal return over days −1..+1 around the earnings 8-K (item 2.02).
4. **Universe portfolio** (`reviewtone portfolio`). At each calendar quarter end,
   firms are sorted on AbnormalTone. The strategy goes long the top group and short
   the bottom group, review-count and equal weighted, and the result is regressed
   on Carhart factors (the paper's Table 6). The latest sort is the live long/short
   list.

### Deliberate changes from the paper and the prompt

| Choice | Why |
|---|---|
| Claude rubric (−2..+2) instead of Naive Bayes; NB still available (`--scorer naive_bayes`) and trainable on Claude labels or your own hand-coded `labels.csv` | The paper's NB used 15k hand-coded summaries we don't have. The prompt asked for "Jev if available, otherwise Claude", and Jev has no public API, so the default is Claude. |
| Title **and** body scored (paper: title only); `include_body: false` restores the paper | Modern reviews are long and the body carries the defect detail ("double left click of doom"). |
| Single-company outcome is the **growth surprise**, not AbnormalSales | AbnormalSales is dominated by seasonality for one firm (Logitech's holiday quarter). The paper absorbed that with pooled fixed effects. AbnormalSales and YoY growth are still in `panel.csv`. |
| Only **negative** swings produce calls (`call_positive_swings: false`) | Tables 9–11: positive swings predicted nothing. |
| Revenue "as first reported", Q4 derived from the 10-K | No look-ahead from restatements. |
| No I/B/E/S consensus, so no earnings-surprise regression | Not freely available. The announcement return captures the market's surprise instead. |
| Per-company pre-registration lock | Companies can be added at any time, but once a company has results it can't be edited or dropped, and the rules can't change. That enforces "never select companies after results". |

## Setup (your computer, one command)

macOS or Linux (on Windows, use WSL). You need Python 3.10+ and git.

```bash
git clone https://github.com/dmiller35/Amazon-Reviews-And-Stock-Pricing.git
cd Amazon-Reviews-And-Stock-Pricing
git checkout claude/focused-pascal-8sss7m
bash scripts/setup.sh
```

`setup.sh` creates `.venv`, installs everything, asks for your Anthropic key,
Apify token and a name+email for SEC requests (hidden input, saved to `.env`,
which only you can read and git ignores), runs the tests, checks every data
source and verifies each company against its SEC filings. Re-run it at any time;
it keeps the existing `.env`. `reviewtone` loads `.env` by itself.

Then `source .venv/bin/activate` and:

## Running it

```bash
reviewtone backfill --all          # 1. history 2018-Sep 2023: products.csv + reviews.csv for every company
                                   #    (streams ~36 GB once; 30-90 min depending on bandwidth; free)
reviewtone fetch --ticker LOGI --source apify --max-reviews 300    # 2. recent reviews (Apify, paid:
                                   #    top 20 products x 300 reviews = 6,000 reviews, about $12)
reviewtone score --ticker LOGI --sample 200                        # 3. try Claude on 200 reviews, check cost
reviewtone run --ticker LOGI --scorer naive_bayes                  # 4. score the rest cheaply, then
                                   #    quarters, charts, SEC join, held-out eval, prediction, report
reviewtone run --all && reviewtone portfolio                       # 5. every company + long/short list
```

`backfill` only scans the categories listed in each company's file
(`mcauley_categories`), streaming each category once for all companies. The
prompt's step-by-step commands (`verify`, `catalog`, `fetch`, `score`) also work
on their own. Each step records what limited it, so a blocked source shows up in
the report's Limitations section and isn't papered over.

**Your first `run` locks the study:** the rules in `config/study.yaml` (including
`apify.top_products` and `product_scope`) and every company file. Decide on
thresholds, companies and how many products to follow before you run it.

Tone is measured on the **tracked products**: the ones followed live, i.e. each
company's 20 most-reviewed plus any in `seed_asins.csv`. That applies to every
quarter, historical and live, so AbnormalTone compares like with like when the
data switches from the historical dataset to the scraper. The historical
dataset ends in September 2023; until live data arrives, quarters from that
date on are marked as a coverage gap.

### History: live scraping vs. the McAuley dataset

Live reviews from Apify are what the current prediction runs on. The actor and its
input format are set in `config/apify.yaml`, so you can switch actors without code
changes. Pick a pay-per-result actor, not a monthly rental. The default is
[stanvanrooy6's Amazon Reviews Scraper (US)](https://apify.com/stanvanrooy6/amazon-review-scraper-us):
about $2 per 1,000 reviews, no Amazon login, up to ~1,000 reviews per product.
Check its Input tab once and fix the field names in `config/apify.yaml` if they
differ. Only each company's 20 most-reviewed products are scraped
(`apify.top_products` in `config/study.yaml`, or `--top`). `--max-reviews` is per
product: about 300 on the first fetch, and the weekly update uses 30. Rough cost:
companies × products × reviews × $0.002, so 7 × 20 × 30 is about $8 a week. Built-in profiles for
Axesso (star-filtered) and Junglee remain: set `actor:` to
`axesso_data~amazon-reviews-scraper` or `junglee~amazon-reviews-scraper` and
remove `input:`. The catch is depth: popular products still only reach back
weeks or months, and
AbnormalTone needs the four quarters before the signal quarter as its baseline.
When a fetch hits the per-product cap, any window between the reviews already
stored and the oldest one fetched is recorded as a gap in `coverage.csv`. Quarters
that overlap a gap, or whose baseline does, are marked `partial_coverage` /
`partial_baseline` and never used for calls. Fetches that overlap stored reviews,
like a normal weekly update, leave no gap. So history grows forward on its own,
and a large `--max-reviews` on the first fetch reaches as far back as the actor can.
For a multi-year backtest without waiting, use the
McAuley Lab **Amazon Reviews 2023** dataset (reviews through Sep 2023; the paper used
the same lab's earlier release):

```bash
reviewtone run --ticker LOGI --source mcauley \
  --meta    https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz \
  --reviews https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz
```

Then layer Apify on top for the quarters since. Both sources de-duplicate into the
same append-only `reviews.csv`. Scoring a whole category history with Claude is the
expensive part. The cheaper route is the paper's: label a sample with Claude
(`reviewtone score --sample 1500`), then `--scorer naive_bayes` trains on those
labels and scores everything (holdout accuracy goes to `models/naive_bayes.json`).

## Adding a company

```bash
reviewtone add-company GRMN --brand "Garmin=Garmin,Garmin Store"
reviewtone add-company HBB --brand "Hamilton Beach" --name "Hamilton Beach Brands" --fye 12-31 --asins B0XXXXXXX1
```

This writes `config/companies/<TICKER>.yaml`. The registrant name and fiscal year
end are looked up on SEC EDGAR unless you pass them. `--brand` takes the brand and,
after `=`, its exact Amazon storefront/`store` names (default: the brand name).
Repeat `--brand` for multi-brand companies. Optional `--asins` seeds
`research/<TICKER>/seed_asins.csv`. You can also just copy an existing file in
`config/companies/`.

The starting universe is the brands shown in the video: Logitech (LOGI),
Shark/Ninja (SN), OXO/Hydro Flask (HELE), Rubbermaid/Sharpie (NWL), Tide (PG),
Colgate (CL) and YETI. With fewer than 8 firms the portfolio uses halves instead of
quartiles, and alphas on a handful of quarters are noise. Adding companies helps
more than anything else.

## Weekly updates

`scripts/weekly_update.sh` (cron example inside) runs `reviewtone update`. It fetches
the newest reviews for every catalogued product, scores only reviews not yet scored,
re-evaluates, rebuilds reports, and snapshots `quarterly.csv`, `prediction.json`,
`evaluation.json` and `status.json` into `research/<TICKER>/history/<date>/`.
`reviews.csv` and `scores.csv` are append-only.

## Output folder (`research/<TICKER>/`)

| File | Contents |
|---|---|
| `README.md` | Prediction, data status (flags **INSUFFICIENT HISTORY**), charts, held-out table with every miss, sources, limitations |
| `products.csv` | ASIN, parent ASIN, title, store, URL, discovery source |
| `reviews.csv` / `coverage.csv` | Dated review text, IDs, ASINs, source URLs; per-product fetch depth and caps |
| `scores.csv`, `rubric.md`, `models/` | Tone, tags, seller/shipping flag, model and rubric version |
| `quarterly.csv` | Per fiscal quarter: counts, mean/sd tone and rating, tag shares, AbnormalTone/Rating/ReviewCnt, sparse/coverage flags |
| `fundamentals.csv` | SEC revenue (first reported, filing date), net income, assets, R&D, EPS, announcement dates, outcomes |
| `panel.csv`, `evaluation_h0.csv`, `evaluation_h1.csv`, `evaluation.json` | Signal↔outcome join, held-out quarters, hits and misses |
| `prediction.json` | The live call |
| `access_log.jsonl`, `status.json` | Every request and its status; what each step did or why it couldn't |

`research/preregistration.json` holds the locked rules and the companies with their
registration dates. `research/_universe/` holds the portfolio positions, returns,
alphas and the latest long/short list.

## Code map

```
config/study.yaml, config/companies/*.yaml   study settings, rules, universe
src/reviewtone/
  fiscal.py            fiscal calendars (FY named by end year; 52/53-week matching)
  sources/apify.py     Apify client: actor from config/apify.yaml, input templates, truncation tracking
  sources/mcauley.py   Amazon Reviews 2023 streaming filter by storefront
  catalog.py reviews.py   products.csv; review schema, de-dup, append-only store
  tone/                rubric, Claude scorer, Naive Bayes (paper), lexicon (tests)
  aggregate.py         quarterly measures and AbnormalTone family
  fundamentals/        SEC XBRL quarterly values, Q4 derivation, 8-K dates, outcomes
  market/              prices, Ken French factors, CAR/BHAR, portfolio alpha
  evaluate.py          pre-registration, fixed rules, held-out evaluation, Eq.1 OLS
  portfolio.py         calendar-time long/short (Table 6)
  predict.py report.py charts.py pipeline.py cli.py
```
