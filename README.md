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
| Labels come from a Claude (Haiku) rubric (−2..+2) on a sample of each company's reviews instead of hand coding; the paper's Naive Bayes learns from them and scores every review (`--scorer claude` scores everything with Claude instead) | The paper's NB used 15k hand-coded summaries we don't have, and Jev (the prompt's first choice) has no public API. Scoring years of history with Claude would cost far more than a labelled sample. |
| Title **and** body scored (paper: title only); `include_body: false` restores the paper | Modern reviews are long and the body carries the defect detail ("double left click of doom"). |
| Single-company outcome is the **growth surprise**, not AbnormalSales | AbnormalSales is dominated by seasonality for one firm (Logitech's holiday quarter). The paper absorbed that with pooled fixed effects. AbnormalSales and YoY growth are still in `panel.csv`. |
| Only **negative** swings produce calls (`call_positive_swings: false`) | Tables 9–11: positive swings predicted nothing. |
| Revenue "as first reported", Q4 derived from the 10-K | No look-ahead from restatements. |
| No I/B/E/S consensus, so no earnings-surprise regression | Not freely available. The announcement return captures the market's surprise instead. |
| Tone compared on **matched products** across a quarter and its baseline | Live scraping can't reach back to 2023 for busy products; comparing only products with complete reviews in both keeps a missing product from looking like a tone change. With complete history (the backtest) every tracked product is used, as in the paper. |
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

**No Linux machine, or WSL keeps crashing?** Use GitHub Codespaces: on the repo
page choose **Code → Codespaces → Create codespace on this branch**. The included
`.devcontainer` installs everything and runs the tests. Add `ANTHROPIC_API_KEY`,
`APIFY_TOKEN` and `SEC_USER_AGENT` as Codespaces secrets (GitHub → Settings →
Codespaces → Secrets), then run the commands below in its terminal.

## Running it

```bash
reviewtone backfill --all          # 1. history 2018-Sep 2023 (free; ~42 GB streamed, resumable)
reviewtone check-live --all        # 2. which top products are still sold (Apify, ~$0.05-0.15 per company)
reviewtone estimate --all          # 3. free: expected costs per company and when live calls can start
reviewtone score --all --scorer claude --sample 1000   # 4. Claude Haiku labels 1,000 reviews per company
                                   #    (Anthropic, ~$0.05 per company; running cost is printed)
reviewtone run --all               # 5. Naive Bayes scores everything (free), then quarters, charts,
                                   #    SEC revenue, held-out evaluation, prediction, report
reviewtone portfolio               # 6. long/short portfolio across companies (needs 4+ companies)
reviewtone update                  # 7. live reviews from Apify (first fetch, then weekly or every 4 weeks)
```

Tested end to end on 2026-10-15 with real data for five companies (Turtle Beach,
YETI, Corsair, Logitech, ACCO): every step above ran against the real sources.
The one exception was the Claude call itself, run through the real SDK against a local stand-in API
because no key was available in the test environment. Measured Apify prices: $0.0018 per review
returned; checks that return nothing are free; there is no per-run fee.

`backfill` only scans the categories listed in each company's file
(`mcauley_categories`), streaming each category once for all companies. It
saves after every category, so if it is interrupted (computer off, Codespace
stopped) just run the same command again: finished categories are skipped and
it picks up at the one it was on. `--restart` rescans everything. The
prompt's step-by-step commands (`verify`, `catalog`, `fetch`, `score`) also work
on their own. Each step records what limited it, so a blocked source shows up in
the report's Limitations section and isn't papered over.

**Your first `run` locks the study:** the rules in `config/study.yaml` (including
`apify.top_products` and `product_scope`) and every company file. Decide on
thresholds, companies and how many products to follow before you run it.

Tone is measured on the **tracked products**: the ones followed live, i.e. each
company's 10 most-reviewed live products plus any in `seed_asins.csv`. That applies to every
quarter, historical and live, so AbnormalTone compares like with like when the
data switches from the historical dataset to the scraper. The historical
dataset ends in September 2023; until live data arrives, quarters from that
date on are marked as a coverage gap.

### History: live scraping vs. the McAuley dataset

Live reviews from Apify are what the current prediction runs on. The actor and its
input format are set in `config/apify.yaml`, so you can switch actors without code
changes. Five actors were tested live on 2026-10-14 from a free Apify account:

| Actor | Result |
|---|---|
| **jdtpnjtp/amazon-reviews** (default) | Works: reads Amazon directly, no login, $0.0018 per review, newest first, past Amazon's 100-per-filter limit |
| stanvanrooy6/amazon-review-scraper-us | Reads a Woot mirror: months stale or empty for many best-sellers |
| axesso_data/amazon-reviews-scraper | "Usage only allowed for paying user" (finishes with zero results) |
| web_wanderer/amazon-reviews-extractor | US full-text reviews need a login; returned nothing |
| junglee/amazon-reviews-scraper | Works, but about $6 per 1,000 reviews on the free tier |

How it is used:
- **Each star rating is a separate request.** Asking for all stars at once silently
  skipped 1-star reviews in testing, and negative reviews are the signal.
- **A preflight check runs before every live fetch.** It fetches a known-active
  product (`check_asin`) and stops the run if nothing recent comes back, so a
  broken or paid-only scraper can't silently store empty results.
- **Products must still be collecting reviews.** Amazon moves reviews to newer
  listings, so a few-review probe checks each candidate's newest review. Listings
  with nothing in the last 120 days are marked `live=False` and replaced by the
  next most-reviewed product. Every tracked product therefore has both historical
  and current reviews.
- **Requests follow each product's review rate.** This actor has no date
  filter, so every run pays for the newest N reviews per star. N is sized to
  about 1.5× the reviews expected since the last fetch (minimum 2), and a star
  whose list fills before reaching the stored reviews is topped up. The first
  fetch reaches back to the 2023 history only for products where that can work;
  for the rest it collects the current quarter.
- **Only each company's 10 most-reviewed live products are followed**
  (`apify.top_products`).

**What live data costs and when it pays off.** Run `reviewtone estimate --all`
after `check-live`: it reads your own data and prints, per company, the first
live fetch, the monthly cost with weekly or 4-weekly updates, and the first
quarter a live call is possible. For the five test companies it estimated $4.90
for the first fetch, then about $7.40 a month with weekly updates or $3.40 every
4 weeks. Updating every 4 weeks catches the same reviews, because request sizes
follow the time since the last fetch, and it pays the per-product minimum less
often. Products with more than about 3 reviews a day at one star level should
stay weekly.

**Why live calls take about five quarters.** The scraper only returns each
product's newest ~100 reviews per star, and the free history ends in September
2023. A product busy enough to matter can't be bridged across that gap: its 100
newest five-star reviews may span only a few months. Such a product rejoins the
tone measure once five complete quarters (a signal quarter and its four-quarter
baseline) have been collected. Tone is compared on **matched products**: each
quarter is compared with its baseline using only products with complete reviews
in all five quarters, so a missing product never looks like a change in tone.
Quiet products that can be bridged are fetched back to 2023 (top-ups chase the
gap only when that can close it), but they rarely have enough reviews for a call on
their own. Expect the first live calls about five quarters after you start
collecting. Until then, the historical backtest and the portfolio are the results.

How gaps are recorded: when a fetch is cut off before it reaches the reviews
already stored, the window in between is recorded in `coverage.csv`, and a
product with a gap is left out of the quarters it touches. Weekly updates that
overlap what is stored leave no gap; if a week is busier than expected, that star
is topped up to the full 100 first.
For a multi-year backtest without waiting, use the
McAuley Lab **Amazon Reviews 2023** dataset (reviews through Sep 2023; the paper used
the same lab's earlier release):

```bash
reviewtone run --ticker LOGI --source mcauley \
  --meta    https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz \
  --reviews https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz
```

Then layer Apify on top for the quarters since. Both sources de-duplicate into the
same append-only `reviews.csv`. Scoring a whole category history with Claude would
be the expensive part, so the default is the paper's route: label a sample with
Claude (`reviewtone score --all --scorer claude --sample 1000`), and Naive Bayes
trains on those labels and scores everything (holdout accuracy goes to
`models/naive_bayes.json`). Run `check-live` before the first `run`: it fixes
which products are tracked, and those decide which reviews the backtest uses.

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

### The universe (15 companies)

**From the video (7):** Logitech (LOGI), SharkNinja (SN), Helen of Troy (OXO,
Hydro Flask; HELE), Newell (Rubbermaid, Sharpie; NWL), Procter & Gamble (Tide;
PG), Colgate-Palmolive (CL) and YETI.

**Added after research (8).** Each one's latest 10-K names Amazon as a major
customer, it sells under a few recognisable brands with their own Amazon
storefronts, and it has heavy review volume in the historical dataset:

| Ticker | Company | Amazon's share of sales (10-K) | Brands tracked |
|---|---|---|---|
| CRSR | Corsair Gaming | 27.4% (2025) | Corsair, Elgato, SCUF |
| HBB | Hamilton Beach Brands | 19% (2025) | Hamilton Beach, Proctor Silex |
| PBH | Prestige Consumer Healthcare | 15% (FY2026) | Summer's Eve, DenTek, Debrox, Monistat, Compound W, Dramamine, Clear Eyes, Fleet, Luden's, Chloraseptic |
| TBCH | Turtle Beach | 14–28% (2025; one of three largest customers) | Turtle Beach, PDP |
| ELF | e.l.f. Beauty | 11% (FY2026) | e.l.f., Naturium |
| MAT | Mattel | about 10% ($0.52B of 2025 sales) | Barbie, Fisher-Price, Hot Wheels, Mattel, Mega Bloks, UNO |
| ACCO | ACCO Brands | 10% (2025) | AT-A-GLANCE, Mead, Five Star, Swingline, Kensington, GBC, PowerA |
| BRBR | BellRing Brands | Walmart, Costco and Amazon together: 74% | Premier Protein, Dymatize |

Considered and left out: Simply Good Foods (Quest, Atkins). Amazon is 18% of its
sales, but its snack and shake listings turn over so fast that only 1 of its 60
most-reviewed 2023 products still collects reviews, so there's no continuous
product set to measure. Also Hasbro, Funko, GoPro, Sonos and Garmin (Amazon not
disclosed as a major customer), Lifetime Brands (licensed brands like KitchenAid
make storefronts ambiguous), JAKKS Pacific (mostly licensed character toys), and
Kimberly-Clark, Church & Dwight and Clorox (Amazon too small a share to matter).
With 15 firms the portfolio uses quartiles, as in the paper.

## Weekly updates

`scripts/weekly_update.sh` (cron example inside) runs `reviewtone update`. Run it weekly,
or every 4 weeks to roughly halve the Apify cost (see `reviewtone estimate`). It fetches
the newest reviews for every tracked product, scores only reviews not yet scored,
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
