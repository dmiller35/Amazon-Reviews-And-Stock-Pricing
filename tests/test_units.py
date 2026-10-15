from datetime import date

import numpy as np
import pandas as pd
import pytest

from reviewtone import aggregate, reviews as rv
from reviewtone.evaluate import PreregistrationError, check_preregistration, make_call
from reviewtone.fiscal import FiscalCalendar
from reviewtone.fundamentals import sec
from reviewtone.fundamentals.outcomes import add_outcomes
from reviewtone.market.data import parse_french_csv
from reviewtone.market.models import announcement_return, portfolio_alpha
from reviewtone.sources.apify import normalize_items, parse_review_date
from reviewtone.tone.claude import ClaudeToneScorer

LOGI = FiscalCalendar(3)


def test_logitech_fiscal_calendar():
    assert LOGI.quarter("2026-03-15").label == "FY26Q4"
    assert LOGI.quarter("2025-04-01").label == "FY26Q1"
    assert LOGI.quarter("2025-12-31").label == "FY26Q3"
    q = LOGI.from_label("FY26Q4")
    assert (q.start, q.end) == (date(2026, 1, 1), date(2026, 3, 31))
    assert q.shift(1).label == "FY27Q1" and q.shift(-4).label == "FY25Q4"


def test_sec_fiscal_year_end_parsing():
    from reviewtone.fiscal import parse_sec_fiscal_year_end as p
    assert (p("0331"), p("0630"), p("1231"), p("0228"), p("0229")) == (3, 6, 12, 2, 2)
    assert p("0102") == 12 and p("1001") == 9 and p("0928") == 9  # 52/53-week year ends
    assert p("") is None and p("abc") is None


def test_period_end_matching_for_52_53_week_years():
    dec = FiscalCalendar(12)
    assert dec.quarter_for_period_end("2025-09-27").label == "FY25Q3"
    assert dec.quarter_for_period_end("2026-01-02").label == "FY25Q4"
    assert dec.quarter_for_period_end("2025-08-10") is None
    feb = FiscalCalendar(2)  # Helen of Troy
    assert feb.quarter("2024-02-29").label == "FY24Q4"
    assert feb.quarter("2024-03-01").label == "FY25Q1"


def test_dedupe_and_append_only(tmp_path):
    path = tmp_path / "reviews.csv"
    base = dict(ticker="LOGI", asin="A1", review_date="2026-01-05", rating=5, source="t")
    first = pd.DataFrame([{**base, "review_id": "1", "title": "Good", "text": "Works great for my daily use at work"}])
    assert len(rv.append_new(path, first)) == 1
    again = pd.DataFrame([
        {**base, "review_id": "1", "title": "Good", "text": "Works great for my daily use at work"},  # same id
        {**base, "review_id": "2", "title": "Good", "text": "Works great for my daily use at work"},  # same text
        {**base, "review_id": "3", "title": "Bad", "text": "Double click problem after six months"},
    ])
    added = rv.append_new(path, again)
    assert added["review_id"].tolist() == ["3"]
    assert rv.load(path)["review_id"].tolist() == ["1", "3"]


def test_apify_normalisation():
    items = [{"reviewId": "R1", "productAsin": "B003", "ratingScore": "2.0 out of 5 stars",
              "reviewTitle": "Modern Versions Are Inferior", "reviewDescription": "These used to be good.",
              "date": "Reviewed in the United States on September 22, 2026", "isVerified": True, "extra": 1},
             {"reviewId": "R2", "productAsin": "B003", "ratingScore": 5, "reviewTitle": "x", "reviewDescription": "y",
              "date": "Reviewed in Germany on 3 March 2026", "reviewedIn": "Germany"}]
    df, unmapped = normalize_items(items, ticker="LOGI", brand="Logitech", fetched_at="now")
    assert df["review_id"].tolist() == ["R1"]
    assert df.loc[0, "review_date"] == "2026-09-22" and df.loc[0, "rating"] == 2.0
    assert unmapped == ["extra"]
    assert parse_review_date(1_700_000_000_000) == "2023-11-14"


def test_abnormal_tone_and_flags():
    rows = []
    for k, tone in enumerate([0.5, 0.5, 0.5, 0.5, 0.3]):
        q = LOGI.from_label("FY25Q1").shift(k)
        for i in range(40):
            rows.append({"review_id": f"{k}-{i}", "review_date": q.start.isoformat(), "rating": 4.0,
                         "tone": tone, "tags": "defect" if tone < 0.5 else "", "seller_shipping_only": False})
    rows.append({"review_id": "ship", "review_date": "2024-04-02", "rating": 1.0, "tone": -1.0, "tags": "",
                 "seller_shipping_only": True})
    q = aggregate.quarterly_measures(pd.DataFrame(rows), LOGI, min_reviews=30, as_of=date(2025, 6, 30))
    last = q.iloc[-1]
    assert last["fiscal_quarter"] == "FY26Q1"
    assert last["AbnormalTone"] == pytest.approx(-0.2)
    assert last["Negative_AbnormalTone"] == pytest.approx(-0.2) and pd.isna(last["Positive_AbnormalTone"])
    assert last["share_defect"] == 1.0
    assert q.iloc[0]["n_seller_shipping_only"] == 1 and q.iloc[0]["mean_tone"] == 0.5
    assert q["usable"].tolist() == [False] * 4 + [True]
    assert make_call(-0.2, True, {"negative_swing_threshold": -0.05}) == "slowdown"
    assert make_call(0.2, True, {"negative_swing_threshold": -0.05, "positive_swing_threshold": 0.05}) == "no_call"


def test_sec_first_reported_and_q4_derivation(world):
    _, facts, subs, quarters, _ = world
    rev = sec.flow_quarters(facts, sec.REVENUE_CONCEPTS, LOGI)
    q2 = rev[rev["fiscal_quarter"] == quarters[5].label].iloc[0]
    q4 = rev[rev["fiscal_quarter"] == quarters[7].label].iloc[0]
    assert not q2["derived"] and q4["derived"]
    # restated (1.5x, filed a year later) values are ignored
    restated = [u for u in facts["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"]
                if u["accn"] == f"r{quarters[5].index}"][0]
    assert q2["value"] == pytest.approx(restated["val"] / 1.5)
    ann = sec.earnings_announcements(subs, LOGI)
    assert ann.iloc[0]["fq_index"] == quarters[0].index


def test_growth_surprise_outcome():
    f = pd.DataFrame({"fq_index": range(10), "revenue": [100, 100, 100, 100, 110, 110, 110, 110, 121, 115.5]})
    f[["net_income", "assets", "rnd"]] = np.nan
    out = add_outcomes(f)
    assert out.loc[8, "SameQtrSalesGrowth"] == pytest.approx(0.10)
    assert out.loc[9, "GrowthSurprise"] == pytest.approx(0.05 - 0.10)


def test_preregistration_lock(study):
    check_preregistration(study)
    check_preregistration(study)  # unchanged: fine
    study.raw["rules"]["negative_swing_threshold"] = -0.2
    with pytest.raises(PreregistrationError):
        check_preregistration(study)
    check_preregistration(study, new_study=True)


def test_french_parser_and_carhart_models():
    text = "This file was created\n\n,Mkt-RF,SMB,HML,RF\n20240102,  1.00, 0.10,-0.20, 0.02\n20240103,-0.50, 0.00, 0.10, 0.02\n\nCopyright\n"
    df = parse_french_csv(text)
    assert list(df.columns) == ["mkt_rf", "smb", "hml", "rf"] and df.iloc[0]["mkt_rf"] == 0.01

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-02", periods=400)
    fac = pd.DataFrame(rng.normal(0, 0.01, (400, 4)), index=idx, columns=["mkt_rf", "smb", "hml", "umd"])
    fac["rf"] = 0.0001
    ret = fac["rf"] + 1.2 * fac["mkt_rf"] + rng.normal(0, 0.002, 400)
    event = idx[300]
    ret.loc[event] += 0.05  # announcement jump
    ev = announcement_return(ret, fac, event.strftime("%Y-%m-%d"))
    assert ev.car == pytest.approx(0.05, abs=0.01)

    q = fac.resample("QE").apply(lambda s: (1 + s).prod() - 1)
    port = 0.02 + 1.0 * q["mkt_rf"] + rng.normal(0, 1e-4, len(q))
    assert portfolio_alpha(port, q)["n_quarters"] == len(q)


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Resp:
    def __init__(self, text, stop="end_turn"):
        self.content, self.stop_reason = [_Block(text)], stop


class _FakeClient:
    def __init__(self):
        self.calls = 0
        self.beta = self
        self.messages = self

    def create(self, **kw):
        import json
        self.calls += 1
        if kw["model"].startswith("claude-haiku"):
            assert "fallbacks" not in kw, "Haiku has no server-side refusal fallback"
        items = json.loads(kw["messages"][0]["content"])
        assert all("rating" not in it for it in items), "star rating must be hidden from the scorer"
        scores = [{"id": it["id"], "tone": -2 if "broke" in it["title"].lower() else 1,
                   "tags": ["defect"] if "broke" in it["title"].lower() else ["performance_praise"],
                   "seller_shipping_only": False, "reason": "x"} for it in items if it["id"] != "skip"]
        return _Resp(json.dumps({"scores": scores}))


def test_claude_scorer_parsing():
    client = _FakeClient()
    s = ClaudeToneScorer(model="claude-haiku-5-5", client=client, reviews_per_request=2)
    df = pd.DataFrame({"review_id": ["a", "b", "skip"], "title": ["Broke fast", "Nice", "x"], "text": ["", "", ""],
                       "rating": [1, 5, 3]})
    out = s.score(df, progress=False)
    assert client.calls == 2
    assert out.set_index("review_id")["tone"].to_dict()["a"] == -1.0
    assert out.set_index("review_id").loc["b", "tags"] == "performance_praise"
    assert out.set_index("review_id").loc["skip", "reason"].startswith("UNSCORED")


def test_add_company_and_per_company_lock(study):
    from reviewtone.cli import main
    from reviewtone.config import load_study

    check_preregistration(study)
    assert main(["--config", str(study.path), "add-company", "GRMN", "--brand", "Garmin=Garmin,Garmin Store",
                 "--name", "Garmin Ltd.", "--fye", "12-31", "--asins", "B0TEST0001"]) == 0
    s2 = load_study(study.path)
    g = s2.company("GRMN")
    assert g.brands[0].amazon_store_names == ("Garmin", "Garmin Store") and g.fiscal_year_end == "12-31"
    reg = check_preregistration(s2)  # adding is allowed
    assert "GRMN" in reg["companies"]
    (s2.companies_dir / "YETI.yaml").unlink()  # removing a registered company is not
    with pytest.raises(PreregistrationError):
        check_preregistration(load_study(study.path))


def test_coverage_gaps_only_flag_missing_windows(study):
    from reviewtone.pipeline import Run

    run = Run(study, study.company("LOGI"))
    capped = pd.DataFrame({"review_date": pd.date_range("2026-05-01", periods=100, freq="D").strftime("%Y-%m-%d")})
    # Historical data ends 2023-09-28, first live fetch is capped at 100 reviews starting 2026-05-01: a gap.
    run._record_coverage("A1", capped, True, "2026-05-01", stored_newest="2023-09-28")
    # Next week's capped fetch overlaps what is stored: no new gap.
    run._record_coverage("A1", capped.iloc[10:].reset_index(drop=True), True, "2026-05-11", stored_newest="2026-08-08")
    # A product with no history whose fetch did not hit the cap is complete: no gap.
    run._record_coverage("B2", capped.iloc[:20], False, None, stored_newest=None)
    gaps = aggregate.coverage_gaps(pd.read_csv(run.dir / "coverage.csv"))
    assert gaps == [(date(2023, 9, 28), date(2026, 5, 1))]

    rows = [{"review_id": f"{d}-{i}", "review_date": d, "rating": 4.0, "tone": 0.5, "tags": "",
             "seller_shipping_only": False}
            for d in pd.date_range("2022-01-15", "2027-12-15", freq="MS").strftime("%Y-%m-%d") for i in range(15)]
    q = aggregate.quarterly_measures(pd.DataFrame(rows), LOGI, min_reviews=30, as_of=date(2027, 12, 31), gaps=gaps)
    flags = q.set_index("fiscal_quarter")
    assert not flags.loc["FY24Q1", "partial_coverage"]          # Apr-Jun 2023: before the gap
    assert flags.loc["FY24Q2", "partial_coverage"]              # Jul-Sep 2023: gap starts
    assert flags.loc["FY27Q1", "partial_coverage"]              # Apr-Jun 2026: gap ends inside
    assert not flags.loc["FY27Q2", "partial_coverage"]          # Jul-Sep 2026: complete again
    assert flags.loc["FY28Q1", "partial_baseline"]              # baseline still reaches into the gap
    assert flags.loc["FY28Q3", "usable"]                        # four clean quarters later: callable


def test_axesso_profile_star_filters_and_truncation():
    from reviewtone.sources.apify import BUILTIN, completeness

    prof = BUILTIN["axesso_data~amazon-reviews-scraper"]
    inp = prof.payloads("B003NR57BY", 100)[0]["input"]
    assert [i["filterByStar"] for i in inp] == ["one_star", "two_star", "three_star", "four_star", "five_star"]
    assert all(i["maxPages"] == 10 and i["sortBy"] == "recent" and i["domainCode"] == "com" for i in inp)
    assert prof.cap(100) == 100 and prof.cap(30) == 30

    # 5-star list hit its 100-review cap reaching back to June; 1-star list is complete back to 2024.
    five = pd.DataFrame({"rating": 5.0, "review_date": pd.date_range("2026-06-01", periods=100).strftime("%Y-%m-%d")})
    one = pd.DataFrame({"rating": 1.0, "review_date": ["2024-01-05", "2026-09-01"]})
    hit, complete_from = completeness(pd.concat([five, one]), prof.cap(100), by_star=True)
    assert hit and complete_from == "2026-06-01"
    assert completeness(one, 100, by_star=True) == (False, None)


def test_non_us_reviews_dropped_by_date_line():
    items = [{"reviewId": "a", "asin": "X", "rating": "5.0 out of 5 stars", "title": "t", "text": "x",
              "date": "Reviewed in the United States on May 2, 2026"},
             {"reviewId": "b", "asin": "X", "rating": "1.0 out of 5 stars", "title": "t", "text": "y",
              "date": "Reviewed in the United Kingdom on May 2, 2026"}]
    df, _ = normalize_items(items, ticker="LOGI", brand="Logitech", fetched_at="now")
    assert df["review_id"].tolist() == ["a"]


def test_configured_actor_template(tmp_path, monkeypatch):
    from reviewtone.sources.apify import CONFIG, resolve_profile

    monkeypatch.delenv("APIFY_REVIEWS_ACTOR", raising=False)
    name, prof = resolve_profile(config=CONFIG)  # the shipped config parses
    assert name == "jdtpnjtp~amazon-reviews" and prof.by_star
    runs = prof.payloads("B0TEST", 100, "2026-09-01")
    assert [r["starFilter"] for r in runs] == ["one_star", "two_star", "three_star", "four_star", "five_star"]
    assert runs[0] == {"products": ["B0TEST"], "country": "us", "sort": "recent", "starFilter": "one_star",
                       "maxReviewsPerAsin": 100, "includeTopReviewsFallback": False, "stopOnError": False}
    assert prof.cap(100) == 100

    cfg = tmp_path / "apify.yaml"
    cfg.write_text("actor: someone/cheap-reviews\nsplit_by_star: false\ncap: 100\n"
                   "input: {asin: '{asin}', limit: '{max}', note: 'asin {asin}'}\n")
    name, prof = resolve_profile(config=cfg)
    assert name == "someone~cheap-reviews" and not prof.by_star
    assert prof.payloads("A1", 50) == [{"asin": "A1", "limit": 50, "note": "asin A1"}]
    assert resolve_profile("junglee/amazon-reviews-scraper", cfg)[0] == "junglee~amazon-reviews-scraper"
    with pytest.raises(ValueError):
        resolve_profile("unknown/actor", cfg)


def test_jdtpnjtp_output_maps():
    item = {"asin": "B07SGPJBZ6", "author": "x", "rating": "1", "title": "Lid leaks", "text": "Leaks every time",
            "date": "October 10, 2026", "country": "the United States", "verified_purchase": "True", "vine": "False",
            "helpful_count": "0", "images": "[]", "image_url": "None", "data": "{}"}
    df, unmapped = normalize_items([item], ticker="YETI", brand="YETI", fetched_at="now")
    assert unmapped == [] and df.loc[0, "review_date"] == "2026-10-10" and df.loc[0, "rating"] == 1.0
    uk = dict(item, country="the United Kingdom")
    assert normalize_items([uk], ticker="YETI", brand="YETI", fetched_at="")[0].empty


def test_preflight_stops_on_silent_empty_actor(monkeypatch):
    from reviewtone.sources import apify

    monkeypatch.setenv("APIFY_TOKEN", "t")
    client = apify.ApifyReviews()
    monkeypatch.setattr(client, "run", lambda payload, **kw: [])  # paid-only actor: "succeeds" with nothing
    with pytest.raises(apify.SourceUnavailable, match="Preflight failed"):
        apify.preflight(client, "2026-10-14")
    monkeypatch.setattr(client, "run", lambda payload, **kw: [{"asin": "A", "rating": 5, "title": "t", "text": "x",
                                                               "date": "October 10, 2026"}])
    assert apify.preflight(client, "2026-10-14") == "2026-10-10"


def test_real_actor_output_maps():
    item = {"review_id": "amz-us-ca3f", "id_type": "content_fingerprint", "amazon_review_id": None,
            "asin": "B0BN7M8CH4", "marketplace": "amazon.com", "rating": 1,
            "title": "attachments are poorly designed", "body": "detachable wands are impossible to remove",
            "published_date": "2026-10-12", "published_date_raw": "Reviewed in the United States on October 12, 2026",
            "date_status": "parsed", "review_origin_country": "the United States", "verified_purchase": True,
            "vine_review": False, "helpful_votes": 0, "author": None, "source": "woot_amazon_us_reviews",
            "source_url": "https://www.woot.com/review/Reviews/B0BN7M8CH4", "product_url": "x", "scraped_at": "x",
            "schema_version": 1, "images": [], "videos": []}
    df, unmapped = normalize_items([item], ticker="SN", brand="Ninja", fetched_at="now")
    r = df.iloc[0]
    assert unmapped == []
    assert (r.review_id, r.rating, r.text, r.review_date, r.verified) == (
        "amz-us-ca3f", 1.0, "detachable wands are impossible to remove", "2026-10-12", True)
    assert r.source == "apify:woot_amazon_us_reviews"
    empty, _ = normalize_items([], ticker="SN", brand="Ninja", fetched_at="now")  # no reviews: still has columns
    assert empty.empty and {"asin", "parent_asin", "review_date"} <= set(empty.columns)

def test_resumable_download_survives_dropped_connection(monkeypatch):
    import gzip
    import io as _io
    import json as _json

    from reviewtone.sources import mcauley

    body = gzip.compress(("\n".join(_json.dumps({"i": i}) for i in range(5000)) + "\n").encode())
    calls = []

    class Raw:
        def __init__(self, data, fail_after):
            self.buf, self.fail_after, self.sent = _io.BytesIO(data), fail_after, 0

        def readinto(self, b):
            if self.fail_after is not None and self.sent >= self.fail_after:
                raise ConnectionResetError(104, "Connection reset by peer")
            n = self.buf.readinto(memoryview(b)[: min(len(b), 4096)])
            self.sent += n
            return n

    class Resp:
        def __init__(self, status, data, fail_after):
            self.status_code, self.raw = status, Raw(data, fail_after)

        def close(self):
            pass

    def fake_get(url, stream, timeout, headers):
        start = int(headers["Range"].split("=")[1].rstrip("-")) if headers else 0
        calls.append(start)
        fail = len(body) // 3 if len(calls) == 1 else None  # first connection drops a third of the way in
        return Resp(206 if start else 200, body[start:], fail)

    monkeypatch.setattr(mcauley.requests, "get", fake_get)
    monkeypatch.setattr(mcauley.time, "sleep", lambda s: None)
    rows = list(mcauley.iter_jsonl("https://example.org/x.jsonl.gz"))
    assert [r["i"] for r in rows] == list(range(5000))
    assert len(calls) == 2 and calls[1] > 0


def test_apify_run_polls_retries_and_flags_delisted(monkeypatch):
    import requests as _rq

    from reviewtone.sources import apify

    monkeypatch.setattr(apify.time, "sleep", lambda s: None)
    state = {"post": 0, "drops": 1}

    class R:
        def __init__(self, code, data):
            self.status_code, self._d, self.text = code, data, ""

        def json(self):
            return self._d

    def fake_request(method, url, params, timeout, **kw):
        if state["drops"]:  # first call: connection reset, must be retried
            state["drops"] -= 1
            raise _rq.ConnectionError("Connection reset by peer")
        if method == "POST":
            state["post"] += 1
            return R(201, {"data": {"id": f"run{state['post']}", "status": "RUNNING", "defaultDatasetId": "d"}})
        if "/actor-runs/" in url:
            ok = url.endswith("run1")
            return R(200, {"data": {"id": url[-4:], "status": "SUCCEEDED" if ok else "FAILED", "defaultDatasetId": "d"}})
        return R(200, [{"review_id": "x", "asin": "A", "rating": 4, "title": "t", "body": "b",
                        "published_date": "2026-10-01"}])

    monkeypatch.setattr(apify.requests, "request", fake_request)
    monkeypatch.setenv("APIFY_TOKEN", "t")
    client = apify.ApifyReviews(actor="junglee/amazon-reviews-scraper")
    assert client.run({"x": 1}, asin="A")[0]["review_id"] == "x"
    with pytest.raises(apify.ProductUnavailable):
        client.run({"x": 1}, asin="DEAD")


def test_request_sizes_follow_review_rate(study):
    from reviewtone.pipeline import Run
    from reviewtone.sources.apify import BUILTIN, completeness, resolve_profile

    run = Run(study, study.company("YETI"), as_of=date(2026, 10, 14))
    days = pd.date_range("2026-04-16", "2026-10-06", freq="D")  # ~2 five-star a day, 1 one-star a week
    rows = [{"rating": 5, "review_date": d.strftime("%Y-%m-%d")} for d in days for _ in range(2)]
    rows += [{"rating": 1, "review_date": d.strftime("%Y-%m-%d")} for d in days[::7]]
    mine = pd.DataFrame(rows).assign(source="apify:x", fetched_at="2026-10-07T06:00:00Z")
    sizes = run._request_sizes(mine, 100)
    assert 25 <= sizes[5] <= 35       # ~9 days x 2/day x 1.5 + margin
    assert sizes[1] == 5 and sizes[3] == 5   # rare stars: minimum request
    assert run._request_sizes(mine.iloc[0:0], 100) == 100  # first live fetch: full depth
    # a truncated list (20 five-star reviews in the last 5 days) must read as ~4/day, not 20/180
    busy = pd.DataFrame({"rating": [5] * 20 + [1] * 20,
                         "review_date": [f"2026-10-0{3 + i % 5}" for i in range(20)]
                         + pd.date_range("2025-01-01", periods=20, freq="20D").strftime("%Y-%m-%d").tolist()})
    busy = busy.assign(source="apify:x", fetched_at="2026-10-07T06:00:00Z")
    assert run._request_sizes(busy, 100)[5] >= 40

    _, prof = resolve_profile()
    reqs = prof.payloads("A", sizes)
    assert [r["maxReviewsPerAsin"] for r in reqs] == [sizes[i] for i in range(1, 6)]
    full5 = pd.DataFrame({"rating": [5.0] * sizes[5], "review_date": ["2026-10-08"] * sizes[5]})
    assert completeness(full5, prof.cap(sizes), by_star=True) == (True, "2026-10-08")  # filled list -> truncated
    assert BUILTIN["junglee~amazon-reviews-scraper"].cap(50) == 50


def test_apify_402_stops_with_clear_message(monkeypatch):
    from reviewtone.sources import apify

    class R:
        status_code, text = 402, '{"error": {}}'

    monkeypatch.setattr(apify.requests, "request", lambda *a, **k: R())
    monkeypatch.setattr(apify.time, "sleep", lambda s: None)
    monkeypatch.setenv("APIFY_TOKEN", "t")
    with pytest.raises(apify.SourceUnavailable, match="unreachable: HTTP 402"):
        apify.ApifyReviews().run({"x": 1}, asin="A")


def test_claude_scorer_through_real_sdk_http(monkeypatch):
    """Real anthropic SDK over HTTP against a local stand-in API: request shape, parsing, usage, cost."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import anthropic

    from reviewtone.tone.claude import ClaudeToneScorer

    seen = []

    class FakeAPI(BaseHTTPRequestHandler):
        def do_POST(self):
            body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            items = _json.loads(body["messages"][0]["content"])
            scores = [{"id": it["id"], "tone": -2 if "broke" in it["text"] else 2, "tags": ["defect"],
                       "seller_shipping_only": False, "reason": "x"} for it in items]
            out = _json.dumps({"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                               "content": [{"type": "text", "text": _json.dumps({"scores": scores})}],
                               "stop_reason": "end_turn", "stop_sequence": None,
                               "usage": {"input_tokens": 1000, "output_tokens": 400}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = anthropic.Anthropic(api_key="test", base_url=f"http://127.0.0.1:{server.server_port}", max_retries=0)
        scorer = ClaudeToneScorer(model="claude-haiku-5-5", effort="low", reviews_per_request=2, client=client)
        reviews = pd.DataFrame({"review_id": ["a", "b", "c"], "title": ["Great", "Bad", "Ok"],
                                "text": ["love it", "it broke in a week", "fine"]})
        out = scorer.score(reviews, progress=False)
    finally:
        server.shutdown()
    path, body = seen[0]
    assert path == "/v1/messages" and len(seen) == 2  # 3 reviews in batches of 2
    assert body["model"] == "claude-haiku-5-5" and "fallbacks" not in body
    assert body["output_config"]["effort"] == "low" and body["output_config"]["format"]["type"] == "json_schema"
    assert "rating" not in _json.dumps(body["messages"])  # star rating never shown to the scorer
    assert out.set_index("review_id")["tone_raw"].to_dict() == {"a": 2, "b": -2, "c": 2}
    assert scorer.tokens == {"input": 2000, "output": 800}
    assert abs(scorer.cost() - (2000 * 0.10 + 800 * 0.50) / 1e6) < 1e-12
