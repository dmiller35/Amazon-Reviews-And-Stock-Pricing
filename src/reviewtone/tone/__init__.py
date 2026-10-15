"""Tone scoring with an append-only score store (scores.csv).

Scores are keyed by (review_id, scorer, rubric_version). Only reviews without a
valid score are sent to a scorer, so weekly updates pay for new reviews only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Study
from .rubric import RUBRIC, RUBRIC_VERSION

SCORE_COLUMNS = ["review_id", "tone_raw", "tone", "tags", "seller_shipping_only", "reason",
                 "scorer", "model", "rubric_version", "scored_at"]
MIN_NB_LABELS = 200


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SCORE_COLUMNS)
    return pd.read_csv(path, dtype={"review_id": str, "tags": str}, keep_default_na=False,
                       na_values={"tone": [""], "tone_raw": [""]})


def latest_valid(scores: pd.DataFrame, scorer: str) -> pd.DataFrame:
    s = scores[(scores["scorer"] == scorer) & (scores["rubric_version"] == RUBRIC_VERSION)].copy()
    s["tone"] = pd.to_numeric(s["tone"], errors="coerce")
    s = s.dropna(subset=["tone"])
    s["seller_shipping_only"] = s["seller_shipping_only"].astype(str).str.lower().eq("true")
    return s.drop_duplicates("review_id", keep="last")


def _naive_bayes(study: Study, ticker_dir: Path, reviews: pd.DataFrame, scores: pd.DataFrame):
    from .naive_bayes import NaiveBayesToneModel

    labels_csv = ticker_dir / "labels.csv"  # optional manual coding: review_id,label (-1/0/1)
    if labels_csv.exists():
        labels = pd.read_csv(labels_csv, dtype={"review_id": str})
        source = f"manual labels ({labels_csv.name})"
    else:
        claude = latest_valid(scores, "claude")
        labels = pd.DataFrame({"review_id": claude["review_id"], "label": np.sign(claude["tone_raw"].astype(float))})
        source = "Claude rubric scores collapsed to sign"
    train = reviews.merge(labels, on="review_id")
    if len(train) < MIN_NB_LABELS:
        raise RuntimeError(
            f"Naive Bayes needs >= {MIN_NB_LABELS} labelled reviews, found {len(train)}. Score a sample with "
            f"`reviewtone score --scorer claude --sample 1500` or provide {labels_csv}.")
    model = NaiveBayesToneModel(include_body=study.tone.get("include_body", False))
    metrics = model.fit(train, train["label"])
    metrics["label_source"] = source
    model.save(ticker_dir / "models" / "naive_bayes.pkl")
    (ticker_dir / "models" / "naive_bayes.json").write_text(json.dumps(metrics, indent=2))
    return model


def get_scorer(study: Study, name: str, ticker_dir: Path, reviews: pd.DataFrame, scores: pd.DataFrame):
    if name == "claude":
        from .claude import ClaudeToneScorer
        t = study.tone
        return ClaudeToneScorer(model=t["claude_model"], effort=t.get("claude_effort", "low"),
                                reviews_per_request=int(t.get("reviews_per_request", 25)),
                                include_body=bool(t.get("include_body", True)),
                                concurrency=int(t.get("concurrency", 4)))
    if name == "naive_bayes":
        return _naive_bayes(study, ticker_dir, reviews, scores)
    if name == "lexicon":
        from .lexicon import LexiconToneScorer
        return LexiconToneScorer()
    raise ValueError(f"unknown scorer {name!r} (claude, naive_bayes, lexicon)")


def write_rubric(ticker_dir: Path, scorer_name: str, model: str) -> None:
    (ticker_dir / "rubric.md").write_text(
        f"# Tone rubric {RUBRIC_VERSION}\n\nScorer: `{scorer_name}`  \nModel: `{model}`\n\n```\n{RUBRIC}```\n")


def score_reviews(study: Study, ticker_dir: Path, reviews: pd.DataFrame, scorer_name: str | None = None,
                  sample: int | None = None, seed: int = 7, block: int = 500) -> pd.DataFrame:
    """Score unscored reviews, append to scores.csv, return latest valid scores for the scorer."""
    name = scorer_name or study.tone["scorer"]
    path = ticker_dir / "scores.csv"
    scores = load_scores(path)
    scorer = get_scorer(study, name, ticker_dir, reviews, scores)
    model = getattr(scorer, "model", None)
    write_rubric(ticker_dir, name, model if isinstance(model, str) else name)

    if name == "naive_bayes":
        # Retrained on every run and cheap to apply, so its scores live in their
        # own file that is regenerated; scores.csv stays append-only.
        nb = scorer.score(reviews)[SCORE_COLUMNS]
        nb.to_csv(ticker_dir / "scores_naive_bayes.csv", index=False)
        return latest_valid(nb, name)

    done = set(latest_valid(scores, name)["review_id"])
    todo = reviews[~reviews["review_id"].isin(done)]
    if sample is not None and len(todo) > sample:
        todo = todo.sample(sample, random_state=seed)
    # Saved every `block` reviews, so an interrupted run keeps what it paid for and the
    # next run scores only what is left.
    for start in range(0, len(todo), block):
        new = scorer.score(todo.iloc[start:start + block])[SCORE_COLUMNS]
        new.to_csv(path, mode="a", header=not path.exists(), index=False)
        scores = pd.concat([scores, new], ignore_index=True)
    return latest_valid(scores, name)
