"""Paper-faithful scorer: Naive Bayes on review summaries (Jin 2022, Appendix C).

Jin hand-coded 15,000 summaries as positive / neutral / negative, trained
scikit-learn's Naive Bayes on 80%, checked the other 20% (<8% error), then
scored every review: positive -> (0, 1], negative -> [-1, 0), neutral -> 0.
Here the labelled sample is either a manual labels CSV or Claude rubric scores
collapsed to their sign.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ..http import utcnow
from .rubric import RUBRIC_VERSION


def review_text(df: pd.DataFrame, include_body: bool) -> pd.Series:
    title = df["title"].fillna("").astype(str)
    return (title + ". " + df["text"].fillna("").astype(str)) if include_body else title


class NaiveBayesToneModel:
    name = "naive_bayes"

    def __init__(self, include_body: bool = False):
        self.include_body = include_body
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("nb", MultinomialNB(alpha=0.5)),
        ])
        self.metrics: dict[str, Any] = {}

    def fit(self, reviews: pd.DataFrame, labels: pd.Series, seed: int = 7) -> dict[str, Any]:
        """labels: -1 / 0 / +1 aligned with `reviews`."""
        x = review_text(reviews, self.include_body)
        y = np.sign(labels.astype(float)).astype(int)
        stratify = y if y.value_counts().min() >= 2 else None
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=stratify)
        self.pipeline.fit(x_tr, y_tr)
        pred = self.pipeline.predict(x_te)
        classes = [-1, 0, 1]
        self.metrics = {
            "trained_at": utcnow(),
            "n_labelled": int(len(y)),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "test_accuracy": float(accuracy_score(y_te, pred)),
            "confusion_rows_true_cols_pred[-1,0,1]": confusion_matrix(y_te, pred, labels=classes).tolist(),
            "label_balance": {int(k): int(v) for k, v in y.value_counts().items()},
            "text": "title+body" if self.include_body else "title (review summary), as in the paper",
            "rubric_version": RUBRIC_VERSION,
        }
        self.pipeline.fit(x, y)  # final model uses every labelled review
        return self.metrics

    def predict_tone(self, reviews: pd.DataFrame) -> np.ndarray:
        x = review_text(reviews, self.include_body)
        proba = self.pipeline.predict_proba(x)
        classes = list(self.pipeline.classes_)
        best = proba.argmax(axis=1)
        tone = np.zeros(len(x))
        for i, (k, p) in enumerate(zip(best, proba.max(axis=1))):
            tone[i] = {1: p, -1: -p}.get(classes[k], 0.0)
        return tone

    def score(self, reviews: pd.DataFrame, progress: bool = False) -> pd.DataFrame:
        tone = self.predict_tone(reviews)
        return pd.DataFrame({
            "review_id": reviews["review_id"].values,
            "tone_raw": np.round(tone * 2).astype(int),
            "tone": tone,
            "tags": "",
            "seller_shipping_only": False,
            "reason": "",
            "scorer": self.name,
            "model": "sklearn MultinomialNB (tf-idf 1-2 grams)",
            "rubric_version": RUBRIC_VERSION,
            "scored_at": utcnow(),
        })

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "NaiveBayesToneModel":
        with path.open("rb") as f:
            return pickle.load(f)
