"""Claude rubric scorer: tone -2..+2, reason tags, seller/shipping-only flag."""

from __future__ import annotations

import json
from typing import Any

import anthropic
import pandas as pd

from ..http import utcnow
from .rubric import OUTPUT_SCHEMA, RUBRIC, RUBRIC_VERSION, TAGS

SYSTEM = (
    "You are a careful annotator scoring Amazon US product reviews for an academic study of "
    "whether review tone anticipates company revenue.\n\n" + RUBRIC +
    "\nYou receive a JSON list of reviews. Return exactly one score object per review id."
)


class ClaudeScorerError(RuntimeError):
    pass


class ClaudeToneScorer:
    name = "claude"

    def __init__(self, model: str, effort: str = "low", reviews_per_request: int = 25,
                 include_body: bool = True, client: Any | None = None):
        self.model = model
        self.effort = effort
        self.batch = reviews_per_request
        self.include_body = include_body
        self.client = client or anthropic.Anthropic(max_retries=5)

    def _request(self, items: list[dict[str, str]]) -> Any:
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            # Server-side fallback: if a classifier declines, the API re-runs the
            # request on a fallback model inside the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    def _score_chunk(self, chunk: pd.DataFrame) -> list[dict[str, Any]]:
        items = [
            {"id": r.review_id, "title": r.title, **({"text": r.text} if self.include_body else {})}
            for r in chunk.itertuples()
        ]
        try:
            resp = self._request(items)
        except anthropic.AuthenticationError as e:
            raise ClaudeScorerError("Claude API authentication failed; set ANTHROPIC_API_KEY.") from e
        except anthropic.APIConnectionError as e:
            raise ClaudeScorerError(f"Claude API unreachable: {e}") from e

        if resp.stop_reason == "max_tokens" and len(chunk) > 1:
            half = len(chunk) // 2
            return self._score_chunk(chunk.iloc[:half]) + self._score_chunk(chunk.iloc[half:])
        if resp.stop_reason == "refusal":
            return [self._unscored(rid, "refusal") for rid in chunk["review_id"]]

        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            parsed = json.loads(text)["scores"]
        except (json.JSONDecodeError, KeyError):
            return [self._unscored(rid, "unparseable response") for rid in chunk["review_id"]]

        by_id = {str(s["id"]): s for s in parsed}
        out = []
        for rid in chunk["review_id"]:
            s = by_id.get(str(rid))
            if s is None or s.get("tone") not in (-2, -1, 0, 1, 2):
                out.append(self._unscored(rid, "missing from response"))
                continue
            out.append({
                "review_id": rid,
                "tone_raw": int(s["tone"]),
                "tone": s["tone"] / 2.0,
                "tags": ";".join(t for t in s.get("tags", []) if t in TAGS),
                "seller_shipping_only": bool(s.get("seller_shipping_only", False)),
                "reason": str(s.get("reason", ""))[:200],
            })
        return out

    def _unscored(self, rid: str, why: str) -> dict[str, Any]:
        return {"review_id": rid, "tone_raw": None, "tone": None, "tags": "", "seller_shipping_only": False,
                "reason": f"UNSCORED: {why}"}

    def score(self, reviews: pd.DataFrame, progress: bool = True) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for start in range(0, len(reviews), self.batch):
            rows.extend(self._score_chunk(reviews.iloc[start:start + self.batch]))
            if progress:
                print(f"  scored {min(start + self.batch, len(reviews))}/{len(reviews)} reviews", flush=True)
        out = pd.DataFrame(rows)
        out["scorer"], out["model"], out["rubric_version"], out["scored_at"] = (
            self.name, self.model, RUBRIC_VERSION, utcnow())
        return out
