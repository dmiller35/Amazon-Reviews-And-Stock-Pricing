"""Tone rubric shared by every scorer. Saved alongside results (step 4)."""

RUBRIC_VERSION = "2026-10-v1"

TAGS = [
    "defect",               # product malfunctioned, arrived broken, stopped working
    "quality_decline",      # explicitly worse than an earlier version / previous unit
    "durability",           # wore out or failed after a period of normal use
    "price_value",          # price, value for money, price changes
    "usability",            # ergonomics, setup, comfort, ease of use
    "compatibility_software",  # drivers, apps, firmware, device compatibility
    "customer_service",     # warranty, support, returns handled by the manufacturer
    "performance_praise",   # product works well / exceeds expectations
    "shipping_seller",      # delivery, packaging, seller or Amazon logistics
    "other",
]

RUBRIC = """\
Score the TONE of each Amazon customer review toward the PRODUCT on an integer scale:

+2  Strongly positive. Clear enthusiasm or praise; no meaningful complaint.
+1  Positive overall, possibly with minor reservations.
 0  Neutral or evenly mixed, or no opinion about the product itself.
-1  Negative overall: disappointment or problems, though some positives remain.
-2  Strongly negative. Product failed, was defective or unusable, the reviewer
    returned it or warns others away.

Judge the words only. The star rating is deliberately hidden: tone is measured
separately from the rating.

Also return:
- tags: every reason that drives the tone, from this list:
  defect, quality_decline, durability, price_value, usability,
  compatibility_software, customer_service, performance_praise,
  shipping_seller, other.
  Use quality_decline only when the reviewer compares unfavourably with an
  earlier version, an older unit, or how the product "used to be".
- seller_shipping_only: true when the review is ONLY about delivery, packaging,
  the seller, Amazon logistics or receiving the wrong item, and says nothing
  about the product itself. These reviews are excluded from firm tone.
- reason: at most 12 words explaining the score.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "tone": {"type": "integer", "enum": [-2, -1, 0, 1, 2]},
                    "tags": {"type": "array", "items": {"type": "string", "enum": TAGS}},
                    "seller_shipping_only": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "tone", "tags", "seller_shipping_only", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}
