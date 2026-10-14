"""Fiscal calendars: map review dates and SEC period ends to fiscal quarters.

A fiscal year is named by the calendar year in which it ends (Logitech's FY2026
runs April 2025 - March 2026). Quarters end on calendar month ends; companies on
52/53-week years (period ends like 2025-09-27) are matched to the nearest
nominal quarter end.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date

import pandas as pd


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + delta
    return total // 12, total % 12 + 1


@dataclass(frozen=True, order=True)
class FiscalQuarter:
    fy: int
    q: int
    fy_end_month: int = 12

    @property
    def label(self) -> str:
        return f"FY{self.fy % 100:02d}Q{self.q}"

    @property
    def index(self) -> int:
        """Monotonic integer, consecutive across quarters (for lags and sorting)."""
        return self.fy * 4 + (self.q - 1)

    @property
    def end(self) -> date:
        y, m = _add_months(self.fy, self.fy_end_month, -3 * (4 - self.q))
        return _month_end(y, m)

    @property
    def start(self) -> date:
        y, m = _add_months(self.end.year, self.end.month, -2)
        return date(y, m, 1)

    def shift(self, n: int) -> "FiscalQuarter":
        idx = self.index + n
        return FiscalQuarter(idx // 4, idx % 4 + 1, self.fy_end_month)


@dataclass(frozen=True)
class FiscalCalendar:
    fy_end_month: int = 12

    def quarter(self, d: date | pd.Timestamp | str) -> FiscalQuarter:
        d = pd.Timestamp(d).date()
        fy = d.year + 1 if d.month > self.fy_end_month else d.year
        months_into_fy = (d.month - (self.fy_end_month % 12 + 1)) % 12
        return FiscalQuarter(fy, months_into_fy // 3 + 1, self.fy_end_month)

    def quarter_for_period_end(self, end: date | pd.Timestamp | str, tolerance_days: int = 20) -> FiscalQuarter | None:
        """The fiscal quarter whose nominal end is nearest to an SEC period end."""
        end = pd.Timestamp(end).date()
        containing = self.quarter(end)
        best = min((containing, containing.shift(-1)), key=lambda fq: abs((fq.end - end).days))
        return best if abs((best.end - end).days) <= tolerance_days else None

    def from_label(self, label: str) -> FiscalQuarter:
        fy2, q = label.upper().removeprefix("FY").split("Q")
        return FiscalQuarter(2000 + int(fy2), int(q), self.fy_end_month)

    def quarters_between(self, start: date, end: date) -> list[FiscalQuarter]:
        out, fq = [], self.quarter(start)
        while fq.start <= end:
            out.append(fq)
            fq = fq.shift(1)
        return out


CALENDAR_QUARTERS = FiscalCalendar(12)


def parse_sec_fiscal_year_end(value: str | None) -> int | None:
    """Fiscal year-end month from SEC 'MMDD' (e.g. '0331').

    52/53-week years end on a weekday near a month end, so the reported date
    can spill a few days into the next month (YETI: '0102' = the year ending
    around December 31). Those are mapped back to the month they close.
    """
    if not value or len(value) != 4 or not value.isdigit():
        return None
    month, day = int(value[:2]), int(value[2:])
    return 12 if month == 1 and day <= 7 else (month - 1 if day <= 7 else month)

