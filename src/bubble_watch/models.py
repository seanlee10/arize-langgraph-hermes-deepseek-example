"""Domain models shared across the pipeline. Everything is JSON-serializable for the state file."""
from __future__ import annotations

import datetime as dt
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Freshness = Literal["EOD", "latest_snapshot", "late_session_last", "derived", "N/A"]
Condition = Literal["true", "false", "partial", "unknown"]
PUT_FIELDS = ("bid", "ask", "last", "iv", "volume", "oi")


def round1(x: float) -> float:
    # round(x, 6) first strips float noise (8.049999999999999 -> 8.05) before half-up to one decimal.
    return float(Decimal(str(round(x, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


class Close(BaseModel):
    price: float | None = None
    source: str = ""
    freshness: Freshness = "EOD"


class PutQuote(BaseModel):
    strike: int
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    iv: float | None = None  # percent, e.g. 34.9
    volume: int | None = None
    oi: int | None = None
    source_url: str = ""
    freshness: Freshness = "N/A"
    as_of: dt.datetime | None = None

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return round((self.bid + self.ask) / 2, 3)


class Verdict(str, Enum):
    NOT_TRIGGERED = "NOT_TRIGGERED"
    TRIGGERED = "TRIGGERED"
    TRIGGERED_DE_CONFIRMING = "TRIGGERED_DE_CONFIRMING"
    TRIGGERED_FURTHER_DE_CONFIRMING = "TRIGGERED_FURTHER_DE_CONFIRMING"
    CONFIRMED = "CONFIRMED"


# Most cautious first: the verdict that keeps the bubble warning most active.
CAUTION_ORDER = [Verdict.CONFIRMED, Verdict.TRIGGERED, Verdict.TRIGGERED_DE_CONFIRMING,
                 Verdict.TRIGGERED_FURTHER_DE_CONFIRMING, Verdict.NOT_TRIGGERED]


def most_cautious(*verdicts: Verdict) -> Verdict:
    return min(verdicts, key=CAUTION_ORDER.index)


class Catalyst(BaseModel):
    headline: str
    url: str
    published: dt.date | None = None
    direction: Literal["bearish", "bullish", "neutral"]
    weight: Literal["low", "med", "high"] = "med"
    rationale_ko: str = ""


class AnalystView(BaseModel):
    score: float
    score_delta_reasoning_ko: str
    verdict: Verdict
    catalysts: list[Catalyst] = Field(default_factory=list)
    tape_read_ko: str
    watch_conditions_ko: str
    confidence: Literal["low", "med", "high"] = "med"
    rebuttal_ko: str | None = None

    @field_validator("score", mode="before")
    @classmethod
    def _clamp(cls, v: object) -> float:
        return round1(min(10.0, max(0.0, float(v))))  # type: ignore[arg-type]


class IVCompare(BaseModel):
    current: float | None = None
    prior: float | None = None
    prior_date: dt.date | None = None


class Signals(BaseModel):
    base_dates: dict[str, dt.date | None]              # window ("1d", "3d", "anchor") -> base date
    returns: dict[str, dict[str, float | None]]        # window -> symbol -> %
    rel_spread: dict[str, float | None]                # window -> ticker − peer, %p
    put_changes: dict[str, dict[int, float | None]]    # window -> strike -> %
    put_freshness_1d: tuple[str, str] | None = None    # (base freshness, today freshness)
    convexity_order_1d: list[int] = Field(default_factory=list)
    iv: dict[int, IVCompare] = Field(default_factory=dict)
    iv_surface_up: bool | None = None
    conditions: dict[str, Condition] = Field(default_factory=dict)


class MergedCatalyst(Catalyst):
    cited_by: list[str] = Field(default_factory=list)


class Reconciliation(BaseModel):
    mode: Literal["agree", "agree_after_rebuttal", "disagree", "single"]
    score: float
    verdict: Verdict
    scores: dict[str, float]
    verdicts: dict[str, Verdict]
    rebuttal_round: bool = False
    catalysts: list[MergedCatalyst] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class DailyRecord(BaseModel):
    date: dt.date
    closes: dict[str, Close] = Field(default_factory=dict)
    puts: dict[int, PutQuote] = Field(default_factory=dict)
    signals: Signals | None = None
    score: float | None = None
    score_delta: float | None = None
    verdict: Verdict | None = None
    analyst_views: dict[str, AnalystView] = Field(default_factory=dict)
    reconciliation: Reconciliation | None = None
    report_path: str | None = None
    note: str = ""


class Anchor(BaseModel):
    label: str
    date: dt.date
    closes: dict[str, float]
    put_prices: dict[int, float]


class WatchState(BaseModel):
    ticker: str = "NVDA"
    peer: str = "SMH"
    leveraged: str = "SOXL"
    expiry: dt.date = dt.date(2026, 10, 16)
    strikes: list[int] = Field(default_factory=lambda: [200, 210, 220])
    trigger_threshold: float = 8.0
    anchors: list[Anchor] = Field(default_factory=list)
    records: list[DailyRecord] = Field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        return [self.ticker, self.peer, self.leveraged]

    def prior_records(self, day: dt.date) -> list[DailyRecord]:
        return sorted((r for r in self.records if r.date < day), key=lambda r: r.date)

    def upsert(self, record: DailyRecord) -> None:
        kept = [r for r in self.records if r.date != record.date]
        self.records = sorted([*kept, record], key=lambda r: r.date)
