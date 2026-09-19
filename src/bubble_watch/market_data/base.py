"""Snapshot model, provider protocol, and the gap-fill contract (no value without source + freshness)."""
from __future__ import annotations

import datetime as dt
from typing import Protocol

from pydantic import BaseModel, Field

from ..models import PUT_FIELDS, Close, Freshness, PutQuote

_INT_FIELDS = {"volume", "oi"}


class MarketSnapshot(BaseModel):
    date: dt.date
    closes: dict[str, Close]
    puts: dict[int, PutQuote]
    notes: list[str] = Field(default_factory=list)  # e.g. which items came from a fallback source


class MarketDataProvider(Protocol):
    def snapshot(self, day: dt.date, symbols: list[str], ticker: str, expiry: dt.date,
                 strikes: list[int]) -> MarketSnapshot: ...


class GapRequest(BaseModel):
    field: str  # "closes.SOXL" | "puts.210.iv"
    description: str


class GapFill(BaseModel):
    field: str
    value: float | None
    source_url: str = ""
    freshness: Freshness = "N/A"
    note: str = ""


def find_gaps(snapshot: MarketSnapshot, symbols: list[str], strikes: list[int]) -> list[GapRequest]:
    gaps = [GapRequest(field=f"closes.{s}", description=f"{s} official close on {snapshot.date}")
            for s in symbols if snapshot.closes.get(s) is None or snapshot.closes[s].price is None]
    for k in strikes:
        q = snapshot.puts.get(k)
        gaps += [GapRequest(field=f"puts.{k}.{f}", description=f"${k} put {f} on {snapshot.date}")
                 for f in PUT_FIELDS if q is None or getattr(q, f) is None]
    return gaps


def apply_fills(snapshot: MarketSnapshot, fills: list[GapFill]) -> list[str]:
    """Apply accepted fills in place; return a message per rejected fill."""
    rejected = []
    for fill in fills:
        if fill.value is not None and (not fill.source_url.startswith(("http://", "https://"))
                                       or fill.freshness == "N/A"):
            rejected.append(f"{fill.field}: value without source_url/freshness rejected")
            continue
        parts = fill.field.split(".")
        if len(parts) == 2 and parts[0] == "closes":
            if fill.value is not None:
                snapshot.closes[parts[1]] = Close(price=fill.value, source=fill.source_url, freshness=fill.freshness)
        elif len(parts) == 3 and parts[0] == "puts" and parts[1].isdigit() and parts[2] in PUT_FIELDS:
            strike = int(parts[1])
            q = snapshot.puts.setdefault(strike, PutQuote(strike=strike))
            if fill.value is not None:
                setattr(q, parts[2], int(fill.value) if parts[2] in _INT_FIELDS else float(fill.value))
                q.source_url = q.source_url or fill.source_url
                if q.freshness == "N/A":
                    q.freshness = fill.freshness
        else:
            rejected.append(f"{fill.field}: unknown field")
    return rejected


class FallbackProvider:
    """Primary source first; a fallback supplies only WHOLE items the primary lacks (a close, a put quote),
    never individual fields, so one quote never mixes two sources. Every fallback is noted."""

    def __init__(self, primary: MarketDataProvider, fallback: MarketDataProvider, *, fallback_name: str) -> None:
        self.primary, self.fallback, self.fallback_name = primary, fallback, fallback_name

    def snapshot(self, day: dt.date, symbols: list[str], ticker: str, expiry: dt.date,
                 strikes: list[int]) -> MarketSnapshot:
        try:
            snap = self.primary.snapshot(day, symbols, ticker, expiry, strikes)
        except Exception as exc:  # noqa: BLE001 - any primary failure degrades to the fallback, noted
            snap = self.fallback.snapshot(day, symbols, ticker, expiry, strikes)
            snap.notes.append(f"primary failed: {str(exc)[:200]}; all data from {self.fallback_name}")
            return snap
        missing_closes = [s for s in symbols if snap.closes.get(s) is None or snap.closes[s].price is None]
        missing_puts = [k for k in strikes if k not in snap.puts]
        if not (missing_closes or missing_puts):
            return snap
        backup = self.fallback.snapshot(day, symbols, ticker, expiry, strikes)
        for s in missing_closes:
            c = backup.closes.get(s)
            if c is not None and c.price is not None:
                snap.closes[s] = c
                snap.notes.append(f"closes.{s} from {self.fallback_name} (primary had none)")
        for k in missing_puts:
            if k in backup.puts:
                snap.puts[k] = backup.puts[k]
                snap.notes.append(f"puts.{k} from {self.fallback_name} (primary had none)")
        return snap
