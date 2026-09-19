"""Snapshot model, provider protocol, and the gap-fill contract (no value without source + freshness)."""
from __future__ import annotations

import datetime as dt
from typing import Protocol

from pydantic import BaseModel

from ..models import PUT_FIELDS, Close, Freshness, PutQuote

_INT_FIELDS = {"volume", "oi"}


class MarketSnapshot(BaseModel):
    date: dt.date
    closes: dict[str, Close]
    puts: dict[int, PutQuote]


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


def merge_puts(primary: dict[int, PutQuote], secondary: dict[int, PutQuote]) -> dict[int, PutQuote]:
    """Fill fields missing from ``primary`` with ``secondary`` values (e.g. historical IV)."""
    merged = {k: q.model_copy() for k, q in primary.items()}
    for k, sq in secondary.items():
        if k not in merged:
            merged[k] = sq.model_copy()
            continue
        q = merged[k]
        for f in PUT_FIELDS:
            if getattr(q, f) is None and getattr(sq, f) is not None:
                setattr(q, f, getattr(sq, f))
    return merged
