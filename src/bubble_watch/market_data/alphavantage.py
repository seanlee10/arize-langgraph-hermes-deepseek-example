"""Primary provider: Alpha Vantage EOD data — daily closes (TIME_SERIES_DAILY) and the option chain for any
past date (HISTORICAL_OPTIONS). One paid EOD source keeps day-over-day comparisons like-for-like."""
from __future__ import annotations

import datetime as dt
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from ..models import Close, PutQuote
from .base import MarketSnapshot

URL = "https://www.alphavantage.co/query"
DOC_URL = "https://www.alphavantage.co/documentation/#historical-options"
DAILY_URL = "https://www.alphavantage.co/documentation/#daily"
NY = ZoneInfo("America/New_York")


class AlphaVantageError(RuntimeError):
    """Alpha Vantage answered without data (rate limit, premium-only endpoint, bad key, bad symbol)."""


def _f(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class AlphaVantageProvider:
    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        self._key = api_key
        self._client = client or httpx.Client(timeout=60)

    def _query(self, **params: str) -> dict[str, Any]:
        r = self._client.get(URL, params={**params, "apikey": self._key})
        r.raise_for_status()
        data = r.json()
        # AV reports quota/premium/key problems as a 200 with one of these keys and no data.
        message = data.get("Information") or data.get("Note") or data.get("Error Message")
        if message:
            raise AlphaVantageError(f"{params.get('function')}: {message[:200]}")
        return data

    def snapshot(self, day: dt.date, symbols: list[str], ticker: str, expiry: dt.date,
                 strikes: list[int]) -> MarketSnapshot:
        return MarketSnapshot(date=day, closes=self.closes(day, symbols), puts=self.puts(day, ticker, expiry, strikes))

    def closes(self, day: dt.date, symbols: list[str]) -> dict[str, Close]:
        out: dict[str, Close] = {}
        for s in symbols:
            # compact = last 100 trading days; older dates need the full history.
            size = "compact" if (dt.datetime.now(NY).date() - day).days < 120 else "full"
            series = self._query(function="TIME_SERIES_DAILY", symbol=s, outputsize=size).get("Time Series (Daily)", {})
            price = _f((series.get(day.isoformat()) or {}).get("4. close"))
            out[s] = Close(price=None if price is None else round(price, 2), source=DAILY_URL,
                           freshness="EOD" if price is not None else "N/A")
        return out

    def puts(self, day: dt.date, ticker: str, expiry: dt.date, strikes: list[int]) -> dict[int, PutQuote]:
        data = self._query(function="HISTORICAL_OPTIONS", symbol=ticker, date=day.isoformat())
        out: dict[int, PutQuote] = {}
        for row in data.get("data") or []:
            strike = _f(row.get("strike"))
            if (row.get("type") != "put" or row.get("expiration") != expiry.isoformat()
                    or strike is None or int(strike) not in strikes):
                continue
            iv, vol, oi = _f(row.get("implied_volatility")), _f(row.get("volume")), _f(row.get("open_interest"))
            out[int(strike)] = PutQuote(
                strike=int(strike), bid=_f(row.get("bid")), ask=_f(row.get("ask")), last=_f(row.get("last")),
                iv=None if iv is None else round(iv * 100, 2), volume=None if vol is None else int(vol),
                oi=None if oi is None else int(oi), source_url=DOC_URL, freshness="EOD")
        return out
