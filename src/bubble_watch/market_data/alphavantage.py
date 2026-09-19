"""Optional: Alpha Vantage historical option snapshots (EOD), used to fill IV/bid/ask for a past date."""
from __future__ import annotations

import datetime as dt

import httpx

from ..models import PutQuote

URL = "https://www.alphavantage.co/query"
DOC_URL = "https://www.alphavantage.co/documentation/#historical-options"


def _f(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class AlphaVantageOptions:
    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        self._key = api_key
        self._client = client or httpx.Client(timeout=30)

    def puts(self, day: dt.date, ticker: str, expiry: dt.date, strikes: list[int]) -> dict[int, PutQuote]:
        r = self._client.get(URL, params={"function": "HISTORICAL_OPTIONS", "symbol": ticker,
                                          "date": day.isoformat(), "apikey": self._key})
        r.raise_for_status()
        out: dict[int, PutQuote] = {}
        for row in r.json().get("data") or []:
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
