"""Default provider: official closes + today's option chain (yfinance has no historical chains)."""
from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from typing import Any

from ..models import Close, PutQuote
from .base import MarketSnapshot


def _num(v: Any, *, positive: bool = False) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or (positive and f <= 0):
        return None
    return f


def _int(v: Any) -> int | None:
    f = _num(v)
    return None if f is None else int(f)


class YFinanceProvider:
    def __init__(self, download: Callable | None = None, ticker_factory: Callable | None = None) -> None:
        if download is None or ticker_factory is None:
            import yfinance as yf
            download, ticker_factory = download or yf.download, ticker_factory or yf.Ticker
        self._download, self._ticker = download, ticker_factory

    def snapshot(self, day: dt.date, symbols: list[str], ticker: str, expiry: dt.date,
                 strikes: list[int]) -> MarketSnapshot:
        return MarketSnapshot(date=day, closes=self._closes(day, symbols), puts=self._puts(day, ticker, expiry, strikes))

    def _closes(self, day: dt.date, symbols: list[str]) -> dict[str, Close]:
        df = self._download(symbols, start=(day - dt.timedelta(days=6)).isoformat(),
                            end=(day + dt.timedelta(days=1)).isoformat(), progress=False, auto_adjust=False)
        out: dict[str, Close] = {}
        for s in symbols:
            price = None
            try:
                col = df["Close"][s].dropna()
                if len(col) and col.index[-1].date() == day:
                    price = round(float(col.iloc[-1]), 2)
            except KeyError:
                pass
            out[s] = Close(price=price, source=f"https://finance.yahoo.com/quote/{s}/history",
                           freshness="EOD" if price is not None else "N/A")
        return out

    def _puts(self, day: dt.date, ticker: str, expiry: dt.date, strikes: list[int]) -> dict[int, PutQuote]:
        t = self._ticker(ticker)
        if expiry.isoformat() not in t.options:
            return {}
        chain = t.option_chain(expiry.isoformat()).puts
        out: dict[int, PutQuote] = {}
        for k in strikes:
            rows = chain[chain["strike"] == float(k)]
            if rows.empty:
                continue
            r = rows.iloc[0]
            traded = r["lastTradeDate"]
            traded_day = traded.tz_convert("America/New_York").date() if traded.tzinfo else traded.date()
            if traded_day != day:  # the chain is "now"; a row last traded on another day is not today's data
                continue
            out[k] = PutQuote(
                strike=k, bid=_num(r["bid"], positive=True), ask=_num(r["ask"], positive=True),
                last=_num(r["lastPrice"]), iv=None if _num(r["impliedVolatility"]) is None
                else round(float(r["impliedVolatility"]) * 100, 2),
                volume=_int(r["volume"]), oi=_int(r["openInterest"]),
                source_url=f"https://finance.yahoo.com/quote/{r['contractSymbol']}",
                freshness="latest_snapshot", as_of=traded.to_pydatetime(),
            )
        return out
