import datetime as dt

import httpx
import pandas as pd
import pytest

from bubble_watch.market_data.alphavantage import AlphaVantageProvider
from bubble_watch.market_data.base import FallbackProvider, GapFill, MarketSnapshot, apply_fills, find_gaps
from bubble_watch.market_data.yfinance_provider import YFinanceProvider
from bubble_watch.models import Close, PutQuote

DAY = dt.date(2026, 9, 18)
EXP = dt.date(2026, 10, 16)


def _fake_download(symbols, start, end, progress, auto_adjust):
    idx = pd.DatetimeIndex([pd.Timestamp("2026-09-17"), pd.Timestamp("2026-09-18")])
    cols = pd.MultiIndex.from_product([["Close"], symbols])
    data = [[219.34, 560.61, 114.82], [222.27, 573.00, float("nan")]]
    return pd.DataFrame(data, index=idx, columns=cols)


class _FakeTicker:
    options = ("2026-09-25", "2026-10-16")

    def option_chain(self, expiry):
        rows = [
            {"contractSymbol": "NVDA261016P00200000", "strike": 200.0, "lastPrice": 1.34, "bid": 1.30, "ask": 1.35,
             "impliedVolatility": 0.343512, "volume": 3310.0, "openInterest": 43456.0,
             "lastTradeDate": pd.Timestamp("2026-09-18 19:59:58", tz="UTC")},
            {"contractSymbol": "NVDA261016P00210000", "strike": 210.0, "lastPrice": 2.98, "bid": 0.0, "ask": 2.97,
             "impliedVolatility": 0.317451, "volume": float("nan"), "openInterest": 26816.0,
             "lastTradeDate": pd.Timestamp("2026-09-17 19:59:41", tz="UTC")},  # stale: last traded yesterday
        ]
        return type("Chain", (), {"puts": pd.DataFrame(rows)})()


def test_yfinance_snapshot_labels_freshness_and_skips_stale_rows():
    p = YFinanceProvider(download=_fake_download, ticker_factory=lambda t: _FakeTicker())
    snap = p.snapshot(DAY, ["NVDA", "SMH", "SOXL"], "NVDA", EXP, [200, 210, 220])
    assert snap.closes["NVDA"].price == 222.27 and snap.closes["NVDA"].freshness == "EOD"
    assert snap.closes["SOXL"].price is None
    q = snap.puts[200]
    assert (q.bid, q.ask, q.last, q.iv, q.volume, q.oi) == (1.30, 1.35, 1.34, 34.35, 3310, 43456)
    assert q.freshness == "latest_snapshot" and q.source_url.endswith("NVDA261016P00200000")
    assert 210 not in snap.puts and 220 not in snap.puts


def test_find_gaps_and_apply_fills_enforce_source_and_freshness():
    snap = MarketSnapshot(date=DAY, closes={"NVDA": Close(price=222.27), "SOXL": Close(price=None)},
                          puts={200: PutQuote(strike=200, bid=1.3, ask=1.35, last=1.34, iv=34.35, volume=1, oi=2,
                                              freshness="latest_snapshot")})
    gaps = {g.field for g in find_gaps(snap, ["NVDA", "SOXL"], [200, 210])}
    assert gaps == {"closes.SOXL", *(f"puts.210.{f}" for f in ("bid", "ask", "last", "iv", "volume", "oi"))}

    rejected = apply_fills(snap, [
        GapFill(field="closes.SOXL", value=123.67, source_url="https://stockanalysis.com/etf/soxl/", freshness="EOD"),
        GapFill(field="puts.210.last", value=2.98, source_url="", freshness="EOD"),          # no source
        GapFill(field="puts.210.iv", value=31.7, source_url="https://x.test", freshness="N/A"),  # value but N/A
        GapFill(field="puts.210.volume", value=2965, source_url="https://x.test/c", freshness="latest_snapshot"),
        GapFill(field="puts.210.oi", value=None, freshness="N/A"),                           # explicit N/A is fine
        GapFill(field="nonsense", value=1, source_url="https://x.test", freshness="EOD"),
    ])
    assert snap.closes["SOXL"].price == 123.67 and snap.closes["SOXL"].freshness == "EOD"
    assert snap.puts[210].last is None and snap.puts[210].iv is None
    assert snap.puts[210].volume == 2965 and snap.puts[210].freshness == "latest_snapshot"
    assert len(rejected) == 3


AV_OPTIONS = {"endpoint": "Historical Options", "message": "success", "data": [
    {"type": "put", "expiration": "2026-10-16", "strike": "200.00", "last": "1.34", "bid": "1.30", "ask": "1.35",
     "implied_volatility": "0.34658", "volume": "3310", "open_interest": "43456"},
    {"type": "call", "expiration": "2026-10-16", "strike": "200.00", "last": "25", "bid": "1", "ask": "1",
     "implied_volatility": "0.3", "volume": "1", "open_interest": "1"},
]}


def _av(handler):
    return AlphaVantageProvider("k", client=httpx.Client(transport=httpx.MockTransport(handler)))


def _av_handler(request):
    params = dict(request.url.params)
    if params["function"] == "TIME_SERIES_DAILY":
        closes = {"NVDA": "222.2700", "SMH": "573.0000"}
        if params["symbol"] not in closes:
            return httpx.Response(200, json={"Meta Data": {}, "Time Series (Daily)": {"2026-09-17": {"4. close": "1"}}})
        return httpx.Response(200, json={"Meta Data": {}, "Time Series (Daily)": {
            "2026-09-18": {"4. close": closes[params["symbol"]]}}})
    assert params["function"] == "HISTORICAL_OPTIONS" and params["date"] == "2026-09-18"
    return httpx.Response(200, json=AV_OPTIONS)


def test_alphavantage_snapshot_is_eod_for_closes_and_puts():
    snap = _av(_av_handler).snapshot(DAY, ["NVDA", "SMH", "SOXL"], "NVDA", EXP, [200, 210])
    assert snap.closes["NVDA"].price == 222.27 and snap.closes["NVDA"].freshness == "EOD"
    assert "alphavantage" in snap.closes["NVDA"].source
    assert snap.closes["SOXL"].price is None  # no row for the day
    assert list(snap.puts) == [200]
    assert (snap.puts[200].iv, snap.puts[200].oi, snap.puts[200].freshness) == (34.66, 43456, "EOD")


def test_alphavantage_quota_or_error_message_raises():
    from bubble_watch.market_data.alphavantage import AlphaVantageError

    info = _av(lambda r: httpx.Response(200, json={"Information": "API rate limit reached"}))
    with pytest.raises(AlphaVantageError, match="rate limit"):
        info.snapshot(DAY, ["NVDA"], "NVDA", EXP, [200])


class _Fixed:
    def __init__(self, snap=None, exc=None):
        self.snap, self.exc, self.calls = snap, exc, 0

    def snapshot(self, day, symbols, ticker, expiry, strikes):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.snap


def test_fallback_fills_whole_missing_items_and_notes_them():
    primary = _Fixed(MarketSnapshot(date=DAY, closes={"NVDA": Close(price=222.27, source="av"),
                                                      "SOXL": Close(price=None, freshness="N/A")},
                                    puts={200: PutQuote(strike=200, last=1.34, iv=None, freshness="EOD")}))
    fallback = _Fixed(MarketSnapshot(date=DAY, closes={"NVDA": Close(price=999.0), "SOXL": Close(price=123.67)},
                                     puts={200: PutQuote(strike=200, last=9.0, iv=40.0),
                                           210: PutQuote(strike=210, last=2.98, freshness="latest_snapshot")}))
    snap = FallbackProvider(primary, fallback, fallback_name="yfinance").snapshot(
        DAY, ["NVDA", "SOXL"], "NVDA", EXP, [200, 210])
    assert snap.closes["NVDA"].price == 222.27 and snap.closes["SOXL"].price == 123.67
    assert snap.puts[200].last == 1.34 and snap.puts[200].iv is None  # never mix sources inside one quote
    assert snap.puts[210].last == 2.98
    assert sorted(snap.notes) == ["closes.SOXL from yfinance (primary had none)",
                                  "puts.210 from yfinance (primary had none)"]


def test_fallback_takes_over_when_primary_fails_and_skips_when_complete():
    fallback = _Fixed(MarketSnapshot(date=DAY, closes={"NVDA": Close(price=1.0)}, puts={}))
    snap = FallbackProvider(_Fixed(exc=RuntimeError("boom")), fallback, fallback_name="yfinance").snapshot(
        DAY, ["NVDA"], "NVDA", EXP, [])
    assert snap.closes["NVDA"].price == 1.0 and "primary failed: boom" in snap.notes[0]
    complete = _Fixed(MarketSnapshot(date=DAY, closes={"NVDA": Close(price=2.0)}, puts={}))
    fb = _Fixed(MarketSnapshot(date=DAY, closes={}, puts={}))
    FallbackProvider(complete, fb, fallback_name="yfinance").snapshot(DAY, ["NVDA"], "NVDA", EXP, [])
    assert fb.calls == 0
