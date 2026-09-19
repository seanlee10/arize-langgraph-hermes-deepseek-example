import datetime as dt

import httpx
import pandas as pd

from bubble_watch.market_data.alphavantage import AlphaVantageOptions
from bubble_watch.market_data.base import GapFill, MarketSnapshot, apply_fills, find_gaps, merge_puts
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


def test_merge_puts_fills_missing_fields_from_secondary():
    primary = {200: PutQuote(strike=200, last=1.34, freshness="latest_snapshot")}
    secondary = {200: PutQuote(strike=200, last=1.40, iv=34.0, freshness="EOD"),
                 210: PutQuote(strike=210, last=2.9, freshness="EOD")}
    merged = merge_puts(primary, secondary)
    assert merged[200].last == 1.34 and merged[200].iv == 34.0
    assert merged[210].last == 2.9 and merged[210].freshness == "EOD"


def test_alphavantage_parses_historical_puts():
    payload = {"data": [
        {"type": "put", "expiration": "2026-10-16", "strike": "200.00", "last": "1.33", "bid": "1.30", "ask": "1.36",
         "implied_volatility": "0.34201", "volume": "3301", "open_interest": "43456"},
        {"type": "call", "expiration": "2026-10-16", "strike": "200.00", "last": "25", "bid": "1", "ask": "1",
         "implied_volatility": "0.3", "volume": "1", "open_interest": "1"},
    ]}
    seen = {}

    def handler(request):
        seen.update(dict(request.url.params))
        return httpx.Response(200, json=payload)

    av = AlphaVantageOptions("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    puts = av.puts(DAY, "NVDA", EXP, [200, 210])
    assert seen["function"] == "HISTORICAL_OPTIONS" and seen["date"] == "2026-09-18"
    assert list(puts) == [200]
    assert (puts[200].iv, puts[200].oi, puts[200].freshness) == (34.2, 43456, "EOD")
