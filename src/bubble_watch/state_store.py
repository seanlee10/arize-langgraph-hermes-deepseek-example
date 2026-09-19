"""JSON persistence for WatchState, plus the seed transcribed from the 9/15–9/17 reports."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from .models import Anchor, Close, DailyRecord, PutQuote, Verdict, WatchState


def load_state(path: Path) -> WatchState:
    return WatchState.model_validate_json(Path(path).read_text())


def save_state(state: WatchState, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json(indent=2))
    os.replace(tmp, path)


_YF = "https://finance.yahoo.com/quote/{}/history"


def _closes(nvda: float, smh: float, soxl: float) -> dict[str, Close]:
    return {s: Close(price=p, source=_YF.format(s), freshness="EOD")
            for s, p in (("NVDA", nvda), ("SMH", smh), ("SOXL", soxl))}


def _puts(freshness: str, source: str, rows: dict[int, dict]) -> dict[int, PutQuote]:
    return {k: PutQuote(strike=k, freshness=freshness, source_url=source, **v) for k, v in rows.items()}


def seed_state() -> WatchState:
    d = dt.date
    records = [
        DailyRecord(date=d(2026, 9, 11), closes=_closes(218.29, 568.53, 121.82),
                    puts=_puts("derived", "derived from 2026-09-16 report 3-day changes",
                               {200: {"last": 2.38}, 210: {"last": 4.84}, 220: {"last": 9.00}})),
        DailyRecord(date=d(2026, 9, 14), closes=_closes(210.96, 541.50, 101.13),
                    puts=_puts("derived", "derived from 2026-09-17 report 3-day changes",
                               {200: {"last": 3.87}, 210: {"last": 7.47}, 220: {"last": 12.99}}),
                    note="Sep-14: strong $200P > $210P > $220P convexity spike (warning signal)."),
        DailyRecord(date=d(2026, 9, 15), closes=_closes(212.17, 542.11, 102.32),
                    puts=_puts("EOD", "https://www.alphavantage.co/documentation/#historical-options", {
                        200: {"bid": 3.45, "ask": 3.55, "last": 3.53, "iv": 34.66},
                        210: {"bid": 6.85, "ask": 6.90, "last": 6.91, "iv": 33.68},
                        220: {"bid": 12.25, "ask": 12.30, "last": 12.25, "iv": 32.71}}),
                    score=8.7, verdict=Verdict.TRIGGERED),
        DailyRecord(date=d(2026, 9, 16), closes=_closes(213.90, 545.56, 103.97),
                    puts=_puts("late_session_last", "https://oqliv.com/flow?days=30&ticker=NVDA", {
                        200: {"last": 3.15, "volume": 3571, "oi": 43504},
                        210: {"last": 6.30, "volume": 2028, "oi": 24694},
                        220: {"last": 11.80, "volume": 1507, "oi": 35872}}),
                    score=8.4, score_delta=-0.3, verdict=Verdict.TRIGGERED_DE_CONFIRMING,
                    note=("NVDA outperformed SMH (+0.18%p); all puts fell with $200P falling most (-10.8%); "
                          "Apple weighing Nvidia networking for M8 Ultra servers (bullish); Fed hike (macro headwind).")),
        DailyRecord(date=d(2026, 9, 17), closes=_closes(219.34, 560.61, 114.82),
                    puts=_puts("latest_snapshot", "https://whalequant.io/en/stocks/NVDA/options/2026-10-16", {
                        200: {"bid": 3.05, "ask": 3.10, "last": 3.06, "iv": 34.90, "volume": 4222, "oi": 43504},
                        210: {"bid": 6.10, "ask": 6.15, "last": 6.06, "iv": 32.80, "volume": 2262, "oi": 24694},
                        220: {"bid": 11.05, "ask": 11.35, "last": 11.20, "iv": 31.97, "volume": 1609, "oi": 35872}}),
                    score=8.2, score_delta=-0.2, verdict=Verdict.TRIGGERED_FURTHER_DE_CONFIRMING,
                    note=("Semis risk-on (SMH +2.76%, SOXL +9.4%); NVDA −0.22%p vs SMH; puts retraced with no "
                          "OTM convex acceleration; IV not surface-wide. Huawei AI chip roadmap (bearish, China moat); "
                          "Nebius raising Nvidia GPU rental prices (bullish demand).")),
    ]
    anchor = Anchor(label="2026-08-28 spike", date=d(2026, 8, 28),
                    closes={"NVDA": 217.55, "SMH": 553.11, "SOXL": 111.34},
                    put_prices={200: 3.85, 210: 6.95, 220: 11.50})
    return WatchState(anchors=[anchor], records=records)
