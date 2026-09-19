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
AV_OPTIONS_URL = "https://www.alphavantage.co/documentation/#historical-options"

# Alpha Vantage HISTORICAL_OPTIONS EOD rows for the Oct-16 puts: (last, bid, ask, iv %, volume, oi).
# The reports' own option numbers matched these through 9/15; their "9/17" rows were 9/16 EOD.
_AV_PUTS: dict[str, dict[int, tuple]] = {
    "2026-09-11": {200: (2.38, 2.35, 2.41, 33.68, 2516, 42123), 210: (4.84, 4.80, 4.90, 32.71, 1634, 22395),
                   220: (9.00, 9.00, 9.10, 31.73, 23898, 51808)},
    "2026-09-14": {200: (3.87, 3.85, 3.95, 34.66, 5314, 41320), 210: (7.47, 7.45, 7.55, 33.68, 3167, 23294),
                   220: (13.00, 13.05, 13.25, 32.71, 819, 35481)},
    "2026-09-15": {200: (3.53, 3.45, 3.55, 34.66, 1856, 43225), 210: (6.91, 6.85, 6.90, 33.68, 4477, 24335),
                   220: (12.25, 12.25, 12.30, 32.71, 1248, 35654)},
    "2026-09-16": {200: (3.06, 3.05, 3.10, 35.63, 4222, 43504), 210: (6.06, 6.10, 6.15, 33.68, 2262, 24694),
                   220: (11.20, 11.05, 11.35, 32.71, 1609, 35872)},
    "2026-09-17": {200: (1.68, 1.66, 1.68, 33.68, 5516, 44118), 210: (3.65, 3.60, 3.70, 31.73, 4438, 25355),
                   220: (7.50, 7.40, 7.50, 29.78, 11316, 36320)},
}


def _closes(nvda: float, smh: float, soxl: float) -> dict[str, Close]:
    return {s: Close(price=p, source=_YF.format(s), freshness="EOD")
            for s, p in (("NVDA", nvda), ("SMH", smh), ("SOXL", soxl))}


def _puts(day: str) -> dict[int, PutQuote]:
    return {k: PutQuote(strike=k, last=last, bid=bid, ask=ask, iv=iv, volume=vol, oi=oi,
                        source_url=AV_OPTIONS_URL, freshness="EOD")
            for k, (last, bid, ask, iv, vol, oi) in _AV_PUTS[day].items()}


def seed_state() -> WatchState:
    d = dt.date
    records = [
        DailyRecord(date=d(2026, 9, 11), closes=_closes(218.29, 568.53, 121.82),
                    puts=_puts("2026-09-11")),
        DailyRecord(date=d(2026, 9, 14), closes=_closes(210.96, 541.50, 101.13),
                    puts=_puts("2026-09-14"),
                    note="Sep-14: strong $200P > $210P > $220P convexity spike (warning signal)."),
        DailyRecord(date=d(2026, 9, 15), closes=_closes(212.17, 542.11, 102.32),
                    puts=_puts("2026-09-15"),
                    score=8.7, verdict=Verdict.TRIGGERED),
        DailyRecord(date=d(2026, 9, 16), closes=_closes(213.90, 545.56, 103.97),
                    puts=_puts("2026-09-16"),
                    score=8.4, score_delta=-0.3, verdict=Verdict.TRIGGERED_DE_CONFIRMING,
                    note=("NVDA outperformed SMH (+0.18%p); all puts fell on an EOD basis (-13.3% / -12.3% / -8.6%), "
                          "$200P most; Apple weighing Nvidia networking for M8 Ultra servers (bullish); "
                          "Fed hike (macro headwind).")),
        DailyRecord(date=d(2026, 9, 17), closes=_closes(219.34, 560.61, 114.82),
                    puts=_puts("2026-09-17"),
                    score=8.2, score_delta=-0.2, verdict=Verdict.TRIGGERED_FURTHER_DE_CONFIRMING,
                    note=("Semis risk-on (SMH +2.76%, SOXL +9.4%); NVDA −0.22%p vs SMH. Puts collapsed on an EOD "
                          "basis (-45.1% / -39.8% / -33.0%) and IV fell at every strike — a sharp de-confirmation. "
                          "(The original report used one-day-stale 9/16 option data and read this as -3~5%.) "
                          "Huawei AI chip roadmap (bearish, China moat); Nebius raising Nvidia GPU rental prices "
                          "(bullish demand).")),
    ]
    anchor = Anchor(label="2026-08-28 spike", date=d(2026, 8, 28),
                    closes={"NVDA": 217.55, "SMH": 553.11, "SOXL": 111.34},
                    put_prices={200: 3.85, 210: 6.95, 220: 11.50})
    return WatchState(anchors=[anchor], records=records)
