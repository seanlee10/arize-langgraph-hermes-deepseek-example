"""Pure signal computation: every number the report shows comes from here."""
from __future__ import annotations

from urllib.parse import urlparse

from .models import Condition, DailyRecord, IVCompare, PutQuote, Signals, WatchState

_QUOTE_FIELDS = ("last", "bid", "ask", "volume", "oi")


def pct(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return round((new / old - 1) * 100, 2)


def _spread(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else round(a - b, 2)


def _close(rec: DailyRecord | None, sym: str) -> float | None:
    c = rec.closes.get(sym) if rec else None
    return c.price if c else None


def _put_last(rec: DailyRecord | None, strike: int) -> float | None:
    q = rec.puts.get(strike) if rec else None
    return q.last if q else None


def _iv_compare(prior: list[DailyRecord], today: DailyRecord, strike: int) -> IVCompare:
    q = today.puts.get(strike)
    current = q.iv if q else None
    for rec in reversed(prior):
        pq = rec.puts.get(strike)
        if pq and pq.iv is not None:
            return IVCompare(current=current, prior=pq.iv, prior_date=rec.date)
    return IVCompare(current=current)


def _source(q: PutQuote) -> str:
    return urlparse(q.source_url).netloc or q.source_url or "unknown"


def put_comparability(base: PutQuote | None, today: PutQuote | None) -> str | None:
    """Why a 1-day put change is NOT a market move, or None when it is like-for-like.
    Catches a data-source switch and a stale feed repeating the previous day's quote."""
    if base is None or today is None:
        return None
    if _source(base) != _source(today):
        return f"source changed: {_source(base)} → {_source(today)}"
    fields = [(getattr(base, f), getattr(today, f)) for f in _QUOTE_FIELDS]
    if all(b is not None for b, _ in fields) and all(b == t for b, t in fields):
        return "identical to the prior day's quote (stale feed?)"
    return None


def _underperforms(rel_1d: float | None, rel_3d: float | None) -> Condition:
    if rel_1d is None:
        return "unknown"
    if rel_1d >= 0:
        return "false"
    return "true" if rel_3d is not None and rel_3d < 0 else "partial"


def _far_otm_leads(changes: dict[int, float | None], strikes: list[int],
                   comparability: dict[int, str | None]) -> Condition:
    if any(changes.get(k) is None or comparability.get(k) for k in strikes):
        return "unknown"
    lo, others = min(strikes), [k for k in strikes if k != min(strikes)]
    lead = changes[lo]
    return "true" if lead > 0 and all(lead > changes[k] for k in others) else "false"  # type: ignore[operator]


def compute_signals(state: WatchState, today: DailyRecord) -> Signals:
    prior = state.prior_records(today.date)
    bases = {"1d": prior[-1] if prior else None, "3d": prior[-3] if len(prior) >= 3 else None}
    anchor = state.anchors[0] if state.anchors else None
    today_close = {s: _close(today, s) for s in state.symbols}

    base_dates, returns, put_changes = {}, {}, {}
    for window, rec in bases.items():
        base_dates[window] = rec.date if rec else None
        returns[window] = {s: pct(today_close[s], _close(rec, s)) for s in state.symbols}
        put_changes[window] = {k: pct(_put_last(today, k), _put_last(rec, k)) for k in state.strikes}
    base_dates["anchor"] = anchor.date if anchor else None
    returns["anchor"] = {s: pct(today_close[s], anchor.closes.get(s) if anchor else None) for s in state.symbols}
    put_changes["anchor"] = {k: pct(_put_last(today, k), anchor.put_prices.get(k) if anchor else None)
                             for k in state.strikes}
    rel = {w: _spread(r.get(state.ticker), r.get(state.peer)) for w, r in returns.items()}

    base_1d, lo = bases["1d"], min(state.strikes)
    freshness = None
    if base_1d and lo in base_1d.puts and lo in today.puts:
        freshness = (base_1d.puts[lo].freshness, today.puts[lo].freshness)

    comparability = {k: put_comparability(base_1d.puts.get(k) if base_1d else None, today.puts.get(k))
                     for k in state.strikes}
    changes_1d = put_changes["1d"]
    order = sorted((k for k, v in changes_1d.items() if v is not None), key=lambda k: changes_1d[k], reverse=True)
    iv = {k: _iv_compare(prior, today, k) for k in state.strikes}
    surface = (None if any(c.current is None or c.prior is None for c in iv.values())
               else all(c.current > c.prior for c in iv.values()))  # type: ignore[operator]

    return Signals(
        base_dates=base_dates, returns=returns, rel_spread=rel, put_changes=put_changes,
        put_freshness_1d=freshness, put_comparability_1d=comparability, convexity_order_1d=order, iv=iv, iv_surface_up=surface,
        conditions={
            "nvda_underperforms": _underperforms(rel["1d"], rel["3d"]),
            "iv_surface_up": "unknown" if surface is None else ("true" if surface else "false"),
            "far_otm_leads": _far_otm_leads(changes_1d, state.strikes, comparability),
        },
    )
