"""The deterministic tools dsh orchestrates: market data, signal computation, gaps and persistence.

These are plain functions over JSON-able values so they can be tested without an MCP transport;
`mcp_server.py` is the only place that knows about MCP. Nothing here calls an agent — gap filling is
dsh's decision, made by delegating to the Hermes subagent and handing the results to `apply_gap_fills`.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import brief_graph
from .config import Settings, load_settings
from .market_data.base import GapFill, MarketDataProvider, MarketSnapshot, apply_fills
from .models import DailyRecord, Verdict, WatchState, round1
from .signals import compute_signals
from .state_store import load_state, save_state
from .tracing import current_trace_id


@dataclass
class ToolDeps:
    market: MarketDataProvider
    state_path: Path
    #: Only the Hermes analyst needs these; the deterministic tools never read them.
    settings: Settings = field(default_factory=load_settings)
    # One run is one dsh session, so the day's snapshot is held here between prepare_brief,
    # apply_gap_fills and save_run. A cold cache (server restart) re-fetches rather than failing.
    snapshots: dict[dt.date, MarketSnapshot] = field(default_factory=dict)


def _day(date: str) -> dt.date:
    return dt.date.fromisoformat(date)


def _watch(deps: ToolDeps) -> WatchState:
    return load_state(deps.state_path)


def _brief(deps: ToolDeps, watch: WatchState, day: dt.date) -> dict[str, Any]:
    """Run the LangGraph pipeline for the day and project its result into JSON-able facts.

    The snapshot is cached on `deps`, so a later `apply_gap_fills` recomputes from the same market
    data instead of re-fetching it.
    """
    initial: dict[str, Any] = {"day": day, "watch": watch}
    if day in deps.snapshots:
        initial["snapshot"] = deps.snapshots[day]
    state = brief_graph.build_brief_graph(deps.market).invoke(initial)
    deps.snapshots[day] = state["snapshot"]
    record = state["record"]
    return {
        "closes": {s: c.model_dump(mode="json") for s, c in record.closes.items()},
        "puts": {str(k): q.model_dump(mode="json") for k, q in sorted(record.puts.items())},
        "signals": record.signals.model_dump(mode="json") if record.signals else None,
        "gaps": [g.model_dump(mode="json") for g in state["gaps"]],
        "notes": list(state.get("notes") or []),
    }


def prepare_brief(deps: ToolDeps, date: str) -> dict[str, Any]:
    """Fetch the day's market data and compute every signal that can be computed from it.

    Missing values are reported as `gaps` and never estimated; `signals` stays null until the
    ticker's close is known. Fill gaps with `apply_gap_fills`.
    """
    watch, day = _watch(deps), _day(date)
    prior = watch.prior_records(day)
    return {
        "date": date,
        "ticker": watch.ticker, "peer": watch.peer, "leveraged": watch.leveraged,
        "expiry": watch.expiry.isoformat(), "strikes": watch.strikes,
        "trigger_threshold": watch.trigger_threshold,
        # Cited in the report's provenance: the model writes the report before save_run, so it
        # needs the trace id while it still has somewhere to put it.
        "trace_id": current_trace_id(),
        **_brief(deps, watch, day),
        "prior_dates": [r.date.isoformat() for r in prior[-5:]],
    }


def apply_gap_fills(deps: ToolDeps, date: str, fills: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply researched values to the day's snapshot and recompute the signals.

    A value without an `http(s)` `source_url` and a `freshness` label is rejected, not stored.
    """
    watch, day = _watch(deps), _day(date)
    if day not in deps.snapshots:  # cold cache (server restart): re-fetch before applying
        _brief(deps, watch, day)
    parsed = [GapFill.model_validate(f) for f in fills]
    rejected = apply_fills(deps.snapshots[day], parsed)
    facts = _brief(deps, watch, day)
    return {
        "date": date,
        "accepted": len(parsed) - len(rejected),
        "rejected": rejected,
        "remaining_gaps": facts.pop("gaps"),
        **facts,
    }


def prior_state(deps: ToolDeps, date: str) -> dict[str, Any]:
    """The scored history before `date`, for continuity and the score delta."""
    watch, day = _watch(deps), _day(date)
    prior = watch.prior_records(day)
    last = next((r for r in reversed(prior) if r.score is not None), None)
    return {
        "last_date": last.date.isoformat() if last else None,
        "last_score": last.score if last else None,
        "last_verdict": last.verdict.value if last and last.verdict else None,
        "trigger_threshold": watch.trigger_threshold,
        "records": [{"date": r.date.isoformat(), "score": r.score,
                     "verdict": r.verdict.value if r.verdict else None, "note": r.note}
                    for r in prior[-5:]],
    }


def save_run(deps: ToolDeps, date: str, score: float, verdict: str, note: str,
             report_path: str) -> dict[str, Any]:
    """Persist the day's record: market data, computed signals, and dsh's verdict."""
    try:
        parsed_verdict = Verdict(verdict)
    except ValueError:
        raise ValueError(
            f"unknown verdict {verdict!r}; expected one of: {', '.join(v.value for v in Verdict)}") from None
    watch, day = _watch(deps), _day(date)
    snapshot = deps.snapshots.get(day) or deps.market.snapshot(
        day, watch.symbols, watch.ticker, watch.expiry, watch.strikes)
    record = DailyRecord(date=snapshot.date, closes=dict(snapshot.closes),
                         puts={k: q.model_copy() for k, q in snapshot.puts.items()})
    close = record.closes.get(watch.ticker)
    if close is not None and close.price is not None:
        record.signals = compute_signals(watch, record)
    prior = next((r.score for r in reversed(watch.prior_records(day)) if r.score is not None), None)
    record.score = score
    record.score_delta = None if prior is None else round1(score - prior)
    record.verdict = parsed_verdict
    record.note = note[:600]
    record.report_path = report_path
    record.trace_id = current_trace_id()
    watch.upsert(record)
    save_state(watch, deps.state_path)
    return {"saved": True, "date": date, "score": score, "score_delta": record.score_delta,
            "verdict": parsed_verdict.value, "state_path": str(deps.state_path),
            "trace_id": record.trace_id}
