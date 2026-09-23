"""The LangGraph half of the run: fetch the day's market data, then compute every signal from it.

This is all that remains of the original orchestration graph, and deliberately so — it is the part
that must stay deterministic. It runs inside the MCP server process, where
`openinference-instrumentation-langchain` turns each node into a span under the run's trace.
"""
from __future__ import annotations

import datetime as dt
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .market_data.base import GapRequest, MarketDataProvider, MarketSnapshot, find_gaps
from .models import DailyRecord, WatchState
from .signals import compute_signals


class BriefState(TypedDict, total=False):
    day: dt.date
    watch: WatchState
    snapshot: MarketSnapshot       # supplied on a recompute (after gap fills), fetched otherwise
    record: DailyRecord
    gaps: list[GapRequest]
    notes: Annotated[list[str], operator.add]


def build_brief_graph(market: MarketDataProvider):
    def fetch_market_data(state: BriefState) -> dict:
        if state.get("snapshot") is not None:
            return {}  # a recompute after gap fills: the snapshot is already in hand
        watch, day = state["watch"], state["day"]
        try:
            snapshot = market.snapshot(day, watch.symbols, watch.ticker, watch.expiry, watch.strikes)
        except Exception as exc:  # a dead provider degrades to "everything is a gap", never a crash
            return {"snapshot": MarketSnapshot(date=day, closes={}, puts={}),
                    "notes": [f"market data provider failed: {exc}"]}
        return {"snapshot": snapshot, "notes": list(snapshot.notes)}

    def compute(state: BriefState) -> dict:
        watch, snapshot = state["watch"], state["snapshot"]
        record = DailyRecord(date=snapshot.date, closes=dict(snapshot.closes),
                             puts={k: q.model_copy() for k, q in snapshot.puts.items()})
        close = record.closes.get(watch.ticker)
        if close is not None and close.price is not None:
            record.signals = compute_signals(watch, record)
        return {"record": record, "gaps": find_gaps(snapshot, watch.symbols, watch.strikes)}

    graph = StateGraph(BriefState)
    graph.add_node("fetch_market_data", fetch_market_data)
    graph.add_node("compute_signals", compute)
    graph.add_edge(START, "fetch_market_data")
    graph.add_edge("fetch_market_data", "compute_signals")
    graph.add_edge("compute_signals", END)
    return graph.compile()
