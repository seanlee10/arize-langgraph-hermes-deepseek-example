"""LangGraph wiring for one Bubble Watch run (see docs/specs for the topology)."""
from __future__ import annotations

import contextvars
import datetime as dt
import operator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents.base import Analyst, AnalystError, AnalystResult
from .agents.prompts import analyst_brief, gap_fill_brief, rebuttal_brief
from .market_data.base import GapFill, MarketDataProvider, MarketSnapshot, apply_fills, find_gaps
from .models import DailyRecord, Reconciliation, WatchState, round1
from .reconcile import reconcile
from .report import render_report
from .signals import compute_signals
from .state_store import save_state
from .tracing import OUTPUT, agent_span, annotate_result
from .writer import Narrative, fallback_narrative, writer_brief


def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class RunState(TypedDict, total=False):
    day: dt.date
    watch: WatchState
    record: DailyRecord
    brief: str
    results: Annotated[dict[str, AnalystResult], _merge]
    notes: Annotated[list[str], operator.add]
    rebuttal_done: bool
    reconciliation: Reconciliation | None
    report_path: str


class GapFiller(Protocol):
    def fill_gaps(self, prompt: str) -> list[GapFill]: ...


class Writer(Protocol):
    def write(self, prompt: str) -> Narrative: ...


@dataclass
class Deps:
    market: MarketDataProvider
    analysts: dict[str, Analyst]
    writer: Writer | None
    tracer: Any
    reports_dir: Path
    state_path: Path
    gap_filler: GapFiller | None = None
    save: bool = True


def _fills_json(fills: list[GapFill]) -> str:
    return "[" + ", ".join(f.model_dump_json() for f in fills) + "]"


def _prior_score(watch: WatchState, day: dt.date) -> float | None:
    return next((r.score for r in reversed(watch.prior_records(day)) if r.score is not None), None)


def build_graph(deps: Deps):
    def fetch_market_data(state: RunState) -> dict:
        w, day, notes = state["watch"], state["day"], []
        try:
            snap = deps.market.snapshot(day, w.symbols, w.ticker, w.expiry, w.strikes)
        except Exception as exc:
            snap = MarketSnapshot(date=day, closes={}, puts={})
            notes.append(f"market data provider failed: {exc}")
        notes += snap.notes
        return {"record": DailyRecord(date=day, closes=snap.closes, puts=snap.puts), "notes": notes}

    def fill_gaps(state: RunState) -> dict:
        w, rec = state["watch"], state["record"]
        snap = MarketSnapshot(date=rec.date, closes=dict(rec.closes),
                              puts={k: q.model_copy() for k, q in rec.puts.items()})
        gaps = find_gaps(snap, w.symbols, w.strikes)
        if not gaps:
            return {}
        if deps.gap_filler is None:
            return {"notes": [f"{len(gaps)} data fields N/A (no gap filler): {', '.join(g.field for g in gaps)}"]}
        try:
            with agent_span(deps.tracer, "hermes gap fill", input_value=[g.field for g in gaps]) as span:
                fills = deps.gap_filler.fill_gaps(gap_fill_brief(rec.date, w.ticker, w.expiry, gaps))
                span.set_attribute(OUTPUT, _fills_json(fills))
        except AnalystError as exc:
            return {"notes": [f"gap fill failed: {exc}"]}
        rejected = apply_fills(snap, fills)
        still = find_gaps(snap, w.symbols, w.strikes)
        notes = [f"gap fill rejected — {r}" for r in rejected]
        if still:
            notes.append(f"still N/A after gap fill: {', '.join(g.field for g in still)}")
        return {"record": rec.model_copy(update={"closes": snap.closes, "puts": snap.puts}), "notes": notes}

    def compute(state: RunState) -> dict:
        w, rec = state["watch"], state["record"]
        close = rec.closes.get(w.ticker)
        if close is None or close.price is None:
            raise RuntimeError(f"no {w.ticker} close for {rec.date} — not a trading day, or data unavailable")
        sig = compute_signals(w, rec)
        rec = rec.model_copy(update={"signals": sig})
        return {"record": rec, "brief": analyst_brief(w, rec, sig)}

    def route_after_signals(state: RunState) -> list[str] | str:
        return [f"analyst_{name}" for name in deps.analysts] or END

    def make_analyst_node(name: str, analyst: Analyst):
        def node(state: RunState) -> dict:
            try:
                with agent_span(deps.tracer, f"{name} analyst", input_value=state["brief"]) as span:
                    result = analyst.analyze(state["brief"])
                    annotate_result(deps.tracer, span, result)
            except Exception as exc:
                return {"notes": [f"{name} analyst failed: {exc}"]}
            return {"results": {name: result}}
        return node

    def reconcile_node(state: RunState) -> dict:
        views = {k: r.view for k, r in (state.get("results") or {}).items()}
        if not views:
            raise RuntimeError("all analysts failed: " + "; ".join(state.get("notes") or []))
        recon = reconcile(views, prior_score=_prior_score(state["watch"], state["day"]),
                          rebuttal_done=state.get("rebuttal_done", False))
        return {"reconciliation": recon}

    def route_after_reconcile(state: RunState) -> str:
        return "rebuttal" if state.get("reconciliation") is None else "write_report"

    def rebuttal(state: RunState) -> dict:
        results = state["results"]
        a, b = list(results)[:2]

        def run(own: str, other: str) -> AnalystResult:
            prompt = rebuttal_brief(state["brief"], own, results[own].view, other, results[other].view)
            with agent_span(deps.tracer, f"{own} rebuttal", input_value=prompt) as span:
                result = deps.analysts[own].rebut(prompt)
                annotate_result(deps.tracer, span, result)
            return result

        updated, notes = {}, []
        with ThreadPoolExecutor(max_workers=2) as pool:
            # copy_context per task so rebuttal spans nest under this node's span
            futures = {own: pool.submit(contextvars.copy_context().run, run, own, other)
                       for own, other in ((a, b), (b, a))}
            for own, fut in futures.items():
                try:
                    updated[own] = fut.result()
                except Exception as exc:
                    notes.append(f"{own} rebuttal failed, kept original view: {exc}")
        return {"results": updated, "rebuttal_done": True, "notes": notes}

    def write_report(state: RunState) -> dict:
        w, rec, recon = state["watch"], state["record"], state["reconciliation"]
        views = {k: r.view for k, r in state["results"].items()}
        notes = list(state.get("notes") or [])
        narrative = None
        if deps.writer is not None:
            try:
                prompt = writer_brief(w, rec, recon, views)
                with agent_span(deps.tracer, "report writer", input_value=prompt, kind="LLM") as span:
                    narrative = deps.writer.write(prompt)
                    span.set_attribute(OUTPUT, narrative.model_dump_json())
            except Exception as exc:
                notes.append(f"writer failed, used analyst text: {exc}")
        narrative = narrative or fallback_narrative(recon, views)
        prior = _prior_score(w, rec.date)
        rec = rec.model_copy(update={
            "score": recon.score, "score_delta": None if prior is None else round1(recon.score - prior),
            "verdict": recon.verdict, "analyst_views": views, "reconciliation": recon,
            "note": f"{narrative.headline_ko} {narrative.verdict_ko}"[:600]})
        deps.reports_dir.mkdir(parents=True, exist_ok=True)
        path = deps.reports_dir / f"{rec.date}-{w.ticker}.md"
        path.write_text(render_report(w, rec, recon, narrative, views=views, notes=notes))
        rec.report_path = str(path)
        return {"record": rec, "report_path": str(path)}

    def save(state: RunState) -> dict:
        if not deps.save:
            return {}
        watch = state["watch"].model_copy(deep=True)
        watch.upsert(state["record"])
        save_state(watch, deps.state_path)
        return {"watch": watch}

    g = StateGraph(RunState)
    g.add_node("fetch_market_data", fetch_market_data)
    g.add_node("fill_gaps", fill_gaps)
    g.add_node("compute_signals", compute)
    g.add_node("reconcile", reconcile_node)
    g.add_node("rebuttal", rebuttal)
    g.add_node("write_report", write_report)
    g.add_node("save_state", save)
    g.add_edge(START, "fetch_market_data")
    g.add_edge("fetch_market_data", "fill_gaps")
    g.add_edge("fill_gaps", "compute_signals")
    analyst_nodes = []
    for name, analyst in deps.analysts.items():
        g.add_node(f"analyst_{name}", make_analyst_node(name, analyst))
        analyst_nodes.append(f"analyst_{name}")
    g.add_conditional_edges("compute_signals", route_after_signals, [*analyst_nodes, END])
    if analyst_nodes:
        g.add_edge(analyst_nodes, "reconcile")
    g.add_conditional_edges("reconcile", route_after_reconcile, ["rebuttal", "write_report"])
    g.add_edge("rebuttal", "reconcile")
    g.add_edge("write_report", "save_state")
    g.add_edge("save_state", END)
    return g.compile()

