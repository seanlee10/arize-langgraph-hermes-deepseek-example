"""The thin driver around dsh: opens the trace, launches the harness, verifies a report came back.

Every decision about the run belongs to dsh and the `bubble-watch` skill. This module owns only the
things a model cannot be trusted with: the trace root, the W3C `traceparent` both child runtimes
parent on, and the check that a non-empty report actually landed on disk.
"""
from __future__ import annotations

import datetime as dt
import inspect
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from .tracing import OUTPUT, DshSpanBuilder, agent_span, session_context

# dsh's own vocabulary for "the model stopped because it was finished".
COMPLETED = frozenset({"stop", "end_turn", "completed", "tool_calls", ""})


class OrchestratorError(RuntimeError):
    """The run did not produce a trustworthy report."""


@dataclass
class RunOutcome:
    report_path: Path
    final_response: str
    finish_reason: str
    tool_call_count: int
    session_id: str


def build_task(day: dt.date, report_path: Path) -> str:
    """What dsh is asked to do. The procedure itself lives in the `bubble-watch` skill.

    English, like every other prompt here: this text is the root span's input, and the people
    reading these traces do not read Korean. Only the report is Korean — it is the product.
    """
    return f"""Produce today's ({day}) NVDA Bubble Signal Watch report.

First read the `bubble-watch` skill with the `skill` tool, then follow the procedure it describes.

- Target date: {day}
- Report file: {report_path}

Rules:
- Every figure must come from an `mcp__bubble__*` tool. Never compute or recall a number yourself.
- Delegate web research and missing-value lookups to `mcp__hermes__analyst`, form your own
  independent view, then reconcile the two.
- Write the report in KOREAN to the path above, then record the result with
  `mcp__bubble__save_run`.
"""


def _start_harness(harness_for: Any, traceparent: str, session_id: str) -> Any:
    """Call the factory, tolerating the one-argument form used by tests and older callers.

    The arity is decided from the signature rather than by catching TypeError: a TypeError raised
    *inside* the factory would otherwise be mistaken for an arity mismatch, retried — re-running
    every side effect — and finally reported as a misleading "missing argument" error.
    """
    try:
        takes_two = len(inspect.signature(harness_for).parameters) >= 2
    except (TypeError, ValueError):       # builtins and C callables have no introspectable signature
        takes_two = True
    return harness_for(traceparent, session_id) if takes_two else harness_for(traceparent)


def _traceparent() -> str:
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier.get("traceparent", "")


def session_ids(day: dt.date) -> tuple[str, str]:
    """(dsh session id, Arize session key) for one run of `day`.

    They are deliberately different. dsh persists sessions under DSH_HOME and rejects a duplicate
    id, so re-running a day with a fixed id fails outright with `session already exists`; the Arize
    grouping key must stay stable across those re-runs to keep a day's traces together.
    """
    grouping = f"bubble-watch-{day}"
    return f"{grouping}-{uuid.uuid4().hex[:8]}", grouping


def run_day(*, day: dt.date, report_path: Path, report_path_for_dsh: str = "",
            harness_for: Callable[[str, str], Any] | Callable[[str], Any],
            tracer: trace.Tracer) -> RunOutcome:
    """Run one day through dsh and return the verified outcome.

    `harness_for` receives this run's `traceparent`; it must reach the harness environment before
    launch, because the MCP composition rows read it once, at load time.

    `report_path` is where the driver verifies the output; `report_path_for_dsh` is the same file
    as the harness sees it, which differs when the harness runs in a container.
    """
    task = build_task(day, report_path_for_dsh or str(report_path))
    dsh_session, grouping = session_ids(day)
    # One root span, and it is the dsh run. dsh cannot open it itself — it emits OTel logs rather
    # than spans, and the traceparent has to exist before it launches, because the MCP composition
    # reads its `env` once at load. So the driver opens it on dsh's behalf. A second wrapper span
    # above this one would only push the harness a level down for nothing.
    with session_context(grouping), agent_span(
            tracer, "dsh", input_value=task,
            attributes={"bubble_watch.date": day.isoformat(),
                        "dsh.session_id": dsh_session}) as root:
        harness = _start_harness(harness_for, _traceparent(), grouping)
        builder = DshSpanBuilder(tracer)
        try:
            try:
                result = harness.run(task, session_id=dsh_session, on_notification=builder)
            finally:
                builder.close()
        except OrchestratorError:
            raise
        except Exception as exc:
            raise OrchestratorError(f"dsh run failed: {type(exc).__name__}: {str(exc)[:300]}") from exc
        finally:
            harness.close()

        finish_reason = result.finish_reason or ""
        if finish_reason not in COMPLETED:
            raise OrchestratorError(f"dsh stopped early (finish_reason={finish_reason}); report not trusted")
        if not report_path.exists() or not report_path.read_text().strip():
            raise OrchestratorError(f"dsh wrote no report at {report_path}")
        root.set_attribute(OUTPUT, report_path.read_text()[:4000])
        root.set_attribute("bubble_watch.report_path", str(report_path))
        root.set_attribute("dsh.tool_call_count", builder.tool_call_count)
        root.set_attribute("dsh.final_response", (result.final_response or "")[:4000])
        return RunOutcome(report_path=report_path, final_response=result.final_response or "",
                          finish_reason=finish_reason, tool_call_count=builder.tool_call_count,
                          session_id=getattr(result, "session_id", "") or "")
