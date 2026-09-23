"""The thin driver around dsh: opens the trace, launches the harness, verifies a report came back.

Every decision about the run belongs to dsh and the `bubble-watch` skill. This module owns only the
things a model cannot be trusted with: the trace root, the W3C `traceparent` both child runtimes
parent on, and the check that a non-empty report actually landed on disk.
"""
from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from .tracing import OUTPUT, DshSpanBuilder, agent_span

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
    """What dsh is asked to do. The procedure itself lives in the `bubble-watch` skill."""
    return f"""오늘({day})의 NVDA Bubble Signal Watch 리포트를 작성하라.

`skill` 도구로 `bubble-watch` 스킬을 먼저 읽고, 거기 적힌 절차를 그대로 따른다.

- 대상 날짜: {day}
- 리포트 파일: {report_path}

규칙:
- 모든 수치는 `mcp__bubble__*` 도구가 돌려준 계산값만 사용한다. 직접 계산하거나 지어내지 않는다.
- 웹 리서치와 결측값 조사는 `hermes_analyst` 서브에이전트에 위임하고, 너는 독립적으로 자신의 판단을 세운 뒤
  두 견해를 비교·조정한다.
- 마지막에 위 경로에 한국어 리포트를 쓰고 `mcp__bubble__save_run`으로 결과를 저장한다."""


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


def run_day(*, day: dt.date, report_path: Path, harness_for: Callable[[str], Any],
            tracer: trace.Tracer) -> RunOutcome:
    """Run one day through dsh and return the verified outcome.

    `harness_for` receives this run's `traceparent`; it must reach the harness environment before
    launch, because the ACP and MCP composition rows read it once, at load time.
    """
    task = build_task(day, report_path)
    dsh_session, grouping = session_ids(day)
    with agent_span(tracer, "bubble-watch run", input_value=task, kind="CHAIN",
                    attributes={"bubble_watch.date": day.isoformat()}) as root:
        harness = harness_for(_traceparent())
        builder = DshSpanBuilder(tracer)
        try:
            with agent_span(tracer, "dsh session", input_value=task,
                            attributes={"session.id": grouping, "dsh.session_id": dsh_session}) as session:
                try:
                    result = harness.run(task, session_id=dsh_session, on_notification=builder)
                finally:
                    builder.close()
                session.set_attribute(OUTPUT, result.final_response or "")
                session.set_attribute("dsh.tool_call_count", builder.tool_call_count)
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
        return RunOutcome(report_path=report_path, final_response=result.final_response or "",
                          finish_reason=finish_reason, tool_call_count=builder.tool_call_count,
                          session_id=getattr(result, "session_id", "") or "")
