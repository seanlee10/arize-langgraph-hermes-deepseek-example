"""Arize AX tracing: LangGraph auto-instrumentation plus explicit AGENT/TOOL spans for the sub-agents."""
from __future__ import annotations

import contextlib
import json
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

KIND, INPUT, OUTPUT = "openinference.span.kind", "input.value", "output.value"
MAX_ATTR = 16_000


def _text(value: Any) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return s if len(s) <= MAX_ATTR else s[:MAX_ATTR] + f"... [truncated {len(s) - MAX_ATTR} chars]"


def setup_tracing(settings: Any) -> Any:
    """Register an Arize tracer provider and instrument LangGraph; None (no-op) without credentials."""
    if not (settings.arize_space_id and settings.arize_api_key):
        return None
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    provider = register(space_id=settings.arize_space_id, api_key=settings.arize_api_key,
                        project_name=settings.arize_project, endpoint=settings.arize_endpoint,
                        set_global_tracer_provider=False, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    return provider


def get_tracer(provider: Any) -> trace.Tracer:
    return trace.NoOpTracer() if provider is None else provider.get_tracer("bubble_watch")


def shutdown_tracing(provider: Any) -> None:
    if provider is not None:
        provider.force_flush()
        provider.shutdown()


@contextlib.contextmanager
def agent_span(tracer: trace.Tracer, name: str, *, input_value: Any, kind: str = "AGENT",
               attributes: dict[str, Any] | None = None) -> Iterator[trace.Span]:
    with tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False) as span:
        span.set_attribute(KIND, kind)
        span.set_attribute(INPUT, _text(input_value))
        for key, value in (attributes or {}).items():
            if value is not None:
                span.set_attribute(key, value)
        try:
            yield span
        except BaseException as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            raise
        span.set_status(Status(StatusCode.OK))


def _blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    data = event.get("data") or {}
    message = data.get("message") if isinstance(data.get("message"), dict) else data
    content = message.get("content") if isinstance(message, dict) else None
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def dsh_tool_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair dsh ``tool-call`` and ``tool-result`` content blocks by ``toolCallId``, in call order."""
    calls: dict[str, dict[str, Any]] = {}
    for event in events:
        failed = bool((event.get("data") or {}).get("error"))
        for block in _blocks(event):
            kind, cid = block.get("type"), block.get("toolCallId") or block.get("id")
            if kind not in ("tool-call", "tool-result") or not cid:
                continue
            call = calls.setdefault(cid, {"id": cid, "name": block.get("toolName") or "tool", "input": None,
                                          "output": None, "error": False})
            if kind == "tool-call":
                call["input"] = block.get("input", block.get("args"))
            else:
                call["output"] = block.get("output", block.get("result"))
                call["error"] = call["error"] or failed or bool(block.get("isError"))
    return list(calls.values())


def record_dsh_tool_spans(tracer: trace.Tracer, events: list[dict[str, Any]]) -> int:
    calls = dsh_tool_calls(events)
    for call in calls:
        with tracer.start_as_current_span(call["name"]) as span:
            span.set_attribute(KIND, "TOOL")
            span.set_attribute("tool.name", call["name"])
            span.set_attribute("tool.id", call["id"])
            span.set_attribute(INPUT, _text(call["input"]))
            span.set_attribute(OUTPUT, _text(call["output"]))
            span.set_status(Status(StatusCode.ERROR, "tool failed") if call["error"] else Status(StatusCode.OK))
    return len(calls)


def annotate_result(tracer: trace.Tracer, span: trace.Span, result: Any) -> None:
    """Attach an AnalystResult to its AGENT span: validated view as output, session link, dsh tool children."""
    span.set_attribute(OUTPUT, _text(result.view.model_dump(mode="json")))
    span.set_attribute("bubble_watch.retried", result.retried)
    if result.session_id:
        span.set_attribute(f"{result.agent}.session_id", result.session_id)
    if result.events:
        span.set_attribute("dsh.tool_call_count", record_dsh_tool_spans(tracer, result.events))
