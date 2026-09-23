"""Arize AX tracing: LangGraph auto-instrumentation plus explicit AGENT/TOOL spans for the sub-agents."""
from __future__ import annotations

import contextlib
import contextvars
import json
from collections.abc import Iterator
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

KIND, INPUT, OUTPUT = "openinference.span.kind", "input.value", "output.value"
SESSION = "session.id"

#: The Arize grouping key for the current run. `using_session` only reaches spans made by an
#: OpenInference instrumentor (LangGraph's); the spans this module builds by hand read it here.
_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar("bubble_watch_session", default="")
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
def session_context(session_id: str) -> Iterator[None]:
    """Stamp `session.id` on every span opened inside, not just one.

    Arize groups sessions by that attribute, so setting it on a single span yields a partial tree
    in a session-filtered view. Two mechanisms are needed, because they cover different spans:
    `using_session` is read by OpenInference instrumentors (LangGraph), while the spans built here
    by hand read the context variable directly. Empty id is a no-op.
    """
    if not session_id:
        yield
        return
    from openinference.instrumentation import using_session

    token = _SESSION_ID.set(session_id)
    try:
        with using_session(session_id):
            yield
    finally:
        _SESSION_ID.reset(token)


def _stamp_session(span: trace.Span) -> None:
    if session_id := _SESSION_ID.get():
        span.set_attribute(SESSION, session_id)


def current_trace_id() -> str | None:
    """The trace this process is contributing to, as 32 hex characters.

    Works across a process boundary: the MCP server attaches the run's remote parent context at
    startup, so the ambient span context carries the caller's trace id even with no local span.
    """
    ctx = trace.get_current_span().get_span_context()
    return f"{ctx.trace_id:032x}" if ctx.is_valid else None


def remote_parent_context(env: dict[str, str] | None = None) -> Any:
    """The caller's span context, read from a W3C `traceparent` in the environment.

    The MCP server runs in its own process, two hops from the driver (driver → dsh → server), so it
    joins the run's trace through this rather than through an in-process parent. Returns None when
    no usable traceparent is present, which makes the server's spans start their own trace.
    """
    import os

    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    traceparent = (env if env is not None else os.environ).get("TRACEPARENT", "")
    if not traceparent:
        return None
    context = TraceContextTextMapPropagator().extract({"traceparent": traceparent})
    span_context = trace.get_current_span(context).get_span_context()
    return context if span_context.is_valid else None


def _parent_context() -> Any:
    """Parent for a sub-agent span. The LangChain instrumentor tracks node spans by run id without
    making them the current OTel span, so inside a LangGraph node look the node span up explicitly;
    otherwise these spans would each start a separate trace."""
    if trace.get_current_span().get_span_context().is_valid:
        return None  # already inside a traced context: use it
    try:
        # get_current_span() reads langchain_core.runnables.config without importing it; load it first
        # so a span opened before anything else imported LangChain doesn't raise AttributeError.
        import langchain_core.runnables.config  # noqa: F401
        from openinference.instrumentation.langchain import get_current_span
        node_span = get_current_span()
    except ImportError:  # LangChain instrumentor not installed
        return None
    return trace.set_span_in_context(node_span) if node_span is not None else None


@contextlib.contextmanager
def agent_span(tracer: trace.Tracer, name: str, *, input_value: Any, kind: str = "AGENT",
               attributes: dict[str, Any] | None = None) -> Iterator[trace.Span]:
    with tracer.start_as_current_span(name, context=_parent_context(), record_exception=False,
                                      set_status_on_exception=False) as span:
        span.set_attribute(KIND, kind)
        span.set_attribute(INPUT, _text(input_value))
        _stamp_session(span)
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


_INPUT_KEYS, _OUTPUT_KEYS = ("input", "arguments", "args"), ("output", "content", "result")


class DshSpanBuilder:
    """Build dsh's TOOL spans live from the Python SDK's ``on_notification`` stream.

    dsh emits OTel *logs*, not spans, so its tool activity is reconstructed here. Doing it from the
    live callback rather than from the finished event list is what gives the spans real durations:
    a ``tool-call`` block opens a span, the matching ``tool-result`` closes it. Call ``close()`` when
    the run ends so a call that never got a result is still terminated (as ERROR) rather than leaked.
    """

    def __init__(self, tracer: trace.Tracer, context: Any = None) -> None:
        self._tracer = tracer
        self._context = context
        self._open: dict[str, trace.Span] = {}
        self.tool_call_count = 0

    def __call__(self, notification: Any) -> None:
        if getattr(notification, "method", None) != "session.event":
            return
        event = (getattr(notification, "payload", None) or {}).get("event")
        if not isinstance(event, dict):
            return
        failed = bool((event.get("data") or {}).get("error"))
        for block in _blocks(event):
            cid = block.get("toolCallId") or block.get("id")
            if not cid:
                continue
            if block.get("type") == "tool-call":
                self._start(cid, block)
            elif block.get("type") == "tool-result":
                self._finish(cid, block, failed)

    def _start(self, cid: str, block: dict[str, Any]) -> None:
        if cid in self._open:  # a repeated call block for the same id is the same call
            return
        name = block.get("toolName") or block.get("name") or "tool"
        span = self._tracer.start_span(name, context=self._context)
        span.set_attribute(KIND, "TOOL")
        _stamp_session(span)
        span.set_attribute("tool.name", name)
        span.set_attribute("tool.id", cid)
        span.set_attribute(INPUT, _text(next((block[k] for k in _INPUT_KEYS if k in block), None)))
        self._open[cid] = span
        self.tool_call_count += 1

    def _finish(self, cid: str, block: dict[str, Any], failed: bool) -> None:
        span = self._open.pop(cid, None)
        if span is None:
            return
        span.set_attribute(OUTPUT, _text(next((block[k] for k in _OUTPUT_KEYS if k in block), None)))
        errored = failed or bool(block.get("isError"))
        span.set_status(Status(StatusCode.ERROR, "tool failed") if errored else Status(StatusCode.OK))
        span.end()

    def close(self) -> None:
        for span in self._open.values():
            span.set_status(Status(StatusCode.ERROR, "no result (run ended first)"))
            span.end()
        self._open.clear()
