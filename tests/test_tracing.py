import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from bubble_watch.tracing import agent_span, get_tracer, setup_tracing


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    return exp, get_tracer(provider)


def test_setup_tracing_is_noop_without_credentials():
    from types import SimpleNamespace
    assert setup_tracing(SimpleNamespace(arize_space_id="", arize_api_key="")) is None
    with agent_span(get_tracer(None), "x", input_value="y"):
        pass


def test_agent_span_sets_terminal_status(exporter):
    exp, tracer = exporter
    with agent_span(tracer, "ok", input_value="in"):
        pass
    with pytest.raises(RuntimeError), agent_span(tracer, "bad", input_value="in"):
        raise RuntimeError("boom")
    spans = {s.name: s for s in exp.get_finished_spans()}
    assert spans["ok"].status.status_code == StatusCode.OK
    assert spans["ok"].attributes["openinference.span.kind"] == "AGENT"
    assert spans["bad"].status.status_code == StatusCode.ERROR


# --- DshSpanBuilder: dsh's spans, built live from the SDK notification stream -------------------

def _notification(event):
    from types import SimpleNamespace
    return SimpleNamespace(method="session.event", payload={"sessionId": "s1", "event": event})


def _call(cid, name, **extra):
    return _notification({"type": "assistant/message", "data": {"message": {"content": [
        {"type": "tool-call", "toolCallId": cid, "toolName": name, "input": {"q": "x"}}]}, **extra}})


def _result(cid, name, output="done", **extra):
    return _notification({"type": "tool/result", "data": {"message": {"content": [
        {"type": "tool-result", "toolCallId": cid, "toolName": name, "output": output}]}, **extra}})


def test_span_builder_opens_and_closes_a_span_per_tool_call(exporter):
    from bubble_watch.tracing import DshSpanBuilder
    exp, tracer = exporter
    builder = DshSpanBuilder(tracer)
    builder(_call("c1", "mcp__bubble__prepare_brief"))
    assert exp.get_finished_spans() == ()  # still running: a live span, not a post-hoc one
    builder(_result("c1", "mcp__bubble__prepare_brief", output={"gaps": []}))
    builder.close()
    span = exp.get_finished_spans()[0]
    assert span.name == "mcp__bubble__prepare_brief"
    assert span.attributes["openinference.span.kind"] == "TOOL"
    assert "gaps" in span.attributes["output.value"]
    assert span.status.status_code == StatusCode.OK


def test_span_builder_marks_a_failed_tool_result_as_error(exporter):
    from bubble_watch.tracing import DshSpanBuilder
    exp, tracer = exporter
    builder = DshSpanBuilder(tracer)
    builder(_call("c1", "hermes_analyst"))
    builder(_result("c1", "hermes_analyst", error={"code": "ACP_FAILED"}))
    builder.close()
    assert exp.get_finished_spans()[0].status.status_code == StatusCode.ERROR


def test_span_builder_closes_an_unanswered_tool_call_on_close(exporter):
    from bubble_watch.tracing import DshSpanBuilder
    exp, tracer = exporter
    builder = DshSpanBuilder(tracer)
    builder(_call("c1", "web_search"))
    builder.close()
    span = exp.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "no result" in span.status.description


def test_span_builder_nests_tool_spans_under_the_session_span(exporter):
    from bubble_watch.tracing import DshSpanBuilder
    exp, tracer = exporter
    with agent_span(tracer, "dsh session", input_value="task") as session:
        builder = DshSpanBuilder(tracer)
        builder(_call("c1", "web_search"))
        builder(_result("c1", "web_search"))
        builder.close()
    tool = next(s for s in exp.get_finished_spans() if s.name == "web_search")
    assert tool.parent.span_id == session.get_span_context().span_id


def test_span_builder_ignores_notifications_that_are_not_session_events(exporter):
    from types import SimpleNamespace

    from bubble_watch.tracing import DshSpanBuilder
    exp, tracer = exporter
    builder = DshSpanBuilder(tracer)
    builder(SimpleNamespace(method="session.status", payload={"sessionId": "s1", "status": "idle"}))
    builder.close()
    assert exp.get_finished_spans() == ()


def test_span_builder_counts_the_tool_calls_it_recorded(exporter):
    from bubble_watch.tracing import DshSpanBuilder
    _, tracer = exporter
    builder = DshSpanBuilder(tracer)
    builder(_call("c1", "web_search"))
    builder(_result("c1", "web_search"))
    builder(_call("c2", "web_fetch"))
    builder(_result("c2", "web_fetch"))
    builder.close()
    assert builder.tool_call_count == 2


# --- joining the caller's trace from a child process -------------------------------------------

TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


def test_remote_parent_context_is_none_without_a_traceparent():
    from bubble_watch.tracing import remote_parent_context
    assert remote_parent_context({}) is None
    assert remote_parent_context({"TRACEPARENT": ""}) is None


def test_remote_parent_context_adopts_the_callers_trace(exporter):
    from bubble_watch.tracing import remote_parent_context
    exp, tracer = exporter
    context = remote_parent_context({"TRACEPARENT": TRACEPARENT})
    with tracer.start_as_current_span("child", context=context):
        pass
    span = exp.get_finished_spans()[0]
    assert f"{span.context.trace_id:032x}" == "0af7651916cd43dd8448eb211c80319c"
    assert f"{span.parent.span_id:016x}" == "b7ad6b7169203331"


def test_remote_parent_context_ignores_a_malformed_traceparent():
    from bubble_watch.tracing import remote_parent_context
    assert remote_parent_context({"TRACEPARENT": "not-a-traceparent"}) is None


# --- the Arize session key, stamped on every span rather than one ---------------------------------

def test_session_context_stamps_every_span(exporter):
    from bubble_watch.tracing import session_context
    exp, tracer = exporter
    # `with a, b, c` nests exactly as nested `with`s do: inner really is a child of outer, which
    # is the point — a child span must inherit the key too, not just the one opened at the top.
    with (session_context("bubble-watch-2026-09-18"),
          agent_span(tracer, "outer", input_value="x", kind="CHAIN"),
          agent_span(tracer, "inner", input_value="y")):
        pass
    ids = {s.attributes.get("session.id") for s in exp.get_finished_spans()}
    assert ids == {"bubble-watch-2026-09-18"}


def test_session_context_is_a_noop_without_an_id(exporter):
    from bubble_watch.tracing import session_context
    exp, tracer = exporter
    with session_context(""), agent_span(tracer, "solo", input_value="x"):
        pass
    assert exp.get_finished_spans()[0].attributes.get("session.id") is None
