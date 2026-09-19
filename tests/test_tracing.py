import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from bubble_watch.agents.base import AnalystResult
from bubble_watch.models import AnalystView
from bubble_watch.tracing import agent_span, annotate_result, dsh_tool_calls, get_tracer, setup_tracing

EVENTS = [
    {"type": "assistant/message", "data": {"message": {"role": "assistant", "content": [
        {"type": "text", "text": "searching"},
        {"type": "tool-call", "toolCallId": "c1", "toolName": "web_search", "input": {"query": "nvda huawei"}}]}}},
    {"type": "tool/result", "data": {"message": {"source": {"kind": "tool", "callId": "c1"}, "content": [
        {"type": "tool-result", "toolCallId": "c1", "toolName": "web_search", "output": {"results": 3}}]}}},
    {"type": "tool/result", "data": {"error": {"code": "DENIED"}, "message": {"content": [
        {"type": "tool-result", "toolCallId": "c2", "toolName": "bash", "output": None}]}}},
]


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


def test_dsh_tool_calls_pairs_calls_and_results():
    calls = dsh_tool_calls(EVENTS)
    assert [(c["name"], c["error"]) for c in calls] == [("web_search", False), ("bash", True)]
    assert calls[0]["input"] == {"query": "nvda huawei"} and calls[0]["output"] == {"results": 3}


def test_annotate_result_adds_output_session_and_tool_children(exporter):
    exp, tracer = exporter
    view = AnalystView.model_validate({"score": 8, "score_delta_reasoning_ko": "r", "verdict": "TRIGGERED",
                                       "tape_read_ko": "t", "watch_conditions_ko": "w"})
    result = AnalystResult(agent="dsh", view=view, session_id="session-9", events=EVENTS)
    with agent_span(tracer, "dsh analyst", input_value="brief") as span:
        annotate_result(tracer, span, result)
    spans = {s.name: s for s in exp.get_finished_spans()}
    parent = spans["dsh analyst"]
    assert parent.attributes["dsh.session_id"] == "session-9"
    assert '"score": 8.0' in parent.attributes["output.value"]
    tool = spans["web_search"]
    assert tool.parent.span_id == parent.context.span_id
    assert tool.attributes["openinference.span.kind"] == "TOOL" and tool.status.status_code == StatusCode.OK
    assert spans["bash"].status.status_code == StatusCode.ERROR
