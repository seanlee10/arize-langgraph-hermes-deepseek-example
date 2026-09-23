import datetime as dt
import re
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from bubble_watch.orchestrator import OrchestratorError, run_day
from bubble_watch.tracing import get_tracer

DAY = dt.date(2026, 9, 18)


def _event(cid, name, kind="tool-call", **block):
    return SimpleNamespace(method="session.event", payload={"sessionId": "s", "event": {
        "type": "assistant/message",
        "data": {"message": {"content": [{"type": kind, "toolCallId": cid, "toolName": name, **block}]}}}})


class FakeHarness:
    """Stands in for DeepSeekHarness: replays notifications, optionally writes the report dsh would."""

    def __init__(self, *, report_path=None, body="# report", finish_reason="stop",
                 notifications=(), fail=None):
        self.report_path, self.body, self.finish_reason = report_path, body, finish_reason
        self.notifications, self.fail = notifications, fail
        self.calls, self.closed = [], False

    def run(self, task, *, session_id=None, on_notification=None):
        self.calls.append((task, session_id))
        for notification in self.notifications:
            on_notification(notification)
        if self.fail is not None:
            raise self.fail
        if self.report_path is not None:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_path.write_text(self.body)
        return SimpleNamespace(final_response="done", session_id="s", finish_reason=self.finish_reason,
                               events=[])

    def close(self):
        self.closed = True


@pytest.fixture
def traced():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, get_tracer(provider)


def _run(tracer, harness, report_path, **kwargs):
    seen = []

    def factory(traceparent):
        seen.append(traceparent)
        return harness

    outcome = run_day(day=DAY, report_path=report_path, harness_for=factory, tracer=tracer, **kwargs)
    return outcome, seen


def test_run_day_returns_the_report_dsh_wrote(traced, tmp_path):
    _, tracer = traced
    report = tmp_path / "reports" / "2026-09-18-NVDA.md"
    outcome, _ = _run(tracer, FakeHarness(report_path=report), report)
    assert outcome.report_path == report
    assert outcome.finish_reason == "stop"


def test_run_day_fails_when_dsh_wrote_no_report(traced, tmp_path):
    _, tracer = traced
    report = tmp_path / "reports" / "2026-09-18-NVDA.md"
    with pytest.raises(OrchestratorError, match="no report"):
        _run(tracer, FakeHarness(report_path=None), report)


def test_run_day_fails_when_dsh_wrote_an_empty_report(traced, tmp_path):
    _, tracer = traced
    report = tmp_path / "reports" / "2026-09-18-NVDA.md"
    with pytest.raises(OrchestratorError, match="no report"):
        _run(tracer, FakeHarness(report_path=report, body="   \n"), report)


def test_run_day_fails_on_a_truncated_run(traced, tmp_path):
    _, tracer = traced
    report = tmp_path / "reports" / "2026-09-18-NVDA.md"
    with pytest.raises(OrchestratorError, match="max_tokens"):
        _run(tracer, FakeHarness(report_path=report, finish_reason="max_tokens"), report)


def test_run_day_asks_for_the_day_and_the_report_path(traced, tmp_path):
    _, tracer = traced
    report = tmp_path / "reports" / "2026-09-18-NVDA.md"
    harness = FakeHarness(report_path=report)
    _run(tracer, harness, report)
    task, session_id = harness.calls[0]
    assert "2026-09-18" in task
    assert str(report) in task
    assert "bubble-watch" in task  # the skill that carries the procedure
    assert session_id.startswith("bubble-watch-2026-09-18-")


def test_run_day_hands_the_root_traceparent_to_the_harness(traced, tmp_path):
    exporter, tracer = traced
    report = tmp_path / "reports" / "2026-09-18-NVDA.md"
    _, seen = _run(tracer, FakeHarness(report_path=report), report)
    assert re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}", seen[0])
    root = next(s for s in exporter.get_finished_spans() if s.name == "bubble-watch run")
    assert f"{root.context.trace_id:032x}" in seen[0]


def test_run_day_builds_dsh_tool_spans_under_a_session_span(traced, tmp_path):
    exporter, tracer = traced
    report = tmp_path / "reports" / "2026-09-18-NVDA.md"
    notifications = [_event("c1", "mcp__bubble__prepare_brief", input={"date": "2026-09-18"}),
                     _event("c1", "mcp__bubble__prepare_brief", kind="tool-result", output={"gaps": []}),
                     _event("c2", "hermes_analyst", input={"task": "research"}),
                     _event("c2", "hermes_analyst", kind="tool-result", output="view")]
    outcome, _ = _run(tracer, FakeHarness(report_path=report, notifications=notifications), report)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert outcome.tool_call_count == 2
    assert spans["mcp__bubble__prepare_brief"].parent.span_id == spans["dsh session"].context.span_id
    assert spans["dsh session"].parent.span_id == spans["bubble-watch run"].context.span_id
    assert spans["hermes_analyst"].attributes["openinference.span.kind"] == "TOOL"


def test_run_day_closes_the_harness_and_marks_the_span_when_dsh_fails(traced, tmp_path):
    exporter, tracer = traced
    report = tmp_path / "reports" / "2026-09-18-NVDA.md"
    harness = FakeHarness(fail=RuntimeError("stdout closed"))
    with pytest.raises(OrchestratorError, match="stdout closed"):
        _run(tracer, harness, report)
    assert harness.closed
    root = next(s for s in exporter.get_finished_spans() if s.name == "bubble-watch run")
    assert root.status.status_code == StatusCode.ERROR


def test_each_run_of_a_day_gets_its_own_dsh_session(traced, tmp_path):
    """dsh persists sessions in DSH_HOME and rejects a duplicate id, so re-running a day with a
    fixed id fails with `session already exists`. The Arize grouping key stays stable instead."""
    exporter, tracer = traced
    report = tmp_path / "reports" / "2026-09-18-NVDA.md"
    seen = []
    for _ in range(2):
        harness = FakeHarness(report_path=report)
        _run(tracer, harness, report)
        seen.append(harness.calls[0][1])
    assert seen[0] != seen[1]
    session_spans = [s for s in exporter.get_finished_spans() if s.name == "dsh session"]
    assert {s.attributes["session.id"] for s in session_spans} == {"bubble-watch-2026-09-18"}
