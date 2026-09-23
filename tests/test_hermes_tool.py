"""Hermes as an MCP tool, shelling out to `hermes chat -Q`.

Driven over its CLI rather than ACP deliberately: the ACP adapter builds an AIAgent directly and
never touches hermes_cli's plugin dispatch, so no plugin hook — including observability/arize —
fires over that path. The CLI path does, so Hermes traces itself into this run's trace.
"""
import subprocess

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from bubble_watch.config import load_settings
from bubble_watch.hermes_tool import HermesError, hermes_analyst
from bubble_watch.tracing import get_tracer


@pytest.fixture
def traced():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, get_tracer(provider)


def _settings(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("XAI_API_KEY", "xai-k")
    monkeypatch.setenv("HERMES_ANALYST_HOME", str(tmp_path / "hh"))
    monkeypatch.setenv("HERMES_BIN", str(tmp_path / "hermes"))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return load_settings(tmp_path / "none.env")


def _runner(stdout="답변입니다.", stderr="session_id: 20260923_abc\n", returncode=0, capture=None):
    def run(cmd, **kwargs):
        if capture is not None:
            capture.append({"cmd": cmd, "env": kwargs.get("env", {})})
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    return run


def test_returns_the_answer_and_the_session_id(tmp_path, monkeypatch, traced):
    _, tracer = traced
    result = hermes_analyst(_settings(tmp_path, monkeypatch), "리서치해라",
                            tracer=tracer, runner=_runner())
    assert result["text"] == "답변입니다."
    assert result["session_id"] == "20260923_abc"


def test_opens_an_agent_span_per_call(tmp_path, monkeypatch, traced):
    exporter, tracer = traced
    hermes_analyst(_settings(tmp_path, monkeypatch), "리서치해라", tracer=tracer, runner=_runner())
    span = next(s for s in exporter.get_finished_spans() if s.name == "hermes analyst")
    assert span.attributes["openinference.span.kind"] == "AGENT"
    assert span.status.status_code == StatusCode.OK


def test_hands_hermes_the_traceparent_of_its_own_span(tmp_path, monkeypatch, traced):
    """The whole point of shelling out from here: this process has a real OTel context, so Hermes'
    spans nest under THIS call rather than under the run root."""
    exporter, tracer = traced
    calls = []
    hermes_analyst(_settings(tmp_path, monkeypatch), "리서치해라", tracer=tracer,
                   runner=_runner(capture=calls))
    span = next(s for s in exporter.get_finished_spans() if s.name == "hermes analyst")
    traceparent = calls[0]["env"]["HERMES_ARIZE_TRACEPARENT"]
    assert f"{span.context.trace_id:032x}" in traceparent
    assert f"{span.context.span_id:016x}" in traceparent


def test_passes_the_isolated_home_and_strips_platform_credentials(tmp_path, monkeypatch, traced):
    _, tracer = traced
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "should-not-leak")
    calls = []
    settings = _settings(tmp_path, monkeypatch)
    hermes_analyst(settings, "x", tracer=tracer, runner=_runner(capture=calls))
    env = calls[0]["env"]
    assert env["HERMES_HOME"] == str(settings.hermes_home)
    assert "TELEGRAM_BOT_TOKEN" not in env
    assert env["TIRITH_ENABLED"] == "false"


def test_uses_the_web_toolset_and_one_shot_flags(tmp_path, monkeypatch, traced):
    _, tracer = traced
    calls = []
    hermes_analyst(_settings(tmp_path, monkeypatch), "x", tracer=tracer, runner=_runner(capture=calls))
    cmd = calls[0]["cmd"]
    assert cmd[1] == "chat" and "-Q" in cmd
    assert cmd[cmd.index("-t") + 1] == "web"     # research only: no terminal toolset
    assert "--query-file" in cmd                  # avoids arg-length and escaping limits


def test_resumes_a_session_when_one_is_given(tmp_path, monkeypatch, traced):
    _, tracer = traced
    calls = []
    hermes_analyst(_settings(tmp_path, monkeypatch), "반박해라", session_id="20260923_abc",
                   tracer=tracer, runner=_runner(capture=calls))
    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--resume") + 1] == "20260923_abc"


def test_a_nonzero_exit_raises_with_the_useful_stderr_lines(tmp_path, monkeypatch, traced):
    _, tracer = traced
    runner = _runner(stdout="", stderr="boom: provider refused\nsession_id: x\n", returncode=1)
    with pytest.raises(HermesError, match="provider refused"):
        hermes_analyst(_settings(tmp_path, monkeypatch), "x", tracer=tracer, runner=runner)


def test_an_empty_answer_raises(tmp_path, monkeypatch, traced):
    _, tracer = traced
    with pytest.raises(HermesError, match="no answer"):
        hermes_analyst(_settings(tmp_path, monkeypatch), "x", tracer=tracer, runner=_runner(stdout="   "))


def test_a_failure_marks_the_span_as_error(tmp_path, monkeypatch, traced):
    exporter, tracer = traced
    with pytest.raises(HermesError):
        hermes_analyst(_settings(tmp_path, monkeypatch), "x", tracer=tracer,
                       runner=_runner(stdout="", returncode=1, stderr="nope"))
    span = next(s for s in exporter.get_finished_spans() if s.name == "hermes analyst")
    assert span.status.status_code == StatusCode.ERROR


def test_a_timeout_raises_rather_than_hanging(tmp_path, monkeypatch, traced):
    _, tracer = traced

    def run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 5)

    with pytest.raises(HermesError, match="timed out"):
        hermes_analyst(_settings(tmp_path, monkeypatch), "x", tracer=tracer, runner=run)


def test_gives_hermes_a_stream_idle_timeout_fit_for_a_reasoning_model(tmp_path, monkeypatch, traced):
    """Hermes' stream watchdog arms on the first parsed event and kills after its idle timeout.
    The default resolved to 12s against xAI, and grok-4.6 reasons silently for far longer — every
    analyst call died with `no SSE events for 12s` and was retried three times before failing."""
    _, tracer = traced
    calls = []
    hermes_analyst(_settings(tmp_path, monkeypatch), "x", tracer=tracer, runner=_runner(capture=calls))
    assert int(calls[0]["env"]["HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS"]) >= 120


def test_the_stream_idle_timeout_is_overridable(tmp_path, monkeypatch, traced):
    _, tracer = traced
    calls = []
    settings = _settings(tmp_path, monkeypatch, HERMES_STREAM_IDLE_S="45")
    hermes_analyst(settings, "x", tracer=tracer, runner=_runner(capture=calls))
    assert calls[0]["env"]["HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS"] == "45"
