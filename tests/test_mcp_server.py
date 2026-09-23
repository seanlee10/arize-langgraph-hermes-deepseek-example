import asyncio
import datetime as dt
import json

from test_mcp_tools import DAY, FakeMarket

from bubble_watch.mcp_server import build_server
from bubble_watch.mcp_tools import ToolDeps
from bubble_watch.state_store import load_state, save_state, seed_state


def _deps(tmp_path, market=None):
    state_path = tmp_path / "state" / "NVDA.json"
    save_state(seed_state(), state_path)
    return ToolDeps(market=market or FakeMarket(), state_path=state_path)


def _call(server, name, arguments):
    return asyncio.run(server.call_tool(name, arguments))


def test_server_exposes_the_deterministic_tools(tmp_path):
    server = build_server(_deps(tmp_path))
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert names == {"prepare_brief", "apply_gap_fills", "prior_state", "save_run"}


def test_every_tool_describes_itself_for_the_model(tmp_path):
    server = build_server(_deps(tmp_path))
    for tool in asyncio.run(server.list_tools()):
        assert tool.description and len(tool.description) > 30


def test_prepare_brief_tool_returns_the_computed_signals(tmp_path):
    server = build_server(_deps(tmp_path))
    result = _call(server, "prepare_brief", {"date": DAY.isoformat()})
    assert result.structured_content["gaps"] == []
    assert result.structured_content["signals"]["returns"]["1d"]["NVDA"] is not None


def test_apply_gap_fills_tool_rejects_an_unsourced_value(tmp_path):
    server = build_server(_deps(tmp_path, FakeMarket(drop=("puts.220.iv",))))
    _call(server, "prepare_brief", {"date": DAY.isoformat()})
    result = _call(server, "apply_gap_fills",
                   {"date": DAY.isoformat(), "fills": [{"field": "puts.220.iv", "value": 30.21}]})
    assert result.structured_content["accepted"] == 0


def test_save_run_tool_persists_the_day(tmp_path):
    deps = _deps(tmp_path)
    server = build_server(deps)
    _call(server, "prepare_brief", {"date": DAY.isoformat()})
    _call(server, "save_run", {"date": DAY.isoformat(), "score": 7.9,
                               "verdict": "TRIGGERED_FURTHER_DE_CONFIRMING",
                               "note": "n", "report_path": "reports/x.md"})
    record = next(r for r in load_state(deps.state_path).records if r.date == dt.date(2026, 9, 18))
    assert record.score == 7.9


def test_server_instructions_forbid_recomputing_numbers(tmp_path):
    server = build_server(_deps(tmp_path))
    assert "compute" in (server.instructions or "").lower()


def test_everything_the_model_reads_is_english(tmp_path):
    """These strings land in span attributes; the people reading the traces do not read Korean."""
    server = build_server(_deps(tmp_path))
    surfaces = [server.instructions or ""]
    for tool in asyncio.run(server.list_tools()):
        surfaces += [tool.description or "", json.dumps(tool.input_schema, ensure_ascii=False)]
    hangul = [s for s in surfaces if any("\uac00" <= c <= "\ud7a3" for c in s)]
    assert hangul == []


def test_the_analyst_is_served_separately_from_the_deterministic_tools(tmp_path):
    from bubble_watch.mcp_server import build_hermes_server
    tools = {t.name for t in asyncio.run(build_server(_deps(tmp_path)).list_tools())}
    analyst = {t.name for t in asyncio.run(build_hermes_server(_deps(tmp_path)).list_tools())}
    assert tools.isdisjoint(analyst)


def test_hermes_analyst_tool_returns_text_and_session(tmp_path, monkeypatch):
    import bubble_watch.mcp_server as srv

    monkeypatch.setattr(srv, "hermes_analyst",
                        lambda settings, task, session_id="", **kw: {"text": f"ok:{task}",
                                                                     "session_id": "s-1"})
    from bubble_watch.mcp_server import build_hermes_server
    result = _call(build_hermes_server(_deps(tmp_path)), "analyst", {"task": "research this"})
    assert result.structured_content == {"text": "ok:research this", "session_id": "s-1"}


def test_hermes_analyst_tool_forwards_the_session_for_a_rebuttal(tmp_path, monkeypatch):
    import bubble_watch.mcp_server as srv

    seen = {}

    def fake(settings, task, session_id="", **kw):
        seen["session_id"] = session_id
        return {"text": "t", "session_id": session_id}

    monkeypatch.setattr(srv, "hermes_analyst", fake)
    from bubble_watch.mcp_server import build_hermes_server
    _call(build_hermes_server(_deps(tmp_path)), "analyst",
          {"task": "rebut", "session_id": "s-1"})
    assert seen["session_id"] == "s-1"


def test_hermes_tool_records_its_span_with_the_servers_tracer(tmp_path):
    """The per-call `hermes analyst` span only exists if the tool is given a real tracer. Falling
    back to the global one silently produces a non-recording span — Hermes then inherits the
    ambient context and parents on the run root instead, which still *looks* connected."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import bubble_watch.mcp_server as srv

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    seen = {}

    def fake(settings, task, session_id="", *, tracer=None, runner=None):
        seen["tracer"] = tracer
        return {"text": "t", "session_id": "s"}

    srv_deps = _deps(tmp_path)
    object.__setattr__(srv_deps, "tracer", provider.get_tracer("t"))
    from unittest import mock
    with mock.patch.object(srv, "hermes_analyst", fake):
        _call(srv.build_hermes_server(srv_deps), "analyst", {"task": "x"})
    assert seen["tracer"] is not None, "the tool must be handed the server's tracer"
    assert seen["tracer"] is srv_deps.tracer


# --- the analyst as its own server, so it can be its own peer container -------------------------

def test_hermes_server_exposes_only_the_analyst(tmp_path):
    from bubble_watch.mcp_server import build_hermes_server
    names = {t.name for t in asyncio.run(build_hermes_server(_deps(tmp_path)).list_tools())}
    assert names == {"analyst"}


def test_hermes_server_analyst_delegates_and_returns_the_session(tmp_path, monkeypatch):
    import bubble_watch.mcp_server as srv
    from bubble_watch.mcp_server import build_hermes_server

    monkeypatch.setattr(srv, "hermes_analyst",
                        lambda settings, task, session_id="", **kw: {"text": f"ok:{task}",
                                                                    "session_id": "s-1"})
    result = _call(build_hermes_server(_deps(tmp_path)), "analyst", {"task": "research"})
    assert result.structured_content == {"text": "ok:research", "session_id": "s-1"}


def test_the_deterministic_server_no_longer_hosts_the_analyst(tmp_path):
    """Split so each runtime can be its own container, one hop from the orchestrator: the tool
    server then never spawns a sibling and needs no Docker socket."""
    names = {t.name for t in asyncio.run(build_server(_deps(tmp_path)).list_tools())}
    assert "hermes_analyst" not in names
    assert names == {"prepare_brief", "apply_gap_fills", "prior_state", "save_run"}
