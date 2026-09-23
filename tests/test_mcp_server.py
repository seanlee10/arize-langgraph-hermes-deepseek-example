import asyncio
import datetime as dt

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


def test_server_exposes_the_four_deterministic_tools(tmp_path):
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
    assert "계산" in (server.instructions or "") or "compute" in (server.instructions or "").lower()
