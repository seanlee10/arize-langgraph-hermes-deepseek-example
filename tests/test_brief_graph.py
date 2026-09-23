import datetime as dt

from test_mcp_tools import DAY, FakeMarket

from bubble_watch.brief_graph import build_brief_graph
from bubble_watch.state_store import seed_state


def _invoke(market, **extra):
    return build_brief_graph(market).invoke({"day": DAY, "watch": seed_state(), **extra})


def test_brief_graph_fetches_then_computes():
    state = _invoke(FakeMarket())
    assert state["record"].signals is not None
    assert state["gaps"] == []
    assert state["record"].closes["NVDA"].price == 222.27


def test_brief_graph_reports_gaps_and_skips_signals_without_the_ticker_close():
    state = _invoke(FakeMarket(drop=("closes.NVDA",)))
    assert state["record"].signals is None
    assert "closes.NVDA" in [g.field for g in state["gaps"]]


def test_brief_graph_recomputes_from_a_supplied_snapshot_without_refetching():
    class Exploding(FakeMarket):
        def snapshot(self, *args, **kwargs):
            raise AssertionError("must not refetch when a snapshot is supplied")

    snapshot = FakeMarket().snapshot(DAY, ["NVDA"], "NVDA", dt.date(2026, 10, 16), [200, 210, 220])
    state = _invoke(Exploding(), snapshot=snapshot)
    assert state["record"].signals is not None


def test_brief_graph_records_a_provider_failure_as_a_note_instead_of_raising():
    class Broken(FakeMarket):
        def snapshot(self, *args, **kwargs):
            raise RuntimeError("alpha vantage 503")

    state = _invoke(Broken())
    assert any("503" in n for n in state["notes"])
    assert state["record"].signals is None


def test_brief_graph_exposes_both_stages_as_named_nodes():
    assert set(build_brief_graph(FakeMarket()).get_graph().nodes) >= {"fetch_market_data", "compute_signals"}


def test_prepare_brief_runs_through_the_graph(monkeypatch, tmp_path):
    from bubble_watch import brief_graph
    from bubble_watch.mcp_tools import ToolDeps, prepare_brief
    from bubble_watch.state_store import save_state

    state_path = tmp_path / "state" / "NVDA.json"
    save_state(seed_state(), state_path)
    built = []
    original = brief_graph.build_brief_graph
    monkeypatch.setattr(brief_graph, "build_brief_graph", lambda m: built.append(1) or original(m))
    brief = prepare_brief(ToolDeps(market=FakeMarket(), state_path=state_path), DAY.isoformat())
    assert built == [1]
    assert brief["signals"] is not None


def test_apply_gap_fills_also_runs_through_the_graph(monkeypatch, tmp_path):
    from bubble_watch import brief_graph
    from bubble_watch.mcp_tools import ToolDeps, apply_gap_fills, prepare_brief
    from bubble_watch.state_store import save_state

    state_path = tmp_path / "state" / "NVDA.json"
    save_state(seed_state(), state_path)
    deps = ToolDeps(market=FakeMarket(drop=("puts.220.iv",)), state_path=state_path)
    prepare_brief(deps, DAY.isoformat())
    built = []
    original = brief_graph.build_brief_graph
    monkeypatch.setattr(brief_graph, "build_brief_graph", lambda m: built.append(1) or original(m))
    apply_gap_fills(deps, DAY.isoformat(), [
        {"field": "puts.220.iv", "value": 30.21, "source_url": "https://x.test", "freshness": "EOD"}])
    assert built == [1]
