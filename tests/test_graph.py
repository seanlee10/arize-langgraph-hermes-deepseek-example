import datetime as dt
from pathlib import Path

import pytest

from bubble_watch.agents.base import AnalystError, AnalystResult
from bubble_watch.graph import Deps, build_graph
from bubble_watch.market_data.base import GapFill, MarketSnapshot
from bubble_watch.models import AnalystView, Close, PutQuote
from bubble_watch.state_store import load_state, save_state, seed_state
from bubble_watch.tracing import get_tracer
from bubble_watch.writer import Narrative

DAY = dt.date(2026, 9, 18)


class FakeMarket:
    def snapshot(self, day, symbols, ticker, expiry, strikes):
        closes = {"NVDA": Close(price=222.27), "SMH": Close(price=573.00), "SOXL": Close(price=123.67)}
        puts = {k: PutQuote(strike=k, bid=b, ask=a, last=l, iv=iv, volume=1000, oi=40000, freshness="latest_snapshot",
                            source_url="https://finance.yahoo.com/x")
                for k, b, a, l, iv in ((200, 1.30, 1.35, 1.34, 34.35), (210, 2.90, 2.97, 2.98, 31.75),
                                       (220, 6.15, 6.30, 6.30, None))}
        return MarketSnapshot(date=day, closes=closes, puts=puts)


class FakeFiller:
    def __init__(self):
        self.prompts = []

    def fill_gaps(self, prompt):
        self.prompts.append(prompt)
        return [GapFill(field="puts.220.iv", value=30.21, source_url="https://x.test/iv", freshness="EOD")]


def _view(score, verdict="TRIGGERED_FURTHER_DE_CONFIRMING"):
    return AnalystView.model_validate({"score": score, "score_delta_reasoning_ko": "r", "verdict": verdict,
                                       "tape_read_ko": f"tape {score}", "watch_conditions_ko": "w",
                                       "catalysts": [{"headline": "h", "url": "https://a.test", "direction": "bullish"}]})


class FakeAnalyst:
    def __init__(self, name, first, rebuttal=None, fail=False):
        self.name, self.first, self.rebuttal, self.fail = name, first, rebuttal, fail
        self.rebut_prompts = []

    def analyze(self, brief):
        if self.fail:
            raise AnalystError(f"{self.name} down")
        assert "2026-09-18" in brief
        return AnalystResult(agent=self.name, view=self.first, session_id=f"{self.name}-s")

    def rebut(self, prompt):
        self.rebut_prompts.append(prompt)
        return AnalystResult(agent=self.name, view=self.rebuttal, session_id=f"{self.name}-s")


class FakeWriter:
    def write(self, prompt):
        return Narrative(headline_ko="H", tape_ko="T", options_ko="O", verdict_ko="V", watch_ko="W")


@pytest.fixture
def paths(tmp_path):
    state_path = tmp_path / "state" / "NVDA.json"
    save_state(seed_state(), state_path)
    return state_path, tmp_path / "reports"


def _run(paths, analysts, *, save=True, filler=None):
    state_path, reports = paths
    deps = Deps(market=FakeMarket(), analysts={a.name: a for a in analysts}, writer=FakeWriter(),
                tracer=get_tracer(None), reports_dir=reports, state_path=state_path, gap_filler=filler, save=save)
    return build_graph(deps).invoke({"day": DAY, "watch": load_state(state_path)})


def test_agree_path_writes_report_and_saves_state(paths):
    filler = FakeFiller()
    final = _run(paths, [FakeAnalyst("hermes", _view(7.9)), FakeAnalyst("dsh", _view(8.2))], filler=filler)
    rec, recon = final["record"], final["reconciliation"]
    assert recon.mode == "agree" and recon.score == 8.1 and rec.score_delta == -0.1
    assert rec.puts[220].iv == 30.21 and "puts.220.iv" in filler.prompts[0]
    assert rec.signals.returns["1d"]["NVDA"] == 1.34
    report = Path(final["report_path"]).read_text()
    assert "2026-09-18" in report and "`8.1 / 10`" in report
    saved = load_state(paths[0])
    assert saved.records[-1].date == DAY and saved.records[-1].score == 8.1
    assert saved.records[-1].analyst_views["dsh"].score == 8.2 and saved.records[-1].note


def test_disagreement_runs_one_rebuttal_round(paths):
    hermes = FakeAnalyst("hermes", _view(7.0, "TRIGGERED_DE_CONFIRMING"), rebuttal=_view(7.9))
    dsh = FakeAnalyst("dsh", _view(8.6, "TRIGGERED"), rebuttal=_view(8.2))
    final = _run(paths, [hermes, dsh])
    assert len(hermes.rebut_prompts) == 1 and len(dsh.rebut_prompts) == 1
    assert "반박" in hermes.rebut_prompts[0] and "tape 8.6" in hermes.rebut_prompts[0]
    assert final["reconciliation"].mode == "agree_after_rebuttal" and final["reconciliation"].score == 8.1


def test_one_failed_analyst_still_ships_single_view(paths):
    final = _run(paths, [FakeAnalyst("hermes", None, fail=True), FakeAnalyst("dsh", _view(8.0))])
    assert final["reconciliation"].mode == "single"
    assert any("hermes analyst failed" in n for n in final["notes"])


def test_all_failed_raises_and_dry_run_does_not_save(paths):
    with pytest.raises(RuntimeError, match="all analysts failed"):
        _run(paths, [FakeAnalyst("hermes", None, fail=True), FakeAnalyst("dsh", None, fail=True)])
    _run(paths, [FakeAnalyst("dsh", _view(8.0))], save=False)
    assert load_state(paths[0]).records[-1].date == dt.date(2026, 9, 17)


def test_no_analysts_stops_after_signals(paths):
    final = _run(paths, [])
    assert final["record"].signals is not None and "report_path" not in final


def test_agent_spans_nest_under_langgraph_node_spans(paths):
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    LangChainInstrumentor().instrument(tracer_provider=provider)
    try:
        state_path, reports = paths
        hermes = FakeAnalyst("hermes", _view(7.0, "TRIGGERED_DE_CONFIRMING"), rebuttal=_view(7.9))
        dsh = FakeAnalyst("dsh", _view(8.6, "TRIGGERED"), rebuttal=_view(8.2))
        deps = Deps(market=FakeMarket(), analysts={"hermes": hermes, "dsh": dsh}, writer=FakeWriter(),
                    tracer=get_tracer(provider), reports_dir=reports, state_path=state_path, save=False)
        build_graph(deps).invoke({"day": DAY, "watch": load_state(state_path)})
    finally:
        LangChainInstrumentor().uninstrument()
    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    parent = {s.name: by_id[s.parent.span_id].name if s.parent and s.parent.span_id in by_id else None for s in spans}
    assert len({s.context.trace_id for s in spans}) == 1  # one run, one trace
    assert parent["hermes analyst"] == "analyst_hermes" and parent["dsh analyst"] == "analyst_dsh"
    assert parent["hermes rebuttal"] == "rebuttal" and parent["report writer"] == "write_report"
