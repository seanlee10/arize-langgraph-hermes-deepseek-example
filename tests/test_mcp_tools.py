import datetime as dt

import pytest

from bubble_watch.market_data.base import MarketSnapshot
from bubble_watch.mcp_tools import ToolDeps, apply_gap_fills, prepare_brief, prior_state, save_run
from bubble_watch.models import Close, PutQuote, Verdict
from bubble_watch.state_store import load_state, save_state, seed_state

DAY = dt.date(2026, 9, 18)
COMPLETE_PUTS = ((200, 1.30, 1.35, 1.34, 34.35), (210, 2.90, 2.97, 2.98, 31.75), (220, 6.15, 6.30, 6.30, 30.21))


class FakeMarket:
    """Complete snapshot unless `drop` names fields to withhold (e.g. "puts.220.iv", "closes.SOXL")."""

    def __init__(self, drop: tuple[str, ...] = ()):
        self.drop = set(drop)

    def snapshot(self, day, symbols, ticker, expiry, strikes):
        closes = {s: Close(price=p, source="https://x.test", freshness="EOD")
                  for s, p in (("NVDA", 222.27), ("SMH", 573.00), ("SOXL", 123.67))
                  if f"closes.{s}" not in self.drop}
        puts = {}
        for k, b, a, last, iv in COMPLETE_PUTS:
            values = {"bid": b, "ask": a, "last": last, "iv": iv, "volume": 1000, "oi": 40000}
            values = {f: (None if f"puts.{k}.{f}" in self.drop else v) for f, v in values.items()}
            puts[k] = PutQuote(strike=k, source_url="https://x.test", freshness="EOD", **values)
        return MarketSnapshot(date=day, closes=closes, puts=puts)


@pytest.fixture
def deps(tmp_path):
    state_path = tmp_path / "state" / "NVDA.json"
    save_state(seed_state(), state_path)
    return lambda market=None: ToolDeps(market=market or FakeMarket(), state_path=state_path)


def test_prepare_brief_computes_signals_when_data_is_complete(deps):
    brief = prepare_brief(deps(), DAY.isoformat())
    assert brief["gaps"] == []
    assert brief["ticker"] == "NVDA"
    assert brief["signals"]["returns"]["1d"]["NVDA"] == pytest.approx(1.34, abs=0.01)
    assert brief["closes"]["NVDA"]["price"] == 222.27


def test_prepare_brief_reports_missing_fields_as_gaps_without_signals(deps):
    brief = prepare_brief(deps(FakeMarket(drop=("closes.NVDA",))), DAY.isoformat())
    assert "closes.NVDA" in [g["field"] for g in brief["gaps"]]
    assert brief["signals"] is None


def test_prepare_brief_keeps_partial_gaps_but_still_computes_signals(deps):
    brief = prepare_brief(deps(FakeMarket(drop=("puts.220.iv",))), DAY.isoformat())
    assert [g["field"] for g in brief["gaps"]] == ["puts.220.iv"]
    assert brief["signals"] is not None


def test_apply_gap_fills_rejects_a_value_without_a_source(deps):
    d = deps(FakeMarket(drop=("puts.220.iv",)))
    prepare_brief(d, DAY.isoformat())
    result = apply_gap_fills(d, DAY.isoformat(), [{"field": "puts.220.iv", "value": 30.21}])
    assert result["accepted"] == 0
    assert "puts.220.iv" in result["rejected"][0]
    assert [g["field"] for g in result["remaining_gaps"]] == ["puts.220.iv"]


def test_apply_gap_fills_accepts_a_sourced_value_and_closes_the_gap(deps):
    d = deps(FakeMarket(drop=("puts.220.iv",)))
    prepare_brief(d, DAY.isoformat())
    result = apply_gap_fills(d, DAY.isoformat(), [
        {"field": "puts.220.iv", "value": 30.21, "source_url": "https://x.test/iv", "freshness": "EOD"}])
    assert result["accepted"] == 1
    assert result["remaining_gaps"] == []
    assert result["puts"]["220"]["iv"] == 30.21


def test_apply_gap_fills_recomputes_signals_after_a_close_is_filled(deps):
    d = deps(FakeMarket(drop=("closes.NVDA",)))
    assert prepare_brief(d, DAY.isoformat())["signals"] is None
    result = apply_gap_fills(d, DAY.isoformat(), [
        {"field": "closes.NVDA", "value": 222.27, "source_url": "https://x.test/c", "freshness": "EOD"}])
    assert result["signals"]["returns"]["1d"]["NVDA"] == pytest.approx(1.34, abs=0.01)


def test_prior_state_returns_the_last_scored_record(deps):
    prior = prior_state(deps(), DAY.isoformat())
    assert prior["last_date"] == "2026-09-17"
    assert prior["last_score"] == 8.2
    assert prior["last_verdict"] == "TRIGGERED_FURTHER_DE_CONFIRMING"


def test_save_run_persists_score_and_verdict_for_the_day(deps):
    d = deps()
    prepare_brief(d, DAY.isoformat())
    save_run(d, DAY.isoformat(), score=7.9, verdict="TRIGGERED_FURTHER_DE_CONFIRMING",
             note="de-confirming continues", report_path="reports/2026-09-18-NVDA.md")
    record = next(r for r in load_state(d.state_path).records if r.date == DAY)
    assert record.score == 7.9
    assert record.verdict is Verdict.TRIGGERED_FURTHER_DE_CONFIRMING
    assert record.report_path == "reports/2026-09-18-NVDA.md"
    assert record.signals is not None


def test_save_run_rejects_an_unknown_verdict(deps):
    with pytest.raises(ValueError, match="verdict"):
        save_run(deps(), DAY.isoformat(), score=5.0, verdict="MAYBE", note="", report_path="x.md")


def test_prepare_brief_reports_the_trace_it_belongs_to(deps):
    """The model writes the report before save_run, so it needs the trace id early enough to cite."""
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("t")
    with tracer.start_as_current_span("run") as span:
        brief = prepare_brief(deps(), DAY.isoformat())
        assert brief["trace_id"] == f"{span.get_span_context().trace_id:032x}"


def test_prepare_brief_reports_no_trace_when_untraced(deps):
    assert prepare_brief(deps(), DAY.isoformat())["trace_id"] is None


def test_save_run_persists_the_trace_id(deps):
    from opentelemetry.sdk.trace import TracerProvider

    d = deps()
    tracer = TracerProvider().get_tracer("t")
    with tracer.start_as_current_span("run") as span:
        prepare_brief(d, DAY.isoformat())
        save_run(d, DAY.isoformat(), score=7.9, verdict="TRIGGERED", note="n", report_path="r.md")
        expected = f"{span.get_span_context().trace_id:032x}"
    record = next(r for r in load_state(d.state_path).records if r.date == DAY)
    assert record.trace_id == expected
