import pytest

from bubble_watch.agents.base import AnalystError, AskResult, ask_for_view, extract_json, parse_view
from bubble_watch.agents.prompts import analyst_brief, gap_fill_brief, rebuttal_brief
from bubble_watch.market_data.base import GapRequest
from bubble_watch.signals import compute_signals
from bubble_watch.state_store import seed_state

VIEW = ('{"score": 8.0, "score_delta_reasoning_ko": "r", "verdict": "TRIGGERED_DE_CONFIRMING", '
        '"catalysts": [{"headline": "h", "url": "https://reuters.com/x", "direction": "bearish"}, '
        '{"headline": "no url", "url": "n/a", "direction": "bullish"}], '
        '"tape_read_ko": "t", "watch_conditions_ko": "w", "confidence": "high"}')


def test_extract_json_prefers_fenced_block_and_tolerates_prose():
    assert extract_json(f"분석 결과:\n```json\n{VIEW}\n```\n끝")["score"] == 8.0
    assert extract_json(f"prose {{not json}} then {VIEW} trailing")["verdict"] == "TRIGGERED_DE_CONFIRMING"
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_parse_view_drops_catalysts_without_urls():
    view = parse_view(VIEW)
    assert [c.headline for c in view.catalysts] == ["h"]


def test_ask_for_view_retries_once_in_same_session():
    calls = []

    def ask(prompt, session_id=None):
        calls.append((prompt, session_id))
        return AskResult(text="not json" if len(calls) == 1 else VIEW, session_id="s-1", events=[{"n": len(calls)}])

    result = ask_for_view("hermes", ask, "brief")
    assert result.retried and result.view.score == 8.0
    assert calls[1][1] == "s-1" and "JSON" in calls[1][0]
    assert result.events == [{"n": 1}, {"n": 2}]


def test_ask_for_view_gives_up_after_retry():
    with pytest.raises(AnalystError, match="hermes"):
        ask_for_view("hermes", lambda p, session_id=None: AskResult(text="nope"), "brief")


def test_briefs_carry_computed_numbers_and_rules():
    s = seed_state()
    today = s.records[-1]
    s.records = s.records[:-1]
    brief = analyst_brief(s, today, compute_signals(s, today))
    assert "2026-09-17" in brief and "\"far_otm_leads\"" in brief and "8.4" in brief
    assert "재계산" in brief and "URL" in brief and "AnalystView" in brief

    view = parse_view(VIEW)
    rb = rebuttal_brief(brief, "hermes", view, "dsh", view)
    assert "반박" in rb and "rebuttal_ko" in rb

    gb = gap_fill_brief(today.date, "NVDA", s.expiry, [GapRequest(field="puts.210.iv", description="d")])
    assert "puts.210.iv" in gb and "추정" in gb and "\"fills\"" in gb
