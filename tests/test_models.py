import datetime as dt

import pytest

from bubble_watch.config import load_settings, missing_required
from bubble_watch.models import (
    AnalystView,
    DailyRecord,
    PutQuote,
    Verdict,
    WatchState,
    most_cautious,
    round1,
)


def _view(**kw):
    base = {"score": 8.26, "score_delta_reasoning_ko": "r", "verdict": "TRIGGERED", "tape_read_ko": "t",
            "watch_conditions_ko": "w"}
    return AnalystView.model_validate({**base, **kw})


def test_score_is_clamped_and_rounded_half_up():
    assert _view(score=8.25).score == 8.3
    assert _view(score=11).score == 10.0
    assert _view(score=-3).score == 0.0


def test_verdict_must_be_known():
    with pytest.raises(ValueError):
        _view(verdict="MAYBE")


def test_most_cautious_prefers_confirmed_then_triggered():
    assert most_cautious(Verdict.TRIGGERED_DE_CONFIRMING, Verdict.TRIGGERED) is Verdict.TRIGGERED
    assert most_cautious(Verdict.NOT_TRIGGERED, Verdict.CONFIRMED) is Verdict.CONFIRMED


def test_put_mid_needs_both_sides():
    assert PutQuote(strike=200, bid=3.05, ask=3.10).mid == 3.075
    assert PutQuote(strike=200, bid=3.05).mid is None


def test_round1_half_up():
    assert round1(8.25) == 8.3
    assert round1(8.24) == 8.2
    assert round1((7.9 + 8.2) / 2) == 8.1  # 8.049999999999999 in floating point


def test_watch_state_upsert_and_prior_records():
    ws = WatchState()
    ws.upsert(DailyRecord(date=dt.date(2026, 9, 17)))
    ws.upsert(DailyRecord(date=dt.date(2026, 9, 16)))
    ws.upsert(DailyRecord(date=dt.date(2026, 9, 17), note="replaced"))
    assert [r.date.day for r in ws.records] == [16, 17]
    assert ws.records[-1].note == "replaced"
    assert [r.date.day for r in ws.prior_records(dt.date(2026, 9, 17))] == [16]
    assert ws.symbols == ["NVDA", "SMH", "SOXL"]


def test_int_strike_keys_survive_json_round_trip():
    rec = DailyRecord(date=dt.date(2026, 9, 17), puts={200: PutQuote(strike=200, last=3.06)})
    back = DailyRecord.model_validate_json(rec.model_dump_json())
    assert back.puts[200].last == 3.06


def test_settings_defaults_and_missing_names(tmp_path, monkeypatch):
    for name in ("HERMES_API_KEY", "HERMES_ANALYST_MODEL", "DSH_ANALYST_MODEL", "WRITER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    env = tmp_path / ".env"
    env.write_text("")
    s = load_settings(env)
    assert s.dsh_model == "grok-4.6" and s.writer_model == "grok-4.6"
    assert s.hermes_model is None
    assert missing_required(s) == ["HERMES_API_KEY"]


def test_inline_comment_after_empty_value_is_not_a_value(tmp_path, monkeypatch):
    for name in ("EXA_API_KEY", "HERMES_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("EXA_API_KEY=                     # dsh web_search via Exa\nHERMES_API_KEY=real-secret  # note\n")
    s = load_settings(env)
    assert s.exa_api_key == ""
    assert s.hermes_api_key == "real-secret"
