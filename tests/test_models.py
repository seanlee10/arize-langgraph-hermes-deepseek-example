import datetime as dt

from bubble_watch.config import load_settings, missing_required
from bubble_watch.models import DailyRecord, PutQuote, WatchState, round1


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
    for name in ("HERMES_ANALYST_MODEL", "DSH_ANALYST_MODEL", "WRITER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    env = tmp_path / ".env"
    env.write_text("")
    s = load_settings(env)
    assert s.dsh_model == "grok-4.6" and s.writer_model == "grok-4.6"
    assert s.hermes_model is None
    assert missing_required(s) == []


def test_inline_comment_after_empty_value_is_not_a_value(tmp_path, monkeypatch):
    for name in ("EXA_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("EXA_API_KEY=                     # dsh web_search via Exa\nTAVILY_API_KEY=real-secret  # note\n")
    s = load_settings(env)
    assert s.exa_api_key == ""
    assert s.tavily_api_key == "real-secret"


def test_alpha_vantage_key_accepts_both_spellings(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "av-k")
    assert load_settings(tmp_path / "none.env").alphavantage_api_key == "av-k"
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "av-primary")
    assert load_settings(tmp_path / "none.env").alphavantage_api_key == "av-primary"
