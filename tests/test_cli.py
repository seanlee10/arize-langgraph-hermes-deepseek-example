import datetime as dt
from zoneinfo import ZoneInfo

import httpx

from bubble_watch.cli import default_day, doctor_checks, main
from bubble_watch.config import load_settings

NY = ZoneInfo("America/New_York")


def test_default_day_uses_last_completed_session():
    assert default_day(dt.datetime(2026, 9, 18, 17, 0, tzinfo=NY)) == dt.date(2026, 9, 18)   # Fri after close
    assert default_day(dt.datetime(2026, 9, 18, 10, 0, tzinfo=NY)) == dt.date(2026, 9, 17)   # Fri before close
    assert default_day(dt.datetime(2026, 9, 20, 12, 0, tzinfo=NY)) == dt.date(2026, 9, 18)   # Sunday


def test_seed_refuses_to_overwrite_without_force(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    assert main(["seed"]) == 0
    assert main(["seed"]) == 1
    assert main(["seed", "--force"]) == 0
    assert (tmp_path / "NVDA.json").exists()


def test_doctor_reports_names_never_values(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-SECRET")
    monkeypatch.setenv("HERMES_API_KEY", "")
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    settings = load_settings(tmp_path / "none.env")

    def handler(req):
        if req.url.host == "api.x.ai":
            return httpx.Response(200, json={"data": [{"id": "grok-4.6"}, {"id": "grok-4.5"}]})
        raise httpx.ConnectError("refused")

    checks = doctor_checks(settings, httpx.Client(transport=httpx.MockTransport(handler)))
    by_name = {c[0]: c for c in checks}
    assert by_name["env HERMES_API_KEY"][1] is False
    assert by_name["xAI model grok-4.6"][1] is True
    assert by_name["Hermes gateway"][1] is False
    assert (tmp_path / "dsh" / "settings.yaml").exists()
    assert not any("xai-SECRET" in c[3] for c in checks)


def test_hermes_gateway_env_is_api_only_and_isolated(tmp_path, monkeypatch):
    from bubble_watch.cli import ensure_hermes_home, hermes_gateway_env

    monkeypatch.setenv("XAI_API_KEY", "xai-k")
    monkeypatch.setenv("HERMES_API_KEY", "gw-k")
    monkeypatch.setenv("HERMES_API_URL", "http://localhost:8650/v1")
    monkeypatch.setenv("EXA_API_KEY", "exa-k")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "should-not-leak")
    settings = load_settings(tmp_path / "none.env")
    home = ensure_hermes_home(tmp_path / "hh", "grok-4.6")
    assert "default: grok-4.6" in (home / "config.yaml").read_text()
    assert "provider: xai" in (home / "config.yaml").read_text()
    env = hermes_gateway_env(settings, home)
    assert env["HERMES_HOME"] == str(home) and env["API_SERVER_ENABLED"] == "true"
    assert env["API_SERVER_KEY"] == "gw-k" and env["API_SERVER_PORT"] == "8650"
    assert env["XAI_API_KEY"] == "xai-k" and env["EXA_API_KEY"] == "exa-k"
    assert "TELEGRAM_BOT_TOKEN" not in env
