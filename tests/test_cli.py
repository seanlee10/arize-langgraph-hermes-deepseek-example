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
    monkeypatch.setenv("HERMES_MODE", "gateway")
    monkeypatch.setenv("XAI_API_KEY", "xai-SECRET")
    monkeypatch.setenv("HERMES_API_KEY", "")
    monkeypatch.setenv("HERMES_ANALYST_HOME", str(tmp_path / "hh"))
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    settings = load_settings(tmp_path / "none.env")

    def handler(req):
        if req.url.host == "api.x.ai":
            return httpx.Response(200, json={"data": [{"id": "grok-4.6"}, {"id": "grok-4.5"}]})
        raise httpx.ConnectError("refused")

    checks = doctor_checks(settings, httpx.Client(transport=httpx.MockTransport(handler)))
    by_name = {c[0]: c for c in checks}
    assert "env HERMES_API_KEY" not in by_name  # generated automatically now
    assert by_name["xAI model grok-4.6"][1] is True
    assert by_name["Hermes gateway"][1] is False and "not reachable" in by_name["Hermes gateway"][3]
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
    cfg = (home / "config.yaml").read_text()
    assert "provider: xai" in cfg
    # web research only: no terminal, so Hermes never needs (or downloads) the tirith scanner
    assert "api_server: [web]" in cfg and "tirith_enabled: false" in cfg
    env = hermes_gateway_env(settings, home)
    assert env["HERMES_HOME"] == str(home) and env["API_SERVER_ENABLED"] == "true"
    assert env["API_SERVER_KEY"] == "gw-k" and env["API_SERVER_PORT"] == "8650"
    assert env["XAI_API_KEY"] == "xai-k" and env["EXA_API_KEY"] == "exa-k"
    assert "TELEGRAM_BOT_TOKEN" not in env
    assert env["TIRITH_ENABLED"] == "false"


def test_gateway_key_is_generated_once_and_shared(tmp_path, monkeypatch):
    import stat

    from bubble_watch.config import missing_required, resolve_hermes_key

    monkeypatch.setenv("XAI_API_KEY", "xai-k")
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_ANALYST_HOME", str(tmp_path / "hh"))
    settings = load_settings(tmp_path / "none.env")
    assert missing_required(settings) == []  # no longer something the user must set
    key = resolve_hermes_key(settings)
    key_file = tmp_path / "hh" / "api_server.key"
    assert len(key) >= 32 and key_file.read_text().strip() == key
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert resolve_hermes_key(settings) == key  # stable across runs

    monkeypatch.setenv("HERMES_API_KEY", "explicit")
    assert resolve_hermes_key(load_settings(tmp_path / "none.env")) == "explicit"


def test_doctor_checks_hermes_executable_in_oneshot_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "")
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    monkeypatch.delenv("HERMES_MODE", raising=False)
    monkeypatch.setenv("HERMES_BIN", "definitely-not-a-hermes-binary")
    checks = {c[0]: c for c in doctor_checks(load_settings(tmp_path / "none.env"), httpx.Client())}
    assert checks["Hermes CLI (oneshot)"][1] is False and "not on PATH" in checks["Hermes CLI (oneshot)"][3]
    assert "Hermes gateway" not in checks
