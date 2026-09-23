import datetime as dt
from zoneinfo import ZoneInfo

import httpx
from test_mcp_tools import FakeMarket

from bubble_watch.cli import default_day, doctor_checks, main, report_path_for
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


def test_report_path_is_the_day_and_the_ticker(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    settings = load_settings(tmp_path / "none.env")
    assert report_path_for(settings, dt.date(2026, 9, 18)).name == "2026-09-18-NVDA.md"


def _doctor_env(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-SECRET")
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    monkeypatch.setenv("HERMES_ANALYST_HOME", str(tmp_path / "hh"))
    return load_settings(tmp_path / "none.env")


def _checks(settings):
    def handler(req):
        if req.url.host == "api.x.ai":
            return httpx.Response(200, json={"data": [{"id": "grok-4.6"}, {"id": "grok-4.5"}]})
        raise httpx.ConnectError("refused")

    return {c[0]: c for c in doctor_checks(settings, httpx.Client(transport=httpx.MockTransport(handler)))}


def test_doctor_reports_names_never_values(tmp_path, monkeypatch):
    checks = _checks(_doctor_env(tmp_path, monkeypatch))
    assert checks["xAI model grok-4.6"][1] is True
    assert (tmp_path / "dsh" / "settings.yaml").exists()
    assert not any("xai-SECRET" in c[3] for c in checks.values())


def test_doctor_checks_the_hermes_launcher(tmp_path, monkeypatch):
    settings = _doctor_env(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_BIN", "definitely-not-a-launcher")
    checks = _checks(load_settings(tmp_path / "none.env"))
    assert checks["Hermes launcher"][1] is False
    assert _checks(settings)["Hermes launcher"][1] is True


def test_doctor_installs_and_reports_the_orchestration_skill(tmp_path, monkeypatch):
    checks = _checks(_doctor_env(tmp_path, monkeypatch))
    assert checks["bubble-watch skill"][1] is True
    assert (tmp_path / "dsh" / "skills" / "bubble-watch" / "SKILL.md").exists()


def test_doctor_no_longer_probes_a_hermes_gateway(tmp_path, monkeypatch):
    assert "Hermes gateway" not in _checks(_doctor_env(tmp_path, monkeypatch))


def test_no_agents_prints_signals_without_launching_dsh(tmp_path, monkeypatch, capsys):
    from bubble_watch import cli

    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(cli, "market_provider", lambda settings: FakeMarket())

    def explode(*args, **kwargs):
        raise AssertionError("--no-agents must not launch dsh")

    monkeypatch.setattr(cli, "make_harness_factory", explode)
    assert main(["seed"]) == 0
    assert main(["run", "--no-agents", "--date", "2026-09-18"]) == 0
    assert '"returns"' in capsys.readouterr().out


def test_mcp_command_serves_the_deterministic_tools(tmp_path, monkeypatch):
    from bubble_watch import cli

    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(cli, "market_provider", lambda settings: FakeMarket())
    served = []
    monkeypatch.setattr(cli, "serve_stdio", lambda server: served.append(server) or 0)
    assert main(["seed"]) == 0
    assert main(["mcp"]) == 0
    assert served[0].name == "bubble-watch"


def test_doctor_flags_missing_profile_plugins(tmp_path, monkeypatch):
    checks = _checks(_doctor_env(tmp_path, monkeypatch))
    check = checks["dsh profile plugins"]
    assert check[1] is False                       # nothing installed into a fresh DSH_HOME
    assert "install-plugins" in check[3]           # and it says how to fix it


def test_install_plugins_command_installs_the_missing_ones(tmp_path, monkeypatch, capsys):
    from bubble_watch import cli

    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    installed = []
    monkeypatch.setattr(cli, "install_profile_plugins", lambda s: installed.append(s) or ["a", "b"])
    assert main(["install-plugins"]) == 0
    assert installed and "a" in capsys.readouterr().out


def test_install_plugins_reports_a_failure_without_traceback(tmp_path, monkeypatch, capsys):
    from bubble_watch import cli

    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))

    def boom(settings):
        raise RuntimeError("installing @x failed: ERR_PNPM_NO_MATCHING_VERSION")

    monkeypatch.setattr(cli, "install_profile_plugins", boom)
    assert main(["install-plugins"]) == 1
    assert "ERR_PNPM" in capsys.readouterr().out


def _fake_runner(returncode, stderr=""):
    import subprocess

    def runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, "", stderr)

    return runner


def test_doctor_runs_the_hermes_self_check(tmp_path, monkeypatch):
    settings = _doctor_env(tmp_path, monkeypatch)
    checks = {c[0]: c for c in doctor_checks(settings, httpx.Client(), runner=_fake_runner(0))}
    assert checks["Hermes runtime"][1] is True


def test_doctor_explains_a_failing_hermes_check(tmp_path, monkeypatch):
    settings = _doctor_env(tmp_path, monkeypatch)
    checks = {c[0]: c for c in doctor_checks(settings, httpx.Client(), runner=_fake_runner(1, "boom"))}
    check = checks["Hermes runtime"]
    assert check[1] is False
    assert "uv sync" in check[3]   # names the fix, not just the error


def test_doctor_treats_a_missing_search_backend_as_a_real_problem(tmp_path, monkeypatch):
    """With no search key, dsh falls back to a backend it has no credentials for: the call fails
    after ~147s every run and the analysts research by fetching pages one at a time. That is a
    required check, not an optional nicety."""
    for name in ("TAVILY_API_KEY", "EXA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    checks = _checks(_doctor_env(tmp_path, monkeypatch))
    check = checks["web search backend"]
    assert check[1] is False
    assert check[2] is True                  # required
    assert "TAVILY_API_KEY" in check[3]


def test_doctor_is_satisfied_by_either_search_backend(tmp_path, monkeypatch):
    settings = _doctor_env(tmp_path, monkeypatch)
    monkeypatch.setenv("EXA_API_KEY", "exa-k")
    assert _checks(load_settings(tmp_path / "none.env"))["web search backend"][1] is True
    monkeypatch.delenv("EXA_API_KEY")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-k")
    assert _checks(load_settings(tmp_path / "none.env"))["web search backend"][1] is True
    assert settings is not None
