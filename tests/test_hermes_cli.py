"""Hermes one-shot CLI analyst: `hermes chat --query-file … -Q` per call, isolated HERMES_HOME, web tools only."""
import json
import subprocess
from pathlib import Path

import pytest

from bubble_watch.agents.base import AnalystError
from bubble_watch.agents.hermes_cli import HermesCliAnalyst, ensure_hermes_home, hermes_env
from bubble_watch.config import load_settings

VIEW = {"score": 8.0, "score_delta_reasoning_ko": "r", "verdict": "TRIGGERED", "catalysts": [],
        "tape_read_ko": "t", "watch_conditions_ko": "w"}


class FakeRunner:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def __call__(self, cmd, **kw):
        query = Path(cmd[cmd.index("--query-file") + 1]).read_text()
        self.calls.append({"cmd": cmd, "env": kw["env"], "query": query, "timeout": kw["timeout"]})
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        code, stdout = out if isinstance(out, tuple) else (0, out)
        return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr="\nsession_id: 20260920_s1\n")


def _analyst(runner, tmp_path):
    return HermesCliAnalyst(model="grok-4.6", env={"HERMES_HOME": str(tmp_path)}, timeout=600, runner=runner,
                            workdir=tmp_path)


def test_analyze_runs_oneshot_with_web_tools_and_parses_session(tmp_path):
    runner = FakeRunner([json.dumps(VIEW)])
    result = _analyst(runner, tmp_path).analyze("brief")
    call = runner.calls[0]
    cmd = call["cmd"]
    assert cmd[:2] == ["hermes", "chat"] and "-Q" in cmd and "--ignore-rules" in cmd
    assert cmd[cmd.index("-t") + 1] == "web"
    assert (cmd[cmd.index("--provider") + 1], cmd[cmd.index("-m") + 1]) == ("xai", "grok-4.6")
    assert cmd[cmd.index("--run-budget") + 1] == "600" and call["timeout"] > 600
    assert call["query"].startswith("너는") and call["query"].endswith("brief")
    assert result.view.score == 8.0 and result.session_id == "20260920_s1" and result.agent == "hermes"
    assert not list(tmp_path.glob("*.md"))  # query files are cleaned up


def test_retry_resumes_the_session_without_repeating_the_role(tmp_path):
    runner = FakeRunner(["not json", json.dumps(VIEW)])
    result = _analyst(runner, tmp_path).analyze("brief")
    retry = runner.calls[1]
    assert result.retried and retry["cmd"][retry["cmd"].index("--resume") + 1] == "20260920_s1"
    assert not retry["query"].startswith("너는")


def test_failures_become_analyst_errors(tmp_path):
    with pytest.raises(AnalystError, match="exited 1"):
        _analyst(FakeRunner([(1, "")]), tmp_path).ask("p")
    with pytest.raises(AnalystError, match="timed out"):
        _analyst(FakeRunner([subprocess.TimeoutExpired("hermes", 660)]), tmp_path).ask("p")
    with pytest.raises(AnalystError, match="no answer"):
        _analyst(FakeRunner(["   "]), tmp_path).ask("p")


def test_fill_gaps_uses_the_researcher_role(tmp_path):
    fills = {"fills": [{"field": "puts.210.iv", "value": 31.7, "source_url": "https://x.test", "freshness": "EOD"}]}
    runner = FakeRunner([json.dumps(fills)])
    out = _analyst(runner, tmp_path).fill_gaps("find it")
    assert out[0].value == 31.7 and "리서처" in runner.calls[0]["query"]


def test_env_is_isolated_web_only_and_bot_free(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-k")
    monkeypatch.setenv("TAVILY_API_KEY", "tv-k")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "must-not-leak")
    settings = load_settings(tmp_path / "none.env")
    home = ensure_hermes_home(tmp_path / "hh", "grok-4.6")
    cfg = (home / "config.yaml").read_text()
    assert "cli: [web]" in cfg and "api_server: [web]" in cfg and "tirith_enabled: false" in cfg
    env = hermes_env(settings, home)
    assert env["HERMES_HOME"] == str(home) and env["TIRITH_ENABLED"] == "false"
    assert env["XAI_API_KEY"] == "xai-k" and env["TAVILY_API_KEY"] == "tv-k"
    assert "TELEGRAM_BOT_TOKEN" not in env and "API_SERVER_ENABLED" not in env
