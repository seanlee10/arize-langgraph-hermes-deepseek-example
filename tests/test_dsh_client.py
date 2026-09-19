import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bubble_watch.agents.base import AnalystError
from bubble_watch.agents.dsh_client import DshAnalyst, ensure_dsh_home

VIEW = {"score": 8.6, "score_delta_reasoning_ko": "r", "verdict": "TRIGGERED", "catalysts": [],
        "tape_read_ko": "t", "watch_conditions_ko": "w"}


class FakeHarness:
    def __init__(self, answers):
        self.answers, self.calls, self.closed = list(answers), [], False

    def run(self, prompt, *, session_id=None):
        self.calls.append((prompt, session_id))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(final_response=answer, session_id=session_id or "session-1", finish_reason="stop",
                               events=[{"type": "assistant/message"}])

    def close(self):
        self.closed = True


def test_analyze_prepends_role_once_and_retries_in_session():
    h = FakeHarness(["no json", json.dumps(VIEW), json.dumps(VIEW)])
    built = []
    analyst = DshAnalyst(lambda: built.append(1) or h)
    result = analyst.analyze("brief")
    assert result.view.score == 8.6 and result.retried and result.agent == "dsh"
    assert h.calls[0][0].startswith("너는") and h.calls[0][0].endswith("brief")
    assert h.calls[1][1] == "session-1" and not h.calls[1][0].startswith("너는")
    assert len(result.events) == 2
    analyst.analyze("again")
    assert built == [1]  # harness reused
    analyst.close()
    assert h.closed


def test_empty_answer_and_runtime_errors_become_analyst_errors():
    with pytest.raises(AnalystError, match="no answer"):
        DshAnalyst(lambda: FakeHarness(["  "])).ask("p")
    with pytest.raises(AnalystError, match="dsh run failed"):
        DshAnalyst(lambda: FakeHarness([RuntimeError("stdout closed")])).ask("p")


def test_ensure_dsh_home_writes_settings_once(tmp_path):
    path = ensure_dsh_home(str(tmp_path / "home"), "grok-4.6")
    text = path.read_text()
    assert "llm-pi-ai:" in text and "https://api.x.ai/v1" in text and "id: grok-4.6" in text
    assert "apiKeyEnv: XAI_API_KEY" in text
    path.write_text("custom: true\n")
    ensure_dsh_home(str(tmp_path / "home"), "grok-4.6")
    assert path.read_text() == "custom: true\n"



def test_search_patch_prefers_tavily_then_exa_then_none(tmp_path, monkeypatch):
    from bubble_watch.agents.dsh_client import search_patches
    from bubble_watch.config import PROJECT_ROOT, load_settings

    for name in ("TAVILY_API_KEY", "EXA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DSH_REPO", str(tmp_path / "repo"))
    assert search_patches(load_settings(tmp_path / "none.env")) == ((), {})

    monkeypatch.setenv("EXA_API_KEY", "exa-k")
    patches, env = search_patches(load_settings(tmp_path / "none.env"))
    assert "searchProvider: exa" in Path(patches[0]).read_text() and env == {"EXA_API_KEY": "exa-k"}

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-k")
    patches, env = search_patches(load_settings(tmp_path / "none.env"))
    text = Path(patches[0]).read_text()
    assert "searchProvider: tavily" in text and "!!js process.env.TAVILY_API_KEY" in text
    assert f"name: '{PROJECT_ROOT / 'dsh' / 'plugins' / 'web-search-tavily.mjs'}'" in text
    assert env == {"TAVILY_API_KEY": "tvly-k"}
