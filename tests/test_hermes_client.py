import json

import httpx
import pytest

from bubble_watch.agents.base import AnalystError
from bubble_watch.agents.hermes_client import HermesAnalyst

VIEW = {"score": 8.1, "score_delta_reasoning_ko": "r", "verdict": "TRIGGERED_DE_CONFIRMING", "catalysts": [],
        "tape_read_ko": "t", "watch_conditions_ko": "w"}


def _completion(content, session="h-1", status=200, extra_headers=None):
    return httpx.Response(status, headers={"X-Hermes-Session-Id": session, **(extra_headers or {})},
                          json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def _client(handler, **kw):
    return HermesAnalyst("http://hermes.test/v1", "k-1", client=httpx.Client(transport=httpx.MockTransport(handler)),
                         **kw)


def test_analyze_sends_auth_unique_run_and_no_model_by_default():
    seen = []

    def handler(req):
        seen.append(req)
        return _completion(f"```json\n{json.dumps(VIEW)}\n```")

    result = _client(handler).analyze("brief")
    req = seen[0]
    body = json.loads(req.content)
    assert req.url.path == "/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer k-1"
    assert "X-Hermes-Session-Id" not in req.headers
    assert "model" not in body and body["stream"] is False
    assert body["messages"][0]["role"] == "system" and "run:" in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": "brief"}
    assert result.view.score == 8.1 and result.session_id == "h-1" and result.agent == "hermes"


def test_retry_continues_session_and_model_override():
    seen = []

    def handler(req):
        seen.append(req)
        return _completion("oops" if len(seen) == 1 else json.dumps(VIEW))

    result = _client(handler, model="grok-4.6").analyze("brief")
    assert result.retried
    assert json.loads(seen[0].content)["model"] == "grok-4.6"
    assert seen[1].headers["X-Hermes-Session-Id"] == "h-1"


def test_http_error_and_incomplete_run_raise_analyst_error():
    with pytest.raises(AnalystError, match="HTTP 500"):
        _client(lambda r: httpx.Response(500, text="boom")).analyze("b")
    with pytest.raises(AnalystError, match="incomplete"):
        _client(lambda r: _completion("partial", extra_headers={"X-Hermes-Completed": "false"})).analyze("b")


def test_fill_gaps_uses_gap_system_prompt_and_parses_fills():
    seen = []
    fills = {"fills": [{"field": "puts.210.iv", "value": 31.7, "source_url": "https://x.test", "freshness": "EOD"}]}

    def handler(req):
        seen.append(json.loads(req.content))
        return _completion(json.dumps(fills))

    out = _client(handler).fill_gaps("find it")
    assert out[0].field == "puts.210.iv" and out[0].value == 31.7
    assert "리서처" in seen[0]["messages"][0]["content"]
