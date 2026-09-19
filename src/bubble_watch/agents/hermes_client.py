"""Hermes Agent as an analyst, via its OpenAI-compatible gateway API (/v1/chat/completions)."""
from __future__ import annotations

import uuid

import httpx

from ..market_data.base import GapFill
from .base import AnalystError, AnalystResult, AskResult, ask_for_view, extract_json
from .prompts import SYSTEM_ANALYST, SYSTEM_GAP_FILL


class HermesAnalyst:
    name = "hermes"

    def __init__(self, base_url: str, api_key: str, *, model: str | None = None, timeout: float = 600.0,
                 client: httpx.Client | None = None) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = api_key
        self._model = model
        self._client = client or httpx.Client(timeout=timeout)
        # Hermes fingerprints system prompt + first user message into a session id; a per-run id keeps
        # a re-run of the same day from continuing the previous run's session.
        self._run_id = uuid.uuid4().hex[:12]

    def ask(self, prompt: str, session_id: str | None = None, *, system: str = SYSTEM_ANALYST) -> AskResult:
        headers = {"Authorization": f"Bearer {self._key}"}
        if session_id:
            headers["X-Hermes-Session-Id"] = session_id
        body: dict = {"messages": [{"role": "system", "content": f"{system}\n\nrun: {self._run_id}"},
                                   {"role": "user", "content": prompt}], "stream": False}
        if self._model:
            body["model"] = self._model
        try:
            r = self._client.post(self._url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise AnalystError(f"hermes request failed: {exc}") from exc
        if r.status_code >= 400:
            raise AnalystError(f"hermes HTTP {r.status_code}: {r.text[:300]}")
        if r.headers.get("X-Hermes-Completed") == "false":
            raise AnalystError(f"hermes run incomplete: {r.headers.get('X-Hermes-Error', '')}")
        try:
            text = r.json()["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError) as exc:
            raise AnalystError(f"hermes returned an unexpected body: {r.text[:300]}") from exc
        return AskResult(text=text, session_id=r.headers.get("X-Hermes-Session-Id") or session_id)

    def analyze(self, brief: str) -> AnalystResult:
        return ask_for_view(self.name, self.ask, brief)

    def rebut(self, prompt: str) -> AnalystResult:
        return ask_for_view(self.name, self.ask, prompt)

    def fill_gaps(self, prompt: str) -> list[GapFill]:
        result = self.ask(prompt, system=SYSTEM_GAP_FILL)
        try:
            data = extract_json(result.text)
            return [GapFill.model_validate(f) for f in data.get("fills") or []]
        except ValueError as exc:
            raise AnalystError(f"hermes gap fill returned invalid JSON: {str(exc)[:300]}") from exc
