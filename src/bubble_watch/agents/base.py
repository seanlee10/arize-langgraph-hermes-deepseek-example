"""The analyst contract: ask → extract JSON → validate AnalystView, with one corrective retry."""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..models import AnalystView
from .prompts import retry_prompt

_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


class AnalystError(RuntimeError):
    """An analyst could not produce a valid view (transport failure or invalid output after retry)."""


class AskResult(BaseModel):
    text: str
    session_id: str | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)


class AnalystResult(BaseModel):
    agent: str
    view: AnalystView
    session_id: str | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    retried: bool = False


class Asker(Protocol):
    def __call__(self, prompt: str, session_id: str | None = None) -> AskResult: ...


class Analyst(Protocol):
    name: str

    def analyze(self, brief: str) -> AnalystResult: ...

    def rebut(self, prompt: str) -> AnalystResult: ...


def extract_json(text: str) -> dict[str, Any]:
    """First JSON object in ``text``: a fenced block wins, else the first decodable ``{...}``."""
    candidates = [m.group(1) for m in _FENCED.finditer(text)]
    decoder = json.JSONDecoder()
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in response")


def parse_view(text: str) -> AnalystView:
    data = extract_json(text)
    cats = data.get("catalysts") or []
    data["catalysts"] = [c for c in cats if isinstance(c, dict)
                         and str(c.get("url", "")).startswith(("http://", "https://"))]
    return AnalystView.model_validate(data)


def ask_for_view(agent: str, ask: Asker, prompt: str) -> AnalystResult:
    first = ask(prompt)
    try:
        return AnalystResult(agent=agent, view=parse_view(first.text), session_id=first.session_id,
                             events=first.events)
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        retry = ask(retry_prompt(str(exc)), session_id=first.session_id)
        try:
            view = parse_view(retry.text)
        except ValueError as exc2:
            raise AnalystError(f"{agent}: invalid AnalystView after retry: {str(exc2)[:300]}") from exc2
        return AnalystResult(agent=agent, view=view, session_id=retry.session_id or first.session_id,
                             events=[*first.events, *retry.events], retried=True)
