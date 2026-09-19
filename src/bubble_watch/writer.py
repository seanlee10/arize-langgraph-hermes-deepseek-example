"""Narrative writer: one direct xAI chat call that turns reconciled views into Korean prose (no numbers of its own)."""
from __future__ import annotations

import httpx
from pydantic import BaseModel

from .agents.base import extract_json
from .facts import korean_facts, leaked_identifiers
from .models import AnalystView, DailyRecord, Reconciliation, WatchState

SYSTEM_WRITER = (
    "너는 한국어 금융 리서치 노트 작가다. 'NVDA Bubble Signal Watch' 일일 노트의 서술 부분만 쓴다. "
    "표와 수치는 코드가 따로 렌더링하므로, 서술에서 인용하는 숫자는 반드시 제공된 값 그대로 써라(새 숫자 금지). "
    "문체: 간결한 한국어 리서치 톤, 핵심 수치와 결론은 **굵게**, de-confirmation·convexity·IV surface 같은 "
    "영어 용어는 그대로 섞어 쓴다. 촉매를 언급할 때는 제공된 번호로 [n] 형태로 인용한다. "
    "변수명·코드 이름(예: rel_spread, far_otm_leads)이나 'x=true', '[220, 210, 200]' 같은 표기는 절대 쓰지 말고, "
    "'NVDA가 SMH를 0.87%p underperform했다', '$200P가 가장 크게 빠졌다'처럼 사람이 읽는 문장으로 쓴다. "
    "JSON 객체 하나만 출력한다."
)


class Narrative(BaseModel):
    headline_ko: str
    tape_ko: str
    options_ko: str
    verdict_ko: str
    watch_ko: str


class WriterError(RuntimeError):
    pass


def writer_brief(state: WatchState, record: DailyRecord, recon: Reconciliation,
                 views: dict[str, AnalystView]) -> str:
    catalysts = "\n".join(
        f"[{i}] {c.headline} ({c.direction}, {c.weight}, cited by {', '.join(c.cited_by)}) — {c.rationale_ko}"
        for i, c in enumerate(recon.catalysts, 1)) or "(없음)"
    opinions = "\n\n".join(
        f"### {name}: score {v.score}, {v.verdict.value}, confidence {v.confidence}\n"
        f"- tape: {v.tape_read_ko}\n- 점수 변화 이유: {v.score_delta_reasoning_ko}\n- watch: {v.watch_conditions_ko}"
        + (f"\n- rebuttal: {v.rebuttal_ko}" if v.rebuttal_ko else "")
        for name, v in views.items())
    return f"""# {state.ticker} {record.date} 노트 서술 작성

## 최종 결론 (확정)
- score {recon.score}, verdict {recon.verdict.value}, reconciliation mode {recon.mode}
- flags: {'; '.join(recon.flags) or '없음'}

## 오늘의 확정 수치 (숫자는 여기서만 인용)
{korean_facts(state, record) if record.signals else '(없음)'}

## 촉매 (번호로 인용)
{catalysts}

## 애널리스트 의견
{opinions}

## 출력
다음 키를 가진 JSON 객체 하나:
- headline_ko: 점수 줄 바로 뒤에 붙는 1–2문장 (오늘 tape의 핵심)
- tape_ko: 점수 판단 근거 문단 (촉매 [n] 인용)
- options_ko: put 가격·IV·convexity 해석 문단
- verdict_ko: 현재 판정과 이유 문단
- watch_ko: 앞으로 볼 재확인/무효화 조건 문단
"""


def fallback_narrative(recon: Reconciliation, views: dict[str, AnalystView]) -> Narrative:
    """Used when the writer call fails: stitch the first analyst's own text so the report still ships."""
    v = next(iter(views.values()))
    return Narrative(headline_ko=f"(writer 실패로 {next(iter(views))} 애널리스트 원문 사용)", tape_ko=v.tape_read_ko,
                     options_ko=v.score_delta_reasoning_ko, verdict_ko=f"판정: {recon.verdict.value}",
                     watch_ko=v.watch_conditions_ko)


class XaiWriter:
    def __init__(self, api_key: str, model: str, base_url: str, client: httpx.Client | None = None,
                 timeout: float = 180.0) -> None:
        self._key, self._model = api_key, model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._client = client or httpx.Client(timeout=timeout)

    def write(self, prompt: str) -> Narrative:
        error = ""
        for attempt in range(2):
            content = prompt if attempt == 0 else f"{prompt}\n\n직전 출력 오류: {error[:500]}. JSON 객체 하나만 다시 출력하라."
            body = {"model": self._model, "messages": [{"role": "system", "content": SYSTEM_WRITER},
                                                       {"role": "user", "content": content}]}
            try:
                r = self._client.post(self._url, json=body, headers={"Authorization": f"Bearer {self._key}"})
            except httpx.HTTPError as exc:
                raise WriterError(f"writer request failed: {exc}") from exc
            if r.status_code >= 400:
                raise WriterError(f"writer HTTP {r.status_code}: {r.text[:300]}")
            try:
                narrative = Narrative.model_validate(extract_json(r.json()["choices"][0]["message"]["content"] or ""))
            except (ValueError, KeyError, IndexError) as exc:
                error = str(exc)
                continue
            leaks = leaked_identifiers(" ".join(narrative.model_dump().values()))
            if not leaks or attempt == 1:
                return narrative  # a leak surviving the retry is reported by the graph, not fatal
            error = f"코드 이름/변수 표기가 문장에 남아 있다: {', '.join(leaks)}. 사람이 읽는 한국어 문장으로 바꿔라"
        raise WriterError(f"writer returned invalid narrative JSON twice: {error[:300]}")
