"""Prompt builders. Numbers arrive pre-computed; agents interpret and research, never recompute."""
from __future__ import annotations

import datetime as dt
import json

from ..models import AnalystView, DailyRecord, Signals, WatchState

SYSTEM_ANALYST = (
    "너는 'NVDA Bubble Signal Watch'의 독립 시니어 애널리스트다. 다른 애널리스트와 독립적으로 판단한다. "
    "제공된 수치는 코드가 계산한 확정값이므로 재계산하거나 새로운 가격·수익률·IV를 만들어내지 마라. "
    "너의 역할은 (1) 마지막 기록 이후 발표된 뉴스·촉매를 웹에서 조사하고, (2) 계산된 tape 신호를 해석하고, "
    "(3) Bubble Signal Score와 판정을 제시하는 것이다. 모든 촉매에는 실제로 열어 확인한 URL을 붙여라. "
    "최종 응답은 지정된 JSON 스키마를 따르는 JSON 객체 하나만 ```json 코드 블록으로 출력한다."
)

SYSTEM_GAP_FILL = (
    "너는 시장 데이터 리서처다. 요청된 값만 웹에서 찾아 출처 URL과 함께 보고한다. "
    "추정·보간·계산으로 값을 만들지 않는다. 찾지 못하면 null로 보고한다."
)


def _fmt(v: object, suffix: str = "") -> str:
    return "N/A" if v is None else f"{v}{suffix}"


def analyst_brief(state: WatchState, today: DailyRecord, signals: Signals) -> str:
    prior = state.prior_records(today.date)
    last = next((r for r in reversed(prior) if r.score is not None), None)
    closes = "\n".join(f"| {s} | {_fmt(c.price)} | {c.freshness} | {c.source} |" for s, c in today.closes.items())
    puts = "\n".join(
        f"| ${k}P | {_fmt(q.bid)} | {_fmt(q.ask)} | {_fmt(q.last)} | {_fmt(q.iv, '%')} | {_fmt(q.volume)} "
        f"| {_fmt(q.oi)} | {q.freshness} |" for k, q in sorted(today.puts.items())) or "| (없음) | | | | | | | |"
    history = "\n".join(
        f"- {r.date}: score={_fmt(r.score)} verdict={r.verdict.value if r.verdict else 'N/A'} — {r.note}"
        for r in prior[-5:] if r.score is not None or r.note)
    since = prior[-1].date if prior else "최근"
    strikes = "/".join(f"${k}" for k in state.strikes)
    schema = json.dumps(AnalystView.model_json_schema(), ensure_ascii=False)
    return f"""# {state.ticker} Bubble Signal Watch — {today.date} 미국장 마감 분석 요청

## 감시 대상
- {state.ticker} vs peer {state.peer}, 레버리지 ETF {state.leveraged}
- {state.expiry} 만기 {strikes} puts
- Trigger zone: score ≥ {state.trigger_threshold}

## 오늘 종가 (코드 확정값)
| 종목 | 종가 | freshness | source |
|---|---:|---|---|
{closes}

## 오늘 put 데이터 (코드 확정값)
| Put | Bid | Ask | Last | IV | Volume | OI | freshness |
|---|---:|---:|---:|---:|---:|---:|---|
{puts}

## 계산된 신호 (코드 확정값)
```json
{signals.model_dump_json(indent=2)}
```
returns / put_changes는 % 변화, rel_spread는 {state.ticker}−{state.peer} %p, base_dates는 각 구간의 비교 기준일,
iv는 strike별 "최근 검증된 IV"와의 비교, conditions는 재확인 조건 3개
(nvda_underperforms: {state.ticker} < {state.peer} 지속, iv_surface_up: IV surface 전체 상승,
far_otm_leads: 가장 OTM put이 다른 put보다 빠르게 상승).
put_comparability_1d가 null이 아닌 strike의 1일 put 변화는 데이터 출처 변경 또는 stale 데이터 때문이므로
시장 움직임으로 해석하지 마라.

## 이전 기록
- 직전 점수: {_fmt(last.score if last else None)} ({last.date if last else '-'}), 판정 {last.verdict.value if last and last.verdict else 'N/A'}
{history}

## 작업
1. {since} 이후 발표된 {state.ticker}·AI 반도체 관련 뉴스와 촉매(경쟁, 수요, 규제, 거시)를 웹에서 조사하라.
   각 촉매를 bubble thesis 기준 bearish / bullish / neutral로 분류하고 URL을 붙여라.
2. 계산된 신호로 오늘 tape가 thesis를 confirm하는지 de-confirm하는지 해석하라. 수치는 재계산하지 말고 인용만 하라.
3. 직전 점수 대비 score(0–10, 소수 첫째 자리)를 제시하고 변화 이유를 설명하라. ±1.0 초과 변화는 강한 근거가 있을 때만.
4. verdict: NOT_TRIGGERED, TRIGGERED, TRIGGERED_DE_CONFIRMING, TRIGGERED_FURTHER_DE_CONFIRMING, CONFIRMED 중 하나.
5. watch_conditions_ko에는 무엇이 thesis를 재확인하거나 무효화하는지 적어라.

## 응답 형식
모든 *_ko 필드는 한국어. 아래 AnalystView JSON Schema를 따르는 JSON 객체 하나만 ```json 블록으로 출력하라.
{schema}
"""


def rebuttal_brief(brief: str, own_name: str, own: AnalystView, other_name: str, other: AnalystView) -> str:
    return f"""{brief}

## 반박 라운드
다른 애널리스트({other_name})가 독립적으로 아래 결론을 냈고, 너({own_name})의 이전 결론과 차이가 크다.

### 너의 이전 결론
```json
{own.model_dump_json(indent=2)}
```

### {other_name}의 결론
```json
{other.model_dump_json(indent=2)}
```

상대의 근거와 촉매를 검토하라. 설득되면 수정하고, 아니면 유지하되 이유를 밝혀라.
rebuttal_ko 필드에 반박 또는 수용의 요지를 반드시 적고, 수정된 전체 AnalystView JSON 하나만 ```json 블록으로 출력하라.
"""


def gap_fill_brief(day: dt.date, ticker: str, expiry: dt.date, gaps: list) -> str:
    wanted = "\n".join(f"- {g.field}: {g.description}" for g in gaps)
    return f"""{ticker} {day} 미국장 기준으로 아래 데이터가 비어 있다 (put은 {expiry} 만기). 웹에서 찾아 채워라.
{wanted}

규칙:
- 추정·보간·계산 금지. 실제 페이지에서 확인한 값만.
- 값마다 확인한 페이지 URL(source_url)과 freshness(EOD | latest_snapshot | late_session_last)를 붙여라.
- 찾지 못하면 value는 null, freshness는 "N/A".
- IV는 퍼센트 숫자(예: 34.9), 가격은 달러 숫자, volume/oi는 정수.

응답: 다음 형태의 JSON 객체 하나만 ```json 블록으로 출력하라.
{{"fills": [{{"field": "puts.210.iv", "value": 31.7, "source_url": "https://...", "freshness": "EOD", "note": ""}}]}}
"""


def retry_prompt(error: str) -> str:
    return (f"직전 응답을 파싱/검증하지 못했다: {error[:800]}\n"
            "스키마를 정확히 지키는 JSON 객체 하나만 ```json 블록으로 다시 출력하라. 다른 텍스트는 쓰지 마라.")
