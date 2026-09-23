---
name: bubble-watch
description: NVDA Bubble Signal Watch 일간 리포트 절차 — 확정 데이터 도구 호출, Hermes 서브에이전트 위임, 두 견해 조정, 한국어 리포트 작성까지. 리포트 작성 요청을 받으면 반드시 이 절차를 따른다.
---

# NVDA Bubble Signal Watch — 일간 절차

너는 이 감시의 **오케스트레이터이자 두 번째 애널리스트**다. 수치는 `mcp__bubble__*` 도구가 계산하고,
웹 리서치는 `hermes_analyst` 서브에이전트가 맡고, 판단과 조정과 작성은 네가 한다.

## 절대 규칙

1. **수치를 만들어내지 마라.** 종가·수익률·IV·put 변화율은 `mcp__bubble__*`가 돌려준 값만 쓴다.
   직접 계산하거나 기억으로 채우지 않는다. 도구가 `N/A`를 주면 리포트에도 `N/A`다.
2. **출처 없는 값은 쓰지 않는다.** 리서치로 찾은 값에는 실제로 열어 확인한 URL과 freshness를 붙인다.
3. **degrade를 숨기지 마라.** `hermes_analyst`가 실패했거나 gap이 남았으면 리포트의 provenance 줄에 적는다.

## 절차

### 1. 맥락과 데이터

```
mcp__bubble__prior_state(date)      → 직전 점수·판정·메모
mcp__bubble__prepare_brief(date)    → closes, puts, signals, gaps
```

`prepare_brief`의 `signals`가 null이면 기준 종가가 없다는 뜻이다. 2단계를 반드시 거쳐야 한다.

### 2. 결측값 채우기 (gaps가 비어 있지 않을 때만)

`gaps`의 각 `field`를 `hermes_analyst`에 위임한다. 과제에 **정확히 어떤 값이 필요한지**와
**출처 URL·freshness가 반드시 필요하다**는 점을 적는다.

```
hermes_analyst("2026-09-18 NVDA $220 put의 IV와 SOXL 종가를 찾아라. 각 값에 출처 URL과
                freshness(EOD | latest_snapshot | late_session_last)를 붙여 JSON으로 보고하라.")
mcp__bubble__apply_gap_fills(date, fills=[{field, value, source_url, freshness}, ...])
```

`rejected`가 비어 있지 않으면 그 값은 저장되지 않은 것이다. 한 번만 다시 시도하고,
그래도 안 되면 `N/A`로 두고 provenance에 적는다.

### 3. 두 견해 만들기

**(a) Hermes 견해** — 리서치를 위임한다. 계산된 signals를 과제에 그대로 넣어주고,
재계산하지 말라고 명시한다.

```
hermes_analyst("<계산된 signals JSON>. 직전 기록 이후 발표된 NVDA/반도체 촉매를 조사하고,
                각 촉매에 URL을 붙여라. 이 tape를 해석해 Bubble Signal Score(0–10)와 판정
                (NOT_TRIGGERED | TRIGGERED | TRIGGERED_DE_CONFIRMING |
                 TRIGGERED_FURTHER_DE_CONFIRMING | CONFIRMED)을 제시하라.")
```

**(b) 너 자신의 견해** — Hermes의 답을 보기 **전에** 네 웹 검색으로 독립적으로 세운다.
같은 score·verdict 형식으로 정리한다.

### 4. 조정

- **합의**: score 차이 ≤ 0.5 **그리고** verdict 동일 → 최종 score는 두 값의 평균, verdict는 공통값.
- **불일치**: 그 외 전부 → **반박 라운드를 한 번** 돈다. `hermes_analyst`에 상대(너)의 견해를 보여주고
  재고 기회를 준다. 너도 Hermes의 논거를 반영해 다시 판단한다.
  - 반박 후에도 불일치면: 최종 score는 평균, verdict는 **더 보수적인 쪽**을 택한다.
    보수적인 순서(앞이 더 보수적): `CONFIRMED` > `TRIGGERED` > `TRIGGERED_DE_CONFIRMING` >
    `TRIGGERED_FURTHER_DE_CONFIRMING` > `NOT_TRIGGERED`.
- **Hermes 실패**: 네 견해 하나로 진행하고 provenance에 "단독 판단"이라고 적는다.
- 직전 점수 대비 1.0 이상 움직였으면 리포트에 그 이유를 명시한다.

### 5. 리포트 작성

과제에 지정된 경로에 한국어 마크다운으로 쓴다. 구성:

1. `# NVDA Bubble Signal Watch — <날짜>` 와 한 줄 헤드라인
2. **판정**: score(직전 대비 delta 포함)와 verdict, 한 문단 근거
3. **Tape**: 종가 표(NVDA/SMH/SOXL, 1일·3일·anchor 수익률, NVDA−SMH 상대 스프레드)
4. **옵션**: put 표(strike별 bid/ask/last/IV/volume/OI, 1일 변화율), IV 비교
5. **재확인 조건 3개**: `nvda_underperforms`, `iv_surface_up`, `far_otm_leads` 각각의 성립 여부
   (`partial`은 "판단 불가"로 쓴다 — Alpha Vantage IV는 약 0.98%p 단위라 한 단계 이내 상승은 noise와
   구분되지 않는다)
6. **촉매**: 두 견해가 인용한 촉매를 URL과 함께 병합, 누가 인용했는지 표시
7. **내일 볼 것**
8. **provenance**: 데이터 출처와 freshness, 조정 방식(합의/반박 후 합의/불일치/단독),
   두 애널리스트의 원 score, 실패하거나 남은 gap

표의 모든 셀은 도구가 돌려준 값을 옮긴 것이어야 한다. 코드 이름·필드 이름
(`iv_surface_up`, `rel_spread` 등)은 조건 항목 외에는 한국어 표현으로 바꿔 쓴다.

### 6. 저장

```
mcp__bubble__save_run(date, score, verdict, note, report_path)
```

`note`는 하루를 한두 문장으로 요약한 것이다. 다음 날 `prior_state`가 이걸 읽는다.
