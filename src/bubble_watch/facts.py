"""Human-readable Korean facts shared by the report tables and the writer's brief: number formatting,
condition labels, and a plain-Korean fact sheet (the writer never sees signal JSON or its key names)."""
from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

from .models import DailyRecord, Verdict, WatchState

VERDICT_LABEL = {
    Verdict.NOT_TRIGGERED: "NOT TRIGGERED",
    Verdict.TRIGGERED: "TRIGGERED",
    Verdict.TRIGGERED_DE_CONFIRMING: "TRIGGERED, BUT DE-CONFIRMING",
    Verdict.TRIGGERED_FURTHER_DE_CONFIRMING: "TRIGGERED, BUT FURTHER DE-CONFIRMING",
    Verdict.CONFIRMED: "CONFIRMED",
}
MODE_LABEL = {"agree": "두 analyst 합의", "agree_after_rebuttal": "반박 라운드 후 합의",
               "disagree": "반박 라운드 후에도 불일치 → 평균 점수, 더 보수적인 판정", "single": "단일 analyst 기준"}

CONDITION_LABEL = {
    "nvda_underperforms": "{t} < {p} (여러 날 지속)",
    "iv_surface_up": "IV surface 전체 상승",
    "far_otm_leads": "${lo}P가 다른 put보다 빠르게 상승",
}
CONDITION_VALUE = {"true": "성립", "false": "불성립", "unknown": "데이터 없음"}
# "partial" means something different per condition.
PARTIAL_LABEL = {"nvda_underperforms": "부분 성립 (1일만)", "iv_surface_up": "판단 불가 (한 단계 이내 상승)"}

# Signal/record field names that must never reach the Korean prose.
_CODE_NAMES = ("rel_spread", "far_otm_leads", "iv_surface_up", "nvda_underperforms", "convexity_order",
               "put_changes", "put_freshness", "base_dates", "score_delta", "reconciliation", "watch_conditions")
_IDENTIFIER = re.compile(r"\b(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b")  # snake / ENUM_CASE
_ASSIGNMENT = re.compile(r"\b[a-z_]+\s*=\s*(?:true|false|\[)")            # iv_surface_up=true, x=[...]


def money(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"${Decimal(str(v)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"


def spct(v: float | None) -> str:
    return "N/A" if v is None else f"{v:+.2f}%"


def pp(v: float | None) -> str:
    return "N/A" if v is None else f"{v:+.2f}%p"


def md(d) -> str:
    return f"{d.month}/{d.day}"


def condition_text(key: str, value: str) -> str:
    return PARTIAL_LABEL.get(key, "부분 성립") if value == "partial" else CONDITION_VALUE[value]


def windows(state: WatchState) -> list[tuple[str, str]]:
    anchor = state.anchors[0] if state.anchors else None
    return [("1d", "1일"), ("3d", "최근 3거래일"), ("anchor", f"{md(anchor.date)} 이후" if anchor else "anchor 이후")]


def korean_facts(state: WatchState, record: DailyRecord) -> str:
    """The day's computed facts as plain Korean lines, in the vocabulary of the original reports."""
    sig = record.signals
    t, p, lo = state.ticker, state.peer, min(state.strikes)
    wins = windows(state)
    lines = [f"기준일: {record.date} 미국장 마감"]
    for s in state.symbols:
        c = record.closes.get(s)
        changes = ", ".join(f"{label} {spct(sig.returns[w][s])}" for w, label in wins)
        lines.append(f"- {s} {money(c.price if c else None)} ({changes})")
    lines.append(f"- {t} − {p} 상대 성과: " + ", ".join(f"{label} {pp(sig.rel_spread[w])}" for w, label in wins))
    lines.append(f"- {state.leveraged}는 daily 3× 레버리지 ETF라 여러 날 수익률을 {t}/{p}와 직접 비교하지 않는다.")
    lines.append(f"- {state.expiry} 만기 put 가격 변화")
    for w, label in wins:
        lines.append(f"  - {label}: " + ", ".join(f"${k}P {spct(sig.put_changes[w][k])}" for k in state.strikes))
    if sig.convexity_order_1d:
        lines.append("  - 1일 변화율 순위 (높은 → 낮은): " + " > ".join(f"${k}P" for k in sig.convexity_order_1d))
    iv = [f"${k}P {c.prior:.2f}% ({c.prior_date}) → {c.current:.2f}%" if c.prior is not None and c.current is not None
          else f"${k}P N/A" for k, c in sorted(sig.iv.items())]
    lines.append("- IV: " + ", ".join(iv) + " (Alpha Vantage IV는 약 0.98%p 단위)")
    lines.append("- 재확인 조건")
    for key, value in sig.conditions.items():
        lines.append(f"  - {CONDITION_LABEL[key].format(t=t, p=p, lo=lo)}: {condition_text(key, value)}")
    return "\n".join(lines)


def leaked_identifiers(text: str) -> list[str]:
    """Code names / variable-style notation that leaked into prose (sorted, unique)."""
    found = {m.group(0) for m in _IDENTIFIER.finditer(text)}
    found |= {m.group(0).split("=")[0].strip() for m in _ASSIGNMENT.finditer(text)}
    found |= {name for name in _CODE_NAMES if name in text and not any(name in f for f in found)}
    return sorted(found)
