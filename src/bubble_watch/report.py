"""Markdown report: every table cell comes from code; the writer's Narrative supplies the prose."""
from __future__ import annotations

from .facts import CONDITION_LABEL, MODE_LABEL, VERDICT_LABEL, condition_text, md, money, pp, spct, windows
from .models import AnalystView, DailyRecord, PutQuote, Reconciliation, WatchState, round1
from .writer import Narrative


def _iv(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.2f}%"


def _num(v: int | None) -> str:
    return "N/A" if v is None else f"{v:,}"




def _put_basis_note(record: DailyRecord) -> str:
    sig = record.signals
    base = sig.base_dates.get("1d")
    if not sig.put_freshness_1d or base is None:
        return "직전 기록이 없어 1일 변화율은 계산하지 않았다."
    before, now = sig.put_freshness_1d
    if before == now == "EOD":
        return f"1일 변화율은 {base} EOD → {record.date} EOD 기준이다."
    return (f"오늘 put 값은 **{now}** 기준이고, 1일 변화율은 {base}의 **{before}** 값과 비교한 "
            f"**recorded-to-recorded 변화**다. 공식 EOD-to-EOD 수익률로 과도하게 해석하면 안 된다.")


def _iv_note(record: DailyRecord, state: WatchState) -> str:
    sig = record.signals
    base = sig.base_dates.get("1d")
    parts = []
    for k in state.strikes:
        c = sig.iv.get(k)
        if c is None or c.current is None:
            parts.append(f"${k}P N/A")
        elif c.prior is None:
            parts.append(f"${k}P {c.current:.2f}% (비교 기준 없음)")
        else:
            parts.append(f"${k}P {c.prior:.2f}% ({c.prior_date}) → **{c.current:.2f}%**")
    verified = {c.prior_date for c in sig.iv.values() if c.prior_date}
    lead = "IV 비교: " if verified <= {base} else "IV 비교 (1일 변화가 아니라 **last-verified 비교**): "
    note = lead + ", ".join(parts)
    if any("alphavantage" in q.source_url for q in record.puts.values()):
        note += (" (Alpha Vantage IV는 약 0.98%p 단위로 제공되므로, 모든 strike가 한 단계 이내로 오른 경우 "
                 "IV surface 조건은 판단 불가(partial)로 본다.)")
    return note


def render_report(state: WatchState, record: DailyRecord, recon: Reconciliation, narrative: Narrative, *,
                  views: dict[str, AnalystView], notes: list[str]) -> str:
    sig = record.signals
    t, p, lev = state.ticker, state.peer, state.leveraged
    prior = next((r.score for r in reversed(state.prior_records(record.date)) if r.score is not None), None)
    delta = None if prior is None else round1(recon.score - prior)
    close = {s: (record.closes[s].price if s in record.closes else None) for s in state.symbols}
    r1 = sig.returns["1d"]

    head = f"**Bubble Signal Score: `{recon.score:.1f} / 10`"
    head += f" — 전회 {prior:.1f} 대비 `{delta:+.1f}`.**" if delta is not None else ".**"
    out = [f"## {t} Bubble Signal Watch — {record.date} 미국장 마감", "", f"{head} {narrative.headline_ko}", "",
           f"| 지표 | {md(record.date)} | 판정 |", "|---|---:|---|",
           f"| **Bubble Signal Score** | **{recon.score:.1f}** | **{'N/A' if delta is None else f'{delta:+.1f}'}** |",
           f"| **{t}** | **{money(close[t])}** | **{spct(r1[t])}** |",
           f"| **{p}** | **{money(close[p])}** | **{spct(r1[p])}** |",
           f"| **{t} − {p}** | — | **{pp(sig.rel_spread['1d'])}** |",
           f"| **{lev}** | **{money(close[lev])}** | **{spct(r1[lev])}** |", "", narrative.tape_ko, ""]

    strikes = " / ".join(f"${k}" for k in state.strikes)
    out += [f"### {state.expiry} {strikes} puts", "", _put_basis_note(record), "",
            "| Put | Bid | Ask | Mid | Last | vs prior recorded last | IV | Volume | OI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in state.strikes:
        q = record.puts.get(k) or PutQuote(strike=k)
        out.append(f"| **${k}P** | {money(q.bid)} | {money(q.ask)} | {money(q.mid)} | {money(q.last)} | "
                   f"{spct(sig.put_changes['1d'][k])} | {_iv(q.iv)} | {_num(q.volume)} | {_num(q.oi)} |")
    out += ["", _iv_note(record, state), "", narrative.options_ko, ""]

    wins = windows(state)
    out += [f"### {' / '.join(label for _, label in wins)}", "",
            "| 구간 | " + " | ".join([*state.symbols, *(f"${k}P" for k in state.strikes)]) + " |",
            "|---|" + "---:|" * (len(state.symbols) + len(state.strikes))]
    for w, label in wins:
        cells = [spct(sig.returns[w][s]) for s in state.symbols] + [spct(sig.put_changes[w][k]) for k in state.strikes]
        out.append(f"| **{label}** | " + " | ".join(cells) + " |")
    bases = ", ".join(f"{label}: {sig.base_dates.get(w) or 'N/A'} 대비" for w, label in wins)
    out += ["", (f"기준일 — {bases}. {lev}는 daily 3× 레버리지 ETF라 multi-day 수익률의 절대 크기를 "
                 f"{t}/{p}와 직접 비교하면 안 된다."), ""]

    lo = min(state.strikes)
    out += [f"### 판정: `{VERDICT_LABEL[recon.verdict]}`", "", narrative.verdict_ko, "", narrative.watch_ko, "",
            "| 재확인 조건 | 오늘 |", "|---|---|"]
    out += [f"| {CONDITION_LABEL[k].format(t=t, p=p, lo=lo)} | {condition_text(k, v)} |"
            for k, v in sig.conditions.items()]

    out += ["", "### Analyst views", "", f"{MODE_LABEL[recon.mode]}.", "",
            "| Analyst | Score | Verdict | Confidence |", "|---|---:|---|---|"]
    out += [f"| {name} | {v.score:.1f} | {VERDICT_LABEL[v.verdict]} | {v.confidence} |" for name, v in views.items()]
    if recon.mode == "disagree":
        out += ["", "#### Analyst disagreement", ""]
        for name, v in views.items():
            rebuttal = f" — 반박: {v.rebuttal_ko}" if v.rebuttal_ko else ""
            out += [f"**{name}** ({v.score:.1f}, {VERDICT_LABEL[v.verdict]}): {v.tape_read_ko}{rebuttal}", ""]
    if recon.flags or notes:
        out += ["", "#### 데이터·실행 노트", ""] + [f"- {n}" for n in [*recon.flags, *notes]]
    if recon.catalysts:
        out += [""] + [f'[{i}]: {c.url} "{c.headline}"' for i, c in enumerate(recon.catalysts, 1)]
    return "\n".join(out) + "\n"
