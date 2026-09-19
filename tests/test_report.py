import json

import httpx
import pytest

from bubble_watch.models import AnalystView
from bubble_watch.reconcile import reconcile
from bubble_watch.report import money, render_report
from bubble_watch.signals import compute_signals
from bubble_watch.state_store import seed_state
from bubble_watch.writer import Narrative, WriterError, XaiWriter, fallback_narrative, writer_brief

NARR = Narrative(headline_ko="오늘은 de-confirmation.", tape_ko="TAPE", options_ko="OPTIONS", verdict_ko="VERDICT",
                 watch_ko="WATCH")


def _view(score, verdict="TRIGGERED_FURTHER_DE_CONFIRMING", url="https://reuters.com/huawei"):
    return AnalystView.model_validate({
        "score": score, "score_delta_reasoning_ko": "r", "verdict": verdict, "tape_read_ko": f"tape {score}",
        "watch_conditions_ko": "w",
        "catalysts": [{"headline": "Huawei AI chips", "url": url, "direction": "bearish", "weight": "high"}]})


def _setup(views):
    s = seed_state()
    rec = s.records[-1]
    s.records = s.records[:-1]
    rec = rec.model_copy(update={"signals": compute_signals(s, rec)})
    recon = reconcile(views, prior_score=8.4, rebuttal_done=True)
    return s, rec, recon


def test_money_rounds_half_up():
    assert money(3.075) == "$3.08" and money(None) == "N/A" and money(1234.5) == "$1,234.50"


def test_report_tables_come_from_code():
    views = {"hermes": _view(8.2), "dsh": _view(8.2)}
    s, rec, recon = _setup(views)
    md = render_report(s, rec, recon, NARR, views=views, notes=["gap note"])
    assert "## NVDA Bubble Signal Watch — 2026-09-17 미국장 마감" in md
    assert "**Bubble Signal Score: `8.2 / 10` — 전회 8.4 대비 `-0.2`.** 오늘은 de-confirmation." in md
    assert "| **NVDA** | **$219.34** | **+2.54%** |" in md
    assert "| **NVDA − SMH** | — | **-0.22%p** |" in md
    assert "| **$200P** | $3.05 | $3.10 | $3.08 | $3.06 | -2.86% | 34.90% | 4,222 | 43,504 |" in md
    assert "recorded-to-recorded" in md and "2026-09-15" in md  # freshness + last-verified IV notes
    assert "-20.93%" in md and "| **최근 3거래일** |" in md
    assert "`TRIGGERED, BUT FURTHER DE-CONFIRMING`" in md
    assert "[1]: https://reuters.com/huawei" in md and "gap note" in md
    assert "Analyst disagreement" not in md and "단일 analyst" not in md


def test_disagreement_and_single_sections():
    views = {"hermes": _view(7.0, "TRIGGERED_DE_CONFIRMING"), "dsh": _view(8.6, "TRIGGERED")}
    s, rec, recon = _setup(views)
    md = render_report(s, rec, recon, NARR, views=views, notes=[])
    assert "Analyst disagreement" in md and "tape 7.0" in md and "tape 8.6" in md
    one = {"dsh": _view(8.1)}
    s, rec, recon = _setup(one)
    assert "단일 analyst 기준" in render_report(s, rec, recon, NARR, views=one, notes=[])


def test_writer_brief_numbers_catalysts_and_fallback():
    views = {"hermes": _view(8.2), "dsh": _view(8.2)}
    s, rec, recon = _setup(views)
    brief = writer_brief(s, rec, recon, views)
    assert "[1] Huawei AI chips" in brief and "headline_ko" in brief
    fb = fallback_narrative(recon, views)
    assert fb.tape_ko == "tape 8.2"


def test_xai_writer_retries_then_fails():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        content = "no json" if len(seen) == 1 else json.dumps(NARR.model_dump())
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    w = XaiWriter("k", "grok-4.6", "https://api.x.ai/v1", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert w.write("brief") == NARR
    assert seen[0]["model"] == "grok-4.6" and len(seen) == 2

    bad = XaiWriter("k", "grok-4.6", "https://api.x.ai/v1",
                    client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="no"))))
    with pytest.raises(WriterError, match="401"):
        bad.write("brief")
