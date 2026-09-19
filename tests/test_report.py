import json

import httpx
import pytest

from bubble_watch.facts import money
from bubble_watch.models import AnalystView
from bubble_watch.reconcile import reconcile
from bubble_watch.report import render_report
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
    assert "| **$200P** | $1.66 | $1.68 | $1.67 | $1.68 | -45.10% | 33.68% | 5,516 | 44,118 |" in md
    assert "EOD → 2026-09-17 EOD" in md and "(2026-09-16)" in md  # freshness + IV comparison notes
    assert "0.98%p" in md and "판단 불가" in md  # Alpha Vantage IV granularity → one-step moves are partial
    assert "-56.59%" in md and "| **최근 3거래일** |" in md
    assert "`TRIGGERED, BUT FURTHER DE-CONFIRMING`" in md
    assert "[1]: https://reuters.com/huawei" in md and "gap note" in md
    assert "Analyst disagreement" not in md and "단일 analyst" not in md
    assert "| hermes | 8.2 | TRIGGERED, BUT FURTHER DE-CONFIRMING | med |" in md
    assert "TRIGGERED_FURTHER" not in md  # no enum names anywhere in the report


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
    assert "NVDA − SMH 상대 성과" in brief  # Korean fact sheet, not raw signal JSON
    assert "TRIGGERED, BUT FURTHER DE-CONFIRMING" in brief and "TRIGGERED_FURTHER" not in brief
    assert "rel_spread" not in brief and "far_otm_leads" not in brief and '"base_dates"' not in brief
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


def test_partial_is_labelled_per_condition():
    views = {"hermes": _view(8.0), "dsh": _view(8.0)}
    s, rec, recon = _setup(views)
    rec.signals.conditions = {"nvda_underperforms": "partial", "iv_surface_up": "partial", "far_otm_leads": "false"}
    md = render_report(s, rec, recon, NARR, views=views, notes=[])
    assert "| NVDA < SMH (여러 날 지속) | 부분 성립 (1일만) |" in md
    assert "| IV surface 전체 상승 | 판단 불가 (한 단계 이내 상승) |" in md
