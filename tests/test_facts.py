"""The Korean fact sheet the writer receives instead of raw signal JSON, and the code-name leak check."""
import json

import httpx

from bubble_watch.facts import korean_facts, leaked_identifiers
from bubble_watch.signals import compute_signals
from bubble_watch.state_store import seed_state
from bubble_watch.writer import Narrative, XaiWriter


def _sep17():
    s = seed_state()
    rec = s.records[-1]
    s.records = s.records[:-1]
    return s, rec.model_copy(update={"signals": compute_signals(s, rec)})


def test_fact_sheet_is_plain_korean_with_the_numbers():
    s, rec = _sep17()
    facts = korean_facts(s, rec)
    assert "NVDA $219.34 (1일 +2.54%, 최근 3거래일 +3.97%, 8/28 이후 +0.82%)" in facts
    assert "NVDA − SMH 상대 성과: 1일 -0.22%p, 최근 3거래일 +0.44%p, 8/28 이후 -0.54%p" in facts
    assert "1일: $200P -45.10%, $210P -39.77%, $220P -33.04%" in facts
    assert "IV: $200P 35.63% (2026-09-16) → 33.68%" in facts
    assert "NVDA < SMH (여러 날 지속): 부분 성립 (1일만)" in facts
    assert leaked_identifiers(facts) == []  # the sheet itself carries no code names


def test_leaked_identifiers_finds_code_names_and_assignments():
    text = "rel_spread 1d -0.87, far_otm_leads=false, convexity_order_1d [220, 210, 200], iv_surface_up=true. 정상 문장."
    assert leaked_identifiers(text) == ["convexity_order_1d", "far_otm_leads", "iv_surface_up", "rel_spread"]
    assert leaked_identifiers("NVDA가 SMH를 -0.87%p underperform했고 de-confirmation이 이어졌다.") == []
    assert leaked_identifiers("판정 TRIGGERED_FURTHER_DE_CONFIRMING 유지") == ["TRIGGERED_FURTHER_DE_CONFIRMING"]
    assert leaked_identifiers("판정 `TRIGGERED, BUT FURTHER DE-CONFIRMING` 유지, NVDA·SMH·SOXL") == []


def test_writer_retries_once_when_code_names_leak():
    leaky = Narrative(headline_ko="rel_spread -0.87", tape_ko="t", options_ko="o", verdict_ko="v", watch_ko="w")
    clean = Narrative(headline_ko="NVDA가 SMH 대비 -0.87%p 약세", tape_ko="t", options_ko="o", verdict_ko="v",
                      watch_ko="w")
    prompts = []

    def handler(req):
        body = json.loads(req.content)
        prompts.append(body["messages"][1]["content"])
        out = leaky if len(prompts) == 1 else clean
        return httpx.Response(200, json={"choices": [{"message": {"content": out.model_dump_json()}}]})

    w = XaiWriter("k", "grok-4.6", "https://api.x.ai/v1", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert w.write("brief") == clean
    assert "rel_spread" in prompts[1]  # the retry names the offending terms
