import pytest

from bubble_watch.models import AnalystView, Verdict
from bubble_watch.reconcile import merge_catalysts, reconcile


def _v(score, verdict="TRIGGERED_DE_CONFIRMING", urls=()):
    cats = [{"headline": u, "url": u, "direction": "bearish"} for u in urls]
    return AnalystView.model_validate({"score": score, "score_delta_reasoning_ko": "r", "verdict": verdict,
                                       "catalysts": cats, "tape_read_ko": "t", "watch_conditions_ko": "w"})


def test_agree_uses_half_up_mean():
    r = reconcile({"hermes": _v(8.2), "dsh": _v(8.3)}, prior_score=8.2, rebuttal_done=False)
    assert r.mode == "agree" and r.score == 8.3 and r.verdict is Verdict.TRIGGERED_DE_CONFIRMING
    assert r.scores == {"hermes": 8.2, "dsh": 8.3} and not r.flags


def test_disagreement_requests_rebuttal_first():
    assert reconcile({"hermes": _v(7.5), "dsh": _v(8.4)}, prior_score=8.2, rebuttal_done=False) is None
    assert reconcile({"hermes": _v(8.2, "TRIGGERED"), "dsh": _v(8.2)}, prior_score=8.2, rebuttal_done=False) is None


def test_after_rebuttal_agree_or_take_most_cautious():
    ok = reconcile({"hermes": _v(8.0), "dsh": _v(8.3)}, prior_score=8.2, rebuttal_done=True)
    assert ok.mode == "agree_after_rebuttal" and ok.rebuttal_round
    split = reconcile({"hermes": _v(7.0, "TRIGGERED_FURTHER_DE_CONFIRMING"), "dsh": _v(8.4, "TRIGGERED")},
                      prior_score=8.2, rebuttal_done=True)
    assert split.mode == "disagree" and split.score == 7.7 and split.verdict is Verdict.TRIGGERED
    assert any("hermes" in f and "-1.2" in f for f in split.flags)


def test_single_analyst_is_flagged():
    r = reconcile({"dsh": _v(8.1)}, prior_score=8.2, rebuttal_done=False)
    assert r.mode == "single" and r.score == 8.1 and "single analyst: dsh" in r.flags


def test_no_views_is_an_error():
    with pytest.raises(ValueError):
        reconcile({}, prior_score=None, rebuttal_done=False)


def test_merge_catalysts_dedupes_by_url_and_tags_citers():
    merged = merge_catalysts({"hermes": _v(8, urls=["https://a", "https://b"]), "dsh": _v(8, urls=["https://b"])})
    assert [(c.url, c.cited_by) for c in merged] == [("https://a", ["hermes"]), ("https://b", ["hermes", "dsh"])]
