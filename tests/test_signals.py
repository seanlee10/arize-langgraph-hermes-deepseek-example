from bubble_watch.signals import compute_signals, pct
from bubble_watch.state_store import seed_state


def _signals_for(day_iso):
    s = seed_state()
    today = next(r for r in s.records if r.date.isoformat() == day_iso)
    s.records = [r for r in s.records if r.date < today.date]
    return compute_signals(s, today)


def test_pct():
    assert pct(219.34, 213.90) == 2.54
    assert pct(None, 1) is None and pct(1, None) is None and pct(1, 0) is None


def test_sep17_matches_report_prices_and_av_options():
    sig = _signals_for("2026-09-17")
    assert sig.returns["1d"]["NVDA"] == 2.54 and sig.returns["1d"]["SMH"] == 2.76
    assert sig.rel_spread["1d"] == -0.22
    assert sig.returns["3d"]["NVDA"] == 3.97 and sig.returns["3d"]["SMH"] == 3.53
    assert sig.rel_spread["3d"] == 0.44
    assert sig.put_changes["1d"] == {200: -45.1, 210: -39.77, 220: -33.04}
    assert sig.put_changes["3d"] == {200: -56.59, 210: -51.14, 220: -42.31}
    assert sig.put_changes["anchor"] == {200: -56.36, 210: -47.48, 220: -34.78}
    assert sig.returns["anchor"]["NVDA"] == 0.82 and sig.returns["anchor"]["SMH"] == 1.36
    assert sig.base_dates["3d"].isoformat() == "2026-09-14"
    assert sig.iv[200].prior == 35.63 and sig.iv[200].prior_date.isoformat() == "2026-09-16"
    assert sig.iv_surface_up is False
    assert sig.convexity_order_1d == [220, 210, 200]
    assert sig.put_freshness_1d == ("EOD", "EOD")
    assert sig.conditions == {"nvda_underperforms": "partial", "iv_surface_up": "false", "far_otm_leads": "false"}


def test_sep16_matches_report_prices_and_av_options():
    sig = _signals_for("2026-09-16")
    assert sig.rel_spread["1d"] == 0.18
    assert sig.put_changes["1d"] == {200: -13.31, 210: -12.3, 220: -8.57}
    assert sig.returns["3d"]["NVDA"] == -2.01 and sig.returns["3d"]["SMH"] == -4.04
    assert sig.put_changes["3d"] == {200: 28.57, 210: 25.21, 220: 24.44}
    assert sig.returns["anchor"]["SOXL"] == -6.62  # official close 103.97 (report used 104.03: -6.57)
    assert sig.iv_surface_up is False and sig.conditions["iv_surface_up"] == "false"  # 200P up, 210P/220P flat
    assert sig.conditions["nvda_underperforms"] == "false"
    assert sig.convexity_order_1d == [220, 210, 200]


def test_far_otm_leads_true_when_200p_rises_fastest():
    s = seed_state()
    today = s.records[-1].model_copy(deep=True)
    today.date = today.date.replace(day=18)
    today.puts[200].last, today.puts[210].last, today.puts[220].last = 4.00, 6.50, 11.50
    sig = compute_signals(s, today)
    assert sig.conditions["far_otm_leads"] == "true"


def _next_day(s, changes=None):
    today = s.records[-1].model_copy(deep=True)
    today.date = today.date.replace(day=18)
    for k, q in today.puts.items():
        for field, value in (changes or {}).get(k, {}).items():
            setattr(q, field, value)
    return today


def test_like_for_like_puts_are_comparable():
    s = seed_state()
    today = _next_day(s, {k: {"last": v} for k, v in ((200, 1.34), (210, 2.98), (220, 6.30))})
    sig = compute_signals(s, today)
    assert sig.put_comparability_1d == {200: None, 210: None, 220: None}


def test_source_change_is_flagged_and_neutralizes_convexity():
    s = seed_state()
    yahoo = "https://finance.yahoo.com/quote/NVDA261016P00200000"
    today = _next_day(s, {200: {"last": 4.00, "source_url": yahoo}, 210: {"last": 6.50}, 220: {"last": 11.50}})
    sig = compute_signals(s, today)
    assert sig.put_comparability_1d[200] == "source changed: www.alphavantage.co → finance.yahoo.com"
    assert sig.put_comparability_1d[210] is None
    assert sig.put_changes["1d"][200] == 138.1  # value kept, but flagged
    assert sig.conditions["far_otm_leads"] == "unknown"


def test_identical_quote_to_prior_day_is_flagged_stale():
    s = seed_state()
    today = _next_day(s)  # 9/17 quotes carried over unchanged, as the stale 9/17 page did with 9/16 data
    sig = compute_signals(s, today)
    assert all(r == "identical to the prior day's quote (stale feed?)" for r in sig.put_comparability_1d.values())
