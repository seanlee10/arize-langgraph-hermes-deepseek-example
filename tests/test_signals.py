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


def test_sep17_matches_report():
    sig = _signals_for("2026-09-17")
    assert sig.returns["1d"]["NVDA"] == 2.54 and sig.returns["1d"]["SMH"] == 2.76
    assert sig.rel_spread["1d"] == -0.22
    assert sig.returns["3d"]["NVDA"] == 3.97 and sig.returns["3d"]["SMH"] == 3.53
    assert sig.rel_spread["3d"] == 0.44
    assert sig.put_changes["1d"] == {200: -2.86, 210: -3.81, 220: -5.08}
    assert sig.put_changes["3d"] == {200: -20.93, 210: -18.88, 220: -13.78}
    assert sig.put_changes["anchor"] == {200: -20.52, 210: -12.81, 220: -2.61}
    assert sig.returns["anchor"]["NVDA"] == 0.82 and sig.returns["anchor"]["SMH"] == 1.36
    assert sig.base_dates["3d"].isoformat() == "2026-09-14"
    # Sep-16 IV was N/A, so IV compares against the last verified snapshot (Sep-15).
    assert sig.iv[200].prior == 34.66 and sig.iv[200].prior_date.isoformat() == "2026-09-15"
    assert sig.iv_surface_up is False
    assert sig.convexity_order_1d == [200, 210, 220]
    assert sig.put_freshness_1d == ("late_session_last", "latest_snapshot")
    assert sig.conditions == {"nvda_underperforms": "partial", "iv_surface_up": "false", "far_otm_leads": "false"}


def test_sep16_matches_report():
    sig = _signals_for("2026-09-16")
    assert sig.rel_spread["1d"] == 0.18
    assert sig.put_changes["1d"] == {200: -10.76, 210: -8.83, 220: -3.67}
    assert sig.returns["3d"]["NVDA"] == -2.01 and sig.returns["3d"]["SMH"] == -4.04
    assert sig.put_changes["3d"] == {200: 32.35, 210: 30.17, 220: 31.11}
    assert sig.returns["anchor"]["SOXL"] == -6.62  # official close 103.97 (report used 104.03: -6.57)
    assert sig.iv_surface_up is None and sig.conditions["iv_surface_up"] == "unknown"
    assert sig.conditions["nvda_underperforms"] == "false"
    assert sig.convexity_order_1d == [220, 210, 200]


def test_far_otm_leads_true_when_200p_rises_fastest():
    s = seed_state()
    today = s.records[-1].model_copy(deep=True)
    today.date = today.date.replace(day=18)
    today.puts[200].last, today.puts[210].last, today.puts[220].last = 4.00, 6.50, 11.50
    sig = compute_signals(s, today)
    assert sig.conditions["far_otm_leads"] == "true"
