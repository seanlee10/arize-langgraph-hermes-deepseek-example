import datetime as dt

from bubble_watch.models import Verdict
from bubble_watch.state_store import load_state, save_state, seed_state


def test_seed_has_history_and_anchor():
    s = seed_state()
    assert [r.date.isoformat() for r in s.records] == [
        "2026-09-11", "2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17"]
    last = s.records[-1]
    assert last.score == 8.2 and last.verdict is Verdict.TRIGGERED_FURTHER_DE_CONFIRMING
    assert last.puts[200].iv == 34.90 and last.puts[200].freshness == "latest_snapshot"
    assert s.records[-2].puts[210].iv is None  # Sep-16 IV was N/A, never estimated
    assert s.anchors[0].date == dt.date(2026, 8, 28)
    assert s.anchors[0].put_prices == {200: 3.85, 210: 6.95, 220: 11.50}


def test_round_trip(tmp_path):
    path = tmp_path / "state" / "NVDA.json"
    save_state(seed_state(), path)
    back = load_state(path)
    assert back == seed_state()
    assert not list(path.parent.glob("*.tmp"))
