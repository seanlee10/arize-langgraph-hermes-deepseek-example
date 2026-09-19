# Bubble Watch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A LangGraph pipeline that produces the daily Korean "NVDA Bubble Signal Watch" report, with Hermes Agent and DeepSeek Harness as two independent analyst sub-agents reconciled in code, traced to Arize AX.

**Architecture:** Deterministic core (market data → gap fill → signals, all numbers computed in Python) feeding two parallel analyst nodes that return schema-validated `AnalystView` JSON; a pure-code reconciler with one conditional rebuttal round; a template renders tables from code and an xAI writer call supplies only narrative paragraphs. State persists as JSON between days.

**Tech Stack:** Python 3.13, uv, LangGraph 1.2, Pydantic 2, httpx, yfinance, deepseek-harness-sdk (local editable), arize-otel + openinference-instrumentation-langchain, pytest.

**Spec:** `docs/specs/2026-09-19-bubble-watch-design.md`

## Global Constraints

- Model for both analysts and the writer: `grok-4.6` via xAI (`https://api.x.ai/v1`), overridable per agent with `HERMES_ANALYST_MODEL`, `DSH_ANALYST_MODEL`, `WRITER_MODEL`.
- Secrets only in `.env` (gitignored); code and `doctor` print variable **names**, never values.
- Every number in a report comes from code; agents never supply prices, returns or IVs to the report.
- Missing data is `None` with freshness `N/A`; never estimated. Gap fills without `source_url` + non-`N/A` freshness are rejected.
- Freshness vocabulary: `EOD`, `latest_snapshot`, `late_session_last`, `derived`, `N/A`.
- Verdicts: `NOT_TRIGGERED`, `TRIGGERED`, `TRIGGERED_DE_CONFIRMING`, `TRIGGERED_FURTHER_DE_CONFIRMING`, `CONFIRMED`; caution order `CONFIRMED > TRIGGERED > TRIGGERED_DE_CONFIRMING > TRIGGERED_FURTHER_DE_CONFIRMING > NOT_TRIGGERED`.
- Agreement: |Δscore| ≤ 0.5 and same verdict. Scores 0–10, one decimal, ROUND_HALF_UP.
- dsh runs under Node ≥ 22.19 (wrapper `bin/dsh` pins `~/.nvm/versions/node/v24.21.0`).
- Tracing is optional: no Arize credentials → no-op.
- Run tests with `uv run pytest -q` from `bubble-watch/`.

## File map

```
bubble-watch/
  pyproject.toml                 (modify: scripts, pytest config)
  .env.example  README.md
  bin/dsh                        (exists: Node 24 launcher)
  dsh/exa-search.patch.yml       dsh overlay: web_search via Exa
  src/bubble_watch/
    __init__.py  config.py  models.py  state_store.py  signals.py
    market_data/__init__.py base.py yfinance_provider.py alphavantage.py
    agents/__init__.py base.py prompts.py hermes_client.py dsh_client.py
    reconcile.py  writer.py  report.py  tracing.py  graph.py  cli.py
  tests/
    test_models.py test_state_store.py test_signals.py test_market_data.py
    test_agents_base.py test_hermes_client.py test_dsh_client.py
    test_reconcile.py test_report.py test_tracing.py test_graph.py test_cli.py
```

---

### Task 1: Project config and domain models

**Files:**
- Modify: `pyproject.toml`
- Create: `src/bubble_watch/config.py`, `src/bubble_watch/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `Settings`, `load_settings(env_file=None) -> Settings`, `missing_required(settings) -> list[str]`, `PROJECT_ROOT: Path`; models `Close`, `PutQuote(.mid)`, `Verdict`, `CAUTION_ORDER`, `most_cautious(*v)`, `Catalyst`, `AnalystView`, `IVCompare`, `Signals`, `MergedCatalyst`, `Reconciliation`, `DailyRecord`, `Anchor`, `WatchState(.symbols, .prior_records(day), .upsert(record))`, `Freshness`, `Condition`, `PUT_FIELDS`, `round1(x)`.

- [ ] **Step 1: Add scripts and pytest config to `pyproject.toml`**

Append:

```toml
[project.scripts]
bubble-watch = "bubble_watch.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 120
```

- [ ] **Step 2: Write the failing test**

**`tests/test_models.py`**
```python
import datetime as dt

import pytest

from bubble_watch.config import load_settings, missing_required
from bubble_watch.models import (
    AnalystView, DailyRecord, PutQuote, Verdict, WatchState, most_cautious, round1,
)


def _view(**kw):
    base = dict(score=8.26, score_delta_reasoning_ko="r", verdict="TRIGGERED", tape_read_ko="t",
                watch_conditions_ko="w")
    return AnalystView.model_validate({**base, **kw})


def test_score_is_clamped_and_rounded_half_up():
    assert _view(score=8.25).score == 8.3
    assert _view(score=11).score == 10.0
    assert _view(score=-3).score == 0.0


def test_verdict_must_be_known():
    with pytest.raises(ValueError):
        _view(verdict="MAYBE")


def test_most_cautious_prefers_confirmed_then_triggered():
    assert most_cautious(Verdict.TRIGGERED_DE_CONFIRMING, Verdict.TRIGGERED) is Verdict.TRIGGERED
    assert most_cautious(Verdict.NOT_TRIGGERED, Verdict.CONFIRMED) is Verdict.CONFIRMED


def test_put_mid_needs_both_sides():
    assert PutQuote(strike=200, bid=3.05, ask=3.10).mid == 3.075
    assert PutQuote(strike=200, bid=3.05).mid is None


def test_round1_half_up():
    assert round1(8.25) == 8.3
    assert round1(8.24) == 8.2
    assert round1((7.9 + 8.2) / 2) == 8.1  # 8.049999999999999 in floating point


def test_watch_state_upsert_and_prior_records():
    ws = WatchState()
    ws.upsert(DailyRecord(date=dt.date(2026, 9, 17)))
    ws.upsert(DailyRecord(date=dt.date(2026, 9, 16)))
    ws.upsert(DailyRecord(date=dt.date(2026, 9, 17), note="replaced"))
    assert [r.date.day for r in ws.records] == [16, 17]
    assert ws.records[-1].note == "replaced"
    assert [r.date.day for r in ws.prior_records(dt.date(2026, 9, 17))] == [16]
    assert ws.symbols == ["NVDA", "SMH", "SOXL"]


def test_int_strike_keys_survive_json_round_trip():
    rec = DailyRecord(date=dt.date(2026, 9, 17), puts={200: PutQuote(strike=200, last=3.06)})
    back = DailyRecord.model_validate_json(rec.model_dump_json())
    assert back.puts[200].last == 3.06


def test_settings_defaults_and_missing_names(tmp_path, monkeypatch):
    for name in ("HERMES_API_KEY", "HERMES_ANALYST_MODEL", "DSH_ANALYST_MODEL", "WRITER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    env = tmp_path / ".env"
    env.write_text("")
    s = load_settings(env)
    assert s.dsh_model == "grok-4.6" and s.writer_model == "grok-4.6"
    assert s.hermes_model is None
    assert missing_required(s) == ["HERMES_API_KEY"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bubble_watch.config'`

- [ ] **Step 4: Write the implementation**

**`src/bubble_watch/config.py`**
```python
"""Runtime settings from the environment (.env). Secrets are never logged: callers get names only."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "grok-4.6"
REQUIRED = ("XAI_API_KEY", "HERMES_API_KEY")


@dataclass(frozen=True)
class Settings:
    xai_api_key: str
    xai_base_url: str
    hermes_api_url: str
    hermes_api_key: str
    hermes_model: str | None
    dsh_bin: str
    dsh_home: str
    dsh_provider: str
    dsh_model: str
    writer_model: str
    analyst_timeout_s: float
    state_dir: Path
    reports_dir: Path
    arize_space_id: str
    arize_api_key: str
    arize_project: str
    arize_endpoint: str
    exa_api_key: str
    alphavantage_api_key: str


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def load_settings(env_file: str | Path | None = None) -> Settings:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    return Settings(
        xai_api_key=_env("XAI_API_KEY"),
        xai_base_url=_env("XAI_BASE_URL", "https://api.x.ai/v1"),
        hermes_api_url=_env("HERMES_API_URL", "http://localhost:8642/v1"),
        hermes_api_key=_env("HERMES_API_KEY"),
        # Unset: the Hermes gateway uses its own configured model (set it to grok-4.6 there).
        hermes_model=_env("HERMES_ANALYST_MODEL") or None,
        dsh_bin=_env("DSH_BIN", str(PROJECT_ROOT / "bin" / "dsh")),
        dsh_home=_env("DSH_HOME", str(PROJECT_ROOT / ".dsh")),
        dsh_provider=_env("DSH_PROVIDER", "xai"),
        dsh_model=_env("DSH_ANALYST_MODEL", DEFAULT_MODEL),
        writer_model=_env("WRITER_MODEL", DEFAULT_MODEL),
        analyst_timeout_s=float(_env("ANALYST_TIMEOUT_S", "600")),
        state_dir=Path(_env("STATE_DIR", str(PROJECT_ROOT / "state"))),
        reports_dir=Path(_env("REPORTS_DIR", str(PROJECT_ROOT / "reports"))),
        arize_space_id=_env("ARIZE_SPACE_ID"),
        arize_api_key=_env("ARIZE_API_KEY"),
        arize_project=_env("ARIZE_PROJECT_NAME", "bubble-watch"),
        arize_endpoint=_env("ARIZE_COLLECTOR_ENDPOINT", "https://otlp.arize.com/v1"),
        exa_api_key=_env("EXA_API_KEY"),
        alphavantage_api_key=_env("ALPHAVANTAGE_API_KEY"),
    )


def missing_required(settings: Settings) -> list[str]:
    values = {"XAI_API_KEY": settings.xai_api_key, "HERMES_API_KEY": settings.hermes_api_key}
    return [name for name in REQUIRED if not values[name]]
```

**`src/bubble_watch/models.py`**
```python
"""Domain models shared across the pipeline. Everything is JSON-serializable for the state file."""
from __future__ import annotations

import datetime as dt
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Freshness = Literal["EOD", "latest_snapshot", "late_session_last", "derived", "N/A"]
Condition = Literal["true", "false", "partial", "unknown"]
PUT_FIELDS = ("bid", "ask", "last", "iv", "volume", "oi")


def round1(x: float) -> float:
    # round(x, 6) first strips float noise (8.049999999999999 -> 8.05) before half-up to one decimal.
    return float(Decimal(str(round(x, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


class Close(BaseModel):
    price: float | None = None
    source: str = ""
    freshness: Freshness = "EOD"


class PutQuote(BaseModel):
    strike: int
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    iv: float | None = None  # percent, e.g. 34.9
    volume: int | None = None
    oi: int | None = None
    source_url: str = ""
    freshness: Freshness = "N/A"
    as_of: dt.datetime | None = None

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return round((self.bid + self.ask) / 2, 3)


class Verdict(str, Enum):
    NOT_TRIGGERED = "NOT_TRIGGERED"
    TRIGGERED = "TRIGGERED"
    TRIGGERED_DE_CONFIRMING = "TRIGGERED_DE_CONFIRMING"
    TRIGGERED_FURTHER_DE_CONFIRMING = "TRIGGERED_FURTHER_DE_CONFIRMING"
    CONFIRMED = "CONFIRMED"


# Most cautious first: the verdict that keeps the bubble warning most active.
CAUTION_ORDER = [Verdict.CONFIRMED, Verdict.TRIGGERED, Verdict.TRIGGERED_DE_CONFIRMING,
                 Verdict.TRIGGERED_FURTHER_DE_CONFIRMING, Verdict.NOT_TRIGGERED]


def most_cautious(*verdicts: Verdict) -> Verdict:
    return min(verdicts, key=CAUTION_ORDER.index)


class Catalyst(BaseModel):
    headline: str
    url: str
    published: dt.date | None = None
    direction: Literal["bearish", "bullish", "neutral"]
    weight: Literal["low", "med", "high"] = "med"
    rationale_ko: str = ""


class AnalystView(BaseModel):
    score: float
    score_delta_reasoning_ko: str
    verdict: Verdict
    catalysts: list[Catalyst] = Field(default_factory=list)
    tape_read_ko: str
    watch_conditions_ko: str
    confidence: Literal["low", "med", "high"] = "med"
    rebuttal_ko: str | None = None

    @field_validator("score", mode="before")
    @classmethod
    def _clamp(cls, v: object) -> float:
        return round1(min(10.0, max(0.0, float(v))))  # type: ignore[arg-type]


class IVCompare(BaseModel):
    current: float | None = None
    prior: float | None = None
    prior_date: dt.date | None = None


class Signals(BaseModel):
    base_dates: dict[str, dt.date | None]              # window ("1d", "3d", "anchor") -> base date
    returns: dict[str, dict[str, float | None]]        # window -> symbol -> %
    rel_spread: dict[str, float | None]                # window -> ticker − peer, %p
    put_changes: dict[str, dict[int, float | None]]    # window -> strike -> %
    put_freshness_1d: tuple[str, str] | None = None    # (base freshness, today freshness)
    convexity_order_1d: list[int] = Field(default_factory=list)
    iv: dict[int, IVCompare] = Field(default_factory=dict)
    iv_surface_up: bool | None = None
    conditions: dict[str, Condition] = Field(default_factory=dict)


class MergedCatalyst(Catalyst):
    cited_by: list[str] = Field(default_factory=list)


class Reconciliation(BaseModel):
    mode: Literal["agree", "agree_after_rebuttal", "disagree", "single"]
    score: float
    verdict: Verdict
    scores: dict[str, float]
    verdicts: dict[str, Verdict]
    rebuttal_round: bool = False
    catalysts: list[MergedCatalyst] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class DailyRecord(BaseModel):
    date: dt.date
    closes: dict[str, Close] = Field(default_factory=dict)
    puts: dict[int, PutQuote] = Field(default_factory=dict)
    signals: Signals | None = None
    score: float | None = None
    score_delta: float | None = None
    verdict: Verdict | None = None
    analyst_views: dict[str, AnalystView] = Field(default_factory=dict)
    reconciliation: Reconciliation | None = None
    report_path: str | None = None
    note: str = ""


class Anchor(BaseModel):
    label: str
    date: dt.date
    closes: dict[str, float]
    put_prices: dict[int, float]


class WatchState(BaseModel):
    ticker: str = "NVDA"
    peer: str = "SMH"
    leveraged: str = "SOXL"
    expiry: dt.date = dt.date(2026, 10, 16)
    strikes: list[int] = Field(default_factory=lambda: [200, 210, 220])
    trigger_threshold: float = 8.0
    anchors: list[Anchor] = Field(default_factory=list)
    records: list[DailyRecord] = Field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        return [self.ticker, self.peer, self.leveraged]

    def prior_records(self, day: dt.date) -> list[DailyRecord]:
        return sorted((r for r in self.records if r.date < day), key=lambda r: r.date)

    def upsert(self, record: DailyRecord) -> None:
        kept = [r for r in self.records if r.date != record.date]
        self.records = sorted([*kept, record], key=lambda r: r.date)
```

Also create empty `src/bubble_watch/__init__.py` content: `"""Bubble Watch: LangGraph NVDA bubble-signal analysis with Hermes and DeepSeek Harness analysts."""`

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -q`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/bubble_watch tests/test_models.py uv.lock .gitignore bin/dsh
git commit -m "feat: settings and domain models"
```

---

### Task 2: State store and seed from the 9/11–9/17 reports

**Files:**
- Create: `src/bubble_watch/state_store.py`
- Test: `tests/test_state_store.py`

**Interfaces:**
- Consumes: `WatchState`, `DailyRecord`, `Close`, `PutQuote`, `Anchor`, `Verdict` (Task 1)
- Produces: `load_state(path) -> WatchState`, `save_state(state, path) -> None`, `seed_state() -> WatchState`

Seed values: NVDA/SMH/SOXL closes are yfinance official EOD closes (NVDA and SMH match the reports to the cent; SOXL 9/16–9/17 in the reports were late-session values). Put values come from the reports; 9/11 and 9/14 put lasts are back-derived from the reports' 3-day percentages (`derived`).

- [ ] **Step 1: Write the failing test**

**`tests/test_state_store.py`**
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_state_store.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bubble_watch.state_store'`

- [ ] **Step 3: Write the implementation**

**`src/bubble_watch/state_store.py`**
```python
"""JSON persistence for WatchState, plus the seed transcribed from the 9/15–9/17 reports."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from .models import Anchor, Close, DailyRecord, PutQuote, Verdict, WatchState


def load_state(path: Path) -> WatchState:
    return WatchState.model_validate_json(Path(path).read_text())


def save_state(state: WatchState, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json(indent=2))
    os.replace(tmp, path)


_YF = "https://finance.yahoo.com/quote/{}/history"


def _closes(nvda: float, smh: float, soxl: float) -> dict[str, Close]:
    return {s: Close(price=p, source=_YF.format(s), freshness="EOD")
            for s, p in (("NVDA", nvda), ("SMH", smh), ("SOXL", soxl))}


def _puts(freshness: str, source: str, rows: dict[int, dict]) -> dict[int, PutQuote]:
    return {k: PutQuote(strike=k, freshness=freshness, source_url=source, **v) for k, v in rows.items()}


def seed_state() -> WatchState:
    d = dt.date
    records = [
        DailyRecord(date=d(2026, 9, 11), closes=_closes(218.29, 568.53, 121.82),
                    puts=_puts("derived", "derived from 2026-09-16 report 3-day changes",
                               {200: {"last": 2.38}, 210: {"last": 4.84}, 220: {"last": 9.00}})),
        DailyRecord(date=d(2026, 9, 14), closes=_closes(210.96, 541.50, 101.13),
                    puts=_puts("derived", "derived from 2026-09-17 report 3-day changes",
                               {200: {"last": 3.87}, 210: {"last": 7.47}, 220: {"last": 12.99}}),
                    note="Sep-14: strong $200P > $210P > $220P convexity spike (warning signal)."),
        DailyRecord(date=d(2026, 9, 15), closes=_closes(212.17, 542.11, 102.32),
                    puts=_puts("EOD", "https://www.alphavantage.co/documentation/#historical-options", {
                        200: {"bid": 3.45, "ask": 3.55, "last": 3.53, "iv": 34.66},
                        210: {"bid": 6.85, "ask": 6.90, "last": 6.91, "iv": 33.68},
                        220: {"bid": 12.25, "ask": 12.30, "last": 12.25, "iv": 32.71}}),
                    score=8.7, verdict=Verdict.TRIGGERED),
        DailyRecord(date=d(2026, 9, 16), closes=_closes(213.90, 545.56, 103.97),
                    puts=_puts("late_session_last", "https://oqliv.com/flow?days=30&ticker=NVDA", {
                        200: {"last": 3.15, "volume": 3571, "oi": 43504},
                        210: {"last": 6.30, "volume": 2028, "oi": 24694},
                        220: {"last": 11.80, "volume": 1507, "oi": 35872}}),
                    score=8.4, score_delta=-0.3, verdict=Verdict.TRIGGERED_DE_CONFIRMING,
                    note=("NVDA outperformed SMH (+0.18%p); all puts fell with $200P falling most (-10.8%); "
                          "Apple weighing Nvidia networking for M8 Ultra servers (bullish); Fed hike (macro headwind).")),
        DailyRecord(date=d(2026, 9, 17), closes=_closes(219.34, 560.61, 114.82),
                    puts=_puts("latest_snapshot", "https://whalequant.io/en/stocks/NVDA/options/2026-10-16", {
                        200: {"bid": 3.05, "ask": 3.10, "last": 3.06, "iv": 34.90, "volume": 4222, "oi": 43504},
                        210: {"bid": 6.10, "ask": 6.15, "last": 6.06, "iv": 32.80, "volume": 2262, "oi": 24694},
                        220: {"bid": 11.05, "ask": 11.35, "last": 11.20, "iv": 31.97, "volume": 1609, "oi": 35872}}),
                    score=8.2, score_delta=-0.2, verdict=Verdict.TRIGGERED_FURTHER_DE_CONFIRMING,
                    note=("Semis risk-on (SMH +2.76%, SOXL +9.4%); NVDA −0.22%p vs SMH; puts retraced with no "
                          "OTM convex acceleration; IV not surface-wide. Huawei AI chip roadmap (bearish, China moat); "
                          "Nebius raising Nvidia GPU rental prices (bullish demand).")),
    ]
    anchor = Anchor(label="2026-08-28 spike", date=d(2026, 8, 28),
                    closes={"NVDA": 217.55, "SMH": 553.11, "SOXL": 111.34},
                    put_prices={200: 3.85, 210: 6.95, 220: 11.50})
    return WatchState(anchors=[anchor], records=records)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_state_store.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/bubble_watch/state_store.py tests/test_state_store.py
git commit -m "feat: JSON state store with seed from 9/11-9/17 reports"
```

---

### Task 3: Signals (golden-tested against the 9/16 and 9/17 reports)

**Files:**
- Create: `src/bubble_watch/signals.py`
- Test: `tests/test_signals.py`

**Interfaces:**
- Consumes: `WatchState`, `DailyRecord`, `Signals`, `IVCompare` (Task 1), `seed_state` (Task 2)
- Produces: `pct(new, old) -> float | None`, `compute_signals(state, today) -> Signals`

Windows: `1d` = previous record; `3d` = 3 records back (records are trading days); `anchor` = `state.anchors[0]`. IV compares against the most recent prior record with a non-null IV for that strike.

- [ ] **Step 1: Write the failing test**

**`tests/test_signals.py`**
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_signals.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bubble_watch.signals'`

- [ ] **Step 3: Write the implementation**

**`src/bubble_watch/signals.py`**
```python
"""Pure signal computation: every number the report shows comes from here."""
from __future__ import annotations

from .models import Condition, DailyRecord, IVCompare, Signals, WatchState


def pct(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return round((new / old - 1) * 100, 2)


def _spread(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else round(a - b, 2)


def _close(rec: DailyRecord | None, sym: str) -> float | None:
    c = rec.closes.get(sym) if rec else None
    return c.price if c else None


def _put_last(rec: DailyRecord | None, strike: int) -> float | None:
    q = rec.puts.get(strike) if rec else None
    return q.last if q else None


def _iv_compare(prior: list[DailyRecord], today: DailyRecord, strike: int) -> IVCompare:
    q = today.puts.get(strike)
    current = q.iv if q else None
    for rec in reversed(prior):
        pq = rec.puts.get(strike)
        if pq and pq.iv is not None:
            return IVCompare(current=current, prior=pq.iv, prior_date=rec.date)
    return IVCompare(current=current)


def _underperforms(rel_1d: float | None, rel_3d: float | None) -> Condition:
    if rel_1d is None:
        return "unknown"
    if rel_1d >= 0:
        return "false"
    return "true" if rel_3d is not None and rel_3d < 0 else "partial"


def _far_otm_leads(changes: dict[int, float | None], strikes: list[int]) -> Condition:
    if any(changes.get(k) is None for k in strikes):
        return "unknown"
    lo, others = min(strikes), [k for k in strikes if k != min(strikes)]
    lead = changes[lo]
    return "true" if lead > 0 and all(lead > changes[k] for k in others) else "false"  # type: ignore[operator]


def compute_signals(state: WatchState, today: DailyRecord) -> Signals:
    prior = state.prior_records(today.date)
    bases = {"1d": prior[-1] if prior else None, "3d": prior[-3] if len(prior) >= 3 else None}
    anchor = state.anchors[0] if state.anchors else None
    today_close = {s: _close(today, s) for s in state.symbols}

    base_dates, returns, put_changes = {}, {}, {}
    for window, rec in bases.items():
        base_dates[window] = rec.date if rec else None
        returns[window] = {s: pct(today_close[s], _close(rec, s)) for s in state.symbols}
        put_changes[window] = {k: pct(_put_last(today, k), _put_last(rec, k)) for k in state.strikes}
    base_dates["anchor"] = anchor.date if anchor else None
    returns["anchor"] = {s: pct(today_close[s], anchor.closes.get(s) if anchor else None) for s in state.symbols}
    put_changes["anchor"] = {k: pct(_put_last(today, k), anchor.put_prices.get(k) if anchor else None)
                             for k in state.strikes}
    rel = {w: _spread(r.get(state.ticker), r.get(state.peer)) for w, r in returns.items()}

    base_1d, lo = bases["1d"], min(state.strikes)
    freshness = None
    if base_1d and lo in base_1d.puts and lo in today.puts:
        freshness = (base_1d.puts[lo].freshness, today.puts[lo].freshness)

    changes_1d = put_changes["1d"]
    order = sorted((k for k, v in changes_1d.items() if v is not None), key=lambda k: changes_1d[k], reverse=True)
    iv = {k: _iv_compare(prior, today, k) for k in state.strikes}
    surface = (None if any(c.current is None or c.prior is None for c in iv.values())
               else all(c.current > c.prior for c in iv.values()))  # type: ignore[operator]

    return Signals(
        base_dates=base_dates, returns=returns, rel_spread=rel, put_changes=put_changes,
        put_freshness_1d=freshness, convexity_order_1d=order, iv=iv, iv_surface_up=surface,
        conditions={
            "nvda_underperforms": _underperforms(rel["1d"], rel["3d"]),
            "iv_surface_up": "unknown" if surface is None else ("true" if surface else "false"),
            "far_otm_leads": _far_otm_leads(changes_1d, state.strikes),
        },
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_signals.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/bubble_watch/signals.py tests/test_signals.py
git commit -m "feat: pure signal computation, golden-tested against 9/16 and 9/17 reports"
```

---

### Task 4: Market data providers and gap handling

**Files:**
- Create: `src/bubble_watch/market_data/__init__.py`, `base.py`, `yfinance_provider.py`, `alphavantage.py`
- Test: `tests/test_market_data.py`

**Interfaces:**
- Consumes: `Close`, `PutQuote`, `Freshness`, `PUT_FIELDS` (Task 1)
- Produces: `MarketSnapshot(date, closes, puts)`, `MarketDataProvider` protocol `.snapshot(day, symbols, ticker, expiry, strikes)`, `GapRequest(field, description)`, `GapFill(field, value, source_url, freshness, note)`, `find_gaps(snapshot, symbols, strikes) -> list[GapRequest]`, `apply_fills(snapshot, fills) -> list[str]` (mutates snapshot, returns rejection messages), `merge_puts(primary, secondary) -> dict[int, PutQuote]`, `YFinanceProvider(download=None, ticker_factory=None)`, `AlphaVantageOptions(api_key, client=None).puts(day, ticker, expiry, strikes)`.

- [ ] **Step 1: Write the failing test**

**`tests/test_market_data.py`**
```python
import datetime as dt

import httpx
import pandas as pd

from bubble_watch.market_data.alphavantage import AlphaVantageOptions
from bubble_watch.market_data.base import GapFill, MarketSnapshot, apply_fills, find_gaps, merge_puts
from bubble_watch.market_data.yfinance_provider import YFinanceProvider
from bubble_watch.models import Close, PutQuote

DAY = dt.date(2026, 9, 18)
EXP = dt.date(2026, 10, 16)


def _fake_download(symbols, start, end, progress, auto_adjust):
    idx = pd.DatetimeIndex([pd.Timestamp("2026-09-17"), pd.Timestamp("2026-09-18")])
    cols = pd.MultiIndex.from_product([["Close"], symbols])
    data = [[219.34, 560.61, 114.82], [222.27, 573.00, float("nan")]]
    return pd.DataFrame(data, index=idx, columns=cols)


class _FakeTicker:
    options = ("2026-09-25", "2026-10-16")

    def option_chain(self, expiry):
        rows = [
            {"contractSymbol": "NVDA261016P00200000", "strike": 200.0, "lastPrice": 1.34, "bid": 1.30, "ask": 1.35,
             "impliedVolatility": 0.343512, "volume": 3310.0, "openInterest": 43456.0,
             "lastTradeDate": pd.Timestamp("2026-09-18 19:59:58", tz="UTC")},
            {"contractSymbol": "NVDA261016P00210000", "strike": 210.0, "lastPrice": 2.98, "bid": 0.0, "ask": 2.97,
             "impliedVolatility": 0.317451, "volume": float("nan"), "openInterest": 26816.0,
             "lastTradeDate": pd.Timestamp("2026-09-17 19:59:41", tz="UTC")},  # stale: last traded yesterday
        ]
        return type("Chain", (), {"puts": pd.DataFrame(rows)})()


def test_yfinance_snapshot_labels_freshness_and_skips_stale_rows():
    p = YFinanceProvider(download=_fake_download, ticker_factory=lambda t: _FakeTicker())
    snap = p.snapshot(DAY, ["NVDA", "SMH", "SOXL"], "NVDA", EXP, [200, 210, 220])
    assert snap.closes["NVDA"].price == 222.27 and snap.closes["NVDA"].freshness == "EOD"
    assert snap.closes["SOXL"].price is None
    q = snap.puts[200]
    assert (q.bid, q.ask, q.last, q.iv, q.volume, q.oi) == (1.30, 1.35, 1.34, 34.35, 3310, 43456)
    assert q.freshness == "latest_snapshot" and q.source_url.endswith("NVDA261016P00200000")
    assert 210 not in snap.puts and 220 not in snap.puts


def test_find_gaps_and_apply_fills_enforce_source_and_freshness():
    snap = MarketSnapshot(date=DAY, closes={"NVDA": Close(price=222.27), "SOXL": Close(price=None)},
                          puts={200: PutQuote(strike=200, bid=1.3, ask=1.35, last=1.34, iv=34.35, volume=1, oi=2,
                                              freshness="latest_snapshot")})
    gaps = {g.field for g in find_gaps(snap, ["NVDA", "SOXL"], [200, 210])}
    assert gaps == {"closes.SOXL", *(f"puts.210.{f}" for f in ("bid", "ask", "last", "iv", "volume", "oi"))}

    rejected = apply_fills(snap, [
        GapFill(field="closes.SOXL", value=123.67, source_url="https://stockanalysis.com/etf/soxl/", freshness="EOD"),
        GapFill(field="puts.210.last", value=2.98, source_url="", freshness="EOD"),          # no source
        GapFill(field="puts.210.iv", value=31.7, source_url="https://x.test", freshness="N/A"),  # value but N/A
        GapFill(field="puts.210.volume", value=2965, source_url="https://x.test/c", freshness="latest_snapshot"),
        GapFill(field="puts.210.oi", value=None, freshness="N/A"),                           # explicit N/A is fine
        GapFill(field="nonsense", value=1, source_url="https://x.test", freshness="EOD"),
    ])
    assert snap.closes["SOXL"].price == 123.67 and snap.closes["SOXL"].freshness == "EOD"
    assert snap.puts[210].last is None and snap.puts[210].iv is None
    assert snap.puts[210].volume == 2965 and snap.puts[210].freshness == "latest_snapshot"
    assert len(rejected) == 3


def test_merge_puts_fills_missing_fields_from_secondary():
    primary = {200: PutQuote(strike=200, last=1.34, freshness="latest_snapshot")}
    secondary = {200: PutQuote(strike=200, last=1.40, iv=34.0, freshness="EOD"),
                 210: PutQuote(strike=210, last=2.9, freshness="EOD")}
    merged = merge_puts(primary, secondary)
    assert merged[200].last == 1.34 and merged[200].iv == 34.0
    assert merged[210].last == 2.9 and merged[210].freshness == "EOD"


def test_alphavantage_parses_historical_puts():
    payload = {"data": [
        {"type": "put", "expiration": "2026-10-16", "strike": "200.00", "last": "1.33", "bid": "1.30", "ask": "1.36",
         "implied_volatility": "0.34201", "volume": "3301", "open_interest": "43456"},
        {"type": "call", "expiration": "2026-10-16", "strike": "200.00", "last": "25", "bid": "1", "ask": "1",
         "implied_volatility": "0.3", "volume": "1", "open_interest": "1"},
    ]}
    seen = {}

    def handler(request):
        seen.update(dict(request.url.params))
        return httpx.Response(200, json=payload)

    av = AlphaVantageOptions("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    puts = av.puts(DAY, "NVDA", EXP, [200, 210])
    assert seen["function"] == "HISTORICAL_OPTIONS" and seen["date"] == "2026-09-18"
    assert list(puts) == [200]
    assert (puts[200].iv, puts[200].oi, puts[200].freshness) == (34.2, 43456, "EOD")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_market_data.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bubble_watch.market_data'`

- [ ] **Step 3: Write the implementation**

**`src/bubble_watch/market_data/__init__.py`**
```python
"""Market data providers and gap handling."""
```

**`src/bubble_watch/market_data/base.py`**
```python
"""Snapshot model, provider protocol, and the gap-fill contract (no value without source + freshness)."""
from __future__ import annotations

import datetime as dt
from typing import Protocol

from pydantic import BaseModel

from ..models import PUT_FIELDS, Close, Freshness, PutQuote

_INT_FIELDS = {"volume", "oi"}


class MarketSnapshot(BaseModel):
    date: dt.date
    closes: dict[str, Close]
    puts: dict[int, PutQuote]


class MarketDataProvider(Protocol):
    def snapshot(self, day: dt.date, symbols: list[str], ticker: str, expiry: dt.date,
                 strikes: list[int]) -> MarketSnapshot: ...


class GapRequest(BaseModel):
    field: str  # "closes.SOXL" | "puts.210.iv"
    description: str


class GapFill(BaseModel):
    field: str
    value: float | None
    source_url: str = ""
    freshness: Freshness = "N/A"
    note: str = ""


def find_gaps(snapshot: MarketSnapshot, symbols: list[str], strikes: list[int]) -> list[GapRequest]:
    gaps = [GapRequest(field=f"closes.{s}", description=f"{s} official close on {snapshot.date}")
            for s in symbols if snapshot.closes.get(s) is None or snapshot.closes[s].price is None]
    for k in strikes:
        q = snapshot.puts.get(k)
        gaps += [GapRequest(field=f"puts.{k}.{f}", description=f"${k} put {f} on {snapshot.date}")
                 for f in PUT_FIELDS if q is None or getattr(q, f) is None]
    return gaps


def apply_fills(snapshot: MarketSnapshot, fills: list[GapFill]) -> list[str]:
    """Apply accepted fills in place; return a message per rejected fill."""
    rejected = []
    for fill in fills:
        if fill.value is not None and (not fill.source_url.startswith(("http://", "https://"))
                                       or fill.freshness == "N/A"):
            rejected.append(f"{fill.field}: value without source_url/freshness rejected")
            continue
        parts = fill.field.split(".")
        if len(parts) == 2 and parts[0] == "closes":
            if fill.value is not None:
                snapshot.closes[parts[1]] = Close(price=fill.value, source=fill.source_url, freshness=fill.freshness)
        elif len(parts) == 3 and parts[0] == "puts" and parts[1].isdigit() and parts[2] in PUT_FIELDS:
            strike = int(parts[1])
            q = snapshot.puts.setdefault(strike, PutQuote(strike=strike))
            if fill.value is not None:
                setattr(q, parts[2], int(fill.value) if parts[2] in _INT_FIELDS else float(fill.value))
                q.source_url = q.source_url or fill.source_url
                if q.freshness == "N/A":
                    q.freshness = fill.freshness
        else:
            rejected.append(f"{fill.field}: unknown field")
    return rejected


def merge_puts(primary: dict[int, PutQuote], secondary: dict[int, PutQuote]) -> dict[int, PutQuote]:
    """Fill fields missing from ``primary`` with ``secondary`` values (e.g. historical IV)."""
    merged = {k: q.model_copy() for k, q in primary.items()}
    for k, sq in secondary.items():
        if k not in merged:
            merged[k] = sq.model_copy()
            continue
        q = merged[k]
        for f in PUT_FIELDS:
            if getattr(q, f) is None and getattr(sq, f) is not None:
                setattr(q, f, getattr(sq, f))
    return merged
```

**`src/bubble_watch/market_data/yfinance_provider.py`**
```python
"""Default provider: official closes + today's option chain (yfinance has no historical chains)."""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Callable

from ..models import Close, PutQuote
from .base import MarketSnapshot


def _num(v: Any, *, positive: bool = False) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or (positive and f <= 0):
        return None
    return f


def _int(v: Any) -> int | None:
    f = _num(v)
    return None if f is None else int(f)


class YFinanceProvider:
    def __init__(self, download: Callable | None = None, ticker_factory: Callable | None = None) -> None:
        if download is None or ticker_factory is None:
            import yfinance as yf
            download, ticker_factory = download or yf.download, ticker_factory or yf.Ticker
        self._download, self._ticker = download, ticker_factory

    def snapshot(self, day: dt.date, symbols: list[str], ticker: str, expiry: dt.date,
                 strikes: list[int]) -> MarketSnapshot:
        return MarketSnapshot(date=day, closes=self._closes(day, symbols), puts=self._puts(day, ticker, expiry, strikes))

    def _closes(self, day: dt.date, symbols: list[str]) -> dict[str, Close]:
        df = self._download(symbols, start=(day - dt.timedelta(days=6)).isoformat(),
                            end=(day + dt.timedelta(days=1)).isoformat(), progress=False, auto_adjust=False)
        out: dict[str, Close] = {}
        for s in symbols:
            price = None
            try:
                col = df["Close"][s].dropna()
                if len(col) and col.index[-1].date() == day:
                    price = round(float(col.iloc[-1]), 2)
            except KeyError:
                pass
            out[s] = Close(price=price, source=f"https://finance.yahoo.com/quote/{s}/history",
                           freshness="EOD" if price is not None else "N/A")
        return out

    def _puts(self, day: dt.date, ticker: str, expiry: dt.date, strikes: list[int]) -> dict[int, PutQuote]:
        t = self._ticker(ticker)
        if expiry.isoformat() not in t.options:
            return {}
        chain = t.option_chain(expiry.isoformat()).puts
        out: dict[int, PutQuote] = {}
        for k in strikes:
            rows = chain[chain["strike"] == float(k)]
            if rows.empty:
                continue
            r = rows.iloc[0]
            traded = r["lastTradeDate"]
            traded_day = traded.tz_convert("America/New_York").date() if traded.tzinfo else traded.date()
            if traded_day != day:  # the chain is "now"; a row last traded on another day is not today's data
                continue
            out[k] = PutQuote(
                strike=k, bid=_num(r["bid"], positive=True), ask=_num(r["ask"], positive=True),
                last=_num(r["lastPrice"]), iv=None if _num(r["impliedVolatility"]) is None
                else round(float(r["impliedVolatility"]) * 100, 2),
                volume=_int(r["volume"]), oi=_int(r["openInterest"]),
                source_url=f"https://finance.yahoo.com/quote/{r['contractSymbol']}",
                freshness="latest_snapshot", as_of=traded.to_pydatetime(),
            )
        return out
```

**`src/bubble_watch/market_data/alphavantage.py`**
```python
"""Optional: Alpha Vantage historical option snapshots (EOD), used to fill IV/bid/ask for a past date."""
from __future__ import annotations

import datetime as dt

import httpx

from ..models import PutQuote

URL = "https://www.alphavantage.co/query"
DOC_URL = "https://www.alphavantage.co/documentation/#historical-options"


def _f(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class AlphaVantageOptions:
    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        self._key = api_key
        self._client = client or httpx.Client(timeout=30)

    def puts(self, day: dt.date, ticker: str, expiry: dt.date, strikes: list[int]) -> dict[int, PutQuote]:
        r = self._client.get(URL, params={"function": "HISTORICAL_OPTIONS", "symbol": ticker,
                                          "date": day.isoformat(), "apikey": self._key})
        r.raise_for_status()
        out: dict[int, PutQuote] = {}
        for row in r.json().get("data") or []:
            strike = _f(row.get("strike"))
            if (row.get("type") != "put" or row.get("expiration") != expiry.isoformat()
                    or strike is None or int(strike) not in strikes):
                continue
            iv, vol, oi = _f(row.get("implied_volatility")), _f(row.get("volume")), _f(row.get("open_interest"))
            out[int(strike)] = PutQuote(
                strike=int(strike), bid=_f(row.get("bid")), ask=_f(row.get("ask")), last=_f(row.get("last")),
                iv=None if iv is None else round(iv * 100, 2), volume=None if vol is None else int(vol),
                oi=None if oi is None else int(oi), source_url=DOC_URL, freshness="EOD")
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_market_data.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/bubble_watch/market_data tests/test_market_data.py
git commit -m "feat: yfinance/Alpha Vantage providers and source-enforced gap fills"
```

---
### Task 5: Analyst contract — prompts, JSON extraction, validated views with one retry

**Files:**
- Create: `src/bubble_watch/agents/__init__.py`, `src/bubble_watch/agents/base.py`, `src/bubble_watch/agents/prompts.py`
- Test: `tests/test_agents_base.py`

**Interfaces:**
- Consumes: `AnalystView`, `WatchState`, `DailyRecord`, `Signals` (Task 1); `GapRequest` (Task 4); `seed_state` (Task 2); `compute_signals` (Task 3)
- Produces: `AnalystError`, `AskResult(text, session_id, events)`, `AnalystResult(agent, view, session_id, events, retried)`, `Analyst` protocol (`name`, `analyze(brief) -> AnalystResult`, `rebut(prompt) -> AnalystResult`), `extract_json(text) -> dict`, `parse_view(text) -> AnalystView`, `ask_for_view(agent, ask, prompt) -> AnalystResult`; prompts `SYSTEM_ANALYST`, `SYSTEM_GAP_FILL`, `analyst_brief(state, today, signals)`, `rebuttal_brief(brief, own_name, own, other_name, other)`, `gap_fill_brief(day, ticker, expiry, gaps)`, `retry_prompt(error)`.

- [ ] **Step 1: Write the failing test**

**`tests/test_agents_base.py`**
```python
import pytest

from bubble_watch.agents.base import AnalystError, AskResult, ask_for_view, extract_json, parse_view
from bubble_watch.agents.prompts import analyst_brief, gap_fill_brief, rebuttal_brief
from bubble_watch.market_data.base import GapRequest
from bubble_watch.signals import compute_signals
from bubble_watch.state_store import seed_state

VIEW = ('{"score": 8.0, "score_delta_reasoning_ko": "r", "verdict": "TRIGGERED_DE_CONFIRMING", '
        '"catalysts": [{"headline": "h", "url": "https://reuters.com/x", "direction": "bearish"}, '
        '{"headline": "no url", "url": "n/a", "direction": "bullish"}], '
        '"tape_read_ko": "t", "watch_conditions_ko": "w", "confidence": "high"}')


def test_extract_json_prefers_fenced_block_and_tolerates_prose():
    assert extract_json(f"분석 결과:\n```json\n{VIEW}\n```\n끝")["score"] == 8.0
    assert extract_json(f"prose {{not json}} then {VIEW} trailing")["verdict"] == "TRIGGERED_DE_CONFIRMING"
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_parse_view_drops_catalysts_without_urls():
    view = parse_view(VIEW)
    assert [c.headline for c in view.catalysts] == ["h"]


def test_ask_for_view_retries_once_in_same_session():
    calls = []

    def ask(prompt, session_id=None):
        calls.append((prompt, session_id))
        return AskResult(text="not json" if len(calls) == 1 else VIEW, session_id="s-1", events=[{"n": len(calls)}])

    result = ask_for_view("hermes", ask, "brief")
    assert result.retried and result.view.score == 8.0
    assert calls[1][1] == "s-1" and "JSON" in calls[1][0]
    assert result.events == [{"n": 1}, {"n": 2}]


def test_ask_for_view_gives_up_after_retry():
    with pytest.raises(AnalystError, match="hermes"):
        ask_for_view("hermes", lambda p, session_id=None: AskResult(text="nope"), "brief")


def test_briefs_carry_computed_numbers_and_rules():
    s = seed_state()
    today = s.records[-1]
    s.records = s.records[:-1]
    brief = analyst_brief(s, today, compute_signals(s, today))
    assert "2026-09-17" in brief and "\"far_otm_leads\"" in brief and "8.4" in brief
    assert "재계산" in brief and "URL" in brief and "AnalystView" in brief

    view = parse_view(VIEW)
    rb = rebuttal_brief(brief, "hermes", view, "dsh", view)
    assert "반박" in rb and "rebuttal_ko" in rb

    gb = gap_fill_brief(today.date, "NVDA", s.expiry, [GapRequest(field="puts.210.iv", description="d")])
    assert "puts.210.iv" in gb and "추정" in gb and "\"fills\"" in gb
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agents_base.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bubble_watch.agents'`

- [ ] **Step 3: Write the implementation**

**`src/bubble_watch/agents/__init__.py`**
```python
"""Analyst sub-agents (Hermes over HTTP, DeepSeek Harness over its Python SDK)."""
```

**`src/bubble_watch/agents/base.py`**
```python
"""The analyst contract: ask → extract JSON → validate AnalystView, with one corrective retry."""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..models import AnalystView
from .prompts import retry_prompt

_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


class AnalystError(RuntimeError):
    """An analyst could not produce a valid view (transport failure or invalid output after retry)."""


class AskResult(BaseModel):
    text: str
    session_id: str | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)


class AnalystResult(BaseModel):
    agent: str
    view: AnalystView
    session_id: str | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    retried: bool = False


class Asker(Protocol):
    def __call__(self, prompt: str, session_id: str | None = None) -> AskResult: ...


class Analyst(Protocol):
    name: str

    def analyze(self, brief: str) -> AnalystResult: ...

    def rebut(self, prompt: str) -> AnalystResult: ...


def extract_json(text: str) -> dict[str, Any]:
    """First JSON object in ``text``: a fenced block wins, else the first decodable ``{...}``."""
    candidates = [m.group(1) for m in _FENCED.finditer(text)]
    decoder = json.JSONDecoder()
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in response")


def parse_view(text: str) -> AnalystView:
    data = extract_json(text)
    cats = data.get("catalysts") or []
    data["catalysts"] = [c for c in cats if isinstance(c, dict)
                         and str(c.get("url", "")).startswith(("http://", "https://"))]
    return AnalystView.model_validate(data)


def ask_for_view(agent: str, ask: Asker, prompt: str) -> AnalystResult:
    first = ask(prompt)
    try:
        return AnalystResult(agent=agent, view=parse_view(first.text), session_id=first.session_id,
                             events=first.events)
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        retry = ask(retry_prompt(str(exc)), session_id=first.session_id)
        try:
            view = parse_view(retry.text)
        except ValueError as exc2:
            raise AnalystError(f"{agent}: invalid AnalystView after retry: {str(exc2)[:300]}") from exc2
        return AnalystResult(agent=agent, view=view, session_id=retry.session_id or first.session_id,
                             events=[*first.events, *retry.events], retried=True)
```

**`src/bubble_watch/agents/prompts.py`**
```python
"""Prompt builders. Numbers arrive pre-computed; agents interpret and research, never recompute."""
from __future__ import annotations

import datetime as dt
import json

from ..models import AnalystView, DailyRecord, Signals, WatchState

SYSTEM_ANALYST = (
    "너는 'NVDA Bubble Signal Watch'의 독립 시니어 애널리스트다. 다른 애널리스트와 독립적으로 판단한다. "
    "제공된 수치는 코드가 계산한 확정값이므로 재계산하거나 새로운 가격·수익률·IV를 만들어내지 마라. "
    "너의 역할은 (1) 마지막 기록 이후 발표된 뉴스·촉매를 웹에서 조사하고, (2) 계산된 tape 신호를 해석하고, "
    "(3) Bubble Signal Score와 판정을 제시하는 것이다. 모든 촉매에는 실제로 열어 확인한 URL을 붙여라. "
    "최종 응답은 지정된 JSON 스키마를 따르는 JSON 객체 하나만 ```json 코드 블록으로 출력한다."
)

SYSTEM_GAP_FILL = (
    "너는 시장 데이터 리서처다. 요청된 값만 웹에서 찾아 출처 URL과 함께 보고한다. "
    "추정·보간·계산으로 값을 만들지 않는다. 찾지 못하면 null로 보고한다."
)


def _fmt(v: object, suffix: str = "") -> str:
    return "N/A" if v is None else f"{v}{suffix}"


def analyst_brief(state: WatchState, today: DailyRecord, signals: Signals) -> str:
    prior = state.prior_records(today.date)
    last = next((r for r in reversed(prior) if r.score is not None), None)
    closes = "\n".join(f"| {s} | {_fmt(c.price)} | {c.freshness} | {c.source} |" for s, c in today.closes.items())
    puts = "\n".join(
        f"| ${k}P | {_fmt(q.bid)} | {_fmt(q.ask)} | {_fmt(q.last)} | {_fmt(q.iv, '%')} | {_fmt(q.volume)} "
        f"| {_fmt(q.oi)} | {q.freshness} |" for k, q in sorted(today.puts.items())) or "| (없음) | | | | | | | |"
    history = "\n".join(
        f"- {r.date}: score={_fmt(r.score)} verdict={r.verdict.value if r.verdict else 'N/A'} — {r.note}"
        for r in prior[-5:] if r.score is not None or r.note)
    since = prior[-1].date if prior else "최근"
    strikes = "/".join(f"${k}" for k in state.strikes)
    schema = json.dumps(AnalystView.model_json_schema(), ensure_ascii=False)
    return f"""# {state.ticker} Bubble Signal Watch — {today.date} 미국장 마감 분석 요청

## 감시 대상
- {state.ticker} vs peer {state.peer}, 레버리지 ETF {state.leveraged}
- {state.expiry} 만기 {strikes} puts
- Trigger zone: score ≥ {state.trigger_threshold}

## 오늘 종가 (코드 확정값)
| 종목 | 종가 | freshness | source |
|---|---:|---|---|
{closes}

## 오늘 put 데이터 (코드 확정값)
| Put | Bid | Ask | Last | IV | Volume | OI | freshness |
|---|---:|---:|---:|---:|---:|---:|---|
{puts}

## 계산된 신호 (코드 확정값)
```json
{signals.model_dump_json(indent=2)}
```
returns / put_changes는 % 변화, rel_spread는 {state.ticker}−{state.peer} %p, base_dates는 각 구간의 비교 기준일,
iv는 strike별 "최근 검증된 IV"와의 비교, conditions는 재확인 조건 3개
(nvda_underperforms: {state.ticker} < {state.peer} 지속, iv_surface_up: IV surface 전체 상승,
far_otm_leads: 가장 OTM put이 다른 put보다 빠르게 상승).

## 이전 기록
- 직전 점수: {_fmt(last.score if last else None)} ({last.date if last else '-'}), 판정 {last.verdict.value if last and last.verdict else 'N/A'}
{history}

## 작업
1. {since} 이후 발표된 {state.ticker}·AI 반도체 관련 뉴스와 촉매(경쟁, 수요, 규제, 거시)를 웹에서 조사하라.
   각 촉매를 bubble thesis 기준 bearish / bullish / neutral로 분류하고 URL을 붙여라.
2. 계산된 신호로 오늘 tape가 thesis를 confirm하는지 de-confirm하는지 해석하라. 수치는 재계산하지 말고 인용만 하라.
3. 직전 점수 대비 score(0–10, 소수 첫째 자리)를 제시하고 변화 이유를 설명하라. ±1.0 초과 변화는 강한 근거가 있을 때만.
4. verdict: NOT_TRIGGERED, TRIGGERED, TRIGGERED_DE_CONFIRMING, TRIGGERED_FURTHER_DE_CONFIRMING, CONFIRMED 중 하나.
5. watch_conditions_ko에는 무엇이 thesis를 재확인하거나 무효화하는지 적어라.

## 응답 형식
모든 *_ko 필드는 한국어. 아래 AnalystView JSON Schema를 따르는 JSON 객체 하나만 ```json 블록으로 출력하라.
{schema}
"""


def rebuttal_brief(brief: str, own_name: str, own: AnalystView, other_name: str, other: AnalystView) -> str:
    return f"""{brief}

## 반박 라운드
다른 애널리스트({other_name})가 독립적으로 아래 결론을 냈고, 너({own_name})의 이전 결론과 차이가 크다.

### 너의 이전 결론
```json
{own.model_dump_json(indent=2)}
```

### {other_name}의 결론
```json
{other.model_dump_json(indent=2)}
```

상대의 근거와 촉매를 검토하라. 설득되면 수정하고, 아니면 유지하되 이유를 밝혀라.
rebuttal_ko 필드에 반박 또는 수용의 요지를 반드시 적고, 수정된 전체 AnalystView JSON 하나만 ```json 블록으로 출력하라.
"""


def gap_fill_brief(day: dt.date, ticker: str, expiry: dt.date, gaps: list) -> str:
    wanted = "\n".join(f"- {g.field}: {g.description}" for g in gaps)
    return f"""{ticker} {day} 미국장 기준으로 아래 데이터가 비어 있다 (put은 {expiry} 만기). 웹에서 찾아 채워라.
{wanted}

규칙:
- 추정·보간·계산 금지. 실제 페이지에서 확인한 값만.
- 값마다 확인한 페이지 URL(source_url)과 freshness(EOD | latest_snapshot | late_session_last)를 붙여라.
- 찾지 못하면 value는 null, freshness는 "N/A".
- IV는 퍼센트 숫자(예: 34.9), 가격은 달러 숫자, volume/oi는 정수.

응답: 다음 형태의 JSON 객체 하나만 ```json 블록으로 출력하라.
{{"fills": [{{"field": "puts.210.iv", "value": 31.7, "source_url": "https://...", "freshness": "EOD", "note": ""}}]}}
"""


def retry_prompt(error: str) -> str:
    return (f"직전 응답을 파싱/검증하지 못했다: {error[:800]}\n"
            "스키마를 정확히 지키는 JSON 객체 하나만 ```json 블록으로 다시 출력하라. 다른 텍스트는 쓰지 마라.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agents_base.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/bubble_watch/agents tests/test_agents_base.py
git commit -m "feat: analyst contract, prompts and validated JSON views with one retry"
```

---

### Task 6: Hermes analyst client (OpenAI-compatible gateway API)

**Files:**
- Create: `src/bubble_watch/agents/hermes_client.py`
- Test: `tests/test_hermes_client.py`

**Interfaces:**
- Consumes: `AskResult`, `AnalystResult`, `AnalystError`, `ask_for_view`, `extract_json` (Task 5); `SYSTEM_ANALYST`, `SYSTEM_GAP_FILL` (Task 5); `GapFill` (Task 4)
- Produces: `HermesAnalyst(base_url, api_key, *, model=None, timeout=600.0, client=None)` with `name = "hermes"`, `ask(prompt, session_id=None, *, system=SYSTEM_ANALYST) -> AskResult`, `analyze(brief)`, `rebut(prompt)`, `fill_gaps(prompt) -> list[GapFill]`.

Hermes derives a session id from the system prompt + first user message, so each client instance adds a unique run id to its system prompt; otherwise re-running a day would continue the previous run's session. `X-Hermes-Session-Id` continues a session (used for the retry).

- [ ] **Step 1: Write the failing test**

**`tests/test_hermes_client.py`**
```python
import json

import httpx
import pytest

from bubble_watch.agents.base import AnalystError
from bubble_watch.agents.hermes_client import HermesAnalyst

VIEW = {"score": 8.1, "score_delta_reasoning_ko": "r", "verdict": "TRIGGERED_DE_CONFIRMING", "catalysts": [],
        "tape_read_ko": "t", "watch_conditions_ko": "w"}


def _completion(content, session="h-1", status=200, extra_headers=None):
    return httpx.Response(status, headers={"X-Hermes-Session-Id": session, **(extra_headers or {})},
                          json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def _client(handler, **kw):
    return HermesAnalyst("http://hermes.test/v1", "k-1", client=httpx.Client(transport=httpx.MockTransport(handler)),
                         **kw)


def test_analyze_sends_auth_unique_run_and_no_model_by_default():
    seen = []

    def handler(req):
        seen.append(req)
        return _completion(f"```json\n{json.dumps(VIEW)}\n```")

    result = _client(handler).analyze("brief")
    req = seen[0]
    body = json.loads(req.content)
    assert req.url.path == "/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer k-1"
    assert "X-Hermes-Session-Id" not in req.headers
    assert "model" not in body and body["stream"] is False
    assert body["messages"][0]["role"] == "system" and "run:" in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": "brief"}
    assert result.view.score == 8.1 and result.session_id == "h-1" and result.agent == "hermes"


def test_retry_continues_session_and_model_override():
    seen = []

    def handler(req):
        seen.append(req)
        return _completion("oops" if len(seen) == 1 else json.dumps(VIEW))

    result = _client(handler, model="grok-4.6").analyze("brief")
    assert result.retried
    assert json.loads(seen[0].content)["model"] == "grok-4.6"
    assert seen[1].headers["X-Hermes-Session-Id"] == "h-1"


def test_http_error_and_incomplete_run_raise_analyst_error():
    with pytest.raises(AnalystError, match="HTTP 500"):
        _client(lambda r: httpx.Response(500, text="boom")).analyze("b")
    with pytest.raises(AnalystError, match="incomplete"):
        _client(lambda r: _completion("partial", extra_headers={"X-Hermes-Completed": "false"})).analyze("b")


def test_fill_gaps_uses_gap_system_prompt_and_parses_fills():
    seen = []
    fills = {"fills": [{"field": "puts.210.iv", "value": 31.7, "source_url": "https://x.test", "freshness": "EOD"}]}

    def handler(req):
        seen.append(json.loads(req.content))
        return _completion(json.dumps(fills))

    out = _client(handler).fill_gaps("find it")
    assert out[0].field == "puts.210.iv" and out[0].value == 31.7
    assert "리서처" in seen[0]["messages"][0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hermes_client.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bubble_watch.agents.hermes_client'`

- [ ] **Step 3: Write the implementation**

**`src/bubble_watch/agents/hermes_client.py`**
```python
"""Hermes Agent as an analyst, via its OpenAI-compatible gateway API (/v1/chat/completions)."""
from __future__ import annotations

import uuid

import httpx

from ..market_data.base import GapFill
from .base import AnalystError, AnalystResult, AskResult, ask_for_view, extract_json
from .prompts import SYSTEM_ANALYST, SYSTEM_GAP_FILL


class HermesAnalyst:
    name = "hermes"

    def __init__(self, base_url: str, api_key: str, *, model: str | None = None, timeout: float = 600.0,
                 client: httpx.Client | None = None) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = api_key
        self._model = model
        self._client = client or httpx.Client(timeout=timeout)
        # Hermes fingerprints system prompt + first user message into a session id; a per-run id keeps
        # a re-run of the same day from continuing the previous run's session.
        self._run_id = uuid.uuid4().hex[:12]

    def ask(self, prompt: str, session_id: str | None = None, *, system: str = SYSTEM_ANALYST) -> AskResult:
        headers = {"Authorization": f"Bearer {self._key}"}
        if session_id:
            headers["X-Hermes-Session-Id"] = session_id
        body: dict = {"messages": [{"role": "system", "content": f"{system}\n\nrun: {self._run_id}"},
                                   {"role": "user", "content": prompt}], "stream": False}
        if self._model:
            body["model"] = self._model
        try:
            r = self._client.post(self._url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise AnalystError(f"hermes request failed: {exc}") from exc
        if r.status_code >= 400:
            raise AnalystError(f"hermes HTTP {r.status_code}: {r.text[:300]}")
        if r.headers.get("X-Hermes-Completed") == "false":
            raise AnalystError(f"hermes run incomplete: {r.headers.get('X-Hermes-Error', '')}")
        try:
            text = r.json()["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError) as exc:
            raise AnalystError(f"hermes returned an unexpected body: {r.text[:300]}") from exc
        return AskResult(text=text, session_id=r.headers.get("X-Hermes-Session-Id") or session_id)

    def analyze(self, brief: str) -> AnalystResult:
        return ask_for_view(self.name, self.ask, brief)

    def rebut(self, prompt: str) -> AnalystResult:
        return ask_for_view(self.name, self.ask, prompt)

    def fill_gaps(self, prompt: str) -> list[GapFill]:
        result = self.ask(prompt, system=SYSTEM_GAP_FILL)
        try:
            data = extract_json(result.text)
            return [GapFill.model_validate(f) for f in data.get("fills") or []]
        except ValueError as exc:
            raise AnalystError(f"hermes gap fill returned invalid JSON: {str(exc)[:300]}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hermes_client.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/bubble_watch/agents/hermes_client.py tests/test_hermes_client.py
git commit -m "feat: Hermes analyst client over the gateway chat completions API"
```

---

### Task 7: DeepSeek Harness analyst client and dsh home setup

**Files:**
- Create: `src/bubble_watch/agents/dsh_client.py`, `dsh/exa-search.patch.yml`
- Test: `tests/test_dsh_client.py`

**Interfaces:**
- Consumes: `AskResult`, `AnalystResult`, `AnalystError`, `ask_for_view` (Task 5); `SYSTEM_ANALYST` (Task 5); `Settings`, `PROJECT_ROOT` (Task 1)
- Produces: `DshAnalyst(harness_factory)` with `name = "dsh"`, `ask(prompt, session_id=None) -> AskResult`, `analyze`, `rebut`, `close()`; `ensure_dsh_home(dsh_home, model, provider="xai", base_url="https://api.x.ai/v1") -> Path`; `make_harness_factory(settings) -> Callable[[], DeepSeekHarness]`; `EXA_PATCH: Path`.

The dsh system prompt is its own (a coding-agent persona), so the analyst role is prepended to the first prompt of a session. `xai` is registered in `$DSH_HOME/settings.yaml` through the `llm-pi-ai` adapter; with `EXA_API_KEY` set, the Exa overlay makes `web_search` use Exa instead of DeepSeek's search API.

- [ ] **Step 1: Write the failing test**

**`tests/test_dsh_client.py`**
```python
import json
from types import SimpleNamespace

import pytest

from bubble_watch.agents.base import AnalystError
from bubble_watch.agents.dsh_client import DshAnalyst, ensure_dsh_home

VIEW = {"score": 8.6, "score_delta_reasoning_ko": "r", "verdict": "TRIGGERED", "catalysts": [],
        "tape_read_ko": "t", "watch_conditions_ko": "w"}


class FakeHarness:
    def __init__(self, answers):
        self.answers, self.calls, self.closed = list(answers), [], False

    def run(self, prompt, *, session_id=None):
        self.calls.append((prompt, session_id))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(final_response=answer, session_id=session_id or "session-1", finish_reason="stop",
                               events=[{"type": "assistant/message"}])

    def close(self):
        self.closed = True


def test_analyze_prepends_role_once_and_retries_in_session():
    h = FakeHarness(["no json", json.dumps(VIEW), json.dumps(VIEW)])
    built = []
    analyst = DshAnalyst(lambda: built.append(1) or h)
    result = analyst.analyze("brief")
    assert result.view.score == 8.6 and result.retried and result.agent == "dsh"
    assert h.calls[0][0].startswith("너는") and h.calls[0][0].endswith("brief")
    assert h.calls[1][1] == "session-1" and not h.calls[1][0].startswith("너는")
    assert len(result.events) == 2
    analyst.analyze("again")
    assert built == [1]  # harness reused
    analyst.close()
    assert h.closed


def test_empty_answer_and_runtime_errors_become_analyst_errors():
    with pytest.raises(AnalystError, match="no answer"):
        DshAnalyst(lambda: FakeHarness(["  "])).ask("p")
    with pytest.raises(AnalystError, match="dsh run failed"):
        DshAnalyst(lambda: FakeHarness([RuntimeError("stdout closed")])).ask("p")


def test_ensure_dsh_home_writes_settings_once(tmp_path):
    path = ensure_dsh_home(str(tmp_path / "home"), "grok-4.6")
    text = path.read_text()
    assert "llm-pi-ai:" in text and "https://api.x.ai/v1" in text and "id: grok-4.6" in text
    assert "apiKeyEnv: XAI_API_KEY" in text
    path.write_text("custom: true\n")
    ensure_dsh_home(str(tmp_path / "home"), "grok-4.6")
    assert path.read_text() == "custom: true\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dsh_client.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bubble_watch.agents.dsh_client'`

- [ ] **Step 3: Write the implementation**

**`dsh/exa-search.patch.yml`**
```yaml
# bubble-watch overlay: route dsh web_search through Exa (EXA_API_KEY) instead of DeepSeek search.
- id: web
  config:
    searchProvider: exa
    fetchProvider: http

- insert:
    - id: web-search-exa
      name: '@deepseek-ai/dsh-web-search-exa'
      config:
        apiKey: !!js process.env.EXA_API_KEY
```

**`src/bubble_watch/agents/dsh_client.py`**
```python
"""DeepSeek Harness as an analyst, driven in-process through its Python SDK (JSON-RPC over stdio)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..config import PROJECT_ROOT, Settings
from .base import AnalystError, AnalystResult, AskResult, ask_for_view
from .prompts import SYSTEM_ANALYST

EXA_PATCH = PROJECT_ROOT / "dsh" / "exa-search.patch.yml"

_SETTINGS_TEMPLATE = """\
# Written by bubble-watch: registers xAI (Grok) for the dsh analyst through the llm-pi-ai adapter.
llm-pi-ai:
  providers:
    {provider}:
      apiKeyEnv: XAI_API_KEY
      api: openai-completions
      baseURL: {base_url}
      models:
        - id: {model}
"""


def ensure_dsh_home(dsh_home: str, model: str, provider: str = "xai",
                    base_url: str = "https://api.x.ai/v1") -> Path:
    """Create DSH_HOME with an xAI provider settings.yaml; never overwrite an existing one."""
    home = Path(dsh_home)
    home.mkdir(parents=True, exist_ok=True)
    path = home / "settings.yaml"
    if not path.exists():
        path.write_text(_SETTINGS_TEMPLATE.format(provider=provider, base_url=base_url, model=model))
    return path


def make_harness_factory(settings: Settings) -> Callable[[], Any]:
    def factory() -> Any:
        from deepseek_harness import DeepSeekHarness

        ensure_dsh_home(settings.dsh_home, settings.dsh_model, settings.dsh_provider, settings.xai_base_url)
        env = {"XAI_API_KEY": settings.xai_api_key}
        patches: tuple[str, ...] = ()
        if settings.exa_api_key:
            env["EXA_API_KEY"] = settings.exa_api_key
            patches = (str(EXA_PATCH),)
        return DeepSeekHarness(
            dsh_bin=settings.dsh_bin, dsh_home=settings.dsh_home, cwd=str(PROJECT_ROOT),
            provider=settings.dsh_provider, model=settings.dsh_model, patches=patches, env=env,
            initialize_timeout_seconds=120, request_timeout_seconds=settings.analyst_timeout_s)
    return factory


class DshAnalyst:
    name = "dsh"

    def __init__(self, harness_factory: Callable[[], Any]) -> None:
        self._factory = harness_factory
        self._harness: Any = None

    def _h(self) -> Any:
        if self._harness is None:
            self._harness = self._factory()
        return self._harness

    def ask(self, prompt: str, session_id: str | None = None) -> AskResult:
        full = prompt if session_id else f"{SYSTEM_ANALYST}\n\n{prompt}"
        try:
            result = self._h().run(full, session_id=session_id)
        except Exception as exc:  # SDK transport/protocol/timeout errors
            raise AnalystError(f"dsh run failed: {type(exc).__name__}: {str(exc)[:300]}") from exc
        if not (result.final_response or "").strip():
            raise AnalystError(f"dsh returned no answer (finish_reason={result.finish_reason})")
        return AskResult(text=result.final_response, session_id=result.session_id, events=list(result.events))

    def analyze(self, brief: str) -> AnalystResult:
        return ask_for_view(self.name, self.ask, brief)

    def rebut(self, prompt: str) -> AnalystResult:
        return ask_for_view(self.name, self.ask, prompt)

    def close(self) -> None:
        if self._harness is not None:
            self._harness.close()
            self._harness = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dsh_client.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/bubble_watch/agents/dsh_client.py dsh/exa-search.patch.yml tests/test_dsh_client.py
git commit -m "feat: DeepSeek Harness analyst client with xAI provider home and Exa search overlay"
```

---

### Task 8: Reconciliation

**Files:**
- Create: `src/bubble_watch/reconcile.py`
- Test: `tests/test_reconcile.py`

**Interfaces:**
- Consumes: `AnalystView`, `Verdict`, `Reconciliation`, `MergedCatalyst`, `most_cautious`, `round1` (Task 1)
- Produces: `AGREE_SCORE_GAP = 0.5`, `views_agree(a, b) -> bool`, `merge_catalysts(views) -> list[MergedCatalyst]`, `reconcile(views, *, prior_score, rebuttal_done) -> Reconciliation | None` (None = a rebuttal round is needed).

- [ ] **Step 1: Write the failing test**

**`tests/test_reconcile.py`**
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reconcile.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bubble_watch.reconcile'`

- [ ] **Step 3: Write the implementation**

**`src/bubble_watch/reconcile.py`**
```python
"""Pure reconciliation of analyst views; the only LLM involvement is the rebuttal round the graph runs."""
from __future__ import annotations

from .models import AnalystView, MergedCatalyst, Reconciliation, most_cautious, round1

AGREE_SCORE_GAP = 0.5
BIG_MOVE = 1.0


def views_agree(a: AnalystView, b: AnalystView) -> bool:
    return abs(a.score - b.score) <= AGREE_SCORE_GAP + 1e-9 and a.verdict == b.verdict


def merge_catalysts(views: dict[str, AnalystView]) -> list[MergedCatalyst]:
    merged: dict[str, MergedCatalyst] = {}
    for agent, view in views.items():
        for c in view.catalysts:
            if c.url not in merged:
                merged[c.url] = MergedCatalyst(**c.model_dump())
            if agent not in merged[c.url].cited_by:
                merged[c.url].cited_by.append(agent)
    return list(merged.values())


def reconcile(views: dict[str, AnalystView], *, prior_score: float | None,
              rebuttal_done: bool) -> Reconciliation | None:
    """Return the reconciliation, or None when the two views disagree and no rebuttal has run yet."""
    if not views:
        raise ValueError("no analyst views to reconcile")
    flags = [f"{a}: score moved {v.score - prior_score:+.1f} from prior {prior_score}"
             for a, v in views.items() if prior_score is not None and abs(v.score - prior_score) > BIG_MOVE]
    common = dict(scores={a: v.score for a, v in views.items()}, verdicts={a: v.verdict for a, v in views.items()},
                  rebuttal_round=rebuttal_done, catalysts=merge_catalysts(views))
    if len(views) == 1:
        (agent, view), = views.items()
        return Reconciliation(mode="single", score=view.score, verdict=view.verdict,
                              flags=[*flags, f"single analyst: {agent}"], **common)
    a, b = list(views.values())[:2]
    mean = round1((a.score + b.score) / 2)
    if views_agree(a, b):
        return Reconciliation(mode="agree_after_rebuttal" if rebuttal_done else "agree", score=mean,
                              verdict=a.verdict, flags=flags, **common)
    if not rebuttal_done:
        return None
    return Reconciliation(mode="disagree", score=mean, verdict=most_cautious(a.verdict, b.verdict),
                          flags=flags, **common)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reconcile.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/bubble_watch/reconcile.py tests/test_reconcile.py
git commit -m "feat: pure reconciliation with rebuttal gate and cautious tie-break"
```

---
### Task 9: Writer and report rendering

**Files:**
- Create: `src/bubble_watch/writer.py`, `src/bubble_watch/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: models (Task 1), `extract_json` (Task 5), `reconcile` (Task 8), `seed_state` (Task 2), `compute_signals` (Task 3)
- Produces: `Narrative(headline_ko, tape_ko, options_ko, verdict_ko, watch_ko)`, `WriterError`, `XaiWriter(api_key, model, base_url, client=None, timeout=180.0).write(prompt) -> Narrative`, `writer_brief(state, record, recon, views) -> str`, `fallback_narrative(recon, views) -> Narrative`; `render_report(state, record, recon, narrative, *, views, notes) -> str`, `VERDICT_LABEL`, `money(v)`.

Tables and headline numbers come only from `record`/`recon`; the writer supplies prose and cites catalysts as `[n]` using the numbering `writer_brief` assigns (same order as `recon.catalysts`, which is also the footer order).

- [ ] **Step 1: Write the failing test**

**`tests/test_report.py`**
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bubble_watch.report'`

- [ ] **Step 3: Write the implementation**

**`src/bubble_watch/writer.py`**
```python
"""Narrative writer: one direct xAI chat call that turns reconciled views into Korean prose (no numbers of its own)."""
from __future__ import annotations

import httpx
from pydantic import BaseModel

from .agents.base import extract_json
from .models import AnalystView, DailyRecord, Reconciliation, WatchState

SYSTEM_WRITER = (
    "너는 한국어 금융 리서치 노트 작가다. 'NVDA Bubble Signal Watch' 일일 노트의 서술 부분만 쓴다. "
    "표와 수치는 코드가 따로 렌더링하므로, 서술에서 인용하는 숫자는 반드시 제공된 값 그대로 써라(새 숫자 금지). "
    "문체: 간결한 한국어 리서치 톤, 핵심 수치와 결론은 **굵게**, de-confirmation·convexity·IV surface 같은 "
    "영어 용어는 그대로 섞어 쓴다. 촉매를 언급할 때는 제공된 번호로 [n] 형태로 인용한다. "
    "JSON 객체 하나만 출력한다."
)


class Narrative(BaseModel):
    headline_ko: str
    tape_ko: str
    options_ko: str
    verdict_ko: str
    watch_ko: str


class WriterError(RuntimeError):
    pass


def writer_brief(state: WatchState, record: DailyRecord, recon: Reconciliation,
                 views: dict[str, AnalystView]) -> str:
    catalysts = "\n".join(
        f"[{i}] {c.headline} ({c.direction}, {c.weight}, cited by {', '.join(c.cited_by)}) — {c.rationale_ko}"
        for i, c in enumerate(recon.catalysts, 1)) or "(없음)"
    opinions = "\n\n".join(
        f"### {name}: score {v.score}, {v.verdict.value}, confidence {v.confidence}\n"
        f"- tape: {v.tape_read_ko}\n- 점수 변화 이유: {v.score_delta_reasoning_ko}\n- watch: {v.watch_conditions_ko}"
        + (f"\n- rebuttal: {v.rebuttal_ko}" if v.rebuttal_ko else "")
        for name, v in views.items())
    return f"""# {state.ticker} {record.date} 노트 서술 작성

## 최종 결론 (확정)
- score {recon.score}, verdict {recon.verdict.value}, reconciliation mode {recon.mode}
- flags: {'; '.join(recon.flags) or '없음'}

## 계산된 신호 (확정값, 숫자는 여기서만 인용)
```json
{record.signals.model_dump_json(indent=2) if record.signals else '{}'}
```

## 촉매 (번호로 인용)
{catalysts}

## 애널리스트 의견
{opinions}

## 출력
다음 키를 가진 JSON 객체 하나:
- headline_ko: 점수 줄 바로 뒤에 붙는 1–2문장 (오늘 tape의 핵심)
- tape_ko: 점수 판단 근거 문단 (촉매 [n] 인용)
- options_ko: put 가격·IV·convexity 해석 문단
- verdict_ko: 현재 판정과 이유 문단
- watch_ko: 앞으로 볼 재확인/무효화 조건 문단
"""


def fallback_narrative(recon: Reconciliation, views: dict[str, AnalystView]) -> Narrative:
    """Used when the writer call fails: stitch the first analyst's own text so the report still ships."""
    v = next(iter(views.values()))
    return Narrative(headline_ko=f"(writer 실패로 {next(iter(views))} 애널리스트 원문 사용)", tape_ko=v.tape_read_ko,
                     options_ko=v.score_delta_reasoning_ko, verdict_ko=f"판정: {recon.verdict.value}",
                     watch_ko=v.watch_conditions_ko)


class XaiWriter:
    def __init__(self, api_key: str, model: str, base_url: str, client: httpx.Client | None = None,
                 timeout: float = 180.0) -> None:
        self._key, self._model = api_key, model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._client = client or httpx.Client(timeout=timeout)

    def write(self, prompt: str) -> Narrative:
        error = ""
        for attempt in range(2):
            content = prompt if attempt == 0 else f"{prompt}\n\n직전 출력 오류: {error[:500]}. JSON 객체 하나만 다시 출력하라."
            body = {"model": self._model, "messages": [{"role": "system", "content": SYSTEM_WRITER},
                                                       {"role": "user", "content": content}]}
            try:
                r = self._client.post(self._url, json=body, headers={"Authorization": f"Bearer {self._key}"})
            except httpx.HTTPError as exc:
                raise WriterError(f"writer request failed: {exc}") from exc
            if r.status_code >= 400:
                raise WriterError(f"writer HTTP {r.status_code}: {r.text[:300]}")
            try:
                return Narrative.model_validate(extract_json(r.json()["choices"][0]["message"]["content"] or ""))
            except (ValueError, KeyError, IndexError) as exc:
                error = str(exc)
        raise WriterError(f"writer returned invalid narrative JSON twice: {error[:300]}")
```

**`src/bubble_watch/report.py`**
```python
"""Markdown report: every table cell comes from code; the writer's Narrative supplies the prose."""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from .models import AnalystView, DailyRecord, PutQuote, Reconciliation, Verdict, WatchState, round1
from .writer import Narrative

VERDICT_LABEL = {
    Verdict.NOT_TRIGGERED: "NOT TRIGGERED",
    Verdict.TRIGGERED: "TRIGGERED",
    Verdict.TRIGGERED_DE_CONFIRMING: "TRIGGERED, BUT DE-CONFIRMING",
    Verdict.TRIGGERED_FURTHER_DE_CONFIRMING: "TRIGGERED, BUT FURTHER DE-CONFIRMING",
    Verdict.CONFIRMED: "CONFIRMED",
}
_CONDITION_LABEL = {
    "nvda_underperforms": "{t} < {p} (여러 날 지속)",
    "iv_surface_up": "IV surface 전체 상승",
    "far_otm_leads": "${lo}P가 다른 put보다 빠르게 상승",
}
_CONDITION_VALUE = {"true": "성립", "partial": "부분 성립 (1일만)", "false": "불성립", "unknown": "데이터 없음"}
_MODE_LABEL = {"agree": "두 analyst 합의", "agree_after_rebuttal": "반박 라운드 후 합의",
               "disagree": "반박 라운드 후에도 불일치 → 평균 점수, 더 보수적인 판정", "single": "단일 analyst 기준"}


def money(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"${Decimal(str(v)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"


def spct(v: float | None) -> str:
    return "N/A" if v is None else f"{v:+.2f}%"


def _pp(v: float | None) -> str:
    return "N/A" if v is None else f"{v:+.2f}%p"


def _iv(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.2f}%"


def _num(v: int | None) -> str:
    return "N/A" if v is None else f"{v:,}"


def _md(d) -> str:
    return f"{d.month}/{d.day}"


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
    return lead + ", ".join(parts)


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
           f"| 지표 | {_md(record.date)} | 판정 |", "|---|---:|---|",
           f"| **Bubble Signal Score** | **{recon.score:.1f}** | **{'N/A' if delta is None else f'{delta:+.1f}'}** |",
           f"| **{t}** | **{money(close[t])}** | **{spct(r1[t])}** |",
           f"| **{p}** | **{money(close[p])}** | **{spct(r1[p])}** |",
           f"| **{t} − {p}** | — | **{_pp(sig.rel_spread['1d'])}** |",
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

    anchor = state.anchors[0] if state.anchors else None
    windows = [("1d", "1일"), ("3d", "최근 3거래일"), ("anchor", f"{_md(anchor.date)} 이후" if anchor else "anchor 이후")]
    out += [f"### {' / '.join(label for _, label in windows)}", "",
            "| 구간 | " + " | ".join([*state.symbols, *(f"${k}P" for k in state.strikes)]) + " |",
            "|---|" + "---:|" * (len(state.symbols) + len(state.strikes))]
    for w, label in windows:
        cells = [spct(sig.returns[w][s]) for s in state.symbols] + [spct(sig.put_changes[w][k]) for k in state.strikes]
        out.append(f"| **{label}** | " + " | ".join(cells) + " |")
    bases = ", ".join(f"{label}: {sig.base_dates.get(w) or 'N/A'} 대비" for w, label in windows)
    out += ["", f"기준일 — {bases}. {lev}는 daily 3× 레버리지 ETF라 multi-day 수익률의 절대 크기를 "
                f"{t}/{p}와 직접 비교하면 안 된다.", ""]

    lo = min(state.strikes)
    out += [f"### 판정: `{VERDICT_LABEL[recon.verdict]}`", "", narrative.verdict_ko, "", narrative.watch_ko, "",
            "| 재확인 조건 | 오늘 |", "|---|---|"]
    out += [f"| {_CONDITION_LABEL[k].format(t=t, p=p, lo=lo)} | {_CONDITION_VALUE[v]} |" for k, v in sig.conditions.items()]

    out += ["", "### Analyst views", "", f"{_MODE_LABEL[recon.mode]}.", "",
            "| Analyst | Score | Verdict | Confidence |", "|---|---:|---|---|"]
    out += [f"| {name} | {v.score:.1f} | {v.verdict.value} | {v.confidence} |" for name, v in views.items()]
    if recon.mode == "disagree":
        out += ["", "#### Analyst disagreement", ""]
        for name, v in views.items():
            out += [f"**{name}** ({v.score:.1f}, {v.verdict.value}): {v.tape_read_ko}"
                    + (f" — 반박: {v.rebuttal_ko}" if v.rebuttal_ko else ""), ""]
    if recon.flags or notes:
        out += ["", "#### 데이터·실행 노트", ""] + [f"- {n}" for n in [*recon.flags, *notes]]
    if recon.catalysts:
        out += [""] + [f'[{i}]: {c.url} "{c.headline}"' for i, c in enumerate(recon.catalysts, 1)]
    return "\n".join(out) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_report.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/bubble_watch/writer.py src/bubble_watch/report.py tests/test_report.py
git commit -m "feat: code-rendered report tables with xAI narrative writer and fallback"
```

---

### Task 10: Tracing helpers

**Files:**
- Create: `src/bubble_watch/tracing.py`
- Test: `tests/test_tracing.py`

**Interfaces:**
- Consumes: `Settings` (Task 1), `AnalystResult` (Task 5)
- Produces: `setup_tracing(settings) -> TracerProvider | None`, `get_tracer(provider) -> Tracer`, `agent_span(tracer, name, *, input_value, kind="AGENT", attributes=None)` context manager (terminal status always set), `annotate_result(tracer, span, result)`, `dsh_tool_calls(events) -> list[dict]`, `record_dsh_tool_spans(tracer, events) -> int`, `shutdown_tracing(provider)`.

dsh has no span export; its tool calls are rebuilt from `RunResult.events` content blocks (`tool-call` / `tool-result`, matched by `toolCallId`). Rebuilt spans have no real timing (the SDK events carry none) — they record what ran and what came back.

- [ ] **Step 1: Write the failing test**

**`tests/test_tracing.py`**
```python
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from bubble_watch.agents.base import AnalystResult
from bubble_watch.models import AnalystView
from bubble_watch.tracing import agent_span, annotate_result, dsh_tool_calls, get_tracer, setup_tracing

EVENTS = [
    {"type": "assistant/message", "data": {"message": {"role": "assistant", "content": [
        {"type": "text", "text": "searching"},
        {"type": "tool-call", "toolCallId": "c1", "toolName": "web_search", "input": {"query": "nvda huawei"}}]}}},
    {"type": "tool/result", "data": {"message": {"source": {"kind": "tool", "callId": "c1"}, "content": [
        {"type": "tool-result", "toolCallId": "c1", "toolName": "web_search", "output": {"results": 3}}]}}},
    {"type": "tool/result", "data": {"error": {"code": "DENIED"}, "message": {"content": [
        {"type": "tool-result", "toolCallId": "c2", "toolName": "bash", "output": None}]}}},
]


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    return exp, get_tracer(provider)


def test_setup_tracing_is_noop_without_credentials():
    from types import SimpleNamespace
    assert setup_tracing(SimpleNamespace(arize_space_id="", arize_api_key="")) is None
    with agent_span(get_tracer(None), "x", input_value="y"):
        pass


def test_agent_span_sets_terminal_status(exporter):
    exp, tracer = exporter
    with agent_span(tracer, "ok", input_value="in"):
        pass
    with pytest.raises(RuntimeError):
        with agent_span(tracer, "bad", input_value="in"):
            raise RuntimeError("boom")
    spans = {s.name: s for s in exp.get_finished_spans()}
    assert spans["ok"].status.status_code == StatusCode.OK
    assert spans["ok"].attributes["openinference.span.kind"] == "AGENT"
    assert spans["bad"].status.status_code == StatusCode.ERROR


def test_dsh_tool_calls_pairs_calls_and_results():
    calls = dsh_tool_calls(EVENTS)
    assert [(c["name"], c["error"]) for c in calls] == [("web_search", False), ("bash", True)]
    assert calls[0]["input"] == {"query": "nvda huawei"} and calls[0]["output"] == {"results": 3}


def test_annotate_result_adds_output_session_and_tool_children(exporter):
    exp, tracer = exporter
    view = AnalystView.model_validate({"score": 8, "score_delta_reasoning_ko": "r", "verdict": "TRIGGERED",
                                       "tape_read_ko": "t", "watch_conditions_ko": "w"})
    result = AnalystResult(agent="dsh", view=view, session_id="session-9", events=EVENTS)
    with agent_span(tracer, "dsh analyst", input_value="brief") as span:
        annotate_result(tracer, span, result)
    spans = {s.name: s for s in exp.get_finished_spans()}
    parent = spans["dsh analyst"]
    assert parent.attributes["dsh.session_id"] == "session-9"
    assert '"score": 8.0' in parent.attributes["output.value"]
    tool = spans["web_search"]
    assert tool.parent.span_id == parent.context.span_id
    assert tool.attributes["openinference.span.kind"] == "TOOL" and tool.status.status_code == StatusCode.OK
    assert spans["bash"].status.status_code == StatusCode.ERROR
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bubble_watch.tracing'`

- [ ] **Step 3: Write the implementation**

**`src/bubble_watch/tracing.py`**
```python
"""Arize AX tracing: LangGraph auto-instrumentation plus explicit AGENT/TOOL spans for the sub-agents."""
from __future__ import annotations

import contextlib
import json
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

KIND, INPUT, OUTPUT = "openinference.span.kind", "input.value", "output.value"
MAX_ATTR = 16_000


def _text(value: Any) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return s if len(s) <= MAX_ATTR else s[:MAX_ATTR] + f"... [truncated {len(s) - MAX_ATTR} chars]"


def setup_tracing(settings: Any) -> Any:
    """Register an Arize tracer provider and instrument LangGraph; None (no-op) without credentials."""
    if not (settings.arize_space_id and settings.arize_api_key):
        return None
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    provider = register(space_id=settings.arize_space_id, api_key=settings.arize_api_key,
                        project_name=settings.arize_project, endpoint=settings.arize_endpoint,
                        set_global_tracer_provider=False, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    return provider


def get_tracer(provider: Any) -> trace.Tracer:
    return trace.NoOpTracer() if provider is None else provider.get_tracer("bubble_watch")


def shutdown_tracing(provider: Any) -> None:
    if provider is not None:
        provider.force_flush()
        provider.shutdown()


@contextlib.contextmanager
def agent_span(tracer: trace.Tracer, name: str, *, input_value: Any, kind: str = "AGENT",
               attributes: dict[str, Any] | None = None) -> Iterator[trace.Span]:
    with tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False) as span:
        span.set_attribute(KIND, kind)
        span.set_attribute(INPUT, _text(input_value))
        for key, value in (attributes or {}).items():
            if value is not None:
                span.set_attribute(key, value)
        try:
            yield span
        except BaseException as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            raise
        span.set_status(Status(StatusCode.OK))


def _blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    data = event.get("data") or {}
    message = data.get("message") if isinstance(data.get("message"), dict) else data
    content = message.get("content") if isinstance(message, dict) else None
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def dsh_tool_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair dsh ``tool-call`` and ``tool-result`` content blocks by ``toolCallId``, in call order."""
    calls: dict[str, dict[str, Any]] = {}
    for event in events:
        failed = bool((event.get("data") or {}).get("error"))
        for block in _blocks(event):
            kind, cid = block.get("type"), block.get("toolCallId") or block.get("id")
            if kind not in ("tool-call", "tool-result") or not cid:
                continue
            call = calls.setdefault(cid, {"id": cid, "name": block.get("toolName") or "tool", "input": None,
                                          "output": None, "error": False})
            if kind == "tool-call":
                call["input"] = block.get("input", block.get("args"))
            else:
                call["output"] = block.get("output", block.get("result"))
                call["error"] = call["error"] or failed or bool(block.get("isError"))
    return list(calls.values())


def record_dsh_tool_spans(tracer: trace.Tracer, events: list[dict[str, Any]]) -> int:
    calls = dsh_tool_calls(events)
    for call in calls:
        with tracer.start_as_current_span(call["name"]) as span:
            span.set_attribute(KIND, "TOOL")
            span.set_attribute("tool.name", call["name"])
            span.set_attribute("tool.id", call["id"])
            span.set_attribute(INPUT, _text(call["input"]))
            span.set_attribute(OUTPUT, _text(call["output"]))
            span.set_status(Status(StatusCode.ERROR, "tool failed") if call["error"] else Status(StatusCode.OK))
    return len(calls)


def annotate_result(tracer: trace.Tracer, span: trace.Span, result: Any) -> None:
    """Attach an AnalystResult to its AGENT span: validated view as output, session link, dsh tool children."""
    span.set_attribute(OUTPUT, _text(result.view.model_dump(mode="json")))
    span.set_attribute("bubble_watch.retried", result.retried)
    if result.session_id:
        span.set_attribute(f"{result.agent}.session_id", result.session_id)
    if result.events:
        span.set_attribute("dsh.tool_call_count", record_dsh_tool_spans(tracer, result.events))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracing.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/bubble_watch/tracing.py tests/test_tracing.py
git commit -m "feat: Arize tracing helpers with dsh tool-span reconstruction"
```

---

### Task 11: LangGraph wiring

**Files:**
- Create: `src/bubble_watch/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `Deps(market, analysts, writer, tracer, reports_dir, state_path, gap_filler=None, options_backup=None, save=True)`, `RunState`, `build_graph(deps) -> CompiledStateGraph`. Invoke with `{"day": date, "watch": WatchState}`; final state has `record`, `reconciliation`, `report_path`, `notes`.

Topology: `START → fetch_market_data → fill_gaps → compute_signals → [analyst_<name> …] → reconcile → (rebuttal → reconcile) → write_report → save_state → END`. With no analysts, the graph ends after `compute_signals` (used by `--no-agents`).

- [ ] **Step 1: Write the failing test**

**`tests/test_graph.py`**
```python
import datetime as dt

import pytest

from bubble_watch.agents.base import AnalystError, AnalystResult
from bubble_watch.graph import Deps, build_graph
from bubble_watch.market_data.base import GapFill, MarketSnapshot
from bubble_watch.models import AnalystView, Close, PutQuote
from bubble_watch.state_store import load_state, save_state, seed_state
from bubble_watch.tracing import get_tracer
from bubble_watch.writer import Narrative

DAY = dt.date(2026, 9, 18)


class FakeMarket:
    def snapshot(self, day, symbols, ticker, expiry, strikes):
        closes = {"NVDA": Close(price=222.27), "SMH": Close(price=573.00), "SOXL": Close(price=123.67)}
        puts = {k: PutQuote(strike=k, bid=b, ask=a, last=l, iv=iv, volume=1000, oi=40000, freshness="latest_snapshot",
                            source_url="https://finance.yahoo.com/x")
                for k, b, a, l, iv in ((200, 1.30, 1.35, 1.34, 34.35), (210, 2.90, 2.97, 2.98, 31.75),
                                       (220, 6.15, 6.30, 6.30, None))}
        return MarketSnapshot(date=day, closes=closes, puts=puts)


class FakeFiller:
    def __init__(self):
        self.prompts = []

    def fill_gaps(self, prompt):
        self.prompts.append(prompt)
        return [GapFill(field="puts.220.iv", value=30.21, source_url="https://x.test/iv", freshness="EOD")]


def _view(score, verdict="TRIGGERED_FURTHER_DE_CONFIRMING"):
    return AnalystView.model_validate({"score": score, "score_delta_reasoning_ko": "r", "verdict": verdict,
                                       "tape_read_ko": f"tape {score}", "watch_conditions_ko": "w",
                                       "catalysts": [{"headline": "h", "url": "https://a.test", "direction": "bullish"}]})


class FakeAnalyst:
    def __init__(self, name, first, rebuttal=None, fail=False):
        self.name, self.first, self.rebuttal, self.fail = name, first, rebuttal, fail
        self.rebut_prompts = []

    def analyze(self, brief):
        if self.fail:
            raise AnalystError(f"{self.name} down")
        assert "2026-09-18" in brief
        return AnalystResult(agent=self.name, view=self.first, session_id=f"{self.name}-s")

    def rebut(self, prompt):
        self.rebut_prompts.append(prompt)
        return AnalystResult(agent=self.name, view=self.rebuttal, session_id=f"{self.name}-s")


class FakeWriter:
    def write(self, prompt):
        return Narrative(headline_ko="H", tape_ko="T", options_ko="O", verdict_ko="V", watch_ko="W")


@pytest.fixture
def paths(tmp_path):
    state_path = tmp_path / "state" / "NVDA.json"
    save_state(seed_state(), state_path)
    return state_path, tmp_path / "reports"


def _run(paths, analysts, *, save=True, filler=None):
    state_path, reports = paths
    deps = Deps(market=FakeMarket(), analysts={a.name: a for a in analysts}, writer=FakeWriter(),
                tracer=get_tracer(None), reports_dir=reports, state_path=state_path, gap_filler=filler, save=save)
    return build_graph(deps).invoke({"day": DAY, "watch": load_state(state_path)})


def test_agree_path_writes_report_and_saves_state(paths):
    filler = FakeFiller()
    final = _run(paths, [FakeAnalyst("hermes", _view(7.9)), FakeAnalyst("dsh", _view(8.2))], filler=filler)
    rec, recon = final["record"], final["reconciliation"]
    assert recon.mode == "agree" and recon.score == 8.1 and rec.score_delta == -0.1
    assert rec.puts[220].iv == 30.21 and "puts.220.iv" in filler.prompts[0]
    assert rec.signals.returns["1d"]["NVDA"] == 1.34
    report = open(final["report_path"]).read()
    assert "2026-09-18" in report and "`8.1 / 10`" in report
    saved = load_state(paths[0])
    assert saved.records[-1].date == DAY and saved.records[-1].score == 8.1
    assert saved.records[-1].analyst_views["dsh"].score == 8.2 and saved.records[-1].note


def test_disagreement_runs_one_rebuttal_round(paths):
    hermes = FakeAnalyst("hermes", _view(7.0, "TRIGGERED_DE_CONFIRMING"), rebuttal=_view(7.9))
    dsh = FakeAnalyst("dsh", _view(8.6, "TRIGGERED"), rebuttal=_view(8.2))
    final = _run(paths, [hermes, dsh])
    assert len(hermes.rebut_prompts) == 1 and len(dsh.rebut_prompts) == 1
    assert "반박" in hermes.rebut_prompts[0] and "tape 8.6" in hermes.rebut_prompts[0]
    assert final["reconciliation"].mode == "agree_after_rebuttal" and final["reconciliation"].score == 8.1


def test_one_failed_analyst_still_ships_single_view(paths):
    final = _run(paths, [FakeAnalyst("hermes", None, fail=True), FakeAnalyst("dsh", _view(8.0))])
    assert final["reconciliation"].mode == "single"
    assert any("hermes analyst failed" in n for n in final["notes"])


def test_all_failed_raises_and_dry_run_does_not_save(paths):
    with pytest.raises(RuntimeError, match="all analysts failed"):
        _run(paths, [FakeAnalyst("hermes", None, fail=True), FakeAnalyst("dsh", None, fail=True)])
    _run(paths, [FakeAnalyst("dsh", _view(8.0))], save=False)
    assert load_state(paths[0]).records[-1].date == dt.date(2026, 9, 17)


def test_no_analysts_stops_after_signals(paths):
    final = _run(paths, [])
    assert final["record"].signals is not None and "report_path" not in final
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bubble_watch.graph'`

- [ ] **Step 3: Write the implementation**

**`src/bubble_watch/graph.py`**
```python
"""LangGraph wiring for one Bubble Watch run (see docs/specs for the topology)."""
from __future__ import annotations

import contextvars
import datetime as dt
import operator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents.base import Analyst, AnalystError, AnalystResult
from .agents.prompts import analyst_brief, gap_fill_brief, rebuttal_brief
from .market_data.base import GapFill, MarketDataProvider, MarketSnapshot, apply_fills, find_gaps, merge_puts
from .models import DailyRecord, Reconciliation, WatchState, round1
from .reconcile import reconcile
from .report import render_report
from .signals import compute_signals
from .state_store import save_state
from .tracing import OUTPUT, agent_span, annotate_result
from .writer import Narrative, fallback_narrative, writer_brief


def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class RunState(TypedDict, total=False):
    day: dt.date
    watch: WatchState
    record: DailyRecord
    brief: str
    results: Annotated[dict[str, AnalystResult], _merge]
    notes: Annotated[list[str], operator.add]
    rebuttal_done: bool
    reconciliation: Reconciliation | None
    report_path: str


class GapFiller(Protocol):
    def fill_gaps(self, prompt: str) -> list[GapFill]: ...


class Writer(Protocol):
    def write(self, prompt: str) -> Narrative: ...


@dataclass
class Deps:
    market: MarketDataProvider
    analysts: dict[str, Analyst]
    writer: Writer | None
    tracer: Any
    reports_dir: Path
    state_path: Path
    gap_filler: GapFiller | None = None
    options_backup: Any = None  # AlphaVantageOptions
    save: bool = True


def _fills_json(fills: list[GapFill]) -> str:
    return "[" + ", ".join(f.model_dump_json() for f in fills) + "]"


def _prior_score(watch: WatchState, day: dt.date) -> float | None:
    return next((r.score for r in reversed(watch.prior_records(day)) if r.score is not None), None)


def build_graph(deps: Deps):
    def fetch_market_data(state: RunState) -> dict:
        w, day, notes = state["watch"], state["day"], []
        try:
            snap = deps.market.snapshot(day, w.symbols, w.ticker, w.expiry, w.strikes)
        except Exception as exc:
            snap = MarketSnapshot(date=day, closes={}, puts={})
            notes.append(f"market data provider failed: {exc}")
        if deps.options_backup is not None:
            try:
                snap.puts = merge_puts(snap.puts, deps.options_backup.puts(day, w.ticker, w.expiry, w.strikes))
            except Exception as exc:
                notes.append(f"historical options backup failed: {exc}")
        return {"record": DailyRecord(date=day, closes=snap.closes, puts=snap.puts), "notes": notes}

    def fill_gaps(state: RunState) -> dict:
        w, rec = state["watch"], state["record"]
        snap = MarketSnapshot(date=rec.date, closes=dict(rec.closes),
                              puts={k: q.model_copy() for k, q in rec.puts.items()})
        gaps = find_gaps(snap, w.symbols, w.strikes)
        if not gaps:
            return {}
        if deps.gap_filler is None:
            return {"notes": [f"{len(gaps)} data fields N/A (no gap filler): {', '.join(g.field for g in gaps)}"]}
        try:
            with agent_span(deps.tracer, "hermes gap fill", input_value=[g.field for g in gaps]) as span:
                fills = deps.gap_filler.fill_gaps(gap_fill_brief(rec.date, w.ticker, w.expiry, gaps))
                span.set_attribute(OUTPUT, _fills_json(fills))
        except AnalystError as exc:
            return {"notes": [f"gap fill failed: {exc}"]}
        rejected = apply_fills(snap, fills)
        still = find_gaps(snap, w.symbols, w.strikes)
        notes = [f"gap fill rejected — {r}" for r in rejected]
        if still:
            notes.append(f"still N/A after gap fill: {', '.join(g.field for g in still)}")
        return {"record": rec.model_copy(update={"closes": snap.closes, "puts": snap.puts}), "notes": notes}

    def compute(state: RunState) -> dict:
        w, rec = state["watch"], state["record"]
        close = rec.closes.get(w.ticker)
        if close is None or close.price is None:
            raise RuntimeError(f"no {w.ticker} close for {rec.date} — not a trading day, or data unavailable")
        sig = compute_signals(w, rec)
        rec = rec.model_copy(update={"signals": sig})
        return {"record": rec, "brief": analyst_brief(w, rec, sig)}

    def route_after_signals(state: RunState) -> list[str] | str:
        return [f"analyst_{name}" for name in deps.analysts] or END

    def make_analyst_node(name: str, analyst: Analyst):
        def node(state: RunState) -> dict:
            try:
                with agent_span(deps.tracer, f"{name} analyst", input_value=state["brief"]) as span:
                    result = analyst.analyze(state["brief"])
                    annotate_result(deps.tracer, span, result)
            except Exception as exc:
                return {"notes": [f"{name} analyst failed: {exc}"]}
            return {"results": {name: result}}
        return node

    def reconcile_node(state: RunState) -> dict:
        views = {k: r.view for k, r in (state.get("results") or {}).items()}
        if not views:
            raise RuntimeError("all analysts failed: " + "; ".join(state.get("notes") or []))
        recon = reconcile(views, prior_score=_prior_score(state["watch"], state["day"]),
                          rebuttal_done=state.get("rebuttal_done", False))
        return {"reconciliation": recon}

    def route_after_reconcile(state: RunState) -> str:
        return "rebuttal" if state.get("reconciliation") is None else "write_report"

    def rebuttal(state: RunState) -> dict:
        results = state["results"]
        a, b = list(results)[:2]

        def run(own: str, other: str) -> AnalystResult:
            prompt = rebuttal_brief(state["brief"], own, results[own].view, other, results[other].view)
            with agent_span(deps.tracer, f"{own} rebuttal", input_value=prompt) as span:
                result = deps.analysts[own].rebut(prompt)
                annotate_result(deps.tracer, span, result)
            return result

        updated, notes = {}, []
        with ThreadPoolExecutor(max_workers=2) as pool:
            # copy_context per task so rebuttal spans nest under this node's span
            futures = {own: pool.submit(contextvars.copy_context().run, run, own, other)
                       for own, other in ((a, b), (b, a))}
            for own, fut in futures.items():
                try:
                    updated[own] = fut.result()
                except Exception as exc:
                    notes.append(f"{own} rebuttal failed, kept original view: {exc}")
        return {"results": updated, "rebuttal_done": True, "notes": notes}

    def write_report(state: RunState) -> dict:
        w, rec, recon = state["watch"], state["record"], state["reconciliation"]
        views = {k: r.view for k, r in state["results"].items()}
        notes = list(state.get("notes") or [])
        narrative = None
        if deps.writer is not None:
            try:
                prompt = writer_brief(w, rec, recon, views)
                with agent_span(deps.tracer, "report writer", input_value=prompt, kind="LLM") as span:
                    narrative = deps.writer.write(prompt)
                    span.set_attribute(OUTPUT, narrative.model_dump_json())
            except Exception as exc:
                notes.append(f"writer failed, used analyst text: {exc}")
        narrative = narrative or fallback_narrative(recon, views)
        prior = _prior_score(w, rec.date)
        rec = rec.model_copy(update={
            "score": recon.score, "score_delta": None if prior is None else round1(recon.score - prior),
            "verdict": recon.verdict, "analyst_views": views, "reconciliation": recon,
            "note": f"{narrative.headline_ko} {narrative.verdict_ko}"[:600]})
        deps.reports_dir.mkdir(parents=True, exist_ok=True)
        path = deps.reports_dir / f"{rec.date}-{w.ticker}.md"
        path.write_text(render_report(w, rec, recon, narrative, views=views, notes=notes))
        rec.report_path = str(path)
        return {"record": rec, "report_path": str(path)}

    def save(state: RunState) -> dict:
        if not deps.save:
            return {}
        watch = state["watch"].model_copy(deep=True)
        watch.upsert(state["record"])
        save_state(watch, deps.state_path)
        return {"watch": watch}

    g = StateGraph(RunState)
    g.add_node("fetch_market_data", fetch_market_data)
    g.add_node("fill_gaps", fill_gaps)
    g.add_node("compute_signals", compute)
    g.add_node("reconcile", reconcile_node)
    g.add_node("rebuttal", rebuttal)
    g.add_node("write_report", write_report)
    g.add_node("save_state", save)
    g.add_edge(START, "fetch_market_data")
    g.add_edge("fetch_market_data", "fill_gaps")
    g.add_edge("fill_gaps", "compute_signals")
    analyst_nodes = []
    for name, analyst in deps.analysts.items():
        g.add_node(f"analyst_{name}", make_analyst_node(name, analyst))
        analyst_nodes.append(f"analyst_{name}")
    g.add_conditional_edges("compute_signals", route_after_signals, [*analyst_nodes, END])
    if analyst_nodes:
        g.add_edge(analyst_nodes, "reconcile")
    g.add_conditional_edges("reconcile", route_after_reconcile, ["rebuttal", "write_report"])
    g.add_edge("rebuttal", "reconcile")
    g.add_edge("write_report", "save_state")
    g.add_edge("save_state", END)
    return g.compile()

```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/bubble_watch/graph.py tests/test_graph.py
git commit -m "feat: LangGraph pipeline with parallel analysts, conditional rebuttal and state save"
```

---

### Task 12: CLI (`seed`, `doctor`, `run`), `.env.example`, README

**Files:**
- Create: `src/bubble_watch/cli.py`, `.env.example`, `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv=None) -> int`; `default_day(now=None) -> date`; `doctor_checks(settings, client) -> list[tuple[str, bool, bool, str]]` (name, ok, required, detail); `build_deps(settings, tracer, *, agents, save) -> Deps`.

- [ ] **Step 1: Write the failing test**

**`tests/test_cli.py`**
```python
import datetime as dt
from zoneinfo import ZoneInfo

import httpx

from bubble_watch.cli import default_day, doctor_checks, main
from bubble_watch.config import load_settings

NY = ZoneInfo("America/New_York")


def test_default_day_uses_last_completed_session():
    assert default_day(dt.datetime(2026, 9, 18, 17, 0, tzinfo=NY)) == dt.date(2026, 9, 18)   # Fri after close
    assert default_day(dt.datetime(2026, 9, 18, 10, 0, tzinfo=NY)) == dt.date(2026, 9, 17)   # Fri before close
    assert default_day(dt.datetime(2026, 9, 20, 12, 0, tzinfo=NY)) == dt.date(2026, 9, 18)   # Sunday


def test_seed_refuses_to_overwrite_without_force(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    assert main(["seed"]) == 0
    assert main(["seed"]) == 1
    assert main(["seed", "--force"]) == 0
    assert (tmp_path / "NVDA.json").exists()


def test_doctor_reports_names_never_values(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-SECRET")
    monkeypatch.setenv("HERMES_API_KEY", "")
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    settings = load_settings(tmp_path / "none.env")

    def handler(req):
        if req.url.host == "api.x.ai":
            return httpx.Response(200, json={"data": [{"id": "grok-4.6"}, {"id": "grok-4.5"}]})
        raise httpx.ConnectError("refused")

    checks = doctor_checks(settings, httpx.Client(transport=httpx.MockTransport(handler)))
    by_name = {c[0]: c for c in checks}
    assert by_name["env HERMES_API_KEY"][1] is False
    assert by_name["xAI model grok-4.6"][1] is True
    assert by_name["Hermes gateway"][1] is False
    assert (tmp_path / "dsh" / "settings.yaml").exists()
    assert not any("xai-SECRET" in c[3] for c in checks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bubble_watch.cli'`

- [ ] **Step 3: Write the implementation**

**`src/bubble_watch/cli.py`**
```python
"""`bubble-watch` command line: seed, doctor, run."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from zoneinfo import ZoneInfo

import httpx

from .agents.dsh_client import DshAnalyst, ensure_dsh_home, make_harness_factory
from .agents.hermes_client import HermesAnalyst
from .config import Settings, load_settings, missing_required
from .graph import Deps, build_graph
from .market_data.alphavantage import AlphaVantageOptions
from .market_data.yfinance_provider import YFinanceProvider
from .state_store import load_state, save_state, seed_state
from .tracing import get_tracer, setup_tracing, shutdown_tracing
from .writer import XaiWriter

NY = ZoneInfo("America/New_York")
CLOSE_BUFFER = dt.time(16, 30)


def default_day(now: dt.datetime | None = None) -> dt.date:
    """Most recent US session that has closed (weekends roll back to Friday; holidays are not modeled)."""
    now = (now or dt.datetime.now(NY)).astimezone(NY)
    day = now.date() if now.time() >= CLOSE_BUFFER else now.date() - dt.timedelta(days=1)
    while day.weekday() >= 5:
        day -= dt.timedelta(days=1)
    return day


def _state_path(settings: Settings):
    return settings.state_dir / "NVDA.json"


def doctor_checks(settings: Settings, client: httpx.Client) -> list[tuple[str, bool, bool, str]]:
    checks: list[tuple[str, bool, bool, str]] = []
    missing = set(missing_required(settings))
    for name in ("XAI_API_KEY", "HERMES_API_KEY"):
        checks.append((f"env {name}", name not in missing, True, "set" if name not in missing else "missing in .env"))
    for name, value, why in (("ARIZE_SPACE_ID", settings.arize_space_id, "tracing"),
                             ("ARIZE_API_KEY", settings.arize_api_key, "tracing"),
                             ("EXA_API_KEY", settings.exa_api_key, "dsh web_search"),
                             ("ALPHAVANTAGE_API_KEY", settings.alphavantage_api_key, "historical option snapshots")):
        checks.append((f"env {name}", bool(value), False, f"set ({why})" if value else f"not set — {why} disabled"))
    state = _state_path(settings)
    checks.append(("state file", state.exists(), True, str(state) if state.exists() else "run `bubble-watch seed`"))
    dsh_ok = os.access(settings.dsh_bin, os.X_OK)
    checks.append(("dsh launcher", dsh_ok, True, settings.dsh_bin))
    home = ensure_dsh_home(settings.dsh_home, settings.dsh_model, settings.dsh_provider, settings.xai_base_url)
    checks.append(("dsh home", True, True, str(home)))

    if settings.xai_api_key:
        try:
            r = client.get(settings.xai_base_url.rstrip("/") + "/models",
                           headers={"Authorization": f"Bearer {settings.xai_api_key}"})
            ids = {m.get("id") for m in r.json().get("data", [])} if r.status_code == 200 else set()
            for model in sorted({settings.dsh_model, settings.writer_model}):
                detail = "served" if model in ids else (
                    f"HTTP {r.status_code}" if r.status_code != 200 else
                    f"not served; grok models: {', '.join(sorted(i for i in ids if i and 'grok' in i))}")
                checks.append((f"xAI model {model}", model in ids, True, detail))
        except httpx.HTTPError as exc:
            checks.append(("xAI API", False, True, f"unreachable: {type(exc).__name__}"))
    try:
        r = client.get(settings.hermes_api_url.rstrip("/") + "/models",
                       headers={"Authorization": f"Bearer {settings.hermes_api_key}"})
        checks.append(("Hermes gateway", r.status_code == 200, True,
                       settings.hermes_api_url if r.status_code == 200 else f"HTTP {r.status_code} (check API_SERVER_KEY)"))
    except httpx.HTTPError:
        checks.append(("Hermes gateway", False, True,
                       f"not reachable at {settings.hermes_api_url} — start it with API_SERVER_ENABLED=true hermes gateway run"))
    return checks


def build_deps(settings: Settings, tracer, *, agents: bool, save: bool) -> Deps:
    hermes = HermesAnalyst(settings.hermes_api_url, settings.hermes_api_key, model=settings.hermes_model,
                           timeout=settings.analyst_timeout_s) if agents else None
    analysts = {"hermes": hermes, "dsh": DshAnalyst(make_harness_factory(settings))} if agents else {}
    return Deps(
        market=YFinanceProvider(), analysts=analysts,
        writer=XaiWriter(settings.xai_api_key, settings.writer_model, settings.xai_base_url) if agents else None,
        tracer=tracer, reports_dir=settings.reports_dir, state_path=_state_path(settings), gap_filler=hermes,
        options_backup=AlphaVantageOptions(settings.alphavantage_api_key) if settings.alphavantage_api_key else None,
        save=save)


def cmd_seed(args, settings: Settings) -> int:
    path = _state_path(settings)
    if path.exists() and not args.force:
        print(f"{path} exists; pass --force to overwrite")
        return 1
    save_state(seed_state(), path)
    print(f"seeded {path}")
    return 0


def cmd_doctor(args, settings: Settings) -> int:
    with httpx.Client(timeout=15) as client:
        checks = doctor_checks(settings, client)
    for name, ok, required, detail in checks:
        print(f"{'✓' if ok else ('✗' if required else '·')} {name} — {detail}")
    return 0 if all(ok for _, ok, required, _ in checks if required) else 1


def cmd_run(args, settings: Settings) -> int:
    path = _state_path(settings)
    if not path.exists():
        print("no state file; run `bubble-watch seed` first")
        return 1
    agents = not args.no_agents
    if agents and (missing := missing_required(settings)):
        print(f"missing required settings: {', '.join(missing)} (set them in .env)")
        return 1
    day = args.date or default_day()
    provider = setup_tracing(settings)
    deps = build_deps(settings, get_tracer(provider), agents=agents, save=agents and not args.dry_run)
    try:
        from openinference.instrumentation import using_session

        with using_session(f"bubble-watch-{day}"):
            final = build_graph(deps).invoke({"day": day, "watch": load_state(path)})
    finally:
        for analyst in deps.analysts.values():
            if hasattr(analyst, "close"):
                analyst.close()
        shutdown_tracing(provider)
    if not agents:
        print(final["record"].signals.model_dump_json(indent=2))
    else:
        recon = final["reconciliation"]
        print(f"report: {final['report_path']}")
        print(f"score {recon.score} ({recon.mode}), verdict {recon.verdict.value}")
    for note in final.get("notes") or []:
        print(f"note: {note}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bubble-watch", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    seed = sub.add_parser("seed", help="write the initial state from the 9/11–9/17 reports")
    seed.add_argument("--force", action="store_true")
    sub.add_parser("doctor", help="check keys (names only), dsh, xAI model and Hermes gateway")
    run = sub.add_parser("run", help="run the pipeline for one trading day")
    run.add_argument("--date", type=dt.date.fromisoformat, help="YYYY-MM-DD (default: last closed session)")
    run.add_argument("--dry-run", action="store_true", help="write the report but do not save state")
    run.add_argument("--no-agents", action="store_true", help="data + signals only; no LLM calls")
    args = parser.parse_args(argv)
    settings = load_settings()
    return {"seed": cmd_seed, "doctor": cmd_doctor, "run": cmd_run}[args.cmd](args, settings)


if __name__ == "__main__":
    sys.exit(main())
```

**`.env.example`**
```bash
# bubble-watch settings. Copy to .env and fill in; .env is gitignored. Never commit keys.

# Required
XAI_API_KEY=                     # Grok 4.6 for the dsh analyst and the report writer
HERMES_API_KEY=                  # = API_SERVER_KEY configured for the Hermes gateway
HERMES_API_URL=http://localhost:8642/v1

# Models (default grok-4.6). Leave HERMES_ANALYST_MODEL empty to use the gateway's configured model.
# HERMES_ANALYST_MODEL=grok-4.6
# DSH_ANALYST_MODEL=grok-4.6
# WRITER_MODEL=grok-4.6

# Tracing (optional; no-op when unset)
ARIZE_SPACE_ID=
ARIZE_API_KEY=
ARIZE_PROJECT_NAME=bubble-watch
# ARIZE_COLLECTOR_ENDPOINT=https://otlp.arize.com/v1

# Optional data/tools
EXA_API_KEY=                     # dsh web_search via Exa (otherwise dsh search needs a DeepSeek key)
ALPHAVANTAGE_API_KEY=            # historical option snapshots (EOD IV for past dates)

# dsh runtime (defaults shown)
# DSH_BIN=./bin/dsh
# DSH_HOME=./.dsh
# DSH_NODE=~/.nvm/versions/node/v24.21.0/bin/node
# DSH_REPO=~/projects/deepseek-harness
# ANALYST_TIMEOUT_S=600
```

**`README.md`**
````markdown
# Bubble Watch

Daily "NVDA Bubble Signal Watch" (Korean) produced by a LangGraph pipeline with two independent analyst
sub-agents — **Hermes Agent** (HTTP gateway) and **DeepSeek Harness** (Python SDK) — both on Grok 4.6.
Every number in the report is computed in code; the agents contribute research, cited catalysts and
judgment, reconciled in code (one rebuttal round when they disagree). Traced to Arize AX.

```
fetch_market_data → fill_gaps (Hermes) → compute_signals
  → hermes_analyst ∥ dsh_analyst → reconcile ─┬→ write_report → save_state
                                   ↑ rebuttal ←┘ (only on disagreement)
```

## Setup

1. `uv sync`
2. dsh runtime: build your DeepSeek Harness checkout under Node ≥ 22.19
   (`PATH=~/.nvm/versions/node/v24.21.0/bin:$PATH pnpm install && pnpm run build` in `~/projects/deepseek-harness`).
   `bin/dsh` launches it with Node 24 (`DSH_NODE` / `DSH_REPO` override the paths).
3. Hermes gateway on Grok 4.6: in `~/.hermes/.env` set `XAI_API_KEY`, `API_SERVER_ENABLED=true`,
   `API_SERVER_KEY=<choose one>`; set the Hermes model to `grok-4.6` (provider `xai`) with `hermes model`;
   then `hermes gateway run`. Optional: enable the `observability/arize` plugin for Hermes' own trace.
4. `cp .env.example .env` and fill in the keys (`HERMES_API_KEY` = the `API_SERVER_KEY` above).
5. `uv run bubble-watch seed` then `uv run bubble-watch doctor`.

## Run

```bash
uv run bubble-watch run --date 2026-09-18            # report → reports/2026-09-18-NVDA.md, state saved
uv run bubble-watch run --date 2026-09-18 --dry-run  # report only
uv run bubble-watch run --no-agents                  # data + signals only, no LLM calls
```

Traces: Arize project `bubble-watch` (one trace per run, `session.id = bubble-watch-<date>`). The Hermes
analyst span carries `hermes.session_id`, which links to Hermes' own trace in the `hermes-agent` project.
dsh tool calls appear as TOOL spans rebuilt from the SDK events (no per-tool timing).

## Data rules

Values carry a freshness label (`EOD`, `latest_snapshot`, `late_session_last`, `derived`, `N/A`). Missing values
stay `N/A` — never estimated; web gap fills need a source URL. yfinance has no historical option chains, so
past-date put data comes from Alpha Vantage (if configured) or the Hermes gap fill.
````

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: 53 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/bubble_watch/cli.py tests/test_cli.py .env.example README.md
git commit -m "feat: bubble-watch CLI (seed, doctor, run), env template and README"
```

---

### Task 13: Live smoke test (needs keys; manual)

**Files:** none (verification only)

- [ ] **Step 1: Offline pipeline check (no keys)**

Run: `uv run bubble-watch seed --force && uv run bubble-watch run --date 2026-09-18 --no-agents`
Expected: signals JSON for 2026-09-18 with `returns.1d.NVDA` = 1.34 and put changes vs the 9/17 record. If yfinance's Oct-16 rows last traded on 9/18, puts are `latest_snapshot`; otherwise a note lists the N/A fields.

- [ ] **Step 2: Verify the dsh Exa overlay resolves** (only if `EXA_API_KEY` is set)

Run: `DSH_HOME=.dsh bin/dsh --profile sdk --patch dsh/exa-search.patch.yml --dump-config | grep -A3 "web-search-exa"`
Expected: the `web-search-exa` plugin appears with the web service's `searchProvider: exa`.

- [ ] **Step 3: User adds keys, starts the Hermes gateway on grok-4.6, then**

Run: `uv run bubble-watch doctor`
Expected: every required check ✓ (xAI `grok-4.6` served, Hermes gateway reachable).

- [ ] **Step 4: Full run**

Run: `uv run bubble-watch run --date 2026-09-18 --dry-run`
Expected: `report: reports/2026-09-18-NVDA.md`, a score/verdict line, and notes for any gaps.

- [ ] **Step 5: Verify the trace in Arize** (arize-trace skill)

Run: `ax spans export bubble-watch --space "$ARIZE_SPACE_ID" --session-id bubble-watch-2026-09-18 --stdout`
Expected: one trace with LangGraph node spans, `hermes analyst` / `dsh analyst` AGENT spans (status OK, output = AnalystView JSON), dsh TOOL children, and a `report writer` LLM span.
