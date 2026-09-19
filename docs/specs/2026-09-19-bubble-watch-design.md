# Bubble Watch — Design Spec

Date: 2026-09-19 · Status: approved in chat, implementing

## Goal

A LangGraph-orchestrated daily analysis agent that reproduces the "NVDA Bubble Signal Watch"
report (Korean, US market close) with two independent sub-agent analysts — **Hermes Agent** and
**DeepSeek Harness (dsh)** — whose views are reconciled in code. Every number in the report comes
from code; agents contribute judgment and cited catalysts only. Runs are traced to Arize AX.

Non-goals: EC2 deployment (config-only later), trace-context propagation into Hermes, historical
backfill beyond the seed, trading advice.

## Runtime assumptions

- Everything runs locally. Hermes is reached over its OpenAI-compatible gateway API
  (`HERMES_API_URL`, default `http://localhost:8642/v1`, bearer `HERMES_API_KEY`).
- dsh is driven in-process through its Python SDK (`deepseek_harness.DeepSeekHarness`), with a
  project-local `DSH_HOME` whose `settings.yaml` registers xAI via `llm-pi-ai`
  (`openai-completions`, `https://api.x.ai/v1`, `apiKeyEnv: XAI_API_KEY`).
- Both analysts and the report writer use **Grok 4.6** (`grok-4.6`), overridable per agent
  (`HERMES_ANALYST_MODEL`, `DSH_ANALYST_MODEL`, `WRITER_MODEL`). Model availability is checked by
  `bubble-watch doctor` (`GET https://api.x.ai/v1/models`).
- Secrets live only in `.env` (gitignored); `.env.example` lists names without values.

## Components (`bubble_watch/`)

| Module | Responsibility |
|---|---|
| `config.py` | Typed settings from env; `doctor()` reports missing names only, never values |
| `models.py` | Pydantic models: `PutQuote`, `Close`, `Anchor`, `Signals`, `AnalystView`, `Catalyst`, `DailyRecord`, `WatchState`, `Reconciliation` |
| `state_store.py` | Load/save `state/<TICKER>.json`; `seed()` from the 9/15–9/17 reports |
| `market_data/base.py` | `MarketDataProvider` protocol; `GapRequest` for missing/stale fields |
| `market_data/yfinance_provider.py` | Closes for NVDA/SMH/SOXL; current put chain (bid/ask/last/IV/vol/OI) — default |
| `market_data/alphavantage.py` | Optional historical option snapshot (`HISTORICAL_OPTIONS`) |
| `signals.py` | Pure: returns (1d / 3 trading days / since anchor), NVDA−SMH spread, put changes, convexity ordering, IV compare vs last-verified, 3 re-confirmation conditions |
| `agents/base.py` | `Analyst` protocol: `analyze(brief) -> AnalystView`, `rebut(brief, other) -> AnalystView`; shared JSON parsing + one validation retry |
| `agents/hermes_client.py` | HTTP to Hermes `/chat/completions`; also `fill_gaps(requests)`; records `X-Hermes-Session-Id` |
| `agents/dsh_client.py` | dsh SDK `.run()`; returns view + events for span reconstruction |
| `agents/prompts.py` | Analyst brief, rebuttal brief, gap-fill brief, writer brief |
| `reconcile.py` | Pure agreement/merge rules |
| `report.py` | Tables from code + writer narrative → Korean markdown with `[n]: url` footers |
| `writer.py` | Direct xAI chat call for the narrative paragraphs |
| `tracing.py` | `arize-otel` register + LangChain instrumentor; dsh event → span builder |
| `graph.py` | LangGraph `StateGraph` wiring |
| `cli.py` | `bubble-watch run [--date] [--dry-run] [--no-agents]`, `doctor`, `seed` |

## Graph

```
load_state → fetch_market_data → fill_gaps → compute_signals
  → [hermes_analyst ∥ dsh_analyst] → reconcile
      ├─ agree → write_report
      └─ disagree (first time) → rebuttal (both, parallel) → reconcile → write_report
  → save_state
```

`fill_gaps` runs only if a required field is missing or stale; Hermes returns each value with
`source_url` and `freshness`. Values without both are rejected; `N/A` is accepted and stored as
`None` with freshness `N/A`. Estimation is never allowed.

## State (`state/NVDA.json`)

`WatchState{ticker, peer, leveraged, expiry, strikes, trigger_threshold=8.0, anchors[], records[]}`.
`DailyRecord{date, closes{sym: Close}, puts{strike: PutQuote}, signals, score, score_delta,
verdict, analyst_views, reconciliation, report_path}`.
`PutQuote{strike, bid, ask, mid, last, iv, volume, oi, source_url, freshness, as_of}` with
`freshness ∈ {EOD, latest_snapshot, late_session_last, N/A}`.

Comparison rules:
- Option 1-day change = vs previous record's `last` ("recorded-to-recorded"), carrying both freshness labels.
- IV compare = vs the most recent record where that strike's IV is non-null ("last-verified"), naming the date.
- 3-trading-day window = vs the record 3 records back (records are trading days).
- Since-anchor = vs the pinned anchor (default: 2026-08-28 spike).

Seed: 9/11 (closes only, for 3-day windows), 9/14, 9/15, 9/16, 9/17 records and the 8/28 anchor,
transcribed from the reports. Values not stated in the reports are `None`.

## Signals

- `rel_1d = r(NVDA) − r(SMH)` in %p; also 3d and since-anchor.
- `convexity_order`: strikes sorted by 1d put % change; `otm_leads = (order[0] == lowest strike and change > 0)`.
- `iv_expansion_surface = all strikes' IV up vs last-verified` (None if any IV unavailable).
- Re-confirmation conditions (as in the reports):
  1. `nvda_underperforms`: rel_1d < 0 (true), and `persistent` if also rel over 3d < 0.
  2. `iv_surface_up`: `iv_expansion_surface`.
  3. `far_otm_leads`: $200P 1d % change > both $210P and $220P, and > 0.
  Each is `true | false | partial | unknown`.

## Analyst contract

Brief (identical for both): today's tables with freshness, prior record, anchors, computed
signals, prior score/verdict/rationale, rules (no recomputing numbers, cite catalysts published
since the last record with URLs, state invalidation conditions, respond in the JSON schema).

`AnalystView{score (0–10, 1dp), score_delta_reasoning_ko, verdict, catalysts[Catalyst],
tape_read_ko, watch_conditions_ko, confidence}`;
`Catalyst{headline, url, published, direction(bearish|bullish|neutral), weight(low|med|high), rationale_ko}`.

Guardrails: schema-validated with one retry carrying the validation error; score clamped; |Δ|>1.0
flagged; catalysts without http(s) URL dropped; per-analyst timeout (default 600 s); a failed
analyst does not fail the run — the report says "단일 analyst 기준".

Verdicts: `NOT_TRIGGERED`, `TRIGGERED`, `TRIGGERED_DE_CONFIRMING`,
`TRIGGERED_FURTHER_DE_CONFIRMING`, `CONFIRMED`. Caution order (most cautious first):
`CONFIRMED > TRIGGERED > TRIGGERED_DE_CONFIRMING > TRIGGERED_FURTHER_DE_CONFIRMING > NOT_TRIGGERED`.

## Reconciliation

1. Agree if |Δscore| ≤ 0.5 and same verdict → score = mean (1dp), verdict shared.
2. Otherwise, if no rebuttal yet → rebuttal round (each sees the other's view, returns revised view + `rebuttal_ko`).
3. After rebuttal: agree → as 1; still disagree → score = mean, verdict = most cautious, report
   includes "Analyst disagreement" section.
4. One analyst only → its view, flagged. None → run fails before writing a report (state not saved).
5. Catalysts merged by URL, each tagged with citing agents.

## Report

Headline score line, main metrics table, put table (bid/ask/mid/last/Δ/IV/vol/OI with freshness
note), multi-window table (1d / 3 trading days / since anchor), narrative sections (tape read,
catalysts, options interpretation, verdict, watch conditions) written by the writer LLM from
reconciled views only, optional disagreement section, `[n]: url` footers. Written to
`reports/<date>-NVDA.md`.

## Tracing

Arize project `bubble-watch` (`ARIZE_*` env). LangGraph auto-instrumented via
`openinference-instrumentation-langchain`. Analyst nodes get AGENT spans; Hermes node records
`hermes.session_id` (its internal trace is in the `hermes-agent` project via the Hermes plugin);
dsh node gets child LLM/TOOL spans rebuilt from SDK events. Tracing is optional: missing Arize
credentials → no-op.

## Testing

- Unit: `signals` against 9/16 and 9/17 report numbers (golden); `reconcile` (agree, rebuttal-agree,
  rebuttal-disagree, single analyst); state round-trip + seed; report rendering; JSON parsing/retry;
  dsh event→span builder.
- Graph: fake analysts + fake market data, offline, both agree and rebuttal paths.
- Live smoke (manual, needs keys): `bubble-watch doctor` then `bubble-watch run --date 2026-09-18`.
