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
