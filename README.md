# Arize × DeepSeek Harness × Hermes Agent × LangGraph

A worked example of **one agent harness orchestrating a second agent runtime and a deterministic
pipeline, with one trace across all three**. DeepSeek Harness (`dsh`) runs the show: it calls the
computation over MCP, delegates web research to Hermes Agent over the Agent Client Protocol, forms
its own view, reconciles the two and writes the report.

The concrete use case is a daily "NVDA Bubble Signal Watch" report in Korean, but the domain is
incidental — the reusable parts are the seams, the composition and the tracing.

**Everything the model reads is English; only the report is Korean.** Prompts, the skill, and every
MCP tool description and parameter land in span attributes, and the people reading these traces do
not read Korean. The report is the product, so it stays Korean — which means the root span's output
and the `write` tool call still carry Korean text, and nothing else does. A test asserts the tool
surface stays English.

```
bubble-watch run --date D                    Python driver: owns the trace root, verifies the report
└── dsh agent loop                           orchestrator AND second analyst
    ├── skill: bubble-watch                  the whole procedure, as instructions
    ├── mcp__bubble__prepare_brief           LangGraph: fetch market data → compute signals
    ├── mcp__bubble__hermes_analyst(task)    Hermes: gap research, catalysts, its own view
    ├── mcp__bubble__apply_gap_fills         sourced values only, re-computed
    ├── (own view → compare → rebuttal round if they diverge)
    ├── write(reports/D-NVDA.md)
    └── mcp__bubble__save_run
```

> **Note.** The diagrams and trace screenshots under `docs/` still show the previous topology, in
> which LangGraph was the orchestrator and both agents were leaf analysts. They need regenerating
> from a live run.

## The three seams

Each runtime is integrated through the seam it actually exposes — that is the whole point of the
example.

| | Hermes Agent | LangGraph | dsh itself |
|---|---|---|---|
| Seam | one-shot CLI subprocess per call, hosted by the tool server | `dsh-mcp-client` → `bubble-watch mcp` | the agent loop |
| Wiring | `hermes_tool.py` | one stdio MCP server | a `SKILL.md` |
| Isolation | own process, session, toolset; `HERMES_HOME` = `.hermes-analyst/` | own process | — |
| Tracing | traces itself, nested under the call's span | `openinference-instrumentation-langchain`; parents on `TRACEPARENT` | spans rebuilt live from the SDK event stream |

LangGraph is not an agent and is not pretended to be one: it is a tool server, which is the shape a
deterministic pipeline actually has.

### Why Hermes is not an ACP subagent

dsh ships an ACP subagent backend and Hermes ships an ACP adapter, so wiring them together is
config-only — that was the first implementation, and it worked. It was replaced because **Hermes'
ACP adapter builds an `AIAgent` directly and never touches `hermes_cli`'s plugin dispatch, so no
plugin hook fires over ACP** — including `observability/arize`, which is how Hermes traces itself.
Proven by running both paths with an identical home, environment, credentials and `traceparent`:
`hermes chat -Q` emitted `Hermes turn` + `LLM call 1`, correctly parented; `hermes-acp` emitted
nothing.

Hosting the analyst inside the MCP tool server instead turned out better than the seam it replaced:
that process owns a real OTel context, so each call gets its own span and Hermes' spans nest
**inside the call** rather than beside it.

## What is deterministic and what is not

`signals.py` and `market_data/` still compute every number, and `apply_gap_fills` still rejects any
researched value that arrives without an `http(s)` source URL and a freshness label.

**The report itself is fully model-authored.** dsh writes the markdown — narrative and tables alike —
so every figure in it is a model transcription of a computed value rather than a rendered one. That
is a deliberate trade for a dsh-native authoring loop; the escape hatch, if transcription errors show
up, is a `verify_report` MCP tool that re-extracts the figures and fails the run on a mismatch.

Two deterministic backstops remain in the driver: the run fails if dsh stops early, and it fails if
no non-empty report file landed at the expected path.

## Setup

Requires Python 3.13 (`uv`), Node ≥ 22.19 for dsh, and local checkouts of both agent runtimes.

1. `uv sync`
2. **dsh runtime** — build your DeepSeek Harness checkout under Node ≥ 22.19:
   `PATH=~/.nvm/versions/node/v24.21.0/bin:$PATH pnpm install && pnpm run build`.
   `bin/dsh` launches the built CLI (`DSH_NODE` / `DSH_REPO` override the paths). Launching the TS
   source via `tsx` mixes src/lib plugin copies and every tool call fails, so the built CLI is used
   deliberately.
3. **Hermes runtime** — `bin/hermes` runs the **repo checkout** (`$HERMES_REPO/.venv/bin/hermes`,
   default `~/projects/hermes-agent`), not an installed `hermes`: that checkout has the Tavily web
   backend and the `observability/arize` plugin. In it run `uv sync`, then
   `uv pip install arize-otel` — the plugin needs it to export spans and Hermes does not install it
   by default. `bubble-watch doctor` checks the runtime starts.
4. `cp .env.example .env` and fill in `XAI_API_KEY`, `TAVILY_API_KEY`, `ALPHA_VANTAGE_API_KEY` and
   the `ARIZE_*` keys.
5. `uv run bubble-watch install-plugins` — installs the two dsh plugins the base bundle does not
   carry into the profile package tree (needs `pnpm` and network).
6. `uv run bubble-watch seed` then `uv run bubble-watch doctor` — doctor reports variable **names**
   only, installs the skill into `DSH_HOME`, and checks the profile plugins.

The first run writes `.dsh/` (xAI provider settings, the rendered patch layers, the installed skill)
and `.hermes-analyst/` (Grok model config, the `web` toolset only, the tirith pre-exec scanner
disabled, the Arize plugin enabled). Your own `~/.hermes` and its messaging platforms are never
touched: platform credentials are stripped from the child environment.

## Run

```bash
uv run bubble-watch run --date 2026-09-18            # dsh orchestrates; report → reports/, state saved
uv run bubble-watch run --date 2026-09-18 --dry-run  # same, then restore the state file
uv run bubble-watch run --no-agents                  # data + signals only: no dsh, no LLM at all
uv run bubble-watch doctor                           # keys, dsh, the ACP launcher, the skill, the model
uv run bubble-watch mcp                              # the deterministic tool server (dsh spawns this)
uv run bubble-watch hermes-mcp                       # the analyst server (dsh spawns this too)
uv run bubble-watch install-plugins                  # one-time: the dsh plugins the bundle lacks
```

Each run gets its own dsh session id (`bubble-watch-<date>-<8 hex>`), because dsh persists sessions
under `DSH_HOME` and rejects a duplicate — a fixed per-day id makes the second run of a day fail with
`session already exists`. The Arize grouping key stays `bubble-watch-<date>` so a day's re-runs still
group together.

## Composition

`dsh/orchestrator.patch.yml` is the whole integration. Rendered into `DSH_HOME` at launch, it
inserts three rows: the ACP subagent provider, the delegation tool that exposes it to the model, and
the MCP client that spawns the tool server.

Four details are load-bearing. Each one fails *silently* — the composition still loads, dsh still
runs, and the tool simply never appears in the model's tool list:

- **The plugins must be installed into the profile.** `dsh-mcp-client` and `dsh-subagent-acp` are
  not in dsh's base bundle, so `name:` cannot resolve them. Do not "fix" that by pointing `name:` at
  the checkout's `lib/index.js`: that loads a *second* copy of the plugin, and its config is then
  validated against the bundle's copy. `bubble-watch install-plugins` does it the supported way.
- **Never declare an env var that is not set.** `!!js process.env.FOO` on an unset `FOO` evaluates
  to `undefined`. `dsh-mcp-client`'s `Config` is a `z.union`, and a branch rejects an `env` dict
  holding one — so the row fails validation and the plugin never activates. The error is unusually
  hard to read: its `but got {...}` is `JSON.stringify`d, which *drops* undefined-valued keys, so
  every field it shows you is valid. `dsh-subagent-acp`'s `Config` is a plain `z.object` and
  tolerates it, which is why a mistake here makes the MCP tools vanish while Hermes keeps working.
  The `env:` blocks are generated from the same source of truth as the child environment.
- **`maxDepth: provider-managed`** is mandatory on the delegation tool. The ACP backend advertises no
  `depthLimit` capability, so a numeric cap — *and the host default that a missing key falls back
  to* — is rejected when the composition activates. Recursion is bounded by the child instead: its
  isolated home enables the web toolset only, so it has no delegation tools.
- **Each child reads a different variable name.** Hermes' plugin reads `HERMES_ARIZE_*`; the Python
  side reads the OTel/Arize names. Both backends scrub credential-shaped ambient variables, so
  everything a row looks up with `!!js process.env` is re-supplied explicitly.

Debugging any of these is harder than it should be, because the SDK captures dsh's stderr
(`stderr=subprocess.PIPE`) rather than inheriting it: the warning naming the entry that failed to
activate never reaches your terminal. It is retained on the client and surfaces in timeout messages.

## Containers

Each runtime can run in its own container. `BUBBLE_WATCH_DEPLOY=containers` swaps the composition
for `dsh/orchestrator.containers.patch.yml`, in which dsh reaches each sibling by spawning
`docker run -i`:

```bash
docker/build.sh                       # three images; two build FROM the other checkouts
BUBBLE_WATCH_DEPLOY=containers \
BUBBLE_WATCH_HOST_ROOT=$PWD \
DSH_BIN=$PWD/bin/dsh-docker \
  uv run bubble-watch run --date 2026-09-18
```

**Why `docker run -i` and not a network transport.** `dsh-subagent-acp` is spawn-only — `command` is
required and there is no URL form — so whatever dsh spawns must be a local process. `docker run -i`
*is* that local process, and it proxies stdio to a container. Both seams keep speaking exactly the
protocol they already spoke; only the far end of the pipe moves. (`dsh-mcp-client` does have a
`streamable-http` transport, but using it for one seam and a spawn-proxy for the other would buy
nothing and split the design in two.)

The same trick makes dsh itself a container: `bin/dsh-docker` is a `DSH_BIN` that execs
`docker run -i`, and the Python SDK's JSON-RPC is none the wiser. The dsh container mounts the
Docker socket, because it spawns its own siblings.

### How the trace context crosses

A container does not inherit its parent's environment, so the traceparent takes two hops per
boundary:

1. **`config.env` on the row** puts the value on the `docker` CLI process. This is required, not
   belt-and-braces: the ACP and MCP backends scrub credential-shaped ambient variables, so
   inheritance alone loses `XAI_API_KEY` and friends.
2. **`-e NAME`** (a bare name, no `=value`) forwards that value from the CLI into the container.

Nothing per-run is baked into an image or into the rendered composition — `-e NAME` carries whatever
dsh's own environment holds for that run, which the driver set before launching dsh.

This is also *why* the containers are per-run. dsh binds a traceparent only when a composition or a
process is created: its MCP client resolves `headers` once at load, exposes no per-request hook, and
has no trace context of its own to inject, because it emits OpenTelemetry logs rather than spans. A
long-lived dsh service would therefore stamp every run with the first run's trace. Per-run
containers sidestep that entirely instead of working around it.

### Paths

`launch_paths()` is the seam: patch layers and the skill are written on the host, into the directory
mounted at `/work/.dsh`, but the paths handed to dsh as argv are the container's. `-v` sources stay
host paths — a sibling container's mount is resolved by the daemon, wherever the CLI issuing it runs
— which is what `BUBBLE_WATCH_HOST_ROOT` is for when the driver is itself containerised.

## Tracing

One run is one trace in the Arize project `bubble-watch`, rooted at the dsh run. The driver opens
that root span *before* launching dsh, which is what makes a `traceparent` available at composition
time — the MCP backend reads its `env` once, when the composition loads.

Verified shape of one run (52 spans, single root):

```
dsh                                 AGENT   ← the only root: the harness run itself
├── mcp__bubble__prior_state        TOOL    built live from the SDK's on_notification callback
├── mcp__bubble__prepare_brief      TOOL    real start/end, from event arrival
├── mcp__hermes__analyst            TOOL    the delegation
├── mcp__bubble__apply_gap_fills    TOOL
├── mcp__hermes__analyst            TOOL    second view
├── … web_fetch / write / mcp__bubble__save_run
├── LangGraph                       CHAIN   ← a different process, via TRACEPARENT
│   ├── fetch_market_data
│   └── compute_signals
├── hermes analyst                  AGENT   ← per-call span, opened by the tool server
│   └── Hermes turn                 AGENT   ← Hermes' own spans, nested inside the call
│       └── LLM call 1              LLM
└── LangGraph                       CHAIN   ← second invocation (apply_gap_fills)
```

**The root is the dsh run.** dsh cannot open that span itself — it emits OpenTelemetry logs rather
than spans, and the `traceparent` has to exist *before* it launches, because the MCP composition
reads its `env` once at load. So the driver opens the root on dsh's behalf and reconstructs dsh's
tool spans beneath it. An earlier version had a separate `bubble-watch run` wrapper above
`dsh session`; the two had the same input and nearly the same lifetime, and the only effect was to
push the harness a level down, so they are now one span.

LangGraph runs in its own process (its own container in containers mode) and its spans still land in
this trace, because the MCP server adopts the run's `TRACEPARENT` as a remote parent.

The shape is identical in containers mode — the variables cross a container boundary instead of a
process boundary, and the spans land in the same trace either way.

dsh emits OpenTelemetry *logs*, not spans, so its tool activity is reconstructed. Building it from
the live callback rather than from the finished event list is what gives those spans real durations.

One fidelity cost, measured rather than assumed: the MCP server's own spans (LangGraph, and the
`hermes analyst` call) parent on the **run root**, not on the `mcp__bubble__*` tool span that
triggered them. The MCP backend's `env` is static, read once at load, so there is no per-call
channel to carry a per-call parent across that boundary. Hermes is unaffected — its parent is
chosen inside the tool server, where a live span exists.

`session.id` is stamped on **every span this repo produces**, through two mechanisms because they
cover different spans: `using_session` is what OpenInference instrumentors (LangGraph) read, while
the spans built here by hand read a context variable. Setting it on one span only — as an earlier
version did — yields a partial tree in a session-filtered view.

One span group stays outside it: **Hermes stamps its own session id** on its own spans, because its
`observability/arize` plugin owns that attribute and exposes no override. So a run looks like this:

```
probe root       session.id=bubble-watch-2026-09-18
└── hermes analyst   session.id=bubble-watch-2026-09-18
    └── Hermes turn      session.id=20260923_150836_e0843e   ← Hermes' own
        └── LLM call 1   session.id=20260923_150836_e0843e
```

The parent/child links are intact and the **trace** is complete — it is only a session-*filtered*
view that splits in two. That is arguably correct (Hermes' spans do belong to a Hermes session), and
changing it would mean editing the Hermes checkout. Open the trace, not the session.

Tracing is optional — without `ARIZE_*` credentials everything runs with a no-op tracer.

### Identifiers: what is propagated, and what is persisted

Four ids are in play, and they are owned by three different systems.

**Trace context travels as environment variables and is persisted nowhere.** Two *different*
traceparents are in flight, which is exactly what gives Hermes its extra nesting level.

To the tool server — the **run root**:

```
driver: agent_span("bubble-watch run")        OTel mints trace_id + span_id
  └─ TraceContextTextMapPropagator().inject()
     TRACEPARENT=00-<trace_id>-<root span_id>-01
        └─ child_env() → DeepSeekHarness(env=…)            dsh process env
           └─ composition: TRACEPARENT: !!js process.env.TRACEPARENT   (static, read at load)
              └─ spawn, or `docker run -e TRACEPARENT`     tool server process env
                 └─ remote_parent_context() → extract → context.attach()
                    every span in that process nests under the root
```

To Hermes — a **per-call** parent, built inside the tool server where a live span exists:

```
tool server: agent_span("hermes analyst")     new span_id, same trace_id
  └─ HERMES_ARIZE_TRACEPARENT=00-<trace_id>-<hermes analyst span_id>-01
     └─ subprocess env, or `docker run -e`    hermes chat process
        └─ observability/arize plugin reads it → "Hermes turn" parents on that span
```

The run's `trace_id` is also **persisted**: `prepare_brief` returns it so the model can cite it in
the report's provenance line, and `save_run` stores it on the day's record. A stored run or a
delivered report therefore leads back to its trace.

**Session ids**, each owned elsewhere:

| Id | Shape | Created by | Persisted by | Propagated by |
|---|---|---|---|---|
| dsh session | `bubble-watch-<date>-<8 hex>` | `orchestrator.session_ids()`, per run | **dsh**, under `$DSH_HOME/sessions/` | `harness.run(session_id=…)` |
| Arize session key | `bubble-watch-<date>`, stable per day | the same call | nothing — a span attribute | `session_context()` in the driver; `BUBBLE_WATCH_SESSION_ID` to the tool server |
| Hermes session | `20260923_145812_84bc3a` | **Hermes** | **Hermes**, in `.hermes-analyst/state.db` | parsed off stderr → returned to the model → passed back → `--resume` |

The two `bubble-watch-<date>*` ids are split deliberately: dsh rejects a duplicate session id, so it
needs a fresh one per run, while Arize needs a stable key to keep a day's re-runs grouped.

## Run time

A run is a few minutes, and **almost all of it is research** — the deterministic half (market data,
LangGraph signal computation, gap application, persistence) takes about **2 seconds** of a 10-minute
run. Measured breakdown of one run:

| | calls | total | share of wall clock |
|---|---:|---:|---:|
| `web_fetch` | 35 | 299s | 51% |
| `hermes_analyst` (including its LLM call) | 2 | 236s | 40% |
| `web_search` | 1 | 104s | 18% |
| everything deterministic | — | ~2s | 0% |

**Set `TAVILY_API_KEY`.** Without a search backend, dsh's `web_search` fails after ~100 seconds
(`DeepSeek search has no API key`) and both the orchestrator and Hermes fall back to fetching pages
one at a time — which is what those 35 `web_fetch` calls are. Tavily is wired for dsh through
`dsh/plugins/web-search-tavily.mjs` and picked up natively by Hermes, so one key fixes both.

The skill also caps the orchestrator's own research (prefer one search over many fetches, at most 8
page fetches for its own view, no repository exploration) so the two analysts do not duplicate each
other's work.

## Hermes stream timeouts

Hermes' stream watchdog arms on the first parsed event and then kills the LLM call after a period of
silence. Against xAI that default resolves to about **12 seconds** — and grok-4.6 is a reasoning
model that spends longer than that thinking between tokens. Left alone, every analyst call dies with
`Codex stream produced no SSE events for 12s`, retries three times, and the turn fails; the run
still completes because the orchestrator retries the delegation, but it is slow and the trace fills
with errors.

`hermes_env()` therefore sets `HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS` to 180s
(`HERMES_STREAM_IDLE_S` to override). A genuinely hung stream is still killed, just not a thinking
one.

## Known gaps

- **A session-filtered view splits in two**, because Hermes stamps its own session id on its own
  spans — see Tracing above. The trace itself is complete.
- **Hermes session continuity rides on the model.** The tool returns `session_id` and `SKILL.md`
  asks for it back on the rebuttal round, but nothing enforces it. If the model drops it, Hermes
  starts a fresh conversation and silently loses the first view — no error, no note in the report.
  The deterministic alternative is for the tool server to remember the last session id per run and
  resume automatically unless told otherwise.
- The MCP server's spans parent on the run root rather than the tool call that triggered them —
  see Tracing above.
- **A trace looks broken while a run is in progress.** The root `dsh` span stays open for the whole
  run, so until it ends it has not been exported and every completed child span shows as orphaned
  ("its parent span is missing"). It resolves when the run finishes. This is
  inherent to exporting a span at its end, not a wiring fault.
- The diagrams and trace screenshots under `docs/` predate all of this.

## Data rules

Alpha Vantage EOD is the single source for closes and option chains (any past date). yfinance is used
only when no Alpha Vantage key is set. Every value carries a freshness label (`EOD`,
`latest_snapshot`, `late_session_last`, `derived`, `N/A`); a missing value stays `N/A` and is never
estimated. Gaps come back from `prepare_brief` as `gaps[]`, dsh delegates the research, and
`apply_gap_fills` rejects anything unsourced.

Alpha Vantage reports IV on a ~0.98%p grid, so a rise within one step is reported as "can't tell"
rather than a signal.

## Layout

```
src/bubble_watch/
  orchestrator.py  the driver: trace root, traceparent, run verification
  harness.py       dsh composition and launch; the isolated Hermes home; the child environment
  mcp_server.py    two MCP stdio servers: the deterministic tools, and the analyst
  mcp_tools.py     the deterministic tools' implementations
  hermes_tool.py   the Hermes analyst: one-shot CLI per call, with its own span
  brief_graph.py   the LangGraph pipeline: fetch market data → compute signals
  signals.py       pure signal computation                      state_store.py  JSON persistence
  market_data/     alphavantage.py (primary) · yfinance_provider.py (fallback) · base.py (gaps)
dsh/
  orchestrator.patch.yml            local: spawn the siblings directly
  orchestrator.containers.patch.yml containers: spawn `docker run -i` for each
  web-search.patch.yml     Tavily/Exa search backend for dsh
  skills/bubble-watch/     SKILL.md — the procedure dsh follows
  plugins/                 web-search-tavily.mjs
docker/
  Dockerfile.mcp-tools     built from this repo
  Dockerfile.hermes        built from the Hermes checkout (copies bubble-watch from mcp-tools)
  Dockerfile.dsh-runner    built from the dsh checkout; carries the Docker CLI
  build.sh                 all three, with the right contexts
bin/              dsh, hermes, dsh-docker launchers
```

## Tests

`uv run pytest -q` — 90 tests, no network. The MCP tools run end-to-end against a fake market data
provider (gaps, rejected fills, recomputation, persistence); the driver runs against a fake harness
replaying a canned notification stream (span tree, the report backstop, early stops); the dsh Tavily
plugin is tested under Node with a mocked `fetch`.

What no unit test covers any more: whether the model actually follows `SKILL.md`. That moved from
graph wiring to eval territory when dsh took over orchestration.

## Status

Analysis, not trade advice. A backtest of the report's own entry logic (2022–2026 NVDA options) did
**not** support it as a timing rule, so no entry/exit decision is emitted.
