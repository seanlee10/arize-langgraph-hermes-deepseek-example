# Arize × DeepSeek Harness × Hermes Agent × LangGraph

A worked example of **one agent harness orchestrating a second agent runtime and a deterministic
pipeline, with one trace across all three**. DeepSeek Harness (`dsh`) runs the show: it calls the
computation over MCP, delegates web research to Hermes Agent over the Agent Client Protocol, forms
its own view, reconciles the two and writes the report.

The concrete use case is a daily "NVDA Bubble Signal Watch" report in Korean, but the domain is
incidental — the reusable parts are the seams, the composition and the tracing.

```
bubble-watch run --date D                    Python driver: owns the trace root, verifies the report
└── dsh agent loop                           orchestrator AND second analyst
    ├── skill: bubble-watch                  the whole procedure, as instructions
    ├── mcp__bubble__prepare_brief           LangGraph: fetch market data → compute signals
    ├── hermes_analyst(task)                 ACP subagent: gap research, catalysts, its own view
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
| Seam | `dsh-subagent-acp` → `bin/hermes-acp` | `dsh-mcp-client` → `bubble-watch mcp` | the agent loop |
| Wiring | **config only** (a patch layer) | one stdio MCP server | a `SKILL.md` |
| Isolation | own process, session, model, tools; `HERMES_HOME` = `.hermes-analyst/` | own process | — |
| Tracing | traces itself; parents on `HERMES_ARIZE_TRACEPARENT` | `openinference-instrumentation-langchain`; parents on `TRACEPARENT` | spans rebuilt live from the SDK event stream |

Hermes ships an ACP adapter and dsh ships an ACP subagent backend, so that integration is a YAML
insert with no glue code. LangGraph is not an agent and is not pretended to be one: it is a tool
server, which is the shape a deterministic pipeline actually has.

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
3. **Hermes runtime** — `bin/hermes-acp` runs the **repo checkout**'s ACP adapter
   (`$HERMES_REPO/.venv/bin/hermes-acp`, default `~/projects/hermes-agent`): that checkout has the
   Tavily web backend and the `observability/arize` plugin. In it run `uv sync`, then
   `uv pip install 'agent-client-protocol==0.9.0' arize-otel`. The ACP package is an optional extra
   Hermes does not install by default, and its adapter imports it lazily — without it `hermes-acp`
   starts and then exits 1 at ACP initialize, which surfaces only as a failed delegation mid-run.
   `bubble-watch doctor` runs `hermes-acp --check` so you find out at setup instead.
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
uv run bubble-watch mcp                              # the tool server (dsh spawns this; stdout is JSON-RPC)
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

One run is one trace in the Arize project `bubble-watch`. The driver opens the root span *before*
launching dsh, which is what makes a single `traceparent` available at composition time — the ACP and
MCP backends read their `env` once, when the composition loads, so there is no per-call channel.

Verified shape of one run (52 spans, single root):

```
bubble-watch run                    CHAIN   ← the only root
├── dsh session                     AGENT   built live from the SDK's on_notification callback
│   ├── mcp__bubble__prior_state    TOOL
│   ├── mcp__bubble__prepare_brief  TOOL    real start/end, from event arrival
│   ├── hermes_analyst              TOOL    the ACP delegation
│   ├── mcp__bubble__apply_gap_fills TOOL
│   ├── hermes_analyst              TOOL    second view
│   └── … web_fetch / write / mcp__bubble__save_run
├── LangGraph                       CHAIN   ← a different process, via TRACEPARENT
│   ├── fetch_market_data
│   └── compute_signals
└── LangGraph                       CHAIN   ← second invocation (apply_gap_fills)
```

LangGraph runs in its own process (its own container in containers mode) and its spans still land in
this trace, because the MCP server adopts the run's `TRACEPARENT` as a remote parent.

The shape is identical in containers mode — the variables cross a container boundary instead of a
process boundary, and the spans land in the same trace either way.

dsh emits OpenTelemetry *logs*, not spans, so its tool activity is reconstructed. Building it from
the live callback rather than from the finished event list is what gives those spans real durations.

Two fidelity costs, both measured rather than assumed:

- Both children parent on the run root, so a child's spans are siblings of the `hermes_analyst` /
  `mcp__bubble__*` tool spans rather than nested inside them. The ACP and MCP backends expose no
  per-call channel, so there is no per-call span to parent on.
- **Hermes contributes no spans of its own over ACP.** Its `observability/arize` plugin instruments
  the chat/gateway turn lifecycle, and the ACP adapter does not go through it. This was isolated by
  running both paths with an identical home, environment, credentials and `traceparent`:
  `hermes chat -Q` emitted `Hermes turn` + `LLM call 1` correctly parented on the supplied
  traceparent, while `hermes-acp` emitted nothing. Hermes' work is therefore attributed and timed in
  the trace as the `hermes_analyst` TOOL span, but not decomposed into its internal LLM and tool
  calls. Closing that gap means instrumenting the ACP surface in the Hermes checkout.

Tracing is optional — without `ARIZE_*` credentials everything runs with a no-op tracer.

## Known gaps

- **Hermes emits no spans over ACP** — see Tracing above. Its turn is visible as one TOOL span.
- **`platform_toolsets.acp` may not be honored.** The isolated home requests the `web` toolset only,
  but an ACP delegation logged `tools.terminal_tool: Shutting down 1 remaining sandbox(es)`, so the
  ACP surface appears to use its own default toolset. The home is still isolated (`HERMES_HOME`,
  platform credentials stripped), but the analyst may have more tools than intended. Worth
  confirming against the Hermes checkout before relying on the narrower surface.
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
  mcp_server.py    the four tools, as an MCP stdio server      mcp_tools.py  their implementations
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
  Dockerfile.hermes-acp    built from the Hermes checkout
  Dockerfile.dsh-runner    built from the dsh checkout; carries the Docker CLI
  build.sh                 all three, with the right contexts
bin/              dsh, hermes-acp, dsh-docker launchers
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
