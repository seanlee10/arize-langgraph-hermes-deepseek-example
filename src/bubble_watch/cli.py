"""`bubble-watch` command line.

Four commands: `seed` and `doctor` as before, `run` (launch dsh as the orchestrator for one trading
day) and `mcp` (serve the deterministic tools over stdio — this is what dsh's MCP client spawns).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from opentelemetry import context

from .config import Settings, load_settings, missing_required
from .harness import (
    ensure_dsh_home,
    install_profile_plugins,
    install_skill,
    make_harness_factory,
    missing_profile_plugins,
)
from .market_data.alphavantage import AlphaVantageProvider
from .market_data.yfinance_provider import YFinanceProvider
from .mcp_server import build_server
from .mcp_tools import ToolDeps, prepare_brief
from .orchestrator import OrchestratorError, run_day
from .state_store import load_state, save_state, seed_state
from .tracing import get_tracer, remote_parent_context, setup_tracing, shutdown_tracing

NY = ZoneInfo("America/New_York")
CLOSE_BUFFER = dt.time(16, 30)


def default_day(now: dt.datetime | None = None) -> dt.date:
    """Most recent US session that has closed (weekends roll back to Friday; holidays are not modeled)."""
    now = (now or dt.datetime.now(NY)).astimezone(NY)
    day = now.date() if now.time() >= CLOSE_BUFFER else now.date() - dt.timedelta(days=1)
    while day.weekday() >= 5:
        day -= dt.timedelta(days=1)
    return day


def state_path(settings: Settings) -> Path:
    return settings.state_dir / "NVDA.json"


def report_path_for(settings: Settings, day: dt.date) -> Path:
    """Where dsh is told to write the day's report; the driver verifies this exact path afterwards."""
    ticker = load_state(state_path(settings)).ticker if state_path(settings).exists() else "NVDA"
    return settings.reports_dir / f"{day}-{ticker}.md"


def market_provider(settings: Settings):
    """Alpha Vantage (paid, EOD, any date) is the single source; anything it lacks stays N/A for the
    Hermes gap fill. yfinance is used only when no Alpha Vantage key is configured."""
    if settings.alphavantage_api_key:
        return AlphaVantageProvider(settings.alphavantage_api_key)
    return YFinanceProvider()


def tool_deps(settings: Settings) -> ToolDeps:
    return ToolDeps(market=market_provider(settings), state_path=state_path(settings))


def serve_stdio(server) -> int:
    asyncio.run(server.run_stdio_async())
    return 0


ACP_HINT = ("hermes-acp cannot start — in $HERMES_REPO run: "
            "uv pip install 'agent-client-protocol==0.9.0' arize-otel")


def doctor_checks(settings: Settings, client: httpx.Client,
                  runner: Any = subprocess.run) -> list[tuple[str, bool, bool, str]]:
    checks: list[tuple[str, bool, bool, str]] = []
    missing = set(missing_required(settings))
    checks.append(("env XAI_API_KEY", "XAI_API_KEY" not in missing, True,
                   "set" if "XAI_API_KEY" not in missing else "missing in .env"))
    for name, value, why in (("ARIZE_SPACE_ID", settings.arize_space_id, "tracing"),
                             ("ARIZE_API_KEY", settings.arize_api_key, "tracing"),
                             ("TAVILY_API_KEY", settings.tavily_api_key, "web search for dsh and Hermes"),
                             ("EXA_API_KEY", settings.exa_api_key, "web search fallback"),
                             ("ALPHAVANTAGE_API_KEY", settings.alphavantage_api_key,
                              "primary market data (EOD closes + options)")):
        checks.append((f"env {name}", bool(value), False, f"set ({why})" if value else f"not set — {why} unavailable"))
    state = state_path(settings)
    checks.append(("state file", state.exists(), True, str(state) if state.exists() else "run `bubble-watch seed`"))
    checks.append(("dsh launcher", os.access(settings.dsh_bin, os.X_OK), True, settings.dsh_bin))
    home = ensure_dsh_home(settings.dsh_home, settings.dsh_model, settings.dsh_provider, settings.xai_base_url)
    checks.append(("dsh home", True, True, str(home)))
    acp_ok = os.access(settings.hermes_acp_bin, os.X_OK)
    checks.append(("Hermes ACP launcher", acp_ok, True,
                   settings.hermes_acp_bin if acp_ok else f"{settings.hermes_acp_bin} not executable"))
    skill = install_skill(settings)
    checks.append(("bubble-watch skill", skill.exists(), True, str(skill)))
    if os.access(settings.hermes_acp_bin, os.X_OK):
        # `hermes-acp --check` imports the adapter and its protocol package: the one failure that
        # otherwise surfaces only mid-run, as a delegation that exits 1 at ACP initialize.
        probe = runner([settings.hermes_acp_bin, "--check"], capture_output=True, text=True,
                       check=False, env={**os.environ, "HERMES_HOME": str(settings.hermes_home)})
        ok = probe.returncode == 0
        checks.append(("Hermes ACP runtime", ok, True, "check OK" if ok else ACP_HINT))
    missing = missing_profile_plugins(settings)
    checks.append(("dsh profile plugins", not missing, True,
                   "installed" if not missing
                   else f"missing {', '.join(missing)} — run `bubble-watch install-plugins`"))

    if settings.xai_api_key:
        try:
            r = client.get(settings.xai_base_url.rstrip("/") + "/models",
                           headers={"Authorization": f"Bearer {settings.xai_api_key}"})
            ids = {m.get("id") for m in r.json().get("data", [])} if r.status_code == 200 else set()
            detail = "served" if settings.dsh_model in ids else (
                f"HTTP {r.status_code}" if r.status_code != 200 else
                f"not served; grok models: {', '.join(sorted(i for i in ids if i and 'grok' in i))}")
            checks.append((f"xAI model {settings.dsh_model}", settings.dsh_model in ids, True, detail))
        except httpx.HTTPError as exc:
            checks.append(("xAI API", False, True, f"unreachable: {type(exc).__name__}"))
    return checks


def cmd_seed(args, settings: Settings) -> int:
    path = state_path(settings)
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


def cmd_install_plugins(args, settings: Settings) -> int:
    """Install the dsh plugins the orchestration composition needs but the base bundle lacks.

    A separate command because it needs pnpm and the network; a run never does this implicitly.
    """
    ensure_dsh_home(settings.dsh_home, settings.dsh_model, settings.dsh_provider, settings.xai_base_url)
    try:
        installed = install_profile_plugins(settings)
    except RuntimeError as exc:
        print(f"{exc}")
        return 1
    for name in installed:
        print(f"installed {name}")
    if not installed:
        print("already installed")
    return 0


def cmd_mcp(args, settings: Settings) -> int:
    """Serve the deterministic tools on stdio. dsh's MCP client spawns this; stdout is the transport.

    Tracing is set up here too: this process owns the LangGraph spans, and joins the run's trace
    through the TRACEPARENT the orchestration patch passes down.
    """
    provider = setup_tracing(settings)
    parent = remote_parent_context()
    token = context.attach(parent) if parent is not None else None
    try:
        return serve_stdio(build_server(tool_deps(settings)))
    finally:
        if token is not None:
            context.detach(token)
        shutdown_tracing(provider)


def _run_without_agents(settings: Settings, day: dt.date) -> int:
    """Data and signals only: the MCP tools called directly, with no dsh and no LLM anywhere."""
    print(json.dumps(prepare_brief(tool_deps(settings), day.isoformat()), indent=2, ensure_ascii=False))
    return 0


def cmd_run(args, settings: Settings) -> int:
    path = state_path(settings)
    if not path.exists():
        print("no state file; run `bubble-watch seed` first")
        return 1
    day = args.date or default_day()
    if args.no_agents:
        return _run_without_agents(settings, day)
    if missing := missing_required(settings):
        print(f"missing required settings: {', '.join(missing)} (set them in .env)")
        return 1

    report_path = report_path_for(settings, day)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.read_bytes() if args.dry_run else None
    provider = setup_tracing(settings)
    try:
        outcome = run_day(day=day, report_path=report_path, tracer=get_tracer(provider),
                          harness_for=lambda traceparent: make_harness_factory(settings, traceparent)())
    except OrchestratorError as exc:
        print(f"run failed: {exc}")
        return 1
    finally:
        if backup is not None:
            path.write_bytes(backup)  # --dry-run: dsh saved state itself, so undo it
        shutdown_tracing(provider)
    print(f"report: {outcome.report_path}")
    print(f"dsh session {outcome.session_id}, {outcome.tool_call_count} tool calls")
    if args.dry_run:
        print("dry run: state file restored")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bubble-watch", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    seed = sub.add_parser("seed", help="write the initial state from the 9/11–9/17 reports")
    seed.add_argument("--force", action="store_true")
    sub.add_parser("doctor", help="check keys (names only), dsh, the Hermes ACP launcher and the xAI model")
    run = sub.add_parser("run", help="run one trading day with dsh as the orchestrator")
    run.add_argument("--date", type=dt.date.fromisoformat, help="YYYY-MM-DD (default: last closed session)")
    run.add_argument("--dry-run", action="store_true", help="write the report but restore the state file")
    run.add_argument("--no-agents", action="store_true", help="data + signals only; no dsh, no LLM calls")
    sub.add_parser("mcp", help="serve the deterministic tools over MCP stdio (dsh spawns this)")
    sub.add_parser("install-plugins", help="install the dsh plugins the base bundle lacks (needs pnpm)")
    args = parser.parse_args(argv)
    settings = load_settings()
    commands = {"seed": cmd_seed, "doctor": cmd_doctor, "run": cmd_run, "mcp": cmd_mcp,
                "install-plugins": cmd_install_plugins}
    return commands[args.cmd](args, settings)


if __name__ == "__main__":
    sys.exit(main())
