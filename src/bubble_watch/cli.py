"""`bubble-watch` command line: seed, doctor, run."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx

from .agents.dsh_client import DshAnalyst, ensure_dsh_home, make_harness_factory
from .agents.hermes_cli import HermesCliAnalyst, ensure_hermes_home, hermes_env
from .agents.hermes_client import HermesAnalyst
from .config import DEFAULT_MODEL, Settings, load_settings, missing_required, resolve_hermes_key
from .graph import Deps, build_graph
from .market_data.alphavantage import AlphaVantageProvider
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
    checks.append(("env XAI_API_KEY", "XAI_API_KEY" not in missing, True,
                   "set" if "XAI_API_KEY" not in missing else "missing in .env"))
    for name, value, why in (("ARIZE_SPACE_ID", settings.arize_space_id, "tracing"),
                             ("ARIZE_API_KEY", settings.arize_api_key, "tracing"),
                             ("TAVILY_API_KEY", settings.tavily_api_key, "web search for both analysts"),
                             ("EXA_API_KEY", settings.exa_api_key, "web search fallback"),
                             ("ALPHAVANTAGE_API_KEY", settings.alphavantage_api_key, "primary market data (EOD closes + options)")):
        checks.append((f"env {name}", bool(value), False, f"set ({why})" if value else f"not set — {why} unavailable"))
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
    if settings.hermes_mode != "gateway":
        path = shutil.which(settings.hermes_bin)
        checks.append(("Hermes CLI (oneshot)", path is not None, True, path or f"{settings.hermes_bin} not on PATH"))
        return checks
    try:
        r = client.get(settings.hermes_api_url.rstrip("/") + "/models",
                       headers={"Authorization": f"Bearer {resolve_hermes_key(settings)}"})
        checks.append(("Hermes gateway", r.status_code == 200, True,
                       settings.hermes_api_url if r.status_code == 200 else f"HTTP {r.status_code} (gateway started with a different key? restart it)"))
    except httpx.HTTPError:
        checks.append(("Hermes gateway", False, True,
                       f"not reachable at {settings.hermes_api_url} — start it with `uv run bubble-watch hermes-gateway`"))
    return checks





def hermes_gateway_env(settings: Settings, home: Path) -> dict[str, str]:
    """hermes_env plus the API server (for HERMES_MODE=gateway)."""
    return {**hermes_env(settings, home), "API_SERVER_ENABLED": "true", "API_SERVER_KEY": resolve_hermes_key(settings),
            "API_SERVER_PORT": str(urlparse(settings.hermes_api_url).port or 8642)}


def cmd_hermes_gateway(args, settings: Settings) -> int:
    missing = missing_required(settings)
    if missing:
        print(f"missing required settings: {', '.join(missing)} (set them in .env)")
        return 1
    home = ensure_hermes_home(settings.hermes_home, DEFAULT_MODEL)
    print(f"starting Hermes analyst gateway (HERMES_HOME={home}, API server only) at {settings.hermes_api_url}")
    os.execvpe("hermes", ["hermes", "gateway", "run"], hermes_gateway_env(settings, home))
    return 0  # unreachable: execvpe replaces the process


def market_provider(settings: Settings):
    """Alpha Vantage (paid, EOD, any date) is the single source; anything it lacks stays N/A for the
    Hermes gap fill. yfinance is used only when no Alpha Vantage key is configured."""
    if settings.alphavantage_api_key:
        return AlphaVantageProvider(settings.alphavantage_api_key)
    return YFinanceProvider()


def hermes_analyst(settings: Settings):
    if settings.hermes_mode == "gateway":
        return HermesAnalyst(settings.hermes_api_url, resolve_hermes_key(settings), model=settings.hermes_model,
                             timeout=settings.analyst_timeout_s)
    home = ensure_hermes_home(settings.hermes_home, DEFAULT_MODEL)
    return HermesCliAnalyst(model=settings.hermes_model or DEFAULT_MODEL, env=hermes_env(settings, home),
                            timeout=settings.analyst_timeout_s, hermes_bin=settings.hermes_bin, workdir=home)


def build_deps(settings: Settings, tracer, *, agents: bool, save: bool) -> Deps:
    hermes = hermes_analyst(settings) if agents else None
    analysts = {"hermes": hermes, "dsh": DshAnalyst(make_harness_factory(settings))} if agents else {}
    return Deps(
        market=market_provider(settings), analysts=analysts,
        writer=XaiWriter(settings.xai_api_key, settings.writer_model, settings.xai_base_url) if agents else None,
        tracer=tracer, reports_dir=settings.reports_dir, state_path=_state_path(settings), gap_filler=hermes,
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
    sub.add_parser("hermes-gateway", help="run the Hermes analyst gateway (isolated home, API server only)")
    args = parser.parse_args(argv)
    settings = load_settings()
    commands = {"seed": cmd_seed, "doctor": cmd_doctor, "run": cmd_run, "hermes-gateway": cmd_hermes_gateway}
    return commands[args.cmd](args, settings)


if __name__ == "__main__":
    sys.exit(main())
