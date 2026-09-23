"""Hermes Agent as a tool, driven over its one-shot CLI (`hermes chat -Q`), one process per call.

**Why the CLI and not ACP.** Hermes' ACP adapter builds an `AIAgent` directly and never touches
`hermes_cli`'s plugin dispatch, so no plugin hook fires over that path — including
`observability/arize`. Driven over the CLI, Hermes traces itself. Verified by running both paths
with an identical home, environment, credentials and traceparent: `hermes chat -Q` emitted
`Hermes turn` + `LLM call 1` correctly parented; `hermes-acp` emitted nothing.

This lives in the MCP server process, which owns a real OTel context, so each call gets its own
AGENT span and Hermes' spans nest under *that* rather than under the run root.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from opentelemetry import trace

from .config import Settings
from .tracing import OUTPUT, agent_span

# Messaging platforms read their credentials from env; never let this home bring a bot online.
PLATFORM_ENV_PREFIXES = ("TELEGRAM_", "DISCORD_", "SLACK_", "WHATSAPP_", "SIGNAL_", "MATRIX_",
                         "MATTERMOST_", "BLUEBUBBLES_", "WEIXIN_", "YUANBAO_", "QQBOT_", "TEAMS_",
                         "MSGRAPH_", "EMAIL_", "SMS_")
SESSION_ID = re.compile(r"session_id:\s*(\S+)")

SYSTEM_ANALYST = (
    "You are an independent senior analyst on the 'NVDA Bubble Signal Watch'. You judge "
    "independently of the orchestrator. The figures you are given are computed values: never "
    "recompute them, and never invent a price, return or IV. Your job is to (1) research the "
    "requested values or catalysts on the web, (2) interpret the computed tape signals, and "
    "(3) answer in the format requested. Attach a URL you actually opened to every claim. "
    "Answer in English unless the task explicitly asks otherwise."
)


_HERMES_CONFIG = """\
# Written by bubble-watch: isolated Hermes home for the analyst the tool server calls.
model:
  default: {model}
  provider: xai
# The analyst only researches: web_search + web_extract. No terminal toolset means no shell
# commands to pre-scan, so the tirith scanner (auto-downloaded from GitHub) is not needed.
platform_toolsets:
  cli: [web]
security:
  tirith_enabled: false
# Hermes traces its own turn/LLM/tool spans into this run's trace (HERMES_ARIZE_TRACEPARENT).
plugins:
  enabled:
    - observability/arize
"""


def ensure_hermes_home(home: Any, model: str) -> Path:
    """Create the isolated HERMES_HOME with a Grok, web-only config; never overwrite an existing one.

    The user's own ~/.hermes and its messaging platforms are never touched.
    """
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    cfg = home / "config.yaml"
    if not cfg.exists():
        cfg.write_text(_HERMES_CONFIG.format(model=model))
    return home


class HermesError(RuntimeError):
    """The Hermes analyst could not be reached, or returned nothing usable."""


def hermes_env(settings: Settings) -> dict[str, str]:
    """Process env for Hermes: isolated home, tirith off, keys, messaging variables stripped."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(PLATFORM_ENV_PREFIXES)}
    env.update(HERMES_HOME=str(settings.hermes_home), XAI_API_KEY=settings.xai_api_key,
               TIRITH_ENABLED="false",
               # Without this the watchdog kills every call after ~12s of stream silence, which
               # a reasoning model spends thinking; the turn then fails after three retries.
               HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS=str(int(settings.hermes_stream_idle_s)))
    if settings.arize_space_id and settings.arize_api_key:
        # Same Arize project as the orchestrator: spans only join one trace within a project.
        env.update(HERMES_ARIZE_SPACE_ID=settings.arize_space_id,
                   HERMES_ARIZE_API_KEY=settings.arize_api_key,
                   HERMES_ARIZE_PROJECT_NAME=settings.arize_project,
                   HERMES_ARIZE_COLLECTOR_ENDPOINT=settings.arize_endpoint)
    # Hermes auto-selects its web_search backend from these keys (Tavily ranks before Exa).
    for name, value in (("TAVILY_API_KEY", settings.tavily_api_key), ("EXA_API_KEY", settings.exa_api_key)):
        if value:
            env[name] = value
    return env


def _traceparent(span: trace.Span) -> dict[str, str]:
    """The W3C traceparent of `span`, under the name Hermes' arize plugin reads."""
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return {}
    flags = "01" if ctx.trace_flags.sampled else "00"
    return {"HERMES_ARIZE_TRACEPARENT": f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{flags}"}


def hermes_analyst(settings: Settings, task: str, session_id: str = "", *,
                   tracer: trace.Tracer | None = None, runner: Any = subprocess.run) -> dict[str, Any]:
    """Ask the Hermes analyst one question and return its answer plus its session id.

    Pass the returned `session_id` back to continue the same Hermes conversation — that is how the
    rebuttal round shows it the other view without repeating the whole brief.
    """
    tracer = tracer or trace.get_tracer(__name__)
    timeout = settings.analyst_timeout_s
    # One-shot mode has no system-prompt flag: the role leads the first query of a session.
    query = task if session_id else f"{SYSTEM_ANALYST}\n\n{task}"
    home = ensure_hermes_home(settings.hermes_home, settings.hermes_model or settings.dsh_model)

    with agent_span(tracer, "hermes analyst", input_value=task,
                    attributes={"hermes.session_id": session_id or None}) as span:
        env = {**hermes_env(settings), **_traceparent(span)}
        with tempfile.NamedTemporaryFile("w", suffix=".md", dir=home, delete=False) as handle:
            handle.write(query)
        cmd = [settings.hermes_bin, "chat", "--query-file", handle.name, "-Q", "-t", "web",
               "--provider", settings.dsh_provider, "-m", settings.hermes_model or settings.dsh_model,
               "--ignore-rules", "--run-budget", str(int(timeout))]
        if session_id:
            cmd += ["--resume", session_id]
        try:
            proc = runner(cmd, capture_output=True, text=True, env=env, timeout=timeout + 60, check=False)
        except subprocess.TimeoutExpired as exc:
            raise HermesError(f"hermes timed out after {int(timeout) + 60}s") from exc
        except FileNotFoundError as exc:
            raise HermesError(f"hermes executable not found: {settings.hermes_bin}") from exc
        finally:
            Path(handle.name).unlink(missing_ok=True)

        if proc.returncode != 0:
            # Hermes prints "session_id: …" last, so a tail alone hides the error.
            lines = [line for line in (proc.stderr or "").splitlines()
                     if line.strip() and not line.startswith("session_id:")]
            detail = " | ".join(lines[-6:])[:600] or (proc.stdout or "")[-300:].strip() or "no stderr"
            raise HermesError(f"hermes exited {proc.returncode}: {detail}")
        text = (proc.stdout or "").strip()
        if not text:
            raise HermesError("hermes returned no answer")
        match = SESSION_ID.search(proc.stderr or "")
        resolved = match.group(1) if match else session_id
        span.set_attribute(OUTPUT, text[:16_000])
        if resolved:
            span.set_attribute("hermes.session_id", resolved)
        return {"text": text, "session_id": resolved}
