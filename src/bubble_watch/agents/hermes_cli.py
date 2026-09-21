"""Hermes Agent as an analyst via its one-shot CLI (`hermes chat --query-file … -Q`), one subprocess per call.

Runs in an isolated HERMES_HOME with only the web toolset: the user's ~/.hermes (and its messaging bots) is never
touched, and with no terminal tools the tirith pre-exec scanner is not needed (nor downloaded)."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..config import Settings
from ..market_data.base import GapFill
from .base import AnalystError, AnalystResult, AskResult, ask_for_view
from .hermes_client import parse_gap_fills
from .prompts import SYSTEM_ANALYST, SYSTEM_GAP_FILL

_HERMES_CONFIG = """\
# Written by bubble-watch: isolated Hermes home for the analyst (one-shot CLI, or the API-server gateway).
model:
  default: {model}
  provider: xai
# The analyst only researches: web_search + web_extract. No terminal toolset means no shell
# commands to pre-scan, so the tirith scanner (auto-downloaded from GitHub) is not needed.
platform_toolsets:
  cli: [web]
  api_server: [web]
security:
  tirith_enabled: false
# Hermes traces its own turn/LLM/tool spans into the caller's trace (HERMES_ARIZE_TRACEPARENT).
plugins:
  enabled:
    - observability/arize
"""
# Messaging platforms read their credentials from env; never let this home bring a bot online.
_PLATFORM_ENV_PREFIXES = ("TELEGRAM_", "DISCORD_", "SLACK_", "WHATSAPP_", "SIGNAL_", "MATRIX_", "MATTERMOST_",
                          "BLUEBUBBLES_", "WEIXIN_", "YUANBAO_", "QQBOT_", "TEAMS_", "MSGRAPH_", "EMAIL_", "SMS_")
_SESSION_ID = re.compile(r"session_id:\s*(\S+)")
_TRACEPARENT = "HERMES_ARIZE_TRACEPARENT"


def ensure_hermes_home(home: Path, model: str) -> Path:
    """Create the isolated HERMES_HOME with a Grok, web-only config; never overwrite an existing one."""
    home.mkdir(parents=True, exist_ok=True)
    cfg = home / "config.yaml"
    if not cfg.exists():
        cfg.write_text(_HERMES_CONFIG.format(model=model))
    return home


def hermes_env(settings: Settings, home: Path) -> dict[str, str]:
    """Process env for Hermes: isolated home, tirith off, model + search keys, messaging variables stripped."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_PLATFORM_ENV_PREFIXES)}
    env.update(HERMES_HOME=str(home), XAI_API_KEY=settings.xai_api_key, TIRITH_ENABLED="false")
    if settings.arize_space_id and settings.arize_api_key:
        # Same Arize project as the orchestrator: nested spans only join one trace within a project.
        env.update(HERMES_ARIZE_SPACE_ID=settings.arize_space_id, HERMES_ARIZE_API_KEY=settings.arize_api_key,
                   HERMES_ARIZE_PROJECT_NAME=settings.arize_project,
                   HERMES_ARIZE_COLLECTOR_ENDPOINT=settings.arize_endpoint)
    # Hermes auto-selects its web_search backend from these keys (Tavily ranks before Exa).
    for name, value in (("TAVILY_API_KEY", settings.tavily_api_key), ("EXA_API_KEY", settings.exa_api_key)):
        if value:
            env[name] = value
    return env


def _traceparent_env() -> dict[str, str]:
    """W3C traceparent for the active span, so the Hermes process traces into this run's trace."""
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return {}
    return {_TRACEPARENT: f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{'01' if ctx.trace_flags.sampled else '00'}"}


class HermesCliAnalyst:
    name = "hermes"

    def __init__(self, *, model: str, env: dict[str, str], timeout: float = 600.0, provider: str = "xai",
                 hermes_bin: str = "hermes", workdir: Path | None = None,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
        self._model, self._provider, self._bin = model, provider, hermes_bin
        self._env, self._timeout, self._run = env, timeout, runner
        self._workdir = workdir

    def ask(self, prompt: str, session_id: str | None = None, *, system: str = SYSTEM_ANALYST) -> AskResult:
        # No system-prompt flag in one-shot mode: the role leads the first query of a session.
        query = prompt if session_id else f"{system}\n\n{prompt}"
        with tempfile.NamedTemporaryFile("w", suffix=".md", dir=self._workdir, delete=False) as f:
            f.write(query)
        cmd = [self._bin, "chat", "--query-file", f.name, "-Q", "-t", "web", "--provider", self._provider,
               "-m", self._model, "--ignore-rules", "--run-budget", str(int(self._timeout))]
        env = {**self._env, **_traceparent_env()}
        if session_id:
            cmd += ["--resume", session_id]
        try:
            proc = self._run(cmd, capture_output=True, text=True, env=env, timeout=self._timeout + 60,
                             check=False)
        except subprocess.TimeoutExpired as exc:
            raise AnalystError(f"hermes timed out after {int(self._timeout) + 60}s") from exc
        except FileNotFoundError as exc:
            raise AnalystError(f"hermes executable not found: {self._bin}") from exc
        finally:
            Path(f.name).unlink(missing_ok=True)
        if proc.returncode != 0:
            # Hermes prints "session_id: …" last, so a tail alone hides the error: keep the lines that matter.
            lines = [line for line in (proc.stderr or "").splitlines()
                     if line.strip() and not line.startswith("session_id:")]
            detail = " | ".join(lines[-6:])[:600] or (proc.stdout or "")[-300:].strip() or "no stderr"
            raise AnalystError(f"hermes exited {proc.returncode}: {detail}")
        text = (proc.stdout or "").strip()
        if not text:
            raise AnalystError("hermes returned no answer")
        match = _SESSION_ID.search(proc.stderr or "")
        return AskResult(text=text, session_id=match.group(1) if match else session_id)

    def analyze(self, brief: str) -> AnalystResult:
        return ask_for_view(self.name, self.ask, brief)

    def rebut(self, prompt: str) -> AnalystResult:
        return ask_for_view(self.name, self.ask, prompt)

    def fill_gaps(self, prompt: str) -> list[GapFill]:
        return parse_gap_fills(self.ask(prompt, system=SYSTEM_GAP_FILL).text)
