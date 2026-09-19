"""Runtime settings from the environment (.env). Secrets are never logged: callers get names only."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "grok-4.6"
REQUIRED = ("XAI_API_KEY",)


@dataclass(frozen=True)
class Settings:
    xai_api_key: str
    xai_base_url: str
    hermes_api_url: str
    hermes_api_key: str
    hermes_model: str | None
    hermes_home: Path
    hermes_mode: str
    hermes_bin: str
    dsh_bin: str
    dsh_repo: str
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
    tavily_api_key: str
    exa_api_key: str
    alphavantage_api_key: str


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, default).strip()
    # python-dotenv reads `KEY=   # comment` (empty value + inline comment) as the comment text.
    return default if value.startswith("#") else value


def load_settings(env_file: str | Path | None = None) -> Settings:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    return Settings(
        xai_api_key=_env("XAI_API_KEY"),
        xai_base_url=_env("XAI_BASE_URL", "https://api.x.ai/v1"),
        hermes_api_url=_env("HERMES_API_URL", "http://localhost:8642/v1"),
        hermes_api_key=_env("HERMES_API_KEY"),
        # Unset: the Hermes gateway uses its own configured model (set it to grok-4.6 there).
        hermes_model=_env("HERMES_ANALYST_MODEL") or None,
        hermes_home=Path(_env("HERMES_ANALYST_HOME", str(PROJECT_ROOT / ".hermes-analyst"))),
        # oneshot: a `hermes chat` subprocess per call; gateway: HTTP to `bubble-watch hermes-gateway` (e.g. on EC2)
        hermes_mode=_env("HERMES_MODE", "oneshot"),
        hermes_bin=_env("HERMES_BIN", "hermes"),
        dsh_bin=_env("DSH_BIN", str(PROJECT_ROOT / "bin" / "dsh")),
        dsh_repo=_env("DSH_REPO", str(Path.home() / "projects" / "deepseek-harness")),
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
        tavily_api_key=_env("TAVILY_API_KEY"),
        exa_api_key=_env("EXA_API_KEY"),
        alphavantage_api_key=_env("ALPHAVANTAGE_API_KEY") or _env("ALPHA_VANTAGE_API_KEY"),
    )


def missing_required(settings: Settings) -> list[str]:
    return [name for name in REQUIRED if not {"XAI_API_KEY": settings.xai_api_key}[name]]


def resolve_hermes_key(settings: Settings) -> str:
    """The analyst gateway's API_SERVER_KEY: HERMES_API_KEY if set, else a random key generated once
    into <hermes_home>/api_server.key (0600). Shared by `hermes-gateway` and the Hermes client."""
    if settings.hermes_api_key:
        return settings.hermes_api_key
    path = settings.hermes_home / "api_server.key"
    if path.exists():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key + "\n")
    return key
