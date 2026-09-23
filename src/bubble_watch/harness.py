"""Composing and launching DeepSeek Harness as the orchestrator.

dsh runs the day: it calls the deterministic tools over MCP, delegates research to the Hermes ACP
subagent, forms its own view, reconciles the two and writes the report. This module owns everything
about *how* dsh is composed and launched; `orchestrator.py` owns what it is asked to do.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, Settings

SEARCH_PATCH_TEMPLATE = PROJECT_ROOT / "dsh" / "web-search.patch.yml"
ORCHESTRATOR_PATCH_TEMPLATE = PROJECT_ROOT / "dsh" / "orchestrator.patch.yml"
CONTAINERS_PATCH_TEMPLATE = PROJECT_ROOT / "dsh" / "orchestrator.containers.patch.yml"
#: The project root as every container sees it (images are built with this as WORKDIR).
CONTAINER_WORKDIR = "/work"
#: Where Dockerfile.dsh-runner puts the dsh checkout inside its image.
CONTAINER_DSH_REPO = "/dsh"
SKILL_SOURCE = PROJECT_ROOT / "dsh" / "skills" / "bubble-watch" / "SKILL.md"
TAVILY_PLUGIN = PROJECT_ROOT / "dsh" / "plugins" / "web-search-tavily.mjs"
EXA_PLUGIN_ENTRY = "packages/web/web-search-exa/lib/index.js"  # inside the dsh checkout

_SETTINGS_TEMPLATE = """\
# Written by bubble-watch: registers xAI (Grok) for the dsh orchestrator through the llm-pi-ai adapter.
llm-pi-ai:
  providers:
    {provider}:
      apiKeyEnv: XAI_API_KEY
      api: openai-completions
      baseURL: {base_url}
      models:
        - id: {model}
"""


_HERMES_CONFIG = """\
# Written by bubble-watch: isolated Hermes home for the analyst dsh delegates to over ACP.
model:
  default: {model}
  provider: xai
# The analyst only researches: web_search + web_extract. No terminal toolset means no shell
# commands to pre-scan, so the tirith scanner (auto-downloaded from GitHub) is not needed.
platform_toolsets:
  cli: [web]
  acp: [web]
  api_server: [web]
security:
  tirith_enabled: false
# Hermes traces its own turn/LLM/tool spans into this run's trace (HERMES_ARIZE_TRACEPARENT).
plugins:
  enabled:
    - observability/arize
"""

# Messaging platforms read their credentials from env; never let this home bring a bot online.
_PLATFORM_ENV_PREFIXES = ("TELEGRAM_", "DISCORD_", "SLACK_", "WHATSAPP_", "SIGNAL_", "MATRIX_", "MATTERMOST_",
                          "BLUEBUBBLES_", "WEIXIN_", "YUANBAO_", "QQBOT_", "TEAMS_", "MSGRAPH_", "EMAIL_", "SMS_")


def ensure_hermes_home(home: Path, model: str) -> Path:
    """Create the isolated HERMES_HOME with a Grok, web-only config; never overwrite an existing one.

    The user's own ~/.hermes and its messaging platforms are never touched.
    """
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    cfg = home / "config.yaml"
    if not cfg.exists():
        cfg.write_text(_HERMES_CONFIG.format(model=model))
    return home


def child_env(settings: Settings, traceparent: str) -> dict[str, str]:
    """The environment dsh is launched with, and through it both children.

    Every value the ACP and MCP composition rows look up with `!!js process.env` must be here:
    those backends scrub credential-shaped ambient variables, so inheritance alone is not enough.
    Hermes' Arize plugin reads `HERMES_ARIZE_*`, the Python side reads the OTel/Arize names, so the
    same traceparent is published under both.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(_PLATFORM_ENV_PREFIXES)}
    env["XAI_API_KEY"] = settings.xai_api_key
    env["TIRITH_ENABLED"] = "false"
    for name, value in (("TAVILY_API_KEY", settings.tavily_api_key), ("EXA_API_KEY", settings.exa_api_key),
                        ("ALPHAVANTAGE_API_KEY", settings.alphavantage_api_key)):
        if value:
            env[name] = value
    if traceparent:
        env["TRACEPARENT"] = env["HERMES_ARIZE_TRACEPARENT"] = traceparent
    if settings.arize_space_id and settings.arize_api_key:
        arize = {"SPACE_ID": settings.arize_space_id, "API_KEY": settings.arize_api_key,
                 "PROJECT_NAME": settings.arize_project, "COLLECTOR_ENDPOINT": settings.arize_endpoint}
        for suffix, value in arize.items():
            # Same Arize project for all three runtimes: spans only join one trace within a project.
            env[f"ARIZE_{suffix}"] = env[f"HERMES_ARIZE_{suffix}"] = value
    return env


def ensure_dsh_home(dsh_home: str, model: str, provider: str = "xai",
                    base_url: str = "https://api.x.ai/v1") -> Path:
    """Create DSH_HOME with an xAI provider settings.yaml; never overwrite an existing one."""
    home = Path(dsh_home)
    home.mkdir(parents=True, exist_ok=True)
    path = home / "settings.yaml"
    if not path.exists():
        path.write_text(_SETTINGS_TEMPLATE.format(provider=provider, base_url=base_url, model=model))
    return path


def _render(template: Path, destination: Path, values: dict[str, str]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = template.read_text()
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    destination.write_text(text)
    return destination


def render_search_patch(dsh_home: str, provider: str, plugin_entry: Path, api_key_env: str) -> Path:
    """Write the web-search overlay into DSH_HOME for one provider (see dsh/web-search.patch.yml)."""
    return _render(SEARCH_PATCH_TEMPLATE, Path(dsh_home) / "web-search.patch.yml",
                   {"provider": provider, "plugin_entry": str(plugin_entry), "api_key_env": api_key_env})


#: Variables each child row may declare. A name is emitted only when `child_env` actually sets it.
HERMES_ENV_NAMES = ("XAI_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY", "TIRITH_ENABLED",
                    "HERMES_ARIZE_TRACEPARENT", "HERMES_ARIZE_SPACE_ID", "HERMES_ARIZE_API_KEY",
                    "HERMES_ARIZE_PROJECT_NAME", "HERMES_ARIZE_COLLECTOR_ENDPOINT")
MCP_ENV_NAMES = ("TRACEPARENT", "ARIZE_SPACE_ID", "ARIZE_API_KEY", "ARIZE_PROJECT_NAME",
                 "ARIZE_COLLECTOR_ENDPOINT", "ALPHAVANTAGE_API_KEY")


def _env_block(names: tuple[str, ...], available: dict[str, str]) -> str:
    """One row's `env:` mapping, holding only the names that have a value.

    Declaring an unset name would emit `!!js process.env.NAME` -> `undefined`, which fails
    dsh-mcp-client's union config and silently drops the whole plugin.
    """
    present = [name for name in names if available.get(name)]
    if not present:
        return "        env: {}"
    lines = [f"          {name}: !!js process.env.{name}" for name in present]
    return "\n".join(["        env:", *lines])


def in_containers(settings: Settings) -> bool:
    return settings.deploy_mode == "containers"


def launch_paths(settings: Settings) -> tuple[str, str, tuple[str, ...]]:
    """(DSH_HOME, workspace cwd, patch paths) as the dsh process will see them.

    In containers mode these are paths inside the dsh container, because the SDK passes them as
    argv to a `dsh` that is really `docker run -i`. The files themselves are written on the host,
    into the directory that is mounted there.
    """
    names = [Path(p).name for p in _patch_names(settings)]
    if in_containers(settings):
        home = f"{CONTAINER_WORKDIR}/.dsh"
        return home, CONTAINER_WORKDIR, tuple(f"{home}/{name}" for name in names)
    return settings.dsh_home, str(PROJECT_ROOT), tuple(str(Path(settings.dsh_home) / n) for n in names)


def _patch_names(settings: Settings) -> tuple[str, ...]:
    """The patch layers this composition uses, in application order."""
    search = ("web-search.patch.yml",) if (settings.tavily_api_key or settings.exa_api_key) else ()
    return (*search, "orchestrator.patch.yml")


def render_orchestrator_patch(settings: Settings, traceparent: str = "") -> Path:
    """Write the orchestration overlay into DSH_HOME: the Hermes ACP subagent and the MCP tool server.

    Both modes produce the same file name, so the composition is selected by how bubble-watch is
    deployed rather than by the dsh profile.
    """
    destination = Path(settings.dsh_home) / "orchestrator.patch.yml"
    available = child_env(settings, traceparent)
    env_values = {"hermes_env": _env_block(HERMES_ENV_NAMES, available),
                  "mcp_env": _env_block(MCP_ENV_NAMES, available)}
    if in_containers(settings):
        return _render(CONTAINERS_PATCH_TEMPLATE, destination, {
            **env_values,
            "hermes_image": settings.hermes_image,
            "mcp_image": settings.mcp_image,
            "host_root": settings.host_root,
            "workdir": CONTAINER_WORKDIR,
        })
    return _render(ORCHESTRATOR_PATCH_TEMPLATE, destination, {
        **env_values,
        "hermes_acp_bin": settings.hermes_acp_bin,
        "hermes_home": str(settings.hermes_home),
        "project_root": str(PROJECT_ROOT),
        "uv_bin": shutil.which("uv") or "uv",
    })


#: Plugins the orchestration composition needs that dsh's base bundle does not carry. They are
#: installed into the profile package tree, where their peer imports reach the bundle; loading them
#: from the checkout by path instead gives a second copy whose config validation does not match.
PROFILE_PLUGINS = ("@deepseek-ai/dsh-mcp-client", "@deepseek-ai/dsh-subagent-acp")
PROFILE = "sdk"


def _profile_manifest(settings: Settings) -> Path:
    return settings.dsh_home_path / "profiles" / PROFILE / "package.json"


def installed_profile_plugins(settings: Settings) -> set[str]:
    path = _profile_manifest(settings)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()).get("dependencies", {}))
    except json.JSONDecodeError:
        return set()


def missing_profile_plugins(settings: Settings) -> list[str]:
    installed = installed_profile_plugins(settings)
    return [name for name in PROFILE_PLUGINS if name not in installed]


def dsh_version(settings: Settings, runner: Any = subprocess.run) -> str:
    """The version of the dsh build in use; profile plugins are pinned to match it."""
    result = runner([settings.dsh_bin, "--version"], capture_output=True, text=True, check=False)
    return (result.stdout or "").strip()


def install_profile_plugins(settings: Settings, version: str = "", runner: Any = subprocess.run) -> list[str]:
    """Install the missing profile plugins, pinned to the dsh build's own version.

    Returns the plugins installed. Needs pnpm and network access; this is a setup step, not
    something a run does, so it is never called implicitly.
    """
    missing = missing_profile_plugins(settings)
    version = version or dsh_version(settings, runner)
    for name in missing:
        cmd = [settings.dsh_bin, "plugin", "--profile", PROFILE, "add", f"{name}@{version}"]
        result = runner(cmd, capture_output=True, text=True, check=False,
                        env={**os.environ, "DSH_HOME": settings.dsh_home})
        output = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
        # `dsh plugin` forwards pnpm's exit code unreliably, so treat pnpm's own error as failure.
        if result.returncode != 0 or "ERR_PNPM" in output:
            raise RuntimeError(f"installing {name} failed: {output[-600:]}")
    return missing


def install_skill(settings: Settings) -> Path:
    """Copy the tracked procedure into DSH_HOME/skills, where dsh's local provider discovers it.

    The tracked source is the single copy that is edited; DSH_HOME is gitignored, so it is refreshed
    on every run rather than treated as user state.
    """
    destination = Path(settings.dsh_home) / "skills" / "bubble-watch" / "SKILL.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(SKILL_SOURCE.read_text())
    return destination


def search_patches(settings: Settings) -> tuple[tuple[str, ...], dict[str, str]]:
    """(patch files, env) for dsh web_search: Tavily if TAVILY_API_KEY, else Exa, else none (dsh's
    default search needs a DeepSeek key, so the orchestrator would have web_fetch only)."""
    if settings.tavily_api_key:
        patch = render_search_patch(settings.dsh_home, "tavily", TAVILY_PLUGIN, "TAVILY_API_KEY")
        return (str(patch),), {"TAVILY_API_KEY": settings.tavily_api_key}
    if settings.exa_api_key:
        entry = Path(settings.dsh_repo).expanduser() / EXA_PLUGIN_ENTRY
        patch = render_search_patch(settings.dsh_home, "exa", entry, "EXA_API_KEY")
        return (str(patch),), {"EXA_API_KEY": settings.exa_api_key}
    return (), {}


def orchestrator_patches(settings: Settings, traceparent: str = "") -> tuple[tuple[str, ...], dict[str, str]]:
    """Every patch layer the orchestrator needs, in application order, plus the env they reference.

    Search comes first so the orchestration layer can rely on `web` already being configured.
    """
    patches, env = search_patches(settings)
    return (*patches, str(render_orchestrator_patch(settings, traceparent))), env


def make_harness_factory(settings: Settings, traceparent: str = "") -> Callable[[], Any]:
    """Build the dsh runtime. `traceparent` is read by the composition's `!!js process.env` lookups,
    so both children parent their spans on this run's root span."""

    def factory() -> Any:
        from deepseek_harness import DeepSeekHarness

        ensure_dsh_home(settings.dsh_home, settings.dsh_model, settings.dsh_provider, settings.xai_base_url)
        ensure_hermes_home(settings.hermes_home, settings.hermes_model or settings.dsh_model)
        install_skill(settings)
        orchestrator_patches(settings, traceparent)   # render every layer onto the host
        dsh_home, cwd, patches = launch_paths(settings)   # …then name them as dsh will see them
        env = child_env(settings, traceparent)
        return DeepSeekHarness(
            dsh_bin=settings.dsh_bin, dsh_home=dsh_home, cwd=cwd,
            provider=settings.dsh_provider, model=settings.dsh_model, patches=patches, env=env,
            initialize_timeout_seconds=120, request_timeout_seconds=settings.analyst_timeout_s)

    return factory
