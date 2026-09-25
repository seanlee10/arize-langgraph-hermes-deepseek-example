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
from .hermes_tool import PLATFORM_ENV_PREFIXES, ensure_hermes_home

SEARCH_PATCH_TEMPLATE = PROJECT_ROOT / "dsh" / "web-search.patch.yml"
NO_SEARCH_PATCH_TEMPLATE = PROJECT_ROOT / "dsh" / "no-web-search.patch.yml"
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



def child_env(settings: Settings, traceparent: str, session_id: str = "") -> dict[str, str]:
    """The environment dsh is launched with, and through it both children.

    Every value the MCP composition row looks up with `!!js process.env` must be here: the backend
    scrubs credential-shaped ambient variables, so inheritance alone is not enough.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(PLATFORM_ENV_PREFIXES)}
    env["XAI_API_KEY"] = settings.xai_api_key
    env["TIRITH_ENABLED"] = "false"
    for name, value in (("TAVILY_API_KEY", settings.tavily_api_key), ("EXA_API_KEY", settings.exa_api_key),
                        ("ALPHAVANTAGE_API_KEY", settings.alphavantage_api_key)):
        if value:
            env[name] = value
    if traceparent:
        env["TRACEPARENT"] = traceparent
    if session_id:
        # The tool server is a different process and cannot derive the grouping key itself.
        env["BUBBLE_WATCH_SESSION_ID"] = session_id
    if settings.arize_space_id and settings.arize_api_key:
        # Same Arize project for every runtime: spans only join one trace within a project. The
        # tool server derives Hermes' own HERMES_ARIZE_* names from these, per call.
        env.update(ARIZE_SPACE_ID=settings.arize_space_id, ARIZE_API_KEY=settings.arize_api_key,
                   ARIZE_PROJECT_NAME=settings.arize_project,
                   ARIZE_COLLECTOR_ENDPOINT=settings.arize_endpoint)
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


def render_search_patch(dsh_home: str, provider: str, plugin_entry: Any, api_key_env: str) -> Path:
    """Write the web-search overlay into DSH_HOME for one provider (see dsh/web-search.patch.yml)."""
    return _render(SEARCH_PATCH_TEMPLATE, Path(dsh_home) / "web-search.patch.yml",
                   {"provider": provider, "plugin_entry": str(plugin_entry), "api_key_env": api_key_env})


#: Variables the MCP row may declare. A name is emitted only when `child_env` actually sets it.
#: The server hosts the Hermes analyst too, so it needs Hermes' keys as well as the data ones.
MCP_ENV_NAMES = ("TRACEPARENT", "BUBBLE_WATCH_SESSION_ID", "ARIZE_SPACE_ID", "ARIZE_API_KEY",
                 "ARIZE_PROJECT_NAME", "ARIZE_COLLECTOR_ENDPOINT", "ALPHAVANTAGE_API_KEY",
                 "XAI_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY", "HERMES_REPO")


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


def launch_kwargs(settings: Settings) -> dict[str, Any]:
    """Where the harness *runs* versus what it is *told* its workspace is.

    These are the same directory locally and different in containers mode, and the SDK keeps them
    apart for exactly that reason: `runtime_cwd` is the host directory the launcher is spawned in,
    while `cwd` is the workspace path sent during `initialize`. Passing the container path as both
    makes the host `Popen` fail with FileNotFoundError before the harness ever starts.
    """
    dsh_home, cwd, patches = launch_paths(settings)
    return {"dsh_home": dsh_home, "cwd": cwd, "runtime_cwd": str(PROJECT_ROOT), "patches": patches}


def _patch_names(settings: Settings) -> tuple[str, ...]:
    """The patch layers this composition uses, in application order."""
    search = (("web-search.patch.yml",) if (settings.tavily_api_key or settings.exa_api_key)
              else ("no-web-search.patch.yml",))
    return (*search, "orchestrator.patch.yml")


def render_orchestrator_patch(settings: Settings, traceparent: str = "",
                              session_id: str = "") -> Path:
    """Write the orchestration overlay into DSH_HOME: the Hermes ACP subagent and the MCP tool server.

    Both modes produce the same file name, so the composition is selected by how bubble-watch is
    deployed rather than by the dsh profile.
    """
    destination = Path(settings.dsh_home) / "orchestrator.patch.yml"
    available = child_env(settings, traceparent, session_id)
    env_values = {"mcp_env": _env_block(MCP_ENV_NAMES, available)}
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
        "project_root": str(PROJECT_ROOT),
        "uv_bin": shutil.which("uv") or "uv",
    })


#: Plugins the orchestration composition needs that dsh's base bundle does not carry. They are
#: installed into the profile package tree, where their peer imports reach the bundle; loading them
#: from the checkout by path instead gives a second copy whose config validation does not match.
PROFILE_PLUGINS = ("@deepseek-ai/dsh-mcp-client",)
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
        # Loaded by absolute path from inside the dsh process, so the path has to be the one that
        # process sees: /work in a container, the checkout on the host.
        entry = (f"{CONTAINER_WORKDIR}/dsh/plugins/web-search-tavily.mjs"
                 if in_containers(settings) else str(TAVILY_PLUGIN))
        patch = render_search_patch(settings.dsh_home, "tavily", entry, "TAVILY_API_KEY")
        return (str(patch),), {"TAVILY_API_KEY": settings.tavily_api_key}
    if settings.exa_api_key:
        repo = CONTAINER_DSH_REPO if in_containers(settings) else str(Path(settings.dsh_repo).expanduser())
        patch = render_search_patch(settings.dsh_home, "exa", f"{repo}/{EXA_PLUGIN_ENTRY}", "EXA_API_KEY")
        return (str(patch),), {"EXA_API_KEY": settings.exa_api_key}
    # No backend: turn the tool off rather than let the model spend ~147s discovering it is broken.
    patch = _render(NO_SEARCH_PATCH_TEMPLATE, Path(settings.dsh_home) / "no-web-search.patch.yml", {})
    return (str(patch),), {}


def orchestrator_patches(settings: Settings, traceparent: str = "",
                         session_id: str = "") -> tuple[tuple[str, ...], dict[str, str]]:
    """Every patch layer the orchestrator needs, in application order, plus the env they reference.

    Search comes first so the orchestration layer can rely on `web` already being configured.
    """
    patches, env = search_patches(settings)
    return (*patches, str(render_orchestrator_patch(settings, traceparent, session_id))), env


def make_harness_factory(settings: Settings, traceparent: str = "",
                         session_id: str = "") -> Callable[[], Any]:
    """Build the dsh runtime. `traceparent` is read by the composition's `!!js process.env` lookups,
    so both children parent their spans on this run's root span."""

    def factory() -> Any:
        from deepseek_harness import DeepSeekHarness

        ensure_dsh_home(settings.dsh_home, settings.dsh_model, settings.dsh_provider, settings.xai_base_url)
        ensure_hermes_home(settings.hermes_home, settings.hermes_model or settings.dsh_model)
        install_skill(settings)
        orchestrator_patches(settings, traceparent, session_id)   # render every layer onto the host
        env = child_env(settings, traceparent, session_id)
        return DeepSeekHarness(
            dsh_bin=settings.dsh_bin, **launch_kwargs(settings),
            provider=settings.dsh_provider, model=settings.dsh_model, env=env,
            initialize_timeout_seconds=120, request_timeout_seconds=settings.run_timeout_s)

    return factory
