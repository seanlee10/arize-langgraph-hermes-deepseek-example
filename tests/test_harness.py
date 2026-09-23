import re
from pathlib import Path

from bubble_watch.config import PROJECT_ROOT, load_settings
from bubble_watch.harness import child_env, orchestrator_patches, render_orchestrator_patch


def _settings(tmp_path, monkeypatch, **env):
    for name in ("TAVILY_API_KEY", "EXA_API_KEY", "ARIZE_SPACE_ID", "ARIZE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DSH_REPO", str(tmp_path / "repo"))
    monkeypatch.setenv("HERMES_ANALYST_HOME", str(tmp_path / "hermes-home"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return load_settings(tmp_path / "none.env")


def test_orchestrator_patch_has_one_child_not_a_subagent(tmp_path, monkeypatch):
    """Hermes is reached through the tool server, not through dsh's ACP subagent backend: its ACP
    adapter fires no plugin hooks, so it would not trace itself. See hermes_tool.py."""
    text = Path(render_orchestrator_patch(_settings(tmp_path, monkeypatch))).read_text()
    rows = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    body = "\n".join(rows)
    assert "subagent" not in body        # the comment explains why; the composition has none
    assert "hermes-acp" not in body
    assert body.count("- id:") == 1


def test_orchestrator_patch_mounts_the_bubble_watch_mcp_server(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    text = Path(render_orchestrator_patch(settings)).read_text()
    assert "name: '@deepseek-ai/dsh-mcp-client'" in text
    assert "serverName: bubble" in text
    assert "'mcp'" in text
    # without the deterministic tools dsh would author numbers from nothing
    assert "failOnStartupError: true" in text


def test_orchestrator_patch_passes_the_traceparent_to_its_one_child(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch, XAI_API_KEY="k")
    text = Path(render_orchestrator_patch(settings, traceparent="00-a-b-01")).read_text()
    declared = [k for k, _ in re.findall(r"^\s+(\w+): !!js process\.env\.(\w+)$", text, re.MULTILINE)]
    assert [v for v in declared if v.endswith("TRACEPARENT")] == ["TRACEPARENT"]


def test_orchestrator_patch_is_written_inside_dsh_home(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    assert Path(render_orchestrator_patch(settings)).parent == tmp_path / "home"


def test_orchestrator_patches_layers_search_before_orchestration(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch, TAVILY_API_KEY="tvly-k")
    patches, env = orchestrator_patches(settings)
    assert [Path(p).name for p in patches] == ["web-search.patch.yml", "orchestrator.patch.yml"]
    assert env["TAVILY_API_KEY"] == "tvly-k"


def test_orchestrator_patches_still_orchestrates_without_a_search_key(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    patches, _ = orchestrator_patches(settings)
    assert [Path(p).name for p in patches] == ["orchestrator.patch.yml"]


def test_orchestrator_patch_template_is_shipped_in_the_repo():
    assert (PROJECT_ROOT / "dsh" / "orchestrator.patch.yml").exists()


def test_install_skill_places_the_procedure_where_dsh_discovers_it(tmp_path, monkeypatch):
    from bubble_watch.harness import install_skill
    settings = _settings(tmp_path, monkeypatch)
    path = install_skill(settings)
    assert path == tmp_path / "home" / "skills" / "bubble-watch" / "SKILL.md"
    assert path.read_text().startswith("---")


def test_install_skill_refreshes_an_outdated_copy(tmp_path, monkeypatch):
    from bubble_watch.harness import install_skill
    settings = _settings(tmp_path, monkeypatch)
    path = install_skill(settings)
    path.write_text("stale\n")
    assert install_skill(settings).read_text() != "stale\n"


def test_skill_source_is_tracked_and_declares_a_kebab_case_name():
    source = PROJECT_ROOT / "dsh" / "skills" / "bubble-watch" / "SKILL.md"
    assert source.exists()
    head = source.read_text().split("---")[1]
    assert "name: bubble-watch" in head
    assert "description:" in head


def test_skill_tells_dsh_to_delegate_research_and_never_compute():
    source = (PROJECT_ROOT / "dsh" / "skills" / "bubble-watch" / "SKILL.md").read_text()
    assert "hermes_analyst" in source
    assert "mcp__bubble__prepare_brief" in source
    assert "mcp__bubble__save_run" in source


def test_hermes_launcher_is_executable():
    import os
    launcher = PROJECT_ROOT / "bin" / "hermes"
    assert launcher.exists()
    assert os.access(launcher, os.X_OK)


# --- the Hermes analyst's isolated home and the dsh child environment ---------------------------

def test_ensure_hermes_home_configures_a_web_only_grok_analyst(tmp_path):
    from bubble_watch.hermes_tool import ensure_hermes_home
    home = ensure_hermes_home(tmp_path / "hh", "grok-4.6")
    cfg = (home / "config.yaml").read_text()
    assert "default: grok-4.6" in cfg and "provider: xai" in cfg
    assert "cli: [web]" in cfg          # research only; the CLI surface is the one we drive
    assert "tirith_enabled: false" in cfg
    assert "observability/arize" in cfg


def test_ensure_hermes_home_never_overwrites_an_existing_config(tmp_path):
    from bubble_watch.hermes_tool import ensure_hermes_home
    home = ensure_hermes_home(tmp_path / "hh", "grok-4.6")
    (home / "config.yaml").write_text("custom: true\n")
    assert (ensure_hermes_home(tmp_path / "hh", "grok-4.6") / "config.yaml").read_text() == "custom: true\n"


def test_child_env_strips_messaging_credentials(tmp_path, monkeypatch):
    from bubble_watch.harness import child_env
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "should-not-leak")
    monkeypatch.setenv("DISCORD_TOKEN", "should-not-leak")
    settings = _settings(tmp_path, monkeypatch, XAI_API_KEY="xai-k")
    env = child_env(settings, "00-abc-def-01")
    assert "TELEGRAM_BOT_TOKEN" not in env and "DISCORD_TOKEN" not in env
    assert env["XAI_API_KEY"] == "xai-k"


def test_child_env_carries_the_run_traceparent(tmp_path, monkeypatch):
    """Only the OTel name: the tool server opens a per-call span for Hermes and derives
    HERMES_ARIZE_TRACEPARENT from that, so Hermes nests under the call, not the run root."""
    from bubble_watch.harness import child_env
    settings = _settings(tmp_path, monkeypatch, XAI_API_KEY="xai-k")
    env = child_env(settings, "00-abc-def-01")
    assert env["TRACEPARENT"] == "00-abc-def-01"
    assert "HERMES_ARIZE_TRACEPARENT" not in env


def test_child_env_forwards_arize_credentials_only_when_configured(tmp_path, monkeypatch):
    from bubble_watch.harness import child_env
    settings = _settings(tmp_path, monkeypatch, XAI_API_KEY="xai-k")
    assert "ARIZE_API_KEY" not in child_env(settings, "")
    settings = _settings(tmp_path, monkeypatch, XAI_API_KEY="xai-k",
                         ARIZE_SPACE_ID="sp", ARIZE_API_KEY="ak")
    env = child_env(settings, "")
    assert env["ARIZE_SPACE_ID"] == "sp" and env["ARIZE_API_KEY"] == "ak"
    assert env["ARIZE_PROJECT_NAME"] == "bubble-watch"


def test_orchestrator_patch_gives_the_tool_server_hermes_keys_too(tmp_path, monkeypatch):
    """The analyst runs inside the tool server now, so its keys travel with the MCP row."""
    settings = _settings(tmp_path, monkeypatch, XAI_API_KEY="k", TAVILY_API_KEY="tv",
                         ARIZE_SPACE_ID="sp", ARIZE_API_KEY="ak")
    text = Path(render_orchestrator_patch(settings, traceparent="00-a-b-01")).read_text()
    for name in ("TRACEPARENT", "ARIZE_API_KEY", "XAI_API_KEY", "TAVILY_API_KEY"):
        assert f"{name}: !!js process.env.{name}" in text


def test_unbundled_plugins_are_referenced_by_package_name(tmp_path, monkeypatch):
    """They are not in dsh's base bundle, so they must be installed into the profile package tree
    (`bubble-watch install-plugins`). Pointing a `name` at the checkout's lib/index.js instead
    loads a second copy of the plugin whose config validation fails against the bundle's copy."""
    text = Path(render_orchestrator_patch(_settings(tmp_path, monkeypatch))).read_text()
    assert "name: '@deepseek-ai/dsh-mcp-client'" in text
    names = re.findall(r"^\s+name: '([^']+)'$", text, re.MULTILINE)
    assert all(not n.endswith(".js") for n in names), names





def test_patch_never_declares_an_env_var_that_is_not_set(tmp_path, monkeypatch):
    """`!!js process.env.FOO` yields `undefined` when FOO is unset, and dsh-mcp-client's Config is a
    z.union whose branch rejects an env dict holding one — the row then fails validation and the
    plugin silently never activates. (dsh-subagent-acp's Config is a plain z.object and tolerates
    it, which is why only the MCP tools went missing.) Declare only variables that exist."""
    settings = _settings(tmp_path, monkeypatch, XAI_API_KEY="xai-k")   # no TAVILY/EXA/ARIZE/ALPHA
    text = Path(render_orchestrator_patch(settings, traceparent="")).read_text()
    declared = re.findall(r"^\s+(\w+): !!js process\.env\.(\w+)$", text, re.MULTILINE)
    assert declared, "expected some env declarations"
    for key, var in declared:
        assert key == var
        assert var in child_env(settings, ""), f"{var} is declared but never set"
    assert "ALPHAVANTAGE_API_KEY: !!js" not in text
    assert "TRACEPARENT: !!js" not in text
    assert "TAVILY_API_KEY: !!js" not in text


def test_patch_declares_the_traceparent_once_the_run_has_one(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch, XAI_API_KEY="xai-k",
                         ARIZE_SPACE_ID="sp", ARIZE_API_KEY="ak")
    text = Path(render_orchestrator_patch(settings, traceparent="00-a-b-01")).read_text()
    assert "TRACEPARENT: !!js process.env.TRACEPARENT" in text
    assert "ARIZE_API_KEY: !!js process.env.ARIZE_API_KEY" in text
