"""Containers mode: dsh spawns each sibling runtime with `docker run -i`, stdio unchanged."""
from pathlib import Path

from bubble_watch.config import PROJECT_ROOT, load_settings
from bubble_watch.harness import CONTAINER_WORKDIR, launch_paths, render_orchestrator_patch


def _settings(tmp_path, monkeypatch, **env):
    for name in ("TAVILY_API_KEY", "EXA_API_KEY", "ARIZE_SPACE_ID", "ARIZE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_ANALYST_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("BUBBLE_WATCH_DEPLOY", "containers")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return load_settings(tmp_path / "none.env")


def _patch(tmp_path, monkeypatch, traceparent="00-a-b-01", **env):
    env.setdefault("XAI_API_KEY", "k")
    env.setdefault("ARIZE_SPACE_ID", "sp")
    env.setdefault("ARIZE_API_KEY", "ak")
    settings = _settings(tmp_path, monkeypatch, **env)
    return Path(render_orchestrator_patch(settings, traceparent)).read_text()


def test_containers_mode_spawns_one_peer_container_per_runtime(tmp_path, monkeypatch):
    text = _patch(tmp_path, monkeypatch)
    assert text.count("command: docker") == 2   # the tool server and the analyst, side by side
    assert text.count("- '-i'") == 2            # stdio stays stdio, both times
    assert text.count("- '--rm'") == 2


def test_containers_mode_needs_no_docker_socket(tmp_path, monkeypatch):
    """Each runtime is one hop from the orchestrator, so no container spawns a sibling — which is
    the only reason one would have needed the socket."""
    assert "docker.sock" not in _patch(tmp_path, monkeypatch)


def test_containers_mode_names_the_tool_server_image(tmp_path, monkeypatch):
    text = _patch(tmp_path, monkeypatch, BUBBLE_WATCH_MCP_IMAGE="reg.test/mcp-tools:v2")
    assert "reg.test/mcp-tools:v2" in text


def test_containers_mode_forwards_trace_variables_by_name(tmp_path, monkeypatch):
    """Two hops. `config.env` gives the docker CLI the value (the MCP backend scrubs ambient
    credentials, so inheritance alone loses it); `-e NAME` then forwards it into the container."""
    text = _patch(tmp_path, monkeypatch)
    assert "- 'TRACEPARENT'" in text                # -e NAME, into the tool server container
    assert "TRACEPARENT: !!js process.env.TRACEPARENT" in text


def test_containers_mode_writes_container_paths_not_host_paths(tmp_path, monkeypatch):
    text = _patch(tmp_path, monkeypatch)
    assert str(tmp_path) not in text
    assert f"cwd: '{CONTAINER_WORKDIR}'" in text


def test_containers_mode_mounts_the_shared_state_and_reports_volume(tmp_path, monkeypatch):
    text = _patch(tmp_path, monkeypatch)
    # the tool server persists state; dsh writes the report — both need the run's shared volume
    assert "- '-v'" in text
    assert f"{CONTAINER_WORKDIR}/state" in text


def test_launch_paths_are_container_paths_in_containers_mode(tmp_path, monkeypatch):
    dsh_home, cwd, patches = launch_paths(_settings(tmp_path, monkeypatch))
    assert dsh_home == f"{CONTAINER_WORKDIR}/.dsh"
    assert cwd == CONTAINER_WORKDIR
    assert all(p.startswith(f"{CONTAINER_WORKDIR}/.dsh/") for p in patches)


def test_launch_paths_stay_on_the_host_in_local_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("BUBBLE_WATCH_DEPLOY", "local")
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "home"))
    settings = load_settings(tmp_path / "none.env")
    dsh_home, cwd, patches = launch_paths(settings)
    assert dsh_home == str(tmp_path / "home")
    assert cwd == str(PROJECT_ROOT)
    assert all(p.startswith(str(tmp_path / "home")) for p in patches)


def test_local_mode_still_spawns_the_tool_server_directly(tmp_path, monkeypatch):
    monkeypatch.setenv("BUBBLE_WATCH_DEPLOY", "local")
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_ANALYST_HOME", str(tmp_path / "hh"))
    text = Path(render_orchestrator_patch(load_settings(tmp_path / "none.env"))).read_text()
    assert "command: docker" not in text
    assert "bubble-watch" in text
