"""dsh-mcp-client and dsh-subagent-acp are not in dsh's base bundle: they must be installed into
the profile package tree before the orchestration composition can resolve them by name."""
import json

import pytest

from bubble_watch.config import load_settings
from bubble_watch.harness import (
    PROFILE_PLUGINS,
    install_profile_plugins,
    missing_profile_plugins,
)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DSH_BIN", str(tmp_path / "dsh"))
    return load_settings(tmp_path / "none.env")


def _write_profile(settings, dependencies):
    path = settings.dsh_home_path / "profiles" / "sdk" / "package.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": "dsh-profile-sdk", "dependencies": dependencies}))


def test_both_plugins_are_missing_before_any_install(settings):
    assert missing_profile_plugins(settings) == list(PROFILE_PLUGINS)


def test_nothing_is_missing_once_both_are_installed(settings):
    _write_profile(settings, {name: "0.1.6-alpha.2" for name in PROFILE_PLUGINS})
    assert missing_profile_plugins(settings) == []


def test_a_partial_install_still_reports_the_rest(settings):
    _write_profile(settings, {PROFILE_PLUGINS[0]: "0.1.6-alpha.2"})
    assert missing_profile_plugins(settings) == [PROFILE_PLUGINS[1]]


def test_install_runs_the_dsh_plugin_command_per_missing_plugin(settings):
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, "", "")

    install_profile_plugins(settings, version="9.9.9", runner=runner)
    assert len(calls) == len(PROFILE_PLUGINS)
    for cmd, name in zip(calls, PROFILE_PLUGINS, strict=True):
        assert cmd[0] == settings.dsh_bin
        assert cmd[1:4] == ["plugin", "--profile", "sdk"]
        assert cmd[-2:] == ["add", f"{name}@9.9.9"]


def test_install_skips_plugins_that_are_already_there(settings):
    _write_profile(settings, {PROFILE_PLUGINS[0]: "9.9.9"})
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, "", "")

    install_profile_plugins(settings, version="9.9.9", runner=runner)
    assert [c[-1] for c in calls] == [f"{PROFILE_PLUGINS[1]}@9.9.9"]


def test_install_raises_with_the_failing_command_output(settings):
    def runner(cmd, **kwargs):
        import subprocess
        return subprocess.CompletedProcess(cmd, 1, "", "ERR_PNPM_WORKSPACE_PKG_NOT_FOUND")

    with pytest.raises(RuntimeError, match="ERR_PNPM"):
        install_profile_plugins(settings, version="9.9.9", runner=runner)
