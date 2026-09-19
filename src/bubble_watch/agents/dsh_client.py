"""DeepSeek Harness as an analyst, driven in-process through its Python SDK (JSON-RPC over stdio)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..config import PROJECT_ROOT, Settings
from .base import AnalystError, AnalystResult, AskResult, ask_for_view
from .prompts import SYSTEM_ANALYST

EXA_PATCH = PROJECT_ROOT / "dsh" / "exa-search.patch.yml"

_SETTINGS_TEMPLATE = """\
# Written by bubble-watch: registers xAI (Grok) for the dsh analyst through the llm-pi-ai adapter.
llm-pi-ai:
  providers:
    {provider}:
      apiKeyEnv: XAI_API_KEY
      api: openai-completions
      baseURL: {base_url}
      models:
        - id: {model}
"""


def ensure_dsh_home(dsh_home: str, model: str, provider: str = "xai",
                    base_url: str = "https://api.x.ai/v1") -> Path:
    """Create DSH_HOME with an xAI provider settings.yaml; never overwrite an existing one."""
    home = Path(dsh_home)
    home.mkdir(parents=True, exist_ok=True)
    path = home / "settings.yaml"
    if not path.exists():
        path.write_text(_SETTINGS_TEMPLATE.format(provider=provider, base_url=base_url, model=model))
    return path


def make_harness_factory(settings: Settings) -> Callable[[], Any]:
    def factory() -> Any:
        from deepseek_harness import DeepSeekHarness

        ensure_dsh_home(settings.dsh_home, settings.dsh_model, settings.dsh_provider, settings.xai_base_url)
        env = {"XAI_API_KEY": settings.xai_api_key}
        patches: tuple[str, ...] = ()
        if settings.exa_api_key:
            env["EXA_API_KEY"] = settings.exa_api_key
            patches = (str(EXA_PATCH),)
        return DeepSeekHarness(
            dsh_bin=settings.dsh_bin, dsh_home=settings.dsh_home, cwd=str(PROJECT_ROOT),
            provider=settings.dsh_provider, model=settings.dsh_model, patches=patches, env=env,
            initialize_timeout_seconds=120, request_timeout_seconds=settings.analyst_timeout_s)
    return factory


class DshAnalyst:
    name = "dsh"

    def __init__(self, harness_factory: Callable[[], Any]) -> None:
        self._factory = harness_factory
        self._harness: Any = None

    def _h(self) -> Any:
        if self._harness is None:
            self._harness = self._factory()
        return self._harness

    def ask(self, prompt: str, session_id: str | None = None) -> AskResult:
        full = prompt if session_id else f"{SYSTEM_ANALYST}\n\n{prompt}"
        try:
            result = self._h().run(full, session_id=session_id)
        except Exception as exc:  # SDK transport/protocol/timeout errors
            raise AnalystError(f"dsh run failed: {type(exc).__name__}: {str(exc)[:300]}") from exc
        if not (result.final_response or "").strip():
            raise AnalystError(f"dsh returned no answer (finish_reason={result.finish_reason})")
        return AskResult(text=result.final_response, session_id=result.session_id, events=list(result.events))

    def analyze(self, brief: str) -> AnalystResult:
        return ask_for_view(self.name, self.ask, brief)

    def rebut(self, prompt: str) -> AnalystResult:
        return ask_for_view(self.name, self.ask, prompt)

    def close(self) -> None:
        if self._harness is not None:
            self._harness.close()
            self._harness = None
