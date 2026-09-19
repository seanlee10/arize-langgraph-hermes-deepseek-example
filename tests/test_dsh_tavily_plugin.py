"""The dsh Tavily search plugin (dsh/plugins/web-search-tavily.mjs), exercised under Node with a mocked fetch."""
import json
import shutil
import subprocess

import pytest

from bubble_watch.config import PROJECT_ROOT

PLUGIN = PROJECT_ROOT / "dsh" / "plugins" / "web-search-tavily.mjs"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

HARNESS = """
const plugin = await import(process.argv[1]);
const calls = [];
globalThis.fetch = async (url, init) => {
  calls.push({ url, init: { ...init, body: JSON.parse(init.body) } });
  if (process.env.MODE === 'error') return new Response(JSON.stringify({ detail: { error: 'Unauthorized: missing or invalid API key.' } }), { status: 401 });
  return new Response(JSON.stringify({ results: [
    { title: 'Nscale files IPO', url: 'https://cnbc.com/nscale', content: ' Nvidia-backed Nscale filed ', published_date: '2026-09-18' },
    { title: 'no snippet', url: 'https://x.test/empty', content: '   ' },
  ] }), { status: 200 });
};
const registered = [];
const ctx = { web: { registerSearchProvider: p => { registered.push(p); return () => {} } } };
plugin.apply(ctx, JSON.parse(process.env.CONFIG || '{}'));
const provider = registered[0];
const out = { id: provider.id, available: provider.available(), name: plugin.name, inject: plugin.inject };
try { out.result = await provider.search({ query: 'nvidia nscale', maxResults: 3 }); }
catch (e) { out.error = { name: e.name, code: e.code, message: e.message }; }
out.calls = calls.map(c => ({ url: c.url, auth: c.init.headers.authorization, body: c.init.body, redirect: c.init.redirect }));
console.log(JSON.stringify(out));
"""


def _run(config, mode="ok", env_key=None):
    env = {"PATH": "/usr/bin:/bin", "CONFIG": json.dumps(config), "MODE": mode,
           **({"TAVILY_API_KEY": env_key} if env_key else {})}
    proc = subprocess.run([shutil.which("node"), "--input-type=module", "-e", HARNESS, str(PLUGIN)],
                          capture_output=True, text=True, env=env, timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_search_maps_results_and_sends_bearer_request():
    out = _run({"apiKey": "tvly-test"})
    assert (out["id"], out["available"], out["name"], out["inject"]) == ("tavily", True, "web-search-tavily", ["web"])
    call = out["calls"][0]
    assert call["url"] == "https://api.tavily.com/search" and call["auth"] == "Bearer tvly-test"
    assert call["redirect"] == "error"
    assert call["body"] == {"query": "nvidia nscale", "max_results": 3, "search_depth": "basic", "topic": "general",
                            "include_answer": False}
    assert out["result"] == {"truncated": False, "sources": [
        {"url": "https://cnbc.com/nscale", "title": "Nscale files IPO", "snippet": "Nvidia-backed Nscale filed",
         "publishedAt": "2026-09-18"}]}


def test_api_key_falls_back_to_env_and_unavailable_without_one():
    assert _run({}, env_key="tvly-env")["calls"][0]["auth"] == "Bearer tvly-env"
    assert _run({})["available"] is False


def test_http_error_becomes_web_provider_error():
    out = _run({"apiKey": "bad"}, mode="error")
    assert out["error"] == {"name": "WebError", "code": "WEB_PROVIDER_ERROR",
                            "message": "Unauthorized: missing or invalid API key."}
