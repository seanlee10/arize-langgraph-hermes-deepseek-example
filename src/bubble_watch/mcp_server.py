"""The `bubble-watch mcp` stdio server: the deterministic half of the run, exposed to dsh as tools.

dsh reaches these as `mcp__bubble__<tool>`. They compute; they never judge, and they never call an
agent. Gap filling is dsh's decision: it delegates the research to the Hermes subagent and hands the
results back through `apply_gap_fills`.
"""
from __future__ import annotations

from typing import Annotated, Any

from mcp.server import MCPServer
from pydantic import Field

from .hermes_tool import HermesError, hermes_analyst
from .mcp_tools import ToolDeps, apply_gap_fills, prepare_brief, prior_state, save_run

INSTRUCTIONS = """\
The confirmed-data layer of the NVDA Bubble Signal Watch.

- Every close, return, IV and put change is computed by these tools. Never compute or estimate one
  yourself.
- The `gaps` returned by `prepare_brief` are values the market data feed does not have. Research
  them, then pass them to `apply_gap_fills` with a source URL and a freshness label. A value
  without a source is rejected.
- A null `signals` means the ticker's own close is still missing. Fill that gap first.
- Delegate web research and missing-value lookups to `hermes_analyst`. It is an independent
  analyst, so take its view before you tell it yours.
"""

_DATE = Annotated[str, Field(description="Trading day to act on (YYYY-MM-DD)")]


def build_server(deps: ToolDeps) -> MCPServer:
    server = MCPServer(name="bubble-watch", instructions=INSTRUCTIONS)

    @server.tool(name="prepare_brief")
    def _prepare_brief(date: _DATE) -> dict[str, Any]:
        """Fetch the day's market data and compute every signal that can be computed from it.

        `closes`/`puts` are confirmed values, `signals` are the computed signals, and `gaps` are the
        fields the feed lacks and you must research. Without the ticker's close, `signals` is null
        and is recomputed once the gap is filled. `trace_id` identifies this run in Arize — cite it
        in the report's provenance.
        """
        return prepare_brief(deps, date)

    @server.tool(name="apply_gap_fills")
    def _apply_gap_fills(
        date: _DATE,
        fills: Annotated[list[dict[str, Any]], Field(description=(
            "The researched values. Each entry is {field, value, source_url, freshness}, where "
            "field looks like 'closes.SOXL' or 'puts.210.iv'. A value without an http(s) "
            "source_url and a freshness label is rejected."))],
    ) -> dict[str, Any]:
        """Apply researched values to the day's data and recompute the signals.

        A value without a source URL and a freshness label comes back in `rejected` and is not
        stored. An empty `remaining_gaps` means the data is complete.
        """
        return apply_gap_fills(deps, date, fills)

    @server.tool(name="hermes_analyst")
    def _hermes_analyst(
        task: Annotated[str, Field(description=(
            "The task to send the Hermes analyst. Say exactly what you need, the output format you "
            "want, and that every claim needs a source URL. Paste computed figures in verbatim and "
            "tell it not to recompute them."))],
        session_id: Annotated[str, Field(description=(
            "A session_id returned by an earlier call. Passing it continues the same conversation "
            "- use it for the rebuttal round so you send only the opposing view, not the whole "
            "brief again."))] = "",
    ) -> dict[str, Any]:
        """Delegate to the independent Hermes analyst: web research, missing values, its own view.

        It runs in its own process, session and toolset, and traces itself. Keep the returned
        session_id and pass it back for the rebuttal round. On failure this returns an error:
        proceed on your own judgement alone and disclose that in the report.
        """
        try:
            return hermes_analyst(deps.settings, task, session_id, tracer=deps.tracer)
        except HermesError as exc:
            raise ValueError(f"hermes analyst unavailable: {exc}") from None

    @server.tool(name="prior_state")
    def _prior_state(date: _DATE) -> dict[str, Any]:
        """The previous records (score, verdict, note). Use them for the score delta and continuity."""
        return prior_state(deps, date)

    @server.tool(name="save_run")
    def _save_run(
        date: _DATE,
        score: Annotated[float, Field(description="Bubble Signal Score, 0-10", ge=0, le=10)],
        verdict: Annotated[str, Field(description=(
            "NOT_TRIGGERED | TRIGGERED | TRIGGERED_DE_CONFIRMING | "
            "TRIGGERED_FURTHER_DE_CONFIRMING | CONFIRMED"))],
        note: Annotated[str, Field(description="One or two sentences summarising the day")],
        report_path: Annotated[str, Field(description="Path of the report you wrote")],
    ) -> dict[str, Any]:
        """Persist the day's record — market data, computed signals, verdict — to the state file.

        Call this last, after the report is written.
        """
        return save_run(deps, date, score=score, verdict=verdict, note=note, report_path=report_path)

    return server
