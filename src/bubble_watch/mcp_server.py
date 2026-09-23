"""The `bubble-watch mcp` stdio server: the deterministic half of the run, exposed to dsh as tools.

dsh reaches these as `mcp__bubble__<tool>`. They compute; they never judge, and they never call an
agent. Gap filling is dsh's decision: it delegates the research to the Hermes subagent and hands the
results back through `apply_gap_fills`.
"""
from __future__ import annotations

from typing import Annotated, Any

from mcp.server import MCPServer
from pydantic import Field

from .mcp_tools import ToolDeps, apply_gap_fills, prepare_brief, prior_state, save_run

INSTRUCTIONS = """\
NVDA Bubble Signal Watch의 확정 데이터 계층이다.

- 모든 종가·수익률·IV·put 변화율은 이 도구들이 계산한다. 직접 계산하거나 추정하지 마라.
- `prepare_brief`가 돌려준 `gaps`는 시장 데이터에 없는 값이다. 웹 리서치로 찾은 뒤
  `apply_gap_fills`에 출처 URL과 freshness를 붙여 전달한다. 출처 없는 값은 거부된다.
- `signals`가 null이면 기준 종가가 아직 없다는 뜻이다. gap을 먼저 채워라.
"""

_DATE = Annotated[str, Field(description="대상 거래일 (YYYY-MM-DD)")]


def build_server(deps: ToolDeps) -> MCPServer:
    server = MCPServer(name="bubble-watch", instructions=INSTRUCTIONS)

    @server.tool(name="prepare_brief")
    def _prepare_brief(date: _DATE) -> dict[str, Any]:
        """그날의 시장 데이터를 받아 계산 가능한 모든 신호를 계산한다.

        closes/puts는 확정값, signals는 계산된 신호, gaps는 시장 데이터에 없어 조사해야 하는
        항목이다. 기준 종가가 없으면 signals는 null이며 gap을 채운 뒤 다시 계산된다.
        """
        return prepare_brief(deps, date)

    @server.tool(name="apply_gap_fills")
    def _apply_gap_fills(
        date: _DATE,
        fills: Annotated[list[dict[str, Any]], Field(description=(
            "조사한 값들. 각 항목은 {field, value, source_url, freshness}. "
            "field는 'closes.SOXL' 또는 'puts.210.iv' 형식. "
            "source_url(http/https)과 freshness가 없는 값은 거부된다."))],
    ) -> dict[str, Any]:
        """조사한 결측값을 적용하고 신호를 다시 계산한다.

        출처 URL과 freshness가 없는 값은 rejected에 담겨 돌아오고 저장되지 않는다.
        remaining_gaps가 비면 데이터가 완성된 것이다.
        """
        return apply_gap_fills(deps, date, fills)

    @server.tool(name="prior_state")
    def _prior_state(date: _DATE) -> dict[str, Any]:
        """직전 기록(점수·판정·메모)을 돌려준다. score delta와 연속성 판단에 쓴다."""
        return prior_state(deps, date)

    @server.tool(name="save_run")
    def _save_run(
        date: _DATE,
        score: Annotated[float, Field(description="Bubble Signal Score, 0–10", ge=0, le=10)],
        verdict: Annotated[str, Field(description=(
            "NOT_TRIGGERED | TRIGGERED | TRIGGERED_DE_CONFIRMING | "
            "TRIGGERED_FURTHER_DE_CONFIRMING | CONFIRMED"))],
        note: Annotated[str, Field(description="하루를 한두 문장으로 요약한 메모")],
        report_path: Annotated[str, Field(description="작성한 리포트 파일 경로")],
    ) -> dict[str, Any]:
        """그날의 기록(시장 데이터·계산된 신호·판정)을 상태 파일에 저장한다. 리포트를 쓴 뒤 마지막에 호출한다."""
        return save_run(deps, date, score=score, verdict=verdict, note=note, report_path=report_path)

    return server
