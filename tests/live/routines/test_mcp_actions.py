"""Independent live scheduled execution, with a typed owner-authorized fixture.

The original ten-case matrix remains unchanged. Only fixture authorization is
seeded here; both scheduled runs use the real model and production composition.
"""

import os
from pathlib import Path

import pytest

from daita.distribution.models import OutcomeState
from tests.support.mcp_live_routine import admit_owner_routine
from tests.support.mcp_routine_harness import (
    AUTHORIZATION,
    COST_ENV,
    NEXT_SLOT,
    RESEARCH_TOKEN,
    SOURCE,
    assert_action,
    assert_completed,
    evaluate,
    live_provider,
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(AUTHORIZATION) != "1",
        reason=(
            f"set {AUTHORIZATION}=1 only after authorizing up to twelve live "
            f"runs per model/repetition, each capped by {COST_ENV}; MCP is simulated"
        ),
    ),
]


async def test_live_owner_admitted_routine_runs_immediate_and_weekly(
    tmp_path: Path,
    model_id: str,
    repetition: int,
    evidence_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    profile, provider = live_provider(model_id)
    async with evaluate(
        tmp_path,
        provider,
        profile,
        evidence_path,
        routine=True,
        case_id=request.node.nodeid,
    ) as scenario:
        await admit_owner_routine(scenario)
        for count in (1, 2):
            if count == 2:
                scenario.clock = NEXT_SLOT
                scenario.agent._embedded._routine_supervisor.wake()
            result, transcript = await scenario.scheduled_result(count)
            assert_completed(result, transcript)
            scenario.check_answer(result, status="server_reported", research=True)
            await assert_action(scenario, count=count)
            assert result.usage.total_tokens > 0
        assert scenario.approvals == []  # exact typed owner authorization is standing
        assert scenario.conversation_id is not None
        for item in await scenario.agent.inbox(
            conversation_id=scenario.conversation_id
        ):
            delivery = await scenario.agent.inspect_delivery(item.delivery_id)
            assert delivery is not None
            assert delivery.delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
        for name, arguments in scenario.server.calls:
            if name == "notify_release":
                assert SOURCE in str(arguments["content"])
                assert RESEARCH_TOKEN in str(arguments["content"])
