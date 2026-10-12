"""Read public analytical evidence and terminal outcomes in owner tests."""

import json
from typing import Any

from daita import Agent
from daita._json import canonical_json
from daita.loop.analysis import AnalysisEvidence
from daita.loop.models import LoopExit


def facts(record: AnalysisEvidence) -> dict[str, Any]:
    return json.loads(canonical_json(record.facts))


async def terminal(agent: Agent, conversation_id: str | None) -> LoopExit:
    assert conversation_id is not None
    result = (await agent.conversation_runs(conversation_id))[0].result
    assert result is not None
    return result


async def usage(agent: Agent, run_id: str) -> dict[str, Any]:
    summary = await agent.analysis_usage(run_id)
    assert summary is not None
    return json.loads(canonical_json(summary))
