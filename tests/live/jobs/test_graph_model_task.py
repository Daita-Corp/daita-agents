"""Authorized live-model smoke test for the unreleased Phase 4 task path."""

from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from daita import create_llm_provider
from daita.context import AgentContextBuilder
from daita.jobs.graph.models import AttemptState, GraphState, TaskRole
from daita.llm.profiles import reviewed_model_profile
from tests.support.conversations import CatalogSpy
from tests.support.model_graph_integration import ModelGraphIntegration

_AUTHORIZATION = "DAITA_RUN_LIVE_GRAPH_TASK"
_MODEL_ID = "DAITA_GRAPH_TASK_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_GRAPH_TASK_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_GRAPH_TASK_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_PROVIDERS = frozenset({"anthropic", "gemini", "grok", "openai"})

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing one bounded "
            "live graph-model-task run"
        ),
    ),
]


def _cost_limit() -> Decimal:
    raw = os.environ.get(_MAX_COST, "0.15")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    if not value.is_finite() or value <= 0:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    return value


async def test_live_model_completes_exact_graph_task_with_lifecycle_terminator(
    tmp_path: Path,
) -> None:
    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    provider_name = model_id.partition(":")[0]
    if provider_name not in _API_PROVIDERS:
        pytest.fail(f"{_MODEL_ID} must name an API-backed provider")
    profile = reviewed_model_profile(model_id)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{_MODEL_ID} must name a reviewed tool-capable model")
    api_key = os.environ.get(_MODEL_KEY)
    if api_key is None or not api_key.strip():
        pytest.fail(f"{_MODEL_KEY} must be set for the authorized live test")
    provider = create_llm_provider(
        model_id,
        api_key=api_key,
        max_output_tokens=min(profile.max_output_tokens, 2_048),
    )
    integration = await ModelGraphIntegration.open(tmp_path)
    loop = integration.supervisor._graph_model_loop
    assert loop is not None
    loop._model = provider
    loop._context_builder = AgentContextBuilder(
        CatalogSpy(),
        profile=profile,
    )
    try:
        admission = integration.build(
            model_route_id=provider.provider_id,
            per_run_max_cost_usd=_cost_limit(),
        )
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id, timeout=120)
        assert terminal.job.state is GraphState.SUCCEEDED
        worker = next(task for task in terminal.tasks if task.role is TaskRole.WORKER)
        attempt = next(
            item for item in terminal.attempts if item.task_id == worker.task_id
        )
        assert attempt.state is AttemptState.SUCCEEDED
        exit = await integration.store.result(attempt.run_id)
        assert exit is not None
        assert exit.kind.value == "machine_terminated"
        assert exit.reason == "task_completed"
        assert len(integration.reader.calls) == 1
    finally:
        await integration.close()
        await provider.close()
