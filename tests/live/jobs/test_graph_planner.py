"""Authorized protocol-only live smoke for the unreleased Phase 5 planner path."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from daita import create_llm_provider
from daita.context import AgentContextBuilder
from daita.jobs.graph.models import (
    BudgetAmount,
    ControlKind,
    ControlState,
    GraphState,
    TaskControl,
    TaskRole,
    TaskState,
    canonical_digest,
)
from daita.llm.profiles import reviewed_model_profile
from tests.support.conversations import CatalogSpy
from tests.support.model_graph_integration import ModelGraphIntegration

_AUTHORIZATION = "DAITA_RUN_LIVE_GRAPH_PLANNER"
_MODEL_ID = "DAITA_GRAPH_PLANNER_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_GRAPH_PLANNER_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_GRAPH_PLANNER_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_LIVE_TOKEN_LIMIT = 100_000
_API_PROVIDERS = frozenset({"anthropic", "gemini", "grok", "openai"})

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing one bounded "
            "integration-only live planner protocol run"
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


async def test_live_planner_uses_typed_replacement_without_production_cutover(
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
    integration = await ModelGraphIntegration.open(
        tmp_path,
        loop_max_total_tokens=_LIVE_TOKEN_LIMIT,
    )
    loop = integration.supervisor._graph_model_loop
    assert loop is not None
    loop._model = provider
    loop._context_builder = AgentContextBuilder(CatalogSpy(), profile=profile)
    try:
        admission = integration.build(
            model_route_id=provider.provider_id,
            per_run_max_tokens=_LIVE_TOKEN_LIMIT,
            per_run_max_cost_usd=_cost_limit(),
        )
        raw_planner_template = admission.job.specification.planner_task_template
        assert raw_planner_template is not None
        planner_template = dict(raw_planner_template)
        raw_planner_specification = planner_template["specification"]
        assert isinstance(raw_planner_specification, Mapping)
        planner_specification = dict(raw_planner_specification)
        planner_specification["description"] = (
            "Inspect the blocked worker, then call graph_supersede_unstarted exactly "
            "once to create an equivalent replacement using the same admitted "
            "initial capability and arguments. Use expected graph revision 1 and a "
            "stable idempotency key. Then finish with task_complete and result kind "
            "graph.plan. Do not call the source capability from this planner task."
        )
        planner_template["specification"] = planner_specification
        job_specification = replace(
            admission.job.specification,
            planner_task_template=planner_template,
        )
        admission = replace(
            admission,
            job=replace(
                admission.job,
                specification=job_specification,
                specification_digest=job_specification.digest,
            ),
        )
        await integration.owner.admit(admission)
        worker = next(task for task in admission.tasks if task.role is TaskRole.WORKER)
        claimed_at = integration.clock()
        attempt = await integration.store.claim_graph_task(
            worker.agent_id,
            worker.job_id,
            worker.task_id,
            attempt_id="attempt-live-discovery",
            claim_token="claim-live-discovery",
            run_id="run-live-discovery",
            executor_id=provider.provider_id,
            claimed_at=claimed_at,
            lease_seconds=30,
            absolute_deadline_at=admission.job.deadline_at,
            budget_reservations=(BudgetAmount("work_units", 1),),
        )
        assert attempt is not None
        running = await integration.store.start_graph_attempt(
            worker.agent_id,
            worker.job_id,
            worker.task_id,
            attempt.attempt_id,
            claim_token=attempt.claim_token,
            fencing_epoch=attempt.fencing_epoch,
            started_at=integration.clock(),
        )
        assert running is not None
        payload = {
            "message": "Discovery requires an equivalent immutable replacement.",
            "details": {"checkpoint": "live-protocol-only"},
        }
        control = TaskControl(
            agent_id=worker.agent_id,
            job_id=worker.job_id,
            task_id=worker.task_id,
            control_id="control-live-replan",
            kind=ControlKind.NEEDS_REPLAN,
            state=ControlState.OPEN,
            requesting_attempt_id=attempt.attempt_id,
            payload=payload,
            created_at=integration.clock(),
            payload_digest=canonical_digest(payload),
        )
        await integration.owner.open_graph_task_control(
            control,
            claim_token=attempt.claim_token,
            fencing_epoch=attempt.fencing_epoch,
        )

        await integration.supervisor.start()
        terminal = await integration.wait_terminal(admission.job.job_id, timeout=180)
        assert terminal.job.state is GraphState.SUCCEEDED
        original = next(
            item for item in terminal.tasks if item.task_id == worker.task_id
        )
        assert original.state is TaskState.SUPERSEDED
        replacement = next(
            item
            for item in terminal.tasks
            if item.task_id == original.superseded_by_task_id
        )
        assert replacement.state is TaskState.SUCCEEDED
        planner = next(item for item in terminal.tasks if item.role is TaskRole.PLANNER)
        assert planner.state is TaskState.SUCCEEDED
        assert len(integration.reader.calls) == 1
    finally:
        await integration.close()
        await provider.close()
