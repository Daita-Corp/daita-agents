from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256

from _distribution_support import inbox_distribution_plan
from _workspace_support import workspace_for

from daita import Agent
from daita._json import FrozenJsonObject
from daita.capabilities import (
    AccessMode,
    AutomationEligibility,
    CapabilityDeclarations,
    ExecutionScope,
    Executor,
    OperationalEffect,
    ToolExecution,
)
from daita.distribution import (
    Delivery,
    DeliveryState,
    DeliverySubjectKind,
    DistributionCapabilityDomain,
    DistributionOwner,
    OutcomeConclusionKind,
    OutcomeReference,
    OutcomeState,
    distribution_capability_declarations,
    logical_delivery_key,
)
from daita.distribution.capabilities import (
    DELIVERY_INSPECT_TOOL_NAME,
    DELIVERY_LIST_TOOL_NAME,
    DISTRIBUTION_DESTINATION_LIST_TOOL_NAME,
    DISTRIBUTION_DOMAIN_OWNER_ID,
)
from daita.llm import ModelSensitivity
from daita.llm.models import FinishReason, ModelResponse, ToolCall, ToolResultBlock
from daita.llm.providers.mock import MockModelProvider
from daita.loop import InstructionAuthority, RunInput, RunOrigin, RunStartEnvelope

NOW = datetime(2026, 8, 28, 18, tzinfo=UTC)
DIGEST = "sha256:" + "a" * 64


class _Store:
    def __init__(self) -> None:
        self.delivery: Delivery | None = None

    async def conversation_exists(self, agent_id: str, conversation_id: str) -> bool:
        return agent_id == "agent-1" and conversation_id == "conversation-1"

    async def load_delivery(
        self,
        agent_id: str,
        delivery_id: str,
    ) -> Delivery | None:
        value = self.delivery
        if (
            value is None
            or value.agent_id != agent_id
            or value.delivery_id != delivery_id
        ):
            return None
        return value

    async def list_deliveries(
        self,
        agent_id: str,
        *,
        conversation_id: str | None = None,
        include_acknowledged: bool = False,
        limit: int = 50,
    ) -> tuple[Delivery, ...]:
        value = self.delivery
        if (
            value is None
            or value.agent_id != agent_id
            or (
                conversation_id is not None and value.conversation_id != conversation_id
            )
            or (
                not include_acknowledged
                and value.visibility_state is DeliveryState.ACKNOWLEDGED
            )
        ):
            return ()
        return (value,)[:limit]

    async def acknowledge_delivery(
        self,
        agent_id: str,
        delivery_id: str,
        *,
        acknowledged_at: datetime,
    ) -> Delivery | None:
        del agent_id, delivery_id, acknowledged_at
        raise AssertionError("no model-visible acknowledgment capability may exist")


def _delivery(owner: DistributionOwner) -> Delivery:
    target = owner.resolve_conversation_inbox(
        "conversation-1",
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
    )
    outcome = OutcomeReference(
        conclusion_kind=OutcomeConclusionKind.TERMINAL_RUN,
        conclusion_state=OutcomeState.SUCCEEDED,
        conclusion_id="run-1",
        conclusion_digest=DIGEST,
        conclusion_preview="Safe report",
        conclusion_preview_truncated=False,
        resulting_run_id="run-1",
        artifact_references=(),
        effective_sensitivity=ModelSensitivity.INTERNAL,
        provenance_digest=DIGEST,
        failure_code=None,
        observed_at=NOW,
    )
    return Delivery(
        delivery_id="delivery-1",
        agent_id="agent-1",
        conversation_id="conversation-1",
        subject_kind=DeliverySubjectKind.AUTONOMOUS_FOLLOWUP,
        subject_id="followup-1",
        logical_key=logical_delivery_key(
            agent_id="agent-1",
            subject_kind=DeliverySubjectKind.AUTONOMOUS_FOLLOWUP,
            subject_id="followup-1",
            target_fingerprint=target.target_fingerprint,
        ),
        target=target,
        outcome=outcome,
        visibility_state=DeliveryState.AVAILABLE,
        acknowledged_at=None,
        blocked_reason_code=None,
        created_at=NOW,
        updated_at=NOW,
    )


def _domain() -> tuple[DistributionCapabilityDomain, tuple[Executor, ...]]:
    store = _Store()
    owner = DistributionOwner(agent_id="agent-1", store=store)
    store.delivery = _delivery(owner)
    bundle = distribution_capability_declarations(owner)
    declarations = CapabilityDeclarations(
        domain_owner_id=DISTRIBUTION_DOMAIN_OWNER_ID,
        capabilities=bundle.capabilities,
        executor_ids=tuple(item.executor_id for item in bundle.capabilities),
        tool_views=bundle.tool_views,
    )
    return DistributionCapabilityDomain(declarations, owner), bundle.executors


async def test_distribution_tools_are_exact_foreground_effect_free_reads() -> None:
    domain, _executors = _domain()
    user = RunInput(
        id="run-user",
        agent_id="agent-1",
        conversation_id="conversation-1",
        message="Show destinations and deliveries.",
        created_at=NOW,
    )
    assert await domain.project(user) == (
        DISTRIBUTION_DESTINATION_LIST_TOOL_NAME,
        DELIVERY_LIST_TOOL_NAME,
        DELIVERY_INSPECT_TOOL_NAME,
    )
    assert all(
        capability.automation_eligibility is AutomationEligibility.INTERACTIVE_ONLY
        and capability.operational_effect is OperationalEffect.NONE
        and capability.access_mode is AccessMode.NONE
        for capability in domain.declarations.capabilities
    )
    assert {item.name for item in domain.declarations.tool_views} == {
        DISTRIBUTION_DESTINATION_LIST_TOOL_NAME,
        DELIVERY_LIST_TOOL_NAME,
        DELIVERY_INSPECT_TOOL_NAME,
    }


async def test_scheduled_scope_cannot_discover_or_inspect_distribution() -> None:
    domain, _executors = _domain()
    instruction = "Read within the frozen scope."
    plan = inbox_distribution_plan("conversation-1")
    scope = ExecutionScope(
        scope_id="scope-routine",
        revision=1,
        agent_id="agent-1",
        principal_id="agent:agent-1",
        grant_id="routine:routine-1",
        job_id=None,
        job_revision=None,
        allowed_source_ids=("source-1",),
        allowed_resource_ids=("resource-1",),
        allowed_capability_ids=("data.sqlite.query",),
        allowed_access_modes=frozenset({AccessMode.READ}),
        allowed_operational_effects=frozenset({OperationalEffect.NONE}),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        eligible_model_routes=("mock",),
        per_run_max_cost_usd=Decimal("0"),
        per_run_max_tokens=1_000,
        distribution_plan_digest=plan.plan_digest,
        routine_id="routine-1",
        routine_revision=1,
        occurrence_id="occurrence-1",
    )
    scheduled = RunInput(
        id="run-scheduled",
        agent_id="agent-1",
        conversation_id="conversation-1",
        message=instruction,
        created_at=NOW,
        start=RunStartEnvelope(
            origin=RunOrigin.SCHEDULED_ROUTINE,
            instruction_authority=InstructionAuthority.FOREGROUND_AUTHORIZED,
            trusted_instruction_id="routine:routine-1:revision:1",
            trusted_instruction=instruction,
            instruction_digest=(
                "sha256:" + sha256(instruction.encode("utf-8")).hexdigest()
            ),
            untrusted_payload={},
            payload_digest="sha256:" + sha256(b"{}").hexdigest(),
            execution_scope=scope,
        ),
    )
    assert await domain.project(scheduled) == ()


async def test_distribution_executors_return_only_safe_bounded_projections() -> None:
    domain, executors = _domain()
    by_id = {item.executor_id: item for item in executors}
    destination = await by_id["distribution.destination.list.executor"].execute(
        ToolExecution(
            run_id="run-user",
            call_id="call-destinations",
            capability_id="distribution.destination.list",
            conversation_id="conversation-1",
            arguments={"_request_sensitivity": "internal"},
        )
    )
    listed = await by_id["distribution.delivery.list.executor"].execute(
        ToolExecution(
            run_id="run-user",
            call_id="call-list",
            capability_id="distribution.delivery.list",
        )
    )
    inspected = await by_id["distribution.delivery.inspect.executor"].execute(
        ToolExecution(
            run_id="run-user",
            call_id="call-inspect",
            capability_id="distribution.delivery.inspect",
            arguments={"delivery_id": "delivery-1"},
        )
    )

    assert destination.data["count"] == 1
    assert listed.data["count"] == 1
    assert inspected.data["delivery_id"] == "delivery-1"
    assert "payload" not in inspected.data
    assert "provider" not in inspected.data


async def test_foreground_model_discovers_destination_through_static_domain(
    tmp_path,
) -> None:
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="call-destinations",
                        name=DISTRIBUTION_DESTINATION_LIST_TOOL_NAME,
                        arguments={},
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="The current conversation Inbox is selectable.",
            ),
        ),
        provider_id="mock:distribution-discovery",
        complete_pricing=True,
    )
    agent = await Agent.create(
        "distribution-discovery",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
    )
    try:
        result = await agent.run("Which distribution destination can I select?")
        transcript = await agent.transcript(result.run_id)
        tool_results = tuple(
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
        assert len(tool_results) == 1
        assert not tool_results[0].is_error, tool_results[0].output
        assert tool_results[0].capability_id == ("distribution.destination.list")
        data = tool_results[0].output["data"]
        assert isinstance(data, FrozenJsonObject)
        assert data["count"] == 1
        assert result.final_text == ("The current conversation Inbox is selectable.")
        assert DISTRIBUTION_DESTINATION_LIST_TOOL_NAME in {
            tool.name for tool in provider.requests[0].tools
        }
    finally:
        await agent.close()
