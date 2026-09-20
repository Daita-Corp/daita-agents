from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from daita.capabilities import (
    AccessMode,
    AutomationEligibility,
    Capability,
    ExecutionAdmissionPolicy,
    ExecutionPreference,
    GraphTaskBinding,
    OperationalEffect,
    ToolExecution,
)
from daita.context import AgentContextBuilder
from daita.jobs.graph.admission import (
    GraphAdmissionBuilder,
    RegistryInitialTaskProposalResolver,
    StartGraphJobExecutor,
    graph_admission_declarations,
)
from daita.jobs.graph.context import TaskContextBundle
from daita.jobs.graph.models import AttemptState, GraphState, TaskRole, TaskState
from daita.jobs.supervisor import _fair_graph_dispatch_order
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
    TextBlock,
    ToolCall,
)
from daita.llm.pricing import CostEstimate
from tests.support.conversations import CatalogSpy
from tests.support.graph import graph_admission
from tests.support.model_graph_integration import (
    AGENT_ID,
    CONVERSATION_ID,
    MODEL_ROUTE_ID,
    READ_CAPABILITY_ID,
    ModelGraphIntegration,
)

pytestmark = pytest.mark.integration


def test_graph_dispatch_never_exceeds_two_when_another_graph_is_ready() -> None:
    first = graph_admission(job_id="job-first").tasks[0]
    second = graph_admission(job_id="job-second").tasks[0]
    ready = tuple(
        [replace(first, task_id=f"first-{index}") for index in range(4)]
        + [replace(second, task_id=f"second-{index}") for index in range(2)]
    )
    ordered = _fair_graph_dispatch_order(ready, last_job_id=None, consecutive=0)
    assert tuple(task.job_id for task in ordered) == (
        "job-first",
        "job-first",
        "job-second",
        "job-second",
        "job-first",
        "job-first",
    )


class _BlockingReadModel:
    provider_id = MODEL_ROUTE_ID

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def model_profile(self) -> ModelProfile:
        return ModelProfile(
            id=self.provider_id,
            context_window_tokens=128_000,
            max_output_tokens=2_048,
            supports_tools=True,
        )

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        assert isinstance(request, ModelRequest)
        self.entered.set()
        await self.release.wait()
        return ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="stale-read",
                    name="graph_read",
                    arguments={
                        "resource_ids": tuple(f"resource-{index}" for index in range(5))
                    },
                ),
            ),
            usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
        )

    async def close(self, *, deadline: float | None = None) -> None:
        del deadline


async def test_model_task_completes_only_through_exact_lifecycle_directive(
    tmp_path: Path,
) -> None:
    integration = await ModelGraphIntegration.open(tmp_path)
    try:
        admission = integration.build()
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)

        assert terminal.job.state is GraphState.SUCCEEDED
        work = next(item for item in terminal.tasks if item.role is TaskRole.WORKER)
        finalizer = next(
            item for item in terminal.tasks if item.role is TaskRole.FINALIZER
        )
        assert work.state is TaskState.SUCCEEDED
        assert finalizer.state is TaskState.SUCCEEDED
        model_attempt = next(
            item for item in terminal.attempts if item.task_id == work.task_id
        )
        assert model_attempt.state is AttemptState.SUCCEEDED
        assert len(integration.reader.calls) == 1
        transcript = await integration.store.load(model_attempt.run_id)
        result = await integration.store.result(model_attempt.run_id)
        assert result is not None
        assert result.kind.value == "machine_terminated"
        assert result.reason == "task_completed"
        assert transcript.messages[-1].role.value == "tool"
        assert len(terminal.results) == 2
        assert len(terminal.delivery_ids) == 1
        integration.provider.assert_consumed()
    finally:
        await integration.close()


async def test_real_context_builder_uses_only_the_immutable_task_bundle(
    tmp_path: Path,
) -> None:
    integration = await ModelGraphIntegration.open(tmp_path)
    loop = integration.supervisor._graph_model_loop
    assert loop is not None
    loop._context_builder = AgentContextBuilder(
        CatalogSpy(),
        profile=integration.provider.model_profile,
    )
    try:
        admission = integration.build()
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)
        assert terminal.job.state is GraphState.SUCCEEDED
        system = integration.provider.requests[0].messages[0].content[0]
        assert isinstance(system, TextBlock)
        assert "Code-owned graph-task protocol" in system.text
        assert '"callable_tool_names":["graph_read"]' in system.text
        assert "untrusted_prior_attempts" in system.text
        assert "unrelated conversation" not in system.text
        complete = next(
            tool
            for tool in integration.provider.requests[0].tools
            if tool.name == "task_complete"
        )
        required = complete.input_schema["required"]
        properties = complete.input_schema["properties"]
        assert isinstance(required, tuple) and isinstance(properties, Mapping)
        assert "evidence_call_ids" not in required
        assert "artifact_ids" not in required
        assert "Never put receipt" in properties["evidence_call_ids"]["description"]
        assert "receipt IDs are never" in properties["artifact_ids"]["description"]
    finally:
        await integration.close()


async def test_normal_assistant_text_is_protocol_violation_and_bounded_retry(
    tmp_path: Path,
) -> None:
    usage = ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0")))
    integration = await ModelGraphIntegration.open(
        tmp_path,
        script=(
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="I am done without committing a lifecycle result.",
                usage=usage,
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Still no lifecycle result.",
                usage=usage,
            ),
        ),
    )
    try:
        admission = integration.build()
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)

        assert terminal.job.state is GraphState.FAILED
        work = next(item for item in terminal.tasks if item.role is TaskRole.WORKER)
        attempts = tuple(
            item for item in terminal.attempts if item.task_id == work.task_id
        )
        assert len(attempts) == 2
        assert all(item.state is AttemptState.PROTOCOL_VIOLATION for item in attempts)
        assert work.attempt_count == 2
        assert terminal.results == ()
        assert integration.reader.calls == []
        integration.provider.assert_consumed()
    finally:
        await integration.close()


async def test_foreground_start_graph_job_rederives_typed_initial_task(
    tmp_path: Path,
) -> None:
    integration = await ModelGraphIntegration.open(tmp_path)
    try:
        resolver = RegistryInitialTaskProposalResolver(
            agent_id=AGENT_ID,
            registry=integration.registry,
            contract_reader=integration.contracts,
            model_route_id=MODEL_ROUTE_ID,
            max_steps=6,
            per_run_max_tokens=10_000,
            per_run_max_cost_usd=Decimal("1"),
        )
        builder = GraphAdmissionBuilder(
            agent_id=AGENT_ID,
            registry=integration.registry,
            distribution=integration.distribution,
            clock=integration.clock,
            id_factory=integration.ids,
        )
        declarations, executors = graph_admission_declarations(
            owner=integration.owner,
            resolver=resolver,
            builder=builder,
        )
        assert declarations.tool_views[0].name == "start_graph_job"
        starter = next(
            item for item in executors if isinstance(item, StartGraphJobExecutor)
        )
        resource_ids = tuple(f"resource-{item}" for item in range(5))
        output = await starter.execute(
            ToolExecution(
                run_id="run-" + "b" * 32,
                call_id="start-typed-graph",
                capability_id="jobs.graph.start",
                conversation_id=CONVERSATION_ID,
                arguments={
                    "objective": "Perform the exact bounded durable read.",
                    "outcome_contract": {"required_result_kind": "test.graph.read"},
                    "deadline_seconds": 300,
                    "initial_task": {
                        "capability_id": READ_CAPABILITY_ID,
                        "arguments": {"resource_ids": resource_ids},
                        "expected_result_contract": {"result_kind": "test.graph.read"},
                        "retained_references": {
                            "source_ids": (),
                            "resource_ids": resource_ids,
                            "connector_binding_ids": (),
                        },
                    },
                },
                request_sensitivity=ModelSensitivity.INTERNAL,
            )
        )
        assert output.kind == "graph.job_started"
        job_id = output.data["job_id"]
        assert isinstance(job_id, str)
        inspection = await integration.owner.inspect_graph(job_id)
        assert inspection is not None
        assert len(inspection.tasks) == 2
        assert {item.execution_kind.value for item in inspection.tasks} == {
            "model",
            "internal_capability",
        }
        assert all(
            item.specification.created_by == "job_owner" for item in inspection.tasks
        )
    finally:
        await integration.close()


async def test_fenced_attempt_racing_model_response_performs_no_tool_io(
    tmp_path: Path,
) -> None:
    integration = await ModelGraphIntegration.open(tmp_path)
    blocking = _BlockingReadModel()
    integration.supervisor._graph_model_loop._model = blocking  # type: ignore[union-attr]
    try:
        admission = integration.build()
        await integration.admit_and_start(admission)
        await asyncio.wait_for(blocking.entered.wait(), timeout=2)
        inspection = await integration.owner.inspect_graph(admission.job.job_id)
        assert inspection is not None
        work = next(item for item in inspection.tasks if item.role is TaskRole.WORKER)
        attempt = next(
            item for item in inspection.attempts if item.task_id == work.task_id
        )
        fenced = await integration.store.fence_graph_attempt(
            AGENT_ID,
            attempt.job_id,
            attempt.task_id,
            attempt.attempt_id,
            fencing_epoch=attempt.fencing_epoch,
            fenced_at=integration.clock(),
            requeue=False,
            reason_code="cancelled_during_model_request",
        )
        assert fenced is not None and fenced.state is AttemptState.FENCED
        blocking.release.set()
        await asyncio.sleep(0.1)
        assert integration.reader.calls == []
        current = await integration.owner.inspect_graph(admission.job.job_id)
        assert current is not None
        current_attempt = next(
            item for item in current.attempts if item.attempt_id == attempt.attempt_id
        )
        assert current_attempt.state is AttemptState.FENCED
    finally:
        blocking.release.set()
        await integration.close()


@pytest.mark.parametrize("revocation", ("contract", "deadline"))
async def test_revoked_or_expired_attempt_racing_model_response_performs_no_tool_io(
    tmp_path: Path,
    revocation: str,
) -> None:
    now = [datetime(2026, 9, 17, 12, tzinfo=UTC)]
    integration = await ModelGraphIntegration.open(tmp_path, clock=lambda: now[0])
    blocking = _BlockingReadModel()
    integration.supervisor._graph_model_loop._model = blocking  # type: ignore[union-attr]
    try:
        admission = integration.build()
        await integration.admit_and_start(admission)
        await asyncio.wait_for(blocking.entered.wait(), timeout=2)
        if revocation == "contract":
            integration.contract_state["revoked"] = True
        else:
            now[0] += timedelta(hours=2)
        blocking.release.set()
        await asyncio.sleep(0.1)
        assert integration.reader.calls == []
        current = await integration.owner.inspect_graph(admission.job.job_id)
        assert current is not None
        assert not any(
            result.task_id
            == next(
                task.task_id for task in current.tasks if task.role is TaskRole.WORKER
            )
            for result in current.results
        )
    finally:
        blocking.release.set()
        await integration.close()


async def test_checkpoint_comment_and_prelimit_warning_remain_bounded(
    tmp_path: Path,
) -> None:
    usage = ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0")))

    def response(
        call_id: str, name: str, arguments: dict[str, object]
    ) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
            usage=usage,
        )

    integration = await ModelGraphIntegration.open(
        tmp_path,
        script=(
            response(
                "checkpoint-1", "task_checkpoint", {"milestone": "one", "payload": {}}
            ),
            response("comment-1", "task_comment", {"body": "first comment"}),
            response(
                "checkpoint-2", "task_checkpoint", {"milestone": "two", "payload": {}}
            ),
            response("comment-2", "task_comment", {"body": "second comment"}),
            response(
                "read-call",
                "graph_read",
                {"resource_ids": tuple(f"resource-{item}" for item in range(5))},
            ),
            response(
                "complete-call",
                "task_complete",
                {
                    "result_kind": "test.graph.read",
                    "summary": "Completed after durable progress records.",
                    "payload": {"value": 42},
                    "evidence_call_ids": ("read-call",),
                    "artifact_ids": (),
                    "residual_risk": None,
                    "downstream_constraints": {},
                },
            ),
        ),
    )
    try:
        admission = integration.build()
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)
        work = next(item for item in terminal.tasks if item.role is TaskRole.WORKER)
        attempt = next(
            item for item in terminal.attempts if item.task_id == work.task_id
        )
        checkpoints = tuple(
            item for item in terminal.checkpoints if item.task_id == work.task_id
        )
        comments = tuple(
            item for item in terminal.comments if item.task_id == work.task_id
        )
        assert len(checkpoints) == 3  # one supervisor start plus two model checkpoints
        assert len(comments) == 2
        transcript = await integration.store.load(attempt.run_id)
        assert any(
            "Code-owned task budget warning" in block.text
            for message in transcript.messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
    finally:
        await integration.close()


@pytest.mark.parametrize(
    ("tool_name", "arguments", "attempt_state", "exit_reason"),
    (
        (
            "task_block",
            {
                "kind": "needs_input",
                "message": "Need one bounded input.",
                "details": {},
            },
            AttemptState.BLOCKED,
            "task_blocked",
        ),
        (
            "task_request_review",
            {
                "result_kind": "test.graph.read",
                "summary": "Review this bounded candidate.",
                "payload": {"candidate": True},
                "evidence_call_ids": (),
                "artifact_ids": (),
                "residual_risk": None,
                "downstream_constraints": {},
                "message": "Review this bounded candidate.",
            },
            AttemptState.REVIEW_REQUESTED,
            "task_review_requested",
        ),
    ),
)
async def test_block_and_review_are_exclusive_machine_terminators(
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, object],
    attempt_state: AttemptState,
    exit_reason: str,
) -> None:
    integration = await ModelGraphIntegration.open(
        tmp_path,
        script=(
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="terminal-control", name=tool_name, arguments=arguments
                    ),
                ),
                usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
            ),
        ),
    )
    try:
        admission = integration.build()
        await integration.admit_and_start(admission)
        deadline = asyncio.get_running_loop().time() + 2
        terminal_attempt = None
        while asyncio.get_running_loop().time() < deadline:
            inspection = await integration.owner.inspect_graph(admission.job.job_id)
            assert inspection is not None
            terminal_attempt = next(
                (item for item in inspection.attempts if item.state is attempt_state),
                None,
            )
            if terminal_attempt is not None:
                break
            await asyncio.sleep(0.005)
        assert terminal_attempt is not None
        exit = None
        while asyncio.get_running_loop().time() < deadline and exit is None:
            exit = await integration.store.result(terminal_attempt.run_id)
            if exit is None:
                await asyncio.sleep(0.005)
        assert exit is not None and exit.reason == exit_reason
        inspection = await integration.owner.inspect_graph(admission.job.job_id)
        assert inspection is not None
        assert len(inspection.controls) == 1
        assert integration.reader.calls == []
    finally:
        await integration.close()


def test_execution_admission_policy_has_exact_auto_and_durable_outcomes() -> None:
    policy = ExecutionAdmissionPolicy(
        shape="relational_read",
        inline_eligible=True,
        graph_v1_eligible=True,
        target_count_argument="resource_ids",
        inline_max_targets=4,
        graph_max_targets=16,
    )
    assert (
        policy.admission_error(
            {"resource_ids": ("a", "b", "c", "d")}, ExecutionPreference.AUTO
        )
        is None
    )
    assert (
        policy.admission_error(
            {"resource_ids": tuple(str(item) for item in range(5))},
            ExecutionPreference.AUTO,
        )
        == "durable_execution_required"
    )
    assert (
        policy.admission_error(
            {"resource_ids": tuple(str(item) for item in range(17))},
            ExecutionPreference.AUTO,
        )
        == "execution_not_supported_for_requested_shape"
    )
    assert (
        policy.admission_error({"resource_ids": ("a",)}, ExecutionPreference.DURABLE)
        == "durable_execution_required"
    )


def test_task_context_is_immutable_bounded_and_excludes_conversation_history() -> None:
    now = datetime(2026, 9, 17, tzinfo=UTC)
    binding = GraphTaskBinding(
        agent_id="agent-a",
        job_id="job-a",
        root_authority_digest="sha256:" + "1" * 64,
        task_id="task-a",
        task_revision=2,
        task_spec_digest="sha256:" + "2" * 64,
        task_scope_digest="sha256:" + "3" * 64,
        attempt_id="attempt-a",
        claim_token_digest="sha256:" + "4" * 64,
        fencing_epoch=1,
        graph_revision_at_claim=1,
        task_role="worker",
        task_deadline_at=now + timedelta(minutes=5),
        job_deadline_at=now + timedelta(hours=1),
        budget_reservation_identity="sha256:" + "5" * 64,
    )
    context = TaskContextBundle(
        binding=binding,
        root_objective="Do the exact bounded work.",
        outcome_contract={"kind": "answer"},
        task_specification={"kind": "model_task"},
        parent_results=({"summary": "untrusted parent"},),
        prior_attempts=({"error_code": "protocol_violation"},) * 2,
        checkpoints=tuple({"ordinal": item} for item in range(8)),
        comments=tuple({"body": str(item)} for item in range(16)),
        created_at=now,
    )
    material = context.material()
    assert "conversation" not in str(material).lower()
    prior_attempts = material["untrusted_prior_attempts"]
    checkpoints = material["untrusted_checkpoints"]
    comments = material["untrusted_comments"]
    assert isinstance(prior_attempts, tuple) and len(prior_attempts) == 2
    assert isinstance(checkpoints, tuple) and len(checkpoints) == 8
    assert isinstance(comments, tuple) and len(comments) == 16
    with pytest.raises(ValueError, match="prior_attempts"):
        TaskContextBundle(
            binding=binding,
            root_objective="Too much retry evidence.",
            outcome_contract={},
            task_specification={},
            prior_attempts=({}, {}, {}),
            created_at=now,
        )


def test_effectful_or_async_policy_is_not_graph_eligible() -> None:
    policy = ExecutionAdmissionPolicy(
        shape="async_remote",
        inline_eligible=False,
        graph_v1_eligible=True,
        target_count_argument=None,
        inline_max_targets=0,
        graph_max_targets=1,
        synchronous_only=False,
    )
    assert (
        policy.admission_error({}, ExecutionPreference.DURABLE)
        == "execution_not_supported_for_requested_shape"
    )
    with pytest.raises(ValueError, match="structurally effect-free"):
        Capability(
            id="test.effectful.graph",
            description="Forbidden graph effect.",
            input_schema={"type": "object"},
            output_kind="test.effect",
            output_schema={"type": "object"},
            executor_id="test.effect.executor",
            access_mode=AccessMode.NONE,
            operational_effect=OperationalEffect.SUBMIT_EXECUTION_GRAPH,
            automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
            execution_admission_policy=ExecutionAdmissionPolicy(
                shape="effect",
                inline_eligible=True,
                graph_v1_eligible=True,
                target_count_argument=None,
                inline_max_targets=1,
                graph_max_targets=1,
            ),
        )
