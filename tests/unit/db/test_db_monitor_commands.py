import pytest

from daita.db import DbAgent, DbRuntime
from daita.db.models import DbIntent, DbIntentKind, DbOperationContract
from daita.db.models import DbOperationResult, DbRequest
from daita.runtime import (
    AccessMode,
    ApprovalRequest,
    ApprovalStatus,
    Operation,
    OperationStatus,
    RiskLevel,
)


class SpyDbRuntime(DbRuntime):
    def __init__(self) -> None:
        super().__init__(runtime_id="db-monitor-command-runtime")
        self.calls: list[tuple[str, object]] = []

    async def create_monitor(self, monitor):
        self.calls.append(("create_monitor", monitor.id))
        return await super().create_monitor(monitor)

    async def list_monitors(self, *, status=None):
        self.calls.append(("list_monitors", status))
        return await super().list_monitors(status=status)

    async def inspect_monitor(self, monitor_id):
        self.calls.append(("inspect_monitor", monitor_id))
        return await super().inspect_monitor(monitor_id)

    async def update_monitor(self, monitor_id, patch):
        self.calls.append(("update_monitor", monitor_id))
        return await super().update_monitor(monitor_id, patch)

    async def pause_monitor(self, monitor_id, *, paused_until=None):
        self.calls.append(("pause_monitor", monitor_id))
        return await super().pause_monitor(
            monitor_id,
            paused_until=paused_until,
        )

    async def resume_monitor(self, monitor_id):
        self.calls.append(("resume_monitor", monitor_id))
        return await super().resume_monitor(monitor_id)

    async def delete_monitor(self, monitor_id):
        self.calls.append(("delete_monitor", monitor_id))
        return await super().delete_monitor(monitor_id)


class RunSpyRuntime(DbRuntime):
    def __init__(self) -> None:
        super().__init__(runtime_id="db-monitor-command-run-spy")
        self.run_requests: list[DbRequest] = []

    async def run(self, request):
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        self.run_requests.append(db_request)
        return DbOperationResult(
            operation_id="ordinary-db-run",
            request=db_request,
            intent=DbIntent(
                kind=DbIntentKind.DATA_QUERY,
                confidence=0.9,
                access=AccessMode.READ,
            ),
            contract=DbOperationContract(
                operation_type="data.query",
                access=AccessMode.READ,
            ),
            status=OperationStatus.SUCCEEDED,
            answer="ordinary path",
        )


async def test_prompt_managed_monitor_crud_uses_typed_apis_and_audits_runtime():
    runtime = SpyDbRuntime()
    agent = DbAgent(runtime=runtime)

    created = await agent.run_detailed(
        "Monitor pending orders every 15 minutes. If pending orders exceed 500 "
        "then inspect freshness."
    )
    assert created.status is OperationStatus.SUCCEEDED
    assert created.contract.operation_type == "monitor.create"
    assert created.diagnostics["monitor"]["id"] == "pending_orders"

    listed = await agent.run("List active monitors.")
    assert "pending_orders" in listed
    inspected = await agent.run("Inspect monitor pending_orders.")
    assert "pending_orders" in inspected
    updated = await agent.run(
        "Make pending_orders monitor less noisy; require two bad checks."
    )
    assert "Updated monitor" in updated
    paused = await agent.run("Please pause the pending_orders monitor until Monday.")
    assert "Paused monitor" in paused
    resumed = await agent.run("Resume pending_orders monitor.")
    assert "Resumed monitor" in resumed
    explained = await agent.run("Why did pending_orders monitor trigger today?")
    assert "no recorded runs" in explained
    deleted = await agent.run("Delete pending_orders monitor.")
    assert "Deleted monitor" in deleted

    assert ("create_monitor", "pending_orders") in runtime.calls
    assert ("list_monitors", "active") in runtime.calls
    assert ("inspect_monitor", "pending_orders") in runtime.calls
    assert ("update_monitor", "pending_orders") in runtime.calls
    assert ("pause_monitor", "pending_orders") in runtime.calls
    assert ("resume_monitor", "pending_orders") in runtime.calls
    assert ("delete_monitor", "pending_orders") in runtime.calls

    operations = await runtime.store.list_operations()
    assert [operation.operation_type for operation in operations] == [
        "monitor.create",
        "monitor.list",
        "monitor.inspect",
        "monitor.update",
        "monitor.pause",
        "monitor.resume",
        "monitor.explain_run",
        "monitor.delete",
    ]
    assert all(
        operation.status is OperationStatus.SUCCEEDED for operation in operations
    )
    for operation in operations:
        evidence = await runtime.store.list_evidence(operation.id)
        events = await runtime.store.list_events(operation.id)
        assert evidence, operation.operation_type
        assert len(events) == 2
        assert evidence[0].owner == "db.monitor"
    assert await runtime.store.list_tasks() == []
    assert await agent.list_monitors() == ()


async def test_prompt_monitor_create_blocks_unsupported_action_and_audits_validation():
    runtime = DbRuntime(runtime_id="db-monitor-validation-runtime")
    agent = DbAgent(runtime=runtime)

    result = await agent.run_detailed(
        "Monitor pending orders every 15 minutes then notify #ops."
    )

    assert result.status is OperationStatus.BLOCKED
    assert result.warnings == ("db_monitor_validation_failed",)
    assert "slack.message.send" in result.answer
    assert await agent.list_monitors() == ()

    operations = await runtime.store.list_operations()
    assert [operation.operation_type for operation in operations] == ["monitor.create"]
    evidence = await runtime.store.list_evidence(operations[0].id)
    events = await runtime.store.list_events(operations[0].id)
    assert evidence[0].kind == "monitor.command.validation"
    assert evidence[0].payload["validation"]["missing_capabilities"] == [
        "slack.message.send"
    ]
    assert len(events) == 2
    assert await runtime.store.list_tasks() == []


async def test_create_monitor_prompt_requires_approval_then_resumes_to_create_monitor():
    runtime = DbRuntime(runtime_id="db-monitor-create-approval-runtime")
    agent = DbAgent(runtime=runtime)

    result = await agent.run_detailed(
        "Create a monitor for the users table. I want to be notified everytime "
        "a new user gets added to the table"
    )

    assert result.status is OperationStatus.BLOCKED
    assert result.contract.operation_type == "monitor.create"
    assert result.warnings == ("db_monitor_approval_required",)
    assert await agent.list_monitors() == ()

    approvals = await runtime.store.list_approval_requests(result.operation_id)
    assert len(approvals) == 1
    assert approvals[0].status is ApprovalStatus.PENDING
    assert approvals[0].proposed_action["operation_type"] == "monitor.create"

    await runtime.approval_channel.approve(approvals[0].approval_id)
    resumed = await runtime.resume_operation(result.operation_id)

    assert resumed.operation.status is OperationStatus.SUCCEEDED
    monitors = await agent.list_monitors()
    assert len(monitors) == 1
    assert monitors[0].id == "users_table"
    evidence = await runtime.store.list_evidence(result.operation_id)
    assert any(item.kind == "monitor.definition" and item.accepted for item in evidence)


async def test_non_monitor_and_ambiguous_prompts_continue_to_db_runtime_run():
    runtime = RunSpyRuntime()
    agent = DbAgent(runtime=runtime)

    ordinary = await agent.run("Generate a churn report for last quarter.")
    ambiguous = await agent.run("Monitor orders.")

    assert ordinary == "ordinary path"
    assert ambiguous == "ordinary path"
    assert [request.prompt for request in runtime.run_requests] == [
        "Generate a churn report for last quarter.",
        "Monitor orders.",
    ]
    assert await agent.list_monitors() == ()
    assert await runtime.store.list_operations() == []


async def test_prompt_monitor_approval_delegates_and_resumes_without_placeholder():
    runtime = DbRuntime(runtime_id="db-monitor-command-approval")
    agent = DbAgent(runtime=runtime)
    await agent.monitor(name="Helper Monitor", monitor_id="helper")
    await runtime.store.save_operation(
        Operation(
            id="monitor-action-op",
            operation_type="monitor.triggered",
            status=OperationStatus.BLOCKED,
            metadata={
                "monitor_action_context": {
                    "monitor_id": "helper",
                    "monitor_run_id": "run-1",
                    "tick_operation_id": "tick-1",
                    "action_kind": "write_proposal",
                    "action_plan_fingerprint": "fingerprint-1",
                }
            },
        )
    )
    approval = ApprovalRequest(
        approval_id="monitor-action-op:approval_required_for_writes:human",
        operation_id="monitor-action-op",
        reason="approve monitor write",
        proposed_action={"approval": "human"},
        risk=RiskLevel.HIGH,
        requested_by_policy_id="approval_required_for_writes",
        owner="runtime",
    )
    await runtime.approval_channel.request(approval)

    answer = await agent.run("Approve the action proposed by the helper monitor.")
    approvals = await runtime.store.list_approval_requests("monitor-action-op")

    assert "Approved monitor approval" in answer
    assert "later phase" not in answer
    assert "no approval state was changed" not in answer
    assert approvals[0].status is ApprovalStatus.APPROVED
    command_ops = [
        operation
        for operation in await runtime.store.list_operations()
        if operation.operation_type == "monitor.approve_action"
    ]
    assert command_ops
    evidence = await runtime.store.list_evidence(command_ops[-1].id)
    assert evidence[-1].kind == "monitor.command.approval"
    assert evidence[-1].payload["approval_status"] == "approved"


@pytest.mark.parametrize(
    ("prompt", "expected_answer", "expected_status"),
    [
        (
            "Reject the action proposed by the helper monitor.",
            "Rejected monitor approval",
            ApprovalStatus.REJECTED,
        ),
        (
            "Cancel the action proposed by the helper monitor.",
            "Cancelled monitor approval",
            ApprovalStatus.CANCELLED,
        ),
    ],
)
async def test_prompt_monitor_terminal_approval_commands_are_real(
    prompt,
    expected_answer,
    expected_status,
):
    runtime = DbRuntime(runtime_id=f"db-monitor-command-{expected_status.value}")
    agent = DbAgent(runtime=runtime)
    await agent.monitor(name="Helper Monitor", monitor_id="helper")
    await runtime.store.save_operation(
        Operation(
            id="monitor-action-op",
            operation_type="monitor.triggered",
            status=OperationStatus.BLOCKED,
            metadata={
                "monitor_action_context": {
                    "monitor_id": "helper",
                    "monitor_run_id": "run-1",
                    "tick_operation_id": "tick-1",
                    "action_kind": "write_proposal",
                    "action_plan_fingerprint": "fingerprint-1",
                }
            },
        )
    )
    approval = ApprovalRequest(
        approval_id=f"monitor-action-op:approval_required_for_writes:{expected_status.value}",
        operation_id="monitor-action-op",
        reason="approve monitor write",
        proposed_action={"approval": "human"},
        risk=RiskLevel.HIGH,
        requested_by_policy_id="approval_required_for_writes",
        owner="runtime",
    )
    await runtime.approval_channel.request(approval)

    answer = await agent.run(prompt)
    approvals = await runtime.store.list_approval_requests("monitor-action-op")

    assert expected_answer in answer
    assert "later phase" not in answer
    assert "no approval state was changed" not in answer
    assert approvals[0].status is expected_status
