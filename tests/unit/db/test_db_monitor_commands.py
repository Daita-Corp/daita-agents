from dataclasses import replace

import pytest

from daita.db import DbAgent, DbMonitor, DbRuntime
from daita.db.monitor_commands import DbMonitorCommand, DbMonitorPlanner
from daita.db.models import DbIntent, DbIntentKind, DbOperationContract
from daita.db.models import DbOperationResult, DbRequest
from daita.runtime import (
    AccessMode,
    ApprovalRequest,
    ApprovalStatus,
    Capability,
    Evidence,
    EvidenceSchema,
    Operation,
    OperationStatus,
    RiskLevel,
    Task,
    TaskDependency,
    TaskStatus,
)
from daita.plugins import RuntimeExtensionPlugin, PluginKind, PluginManifest


class SchemaInspectProbeExecutor:
    id = "schema_probe.schema.inspect"
    capability_ids = frozenset({"db.schema.inspect"})

    def __init__(self, tables):
        self.tables = tuple(tables)
        self.calls = []

    async def execute(self, task: Task, operation: Operation, context):
        requested = tuple(task.input.get("tables") or ())
        self.calls.append(requested)
        tables = [
            table
            for table in self.tables
            if not requested or table["name"] in requested
        ]
        return [
            Evidence(
                kind="schema.asset_profile",
                owner="schema_probe",
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "database_type": "test",
                    "table_count": len(tables),
                    "tables": tables,
                },
            )
        ]


class SchemaInspectProbePlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="schema_probe",
        display_name="Schema Probe",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def __init__(self, tables):
        self.executor = SchemaInspectProbeExecutor(tables)

    def declare_capabilities(self):
        return (
            Capability(
                id="db.schema.inspect",
                owner="schema_probe",
                description="Inspect test schema metadata.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"monitor.create", "schema.query"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"schema.asset_profile"}),
                executor=self.executor.id,
                runtime_only=True,
                side_effecting=False,
            ),
        )

    def get_executors(self):
        return (self.executor,)

    def declare_evidence_schemas(self):
        return (
            EvidenceSchema(
                kind="schema.asset_profile",
                owner="schema_probe",
                json_schema={"type": "object"},
            ),
        )


def _schema_probe(*table_names):
    tables = []
    for table_name in table_names:
        tables.append(
            {
                "name": table_name,
                "columns": [
                    {
                        "name": "id",
                        "data_type": "uuid",
                        "is_primary_key": True,
                    },
                    {
                        "name": "created_at",
                        "data_type": "timestamp",
                        "is_primary_key": False,
                    },
                ],
            }
        )
    return SchemaInspectProbePlugin(tables)


async def _create_helper_monitor(agent: DbAgent) -> None:
    await agent.create_monitor(
        DbMonitor(
            id="helper",
            name="Helper Monitor",
            observation_plan={
                "kind": "metric_sql",
                "metric": "helper_count",
                "sql": "select count(*) as helper_count from helper_events",
                "value_path": "rows.0.helper_count",
            },
        )
    )


def test_legacy_planner_create_monitor_rejects_incomplete_proposal():
    runtime = DbRuntime(runtime_id="db-monitor-legacy-planner-runtime")
    planner = DbMonitorPlanner(registry=runtime.registry, limits={})
    command = DbMonitorCommand(
        kind="create",
        prompt="Create a monitor for the users table when a new user is added.",
    )

    with pytest.raises(ValueError, match="monitor proposal is incomplete"):
        planner.create_monitor(command)


class SpyDbRuntime(DbRuntime):
    def __init__(self, *, plugins=()) -> None:
        super().__init__(
            runtime_id="db-monitor-command-runtime",
            plugins=tuple(plugins),
        )
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


class ResumeTaskSpyRuntime(DbRuntime):
    def __init__(self, *, plugins=()) -> None:
        super().__init__(
            runtime_id="db-monitor-resume-task-spy",
            plugins=tuple(plugins),
        )
        self.executed_capabilities: list[str] = []

    async def execute_task(self, task, operation, context=None):
        self.executed_capabilities.append(task.capability_id)
        return await super().execute_task(task, operation, context=context)


async def _approval_required_create(runtime: DbRuntime):
    agent = DbAgent(runtime=runtime)
    result = await agent.run_detailed(
        "Create a monitor for the users table. I want to be notified everytime "
        "a new user gets added to the table"
    )
    assert result.status is OperationStatus.BLOCKED
    approvals = await runtime.store.list_approval_requests(result.operation_id)
    assert len(approvals) == 1
    tasks = await runtime.store.list_tasks(result.operation_id)
    plan_task = next(
        task for task in tasks if task.capability_id == "db.monitor.plan_create"
    )
    commit_task = next(
        task for task in tasks if task.capability_id == "db.monitor.commit_create"
    )
    evidence = await runtime.store.list_evidence(result.operation_id)
    proposal = next(item for item in evidence if item.kind == "monitor.proposal")
    return agent, result, approvals[0], plan_task, commit_task, proposal


async def test_prompt_managed_monitor_crud_uses_typed_apis_and_audits_runtime():
    schema_probe = _schema_probe("pending_orders")
    runtime = SpyDbRuntime(plugins=(schema_probe,))
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

    assert ("create_monitor", "pending_orders") not in runtime.calls
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
        if operation.operation_type == "monitor.create":
            assert [item.kind for item in evidence] == [
                "schema.asset_profile",
                "monitor.proposal",
                "monitor.definition",
            ]
            proposal = next(
                item for item in evidence if item.kind == "monitor.proposal"
            )
            definition = next(
                item for item in evidence if item.kind == "monitor.definition"
            )
            assert proposal.owner == "db_runtime"
            assert definition.owner == "db_runtime"
            assert proposal.payload["observation_plan"]["kind"] == "planned_read"
            assert proposal.payload["observation_plan"]["cursor"]["field"] == (
                "created_at"
            )
            assert schema_probe.executor.calls == [("pending_orders",)]
            assert "watch" not in definition.payload["monitor"]["observation_plan"]
        elif operation.operation_type in {
            "monitor.update",
            "monitor.pause",
            "monitor.resume",
            "monitor.delete",
        }:
            assert evidence[0].kind == "monitor.proposal"
            assert evidence[0].owner == "db_runtime"
            assert evidence[1].owner == "db_runtime"
            if operation.operation_type == "monitor.update":
                assert evidence[1].kind == "monitor.state_update"
            elif operation.operation_type == "monitor.pause":
                assert evidence[1].kind == "monitor.paused"
            elif operation.operation_type == "monitor.resume":
                assert evidence[1].kind == "monitor.resumed"
            else:
                assert evidence[1].kind == "monitor.deleted"
            assert len(events) >= 2
        else:
            assert len(events) == 2
            assert evidence[0].owner == "db.monitor"
    tasks = await runtime.store.list_tasks()
    assert [task.capability_id for task in tasks] == [
        "db.monitor.plan_create",
        "db.schema.inspect",
        "db.monitor.commit_create",
        "db.monitor.plan_lifecycle",
        "db.monitor.commit_lifecycle",
        "db.monitor.plan_lifecycle",
        "db.monitor.commit_lifecycle",
        "db.monitor.plan_lifecycle",
        "db.monitor.commit_lifecycle",
        "db.monitor.plan_lifecycle",
        "db.monitor.commit_lifecycle",
    ]
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
    assert evidence[0].kind == "monitor.proposal"
    assert evidence[0].accepted is False
    assert evidence[0].payload["validation"]["missing_capabilities"] == [
        "slack.message.send"
    ]
    assert any(event.payload.get("status") == "blocked" for event in events)
    tasks = await runtime.store.list_tasks()
    assert [task.capability_id for task in tasks] == ["db.monitor.plan_create"]


async def test_create_monitor_prompt_requires_approval_then_resumes_to_create_monitor():
    schema_probe = _schema_probe("users")
    runtime = DbRuntime(
        runtime_id="db-monitor-create-approval-runtime",
        plugins=(schema_probe,),
    )
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
    assert monitors[0].observation_plan["cursor"]["field"] == "created_at"
    evidence = await runtime.store.list_evidence(result.operation_id)
    assert any(item.kind == "monitor.definition" and item.accepted for item in evidence)


async def test_approval_resume_skips_plan_and_executes_only_pending_commit():
    schema_probe = _schema_probe("users")
    runtime = ResumeTaskSpyRuntime(plugins=(schema_probe,))
    agent, result, approval, plan_task, commit_task, proposal = (
        await _approval_required_create(runtime)
    )

    assert plan_task.status is TaskStatus.SUCCEEDED
    assert commit_task.status in {TaskStatus.PENDING, TaskStatus.BLOCKED}
    assert runtime.executed_capabilities == [
        "db.monitor.plan_create",
        "db.schema.inspect",
    ]

    runtime.executed_capabilities.clear()
    await runtime.approval_channel.approve(approval.approval_id)
    resumed = await runtime.resume_operation(result.operation_id)

    assert resumed.operation.status is OperationStatus.SUCCEEDED
    assert runtime.executed_capabilities == ["db.monitor.commit_create"]
    tasks = await runtime.store.list_tasks(result.operation_id)
    assert {
        task.capability_id: task.status
        for task in tasks
        if task.capability_id.startswith("db.monitor.")
    } == {
        "db.monitor.plan_create": TaskStatus.SUCCEEDED,
        "db.monitor.commit_create": TaskStatus.SUCCEEDED,
    }
    monitors = await agent.list_monitors()
    assert [monitor.id for monitor in monitors] == [proposal.payload["monitor_id"]]


async def test_monitor_commit_uses_exact_accepted_proposal_evidence():
    schema_probe = _schema_probe("users")
    runtime = DbRuntime(
        runtime_id="db-monitor-exact-proposal-runtime",
        plugins=(schema_probe,),
    )
    agent, result, approval, _plan_task, _commit_task, proposal = (
        await _approval_required_create(runtime)
    )
    operation = await runtime.store.load_operation(result.operation_id)
    assert operation is not None
    await runtime.store.save_operation(
        replace(
            operation,
            metadata={
                **operation.metadata,
                "monitor_id": "operation_metadata_should_not_win",
                "proposal_fingerprint": "operation-metadata-fingerprint",
            },
        )
    )
    fake_payload = {
        **proposal.payload,
        "monitor_id": "newer_fake_proposal",
        "name": "Newer Fake Proposal",
        "proposal_fingerprint": "newer-fake-proposal-fingerprint",
    }
    await runtime.store.save_evidence(
        Evidence(
            id="newer-fake-monitor-proposal",
            kind="monitor.proposal",
            owner="db_runtime",
            operation_id=result.operation_id,
            task_id=proposal.task_id,
            accepted=True,
            payload=fake_payload,
            metadata={
                "payload_fingerprint": fake_payload["proposal_fingerprint"],
                "monitor_id": fake_payload["monitor_id"],
            },
        )
    )

    await runtime.approval_channel.approve(approval.approval_id)
    resumed = await runtime.resume_operation(result.operation_id)

    assert resumed.operation.status is OperationStatus.SUCCEEDED
    monitors = await agent.list_monitors()
    assert [monitor.id for monitor in monitors] == [proposal.payload["monitor_id"]]
    definitions = [
        item
        for item in await runtime.store.list_evidence(result.operation_id)
        if item.kind == "monitor.definition"
    ]
    assert definitions[-1].payload["proposal_evidence_id"] == proposal.id
    assert definitions[-1].payload["proposal_fingerprint"] == (
        proposal.payload["proposal_fingerprint"]
    )


async def test_monitor_commit_rejects_stale_proposal_fingerprint():
    schema_probe = _schema_probe("users")
    runtime = DbRuntime(
        runtime_id="db-monitor-stale-proposal-runtime",
        plugins=(schema_probe,),
    )
    agent, result, approval, _plan_task, commit_task, _proposal = (
        await _approval_required_create(runtime)
    )
    await runtime.store.save_task(
        replace(
            commit_task,
            input={
                **commit_task.input,
                "proposal_fingerprint": "stale-proposal-fingerprint",
            },
        )
    )

    await runtime.approval_channel.approve(approval.approval_id)
    with pytest.raises(RuntimeError, match="monitor proposal fingerprint mismatch"):
        await runtime.resume_operation(result.operation_id)

    assert await agent.list_monitors() == ()


async def test_monitor_commit_rejects_rejected_proposal_evidence():
    schema_probe = _schema_probe("users")
    runtime = DbRuntime(
        runtime_id="db-monitor-rejected-proposal-runtime",
        plugins=(schema_probe,),
    )
    agent, result, approval, _plan_task, commit_task, proposal = (
        await _approval_required_create(runtime)
    )
    rejected_proposal = replace(
        proposal,
        id="rejected-monitor-proposal",
        accepted=False,
    )
    await runtime.store.save_evidence(rejected_proposal)
    await runtime.store.save_task(
        replace(
            commit_task,
            input={
                **commit_task.input,
                "proposal_evidence_id": rejected_proposal.id,
            },
            dependencies=tuple(
                (
                    replace(
                        dependency,
                        evidence_id=rejected_proposal.id,
                        evidence_accepted=False,
                    )
                    if dependency.evidence_kind == "monitor.proposal"
                    else dependency
                )
                for dependency in commit_task.dependencies
            ),
        )
    )

    await runtime.approval_channel.approve(approval.approval_id)
    with pytest.raises(
        RuntimeError, match="monitor proposal evidence was not accepted"
    ):
        await runtime.resume_operation(result.operation_id)

    assert await agent.list_monitors() == ()


async def test_monitor_commit_rejects_missing_proposal_evidence():
    schema_probe = _schema_probe("users")
    runtime = DbRuntime(
        runtime_id="db-monitor-missing-proposal-runtime",
        plugins=(schema_probe,),
    )
    agent, result, approval, _plan_task, commit_task, _proposal = (
        await _approval_required_create(runtime)
    )
    await runtime.store.save_task(
        replace(
            commit_task,
            input={
                **commit_task.input,
                "proposal_evidence_id": "missing-monitor-proposal",
            },
        )
    )

    await runtime.approval_channel.approve(approval.approval_id)
    with pytest.raises(RuntimeError, match="monitor proposal evidence is required"):
        await runtime.resume_operation(result.operation_id)

    assert await agent.list_monitors() == ()


async def test_monitor_commit_replay_is_idempotent_and_preserves_initial_cursor():
    schema_probe = _schema_probe("users")
    runtime = DbRuntime(
        runtime_id="db-monitor-idempotent-commit-runtime",
        plugins=(schema_probe,),
    )
    agent, result, approval, _plan_task, _commit_task, proposal = (
        await _approval_required_create(runtime)
    )

    await runtime.approval_channel.approve(approval.approval_id)
    resumed = await runtime.resume_operation(result.operation_id)
    assert resumed.operation.status is OperationStatus.SUCCEEDED
    inspection = await agent.inspect_monitor(proposal.payload["monitor_id"])
    assert inspection is not None
    assert inspection.state is not None
    assert inspection.state.cursor == proposal.payload["initial_state"]["cursor"]

    replay_task = await runtime.kernel.plan_task(
        operation_id=result.operation_id,
        capability_id="db.monitor.commit_create",
        owner="db_runtime",
        input={
            "proposal_evidence_id": proposal.id,
            "proposal_fingerprint": proposal.payload["proposal_fingerprint"],
        },
        metadata={
            "reason": "monitor_create_commit_replay",
            "sequence": 3,
            "idempotency_key": f"{proposal.payload['proposal_fingerprint']}:replay",
        },
        dependencies=(
            TaskDependency(
                kind="evidence",
                evidence_kind="monitor.proposal",
                evidence_id=proposal.id,
                evidence_owner="db_runtime",
                producer_task_id=proposal.task_id,
                evidence_payload={
                    "proposal_fingerprint": proposal.payload["proposal_fingerprint"],
                },
                evidence_accepted=True,
                operation_id=result.operation_id,
            ),
        ),
    )

    replay_evidence = await runtime.execute_task(replay_task, resumed.operation)

    monitors = await agent.list_monitors()
    assert [monitor.id for monitor in monitors] == [proposal.payload["monitor_id"]]
    definition = next(
        item for item in replay_evidence if item.kind == "monitor.definition"
    )
    assert definition.payload["idempotent_existing"] is True


async def test_monitor_create_metadata_is_ready_for_worker_handoff():
    schema_probe = _schema_probe("users")
    runtime = DbRuntime(
        runtime_id="db-monitor-handoff-metadata-runtime",
        plugins=(schema_probe,),
    )
    _agent, result, approval, plan_task, commit_task, proposal = (
        await _approval_required_create(runtime)
    )
    operation = await runtime.store.load_operation(result.operation_id)
    assert operation is not None

    assert operation.metadata["control_plane"] == "db.monitor"
    assert operation.metadata["command_kind"] == "create"
    assert operation.metadata["monitor_id"] == proposal.payload["monitor_id"]
    assert operation.metadata["proposal_fingerprint"] == (
        proposal.payload["proposal_fingerprint"]
    )
    assert operation.metadata["resume_context"]["request"]["prompt"] == (
        operation.request["prompt"]
    )
    assert plan_task.metadata["sequence"] == 1
    assert plan_task.metadata["idempotency_key"]
    assert plan_task.input["command"]["kind"] == "create"
    assert commit_task.metadata["owner"] == "db_runtime"
    assert commit_task.metadata["reason"] == "monitor_create_commit"
    assert commit_task.metadata["sequence"] == 2
    assert commit_task.metadata["idempotency_key"] == (
        proposal.payload["proposal_fingerprint"]
    )
    assert commit_task.input == {
        "proposal_evidence_id": proposal.id,
        "proposal_fingerprint": proposal.payload["proposal_fingerprint"],
    }
    proposal_dependency = next(
        dependency
        for dependency in commit_task.dependencies
        if dependency.evidence_kind == "monitor.proposal"
    )
    assert proposal_dependency.evidence_id == proposal.id
    assert proposal_dependency.evidence_owner == "db_runtime"
    assert proposal_dependency.producer_task_id == plan_task.id
    assert proposal_dependency.producer_capability_id == "db.monitor.plan_create"
    assert proposal_dependency.evidence_payload == {
        "proposal_fingerprint": proposal.payload["proposal_fingerprint"],
    }
    approval_dependency = next(
        dependency
        for dependency in commit_task.dependencies
        if dependency.approval_id == approval.approval_id
    )
    assert approval_dependency.approval_status is ApprovalStatus.APPROVED
    assert approval.proposed_action["proposal_evidence_id"] == proposal.id
    assert approval.proposed_action["proposal_fingerprint"] == (
        proposal.payload["proposal_fingerprint"]
    )
    assert approval.metadata["commit_task_id"] == commit_task.id
    assert approval.metadata["proposal_evidence_id"] == proposal.id
    assert approval.metadata["proposal_fingerprint"] == (
        proposal.payload["proposal_fingerprint"]
    )
    assert proposal.metadata["monitor_id"] == proposal.payload["monitor_id"]
    assert proposal.metadata["payload_fingerprint"]
    assert proposal.metadata["validation_accepted"] is True
    assert proposal.payload["proposal_fingerprint"]


async def test_prompt_monitor_create_blocks_when_schema_cannot_prove_cursor():
    runtime = DbRuntime(runtime_id="db-monitor-missing-cursor-runtime")
    agent = DbAgent(runtime=runtime)

    result = await agent.run_detailed(
        "Create a monitor for the users table. Notify me when a new user is inserted."
    )

    assert result.status is OperationStatus.BLOCKED
    assert result.warnings == ("db_monitor_validation_failed",)
    assert await agent.list_monitors() == ()
    evidence = await runtime.store.list_evidence(result.operation_id)
    proposal = next(item for item in evidence if item.kind == "monitor.proposal")
    assert proposal.accepted is False
    assert (
        "monitor.proposal_incomplete:cursor" in proposal.payload["validation"]["errors"]
    )
    assert (
        "monitor.proposal_incomplete:observation_sql"
        in proposal.payload["validation"]["errors"]
    )


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
    await _create_helper_monitor(agent)
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
    await _create_helper_monitor(agent)
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
