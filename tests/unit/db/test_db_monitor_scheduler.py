import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import inspect
import sqlite3
import threading

import pytest

import daita.db.monitor_scheduler as monitor_scheduler_module
from daita.db import (
    DbMonitor,
    DbMonitorMutation,
    DbMonitorRun,
    DbMonitorState,
    DbRuntime,
)
from daita.db.monitor_scheduler import DbMonitorScheduler
from daita.db.monitors import SQLiteDbMonitorStore
from daita.db.monitor_commands import DbMonitorValidation
from daita.db.monitor_scheduler.observation import (
    _cursor_updates_from_plan as _observation_cursor_updates_from_plan,
)
from daita.db.monitor_scheduler.state import (
    _cursor_updates_from_plan as _state_cursor_updates_from_plan,
)
from daita.db.runtime.tasks.models import DbTaskSpec
from daita.plugins import RuntimeExtensionPlugin, PluginKind, PluginManifest
from daita.runtime import (
    AccessMode,
    ApprovalRequest,
    Capability,
    Evidence,
    EvidenceSchema,
    Operation,
    OperationStatus,
    ApprovalStatus,
    PolicyDecision,
    PolicyEffect,
    RiskLevel,
    SQLiteRuntimeStore,
    Task,
    TaskStatus,
)

NOW = datetime(2026, 6, 12, 12, 0, tzinfo=timezone.utc)


async def test_sqlite_monitor_store_does_not_block_the_event_loop(
    tmp_path, monkeypatch
):
    store = SQLiteDbMonitorStore(tmp_path / "monitor-responsive.sqlite")
    monitor = DbMonitor(id="responsive-monitor", name="Responsive monitor")
    helper_started = threading.Event()
    helper_released = threading.Event()
    helper_finished = threading.Event()
    release_timed_out = threading.Event()
    save_monitor_sync = store._save_monitor_sync

    def blocking_save_monitor_sync(value):
        helper_started.set()
        if not helper_released.wait(timeout=5):
            release_timed_out.set()
        try:
            return save_monitor_sync(value)
        finally:
            helper_finished.set()

    monkeypatch.setattr(store, "_save_monitor_sync", blocking_save_monitor_sync)
    store_task = asyncio.create_task(store.save_monitor(monitor))
    loop_advanced = asyncio.Event()

    async def advance_loop():
        loop_advanced.set()

    try:
        assert await asyncio.to_thread(helper_started.wait, 5)
        progress_task = asyncio.create_task(advance_loop())
        await progress_task

        assert loop_advanced.is_set()
        assert not helper_released.is_set()
        assert not helper_finished.is_set()
        assert not release_timed_out.is_set()
    finally:
        helper_released.set()
        await store_task

    assert await store.load_monitor(monitor.id) == monitor


async def test_sqlite_monitor_store_closes_read_connection_before_return(
    tmp_path, monkeypatch
):
    store = SQLiteDbMonitorStore(tmp_path / "monitor-read-close.sqlite")
    monitor = DbMonitor(id="read-close-monitor", name="Read close monitor")
    await store.save_monitor(monitor)
    connections = []

    class ConnectionProbe:
        def __init__(self):
            self.connection = sqlite3.connect(store.path)
            self.connection.row_factory = sqlite3.Row
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return self.connection.__exit__(exc_type, exc, tb)

        def execute(self, *args, **kwargs):
            return self.connection.execute(*args, **kwargs)

        def close(self):
            self.closed = True
            self.connection.close()

    def connect():
        connection = ConnectionProbe()
        connections.append(connection)
        return connection

    monkeypatch.setattr(store, "_connect", connect)

    assert await store.load_monitor(monitor.id) == monitor
    assert len(connections) == 1
    assert connections[0].closed is True


def _validation(accepted=True, **kwargs):
    return DbMonitorValidation(accepted=accepted, **kwargs).to_dict()


def test_cursor_updates_capture_non_null_values_once_before_max():
    class PopOnceRow(dict):
        def get(self, key, default=None):
            if key == "id":
                return self.pop(key, default)
            return super().get(key, default)

    plan = {"cursor_update": {"last_id": "max(rows.id)"}}

    def payload():
        return {"rows": [PopOnceRow(id=41), {"id": None}, {"id": 42}]}

    assert _observation_cursor_updates_from_plan(plan, payload()) == {"last_id": 42}
    assert _state_cursor_updates_from_plan(plan, payload()) == {"last_id": 42}


class MonitorReadProbeExecutor:
    id = "monitor_read_probe.sql.execute_read"
    capability_ids = frozenset({"db.sql.execute_read"})

    def __init__(self, rows=None, *, fail=False):
        self.rows = list(rows or [{"pending_count": 4}])
        self.fail = fail
        self.calls = 0

    async def execute(self, task: Task, operation: Operation, context):
        self.calls += 1
        if self.fail:
            raise RuntimeError("read guardrail failed")
        return [
            Evidence(
                kind="query.result",
                owner="monitor_read_probe",
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "sql": task.input["sql"],
                    "rows": list(self.rows),
                },
            )
        ]


class MonitorValidateProbeExecutor:
    id = "monitor_read_probe.sql.validate"
    capability_ids = frozenset({"db.sql.validate"})

    def __init__(self, *, valid=True):
        self.calls = 0
        self.valid = valid

    async def execute(self, task: Task, operation: Operation, context):
        self.calls += 1
        sql = str(task.input["sql"])
        lowered = sql.strip().lower()
        valid = self.valid and not (
            lowered == "select * from"
            or lowered.startswith("delete ")
            or lowered.startswith("drop ")
        )
        is_read = lowered.startswith("select ") or lowered.startswith("with ")
        tables = ["orders"]
        if "runtime_operations" in lowered:
            tables = ["runtime_operations"]
        if "customers" in lowered:
            tables = ["customers"]
        return [
            Evidence(
                kind="sql.validation",
                owner="monitor_read_probe",
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "valid": valid,
                    "sql": sql,
                    "is_read": is_read,
                    "tables": tables,
                    "sql_fingerprint": f"fingerprint:{abs(hash(sql))}",
                },
            )
        ]


class MonitorReadProbePlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="monitor_read_probe",
        display_name="Monitor Read Probe",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def __init__(self, rows=None, *, validation_valid=True, fail_read=False):
        self.setup_called = False
        self.read_executor = MonitorReadProbeExecutor(rows, fail=fail_read)
        self.validate_executor = MonitorValidateProbeExecutor(valid=validation_valid)

    async def setup(self, context):
        self.setup_called = True

    def declare_capabilities(self):
        return [
            Capability(
                id="db.sql.validate",
                owner="monitor_read_probe",
                description="Validate read SQL.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"data.query", "monitor.tick"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"sql.validation"}),
                executor="monitor_read_probe.sql.validate",
                runtime_only=True,
                side_effecting=False,
            ),
            Capability(
                id="db.sql.execute_read",
                owner="monitor_read_probe",
                description="Execute read SQL.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"data.query", "monitor.tick"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"query.result"}),
                executor="monitor_read_probe.sql.execute_read",
                runtime_only=True,
                side_effecting=False,
            ),
        ]

    def get_executors(self):
        return [self.validate_executor, self.read_executor]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="sql.validation",
                owner="monitor_read_probe",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="query.result",
                owner="monitor_read_probe",
                json_schema={"type": "object"},
            ),
        ]


class MonitorWriteProbeExecutor:
    capability_ids = frozenset({"db.sql.execute_write"})

    def __init__(self, *, owner="monitor_write_probe"):
        self.id = f"{owner}.sql.execute_write"
        self.owner = owner
        self.calls = 0
        self.inputs = []

    async def execute(self, task: Task, operation: Operation, context):
        self.calls += 1
        self.inputs.append(dict(task.input))
        return [
            Evidence(
                kind="write.execution",
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "ok": True,
                    "sql": task.input.get("sql"),
                    "validated_evidence_id": task.input.get("validated_evidence_id"),
                },
            )
        ]


class MonitorWriteValidateProbeExecutor:
    capability_ids = frozenset({"db.sql.validate"})

    def __init__(self, *, owner="monitor_write_probe"):
        self.id = f"{owner}.sql.validate"
        self.owner = owner
        self.calls = 0

    async def execute(self, task: Task, operation: Operation, context):
        self.calls += 1
        sql = str(task.input["sql"])
        lowered = sql.strip().lower()
        statement_type = lowered.split(None, 1)[0].upper() if lowered else ""
        destructive = statement_type in {"DELETE", "DROP", "ALTER", "TRUNCATE"}
        return [
            Evidence(
                kind="sql.validation",
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "valid": bool(sql) and not destructive,
                    "sql": sql,
                    "is_read": False,
                    "tables": ["orders"],
                    "statement_type": statement_type,
                    "destructive_statement_classes": (
                        [statement_type] if destructive else []
                    ),
                    "sql_fingerprint": f"write-fingerprint:{abs(hash(sql))}",
                },
            )
        ]


class MonitorWriteProbePlugin(RuntimeExtensionPlugin):
    def __init__(self, *, plugin_id="monitor_write_probe", write_risk=RiskLevel.HIGH):
        self.manifest = PluginManifest(
            id=plugin_id,
            display_name="Monitor Write Probe",
            version="1.0.0",
            kind=PluginKind.RUNTIME_EXTENSION,
            domains=frozenset({"db"}),
        )
        self.validate_executor = MonitorWriteValidateProbeExecutor(owner=plugin_id)
        self.write_executor = MonitorWriteProbeExecutor(owner=plugin_id)
        self.write_risk = write_risk

    async def setup(self, context):
        pass

    def declare_capabilities(self):
        return [
            Capability(
                id="db.sql.validate",
                owner=self.manifest.id,
                description="Validate write SQL.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"monitor.triggered", "write.execute"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"sql.validation"}),
                executor=self.validate_executor.id,
                runtime_only=True,
                side_effecting=False,
            ),
            Capability(
                id="db.sql.execute_write",
                owner=self.manifest.id,
                description="Execute write SQL.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"monitor.triggered", "write.execute"}),
                access=AccessMode.WRITE,
                risk=self.write_risk,
                input_schema={"type": "object"},
                output_evidence=frozenset({"write.execution"}),
                executor=self.write_executor.id,
                runtime_only=True,
                side_effecting=True,
            ),
        ]

    def get_executors(self):
        return [self.validate_executor, self.write_executor]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="sql.validation",
                owner=self.manifest.id,
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="write.execution",
                owner=self.manifest.id,
                json_schema={"type": "object"},
            ),
        ]


class MonitorPluginProbeExecutor:
    def __init__(self, *, evidence_kind, owner, fail=False):
        self.id = f"{owner}.monitor.probe"
        self.capability_ids = frozenset()
        self.evidence_kind = evidence_kind
        self.owner = owner
        self.fail = fail
        self.calls = 0
        self.inputs = []

    async def execute(self, task: Task, operation: Operation, context):
        self.calls += 1
        self.inputs.append(dict(task.input))
        if self.fail:
            raise RuntimeError(f"{self.owner} failed")
        payload = {
            "ok": True,
            "idempotency_key": task.input.get("idempotency_key")
            or task.metadata.get("idempotency_key"),
            "input_summary": {
                "keys": sorted(task.input),
                "target": task.input.get("target"),
                "payload_source": task.input.get("payload_source"),
                "request": task.input.get("request"),
            },
        }
        if self.evidence_kind == "api.http.response":
            payload = {
                "status_code": 200,
                "body": {"external_error_count": 3, "events": [{"id": "evt-1"}]},
                "request": task.input.get("request"),
            }
        return [
            Evidence(
                kind=self.evidence_kind,
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                payload=payload,
            )
        ]


class MonitorDeliveryApprovalPolicy:
    def __init__(self, *, owner, capability_id):
        self.id = "monitor.delivery.requires_approval"
        self.owner = owner
        self.capability_id = capability_id

    def applies_to(self, request, operation_type):
        return True

    def modify_contract(self, contract):
        return contract

    def evaluate_operation(self, operation):
        facts = operation.request.get("governance_facts") or {}
        monitor_effect = facts.get("monitor_effect") or {}
        capability = operation.request.get("capability") or {}
        if (
            monitor_effect.get("phase") != "delivery"
            or capability.get("id") != self.capability_id
        ):
            return None
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=PolicyEffect.REQUIRE_APPROVAL,
            reason="monitor delivery requires approval",
            severity=RiskLevel.LOW,
            required_approvals=("monitor_delivery",),
            metadata={"monitor_effect": monitor_effect},
        )


class TaskSpecSpyRuntime(DbRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec_batches: list[tuple[DbTaskSpec, ...]] = []
        self.planned_task_batches: list[tuple[Task, ...]] = []
        self.write_governance_task_ids: list[str] = []

    async def plan_task_specs(self, operation, specs, *, contract=None):
        materialized = tuple(specs)
        self.spec_batches.append(materialized)
        plan = await super().plan_task_specs(
            operation,
            materialized,
            contract=contract,
        )
        self.planned_task_batches.append(plan.tasks)
        return plan

    async def evaluate_monitor_effect_governance(
        self,
        operation,
        *,
        capability,
        task=None,
        intent,
        phase,
        mutate_approvals=False,
        operation_override=None,
    ):
        if phase == "write_execution" and task is not None:
            self.write_governance_task_ids.append(task.id)
        return await super().evaluate_monitor_effect_governance(
            operation,
            capability=capability,
            task=task,
            intent=intent,
            phase=phase,
            mutate_approvals=mutate_approvals,
            operation_override=operation_override,
        )


class MonitorCapabilityProbePlugin(RuntimeExtensionPlugin):
    def __init__(
        self,
        *,
        plugin_id,
        capability_id,
        evidence_kind,
        role,
        kind,
        access=AccessMode.WRITE,
        side_effecting=True,
        accepted_formats=("markdown",),
        accepted_target_types=(),
        default_target=None,
        requires_approval=False,
        input_schema=None,
        fail=False,
    ):
        self.manifest = PluginManifest(
            id=plugin_id,
            display_name=plugin_id,
            version="1.0.0",
            kind=PluginKind.RUNTIME_EXTENSION,
            domains=frozenset({kind, "monitor"}),
        )
        self.executor = MonitorPluginProbeExecutor(
            evidence_kind=evidence_kind,
            owner=plugin_id,
            fail=fail,
        )
        self.executor.capability_ids = frozenset({capability_id})
        self.capability_id = capability_id
        self.evidence_kind = evidence_kind
        self.role = role
        self.kind = kind
        self.access = access
        self.side_effecting = side_effecting
        self.accepted_formats = tuple(accepted_formats)
        self.accepted_target_types = tuple(accepted_target_types)
        self.default_target = default_target
        self.requires_approval = requires_approval
        self.input_schema = input_schema or {
            "type": "object",
            "properties": {
                "target": {"type": "object"},
                "request": {"type": "object"},
                "payload_source": {"type": "object"},
            },
        }

    def declare_capabilities(self):
        metadata = {
            "monitor_roles": [self.role],
            f"{self.role}_kind": self.kind,
            "accepted_payload_kinds": ["monitor.report", "analysis.synthesis"],
            "accepted_formats": list(self.accepted_formats),
            "supports_idempotency_key": True,
            "supports_dry_run": True,
            "replay_safe": False,
        }
        if self.accepted_target_types:
            metadata["accepted_target_types"] = list(self.accepted_target_types)
        if self.default_target is not None:
            metadata["default_target"] = self.default_target
        return [
            Capability(
                id=self.capability_id,
                owner=self.manifest.id,
                description=f"{self.role} probe",
                domains=frozenset({self.kind, "monitor"}),
                operation_types=frozenset(
                    {"monitor.tick" if self.role == "source" else "monitor.delivery"}
                ),
                access=self.access,
                risk=RiskLevel.LOW,
                input_schema=self.input_schema,
                output_evidence=frozenset({self.evidence_kind}),
                executor=self.executor.id,
                runtime_only=True,
                side_effecting=self.side_effecting,
                idempotent=False,
                replay_safe=False,
                metadata=metadata,
            )
        ]

    def get_executors(self):
        return [self.executor]

    def declare_policies(self):
        if not self.requires_approval:
            return ()
        return (
            MonitorDeliveryApprovalPolicy(
                owner=self.manifest.id,
                capability_id=self.capability_id,
            ),
        )

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind=self.evidence_kind,
                owner=self.manifest.id,
                json_schema={"type": "object"},
            )
        ]


def _metric_observation(sql="select count(*) as pending_count from orders"):
    return {
        "kind": "metric_sql",
        "metric": "pending_count",
        "sql": sql,
        "value_path": "rows.0.pending_count",
        "source_scope": ["orders"],
        "capability_owner": "monitor_read_probe",
    }


def _plugin_source_observation(**overrides):
    value = {
        "kind": "plugin_source",
        "source_kind": "rest",
        "capability_id": "rest.http.get",
        "capability_owner": "rest_source_probe",
        "request": {
            "method": "GET",
            "path": "/billing/events",
            "query": {"since": "monitor.state.cursor.last_event_id"},
        },
        "correlation": {"event_key": "payment_id", "db_key": "payments.id"},
        "expected_evidence": ["api.http.response"],
        "value_path": "body.external_error_count",
    }
    value.update(overrides)
    return value


def _scheduled_report_action(delivery_intent=None):
    return {
        "kind": "scheduled_report",
        "title": "Daily order health",
        "steps": [
            {
                "id": "revenue",
                "kind": "metric_sql",
                "metric": "revenue",
                "sql": "select sum(total) as revenue from orders",
                "source_scope": ["orders"],
                "capability_owner": "monitor_read_probe",
            },
            {
                "id": "report_summary",
                "kind": "synthesis",
                "purpose": "Generate the report narrative",
                "expected_evidence": ["analysis.synthesis"],
            },
        ],
        "output": {"kind": "report", "format": "markdown"},
        "delivery_intent": delivery_intent or {},
    }


def _notification_action(delivery_intent=None):
    return {
        "kind": "notification",
        "title": "New order notification",
        "delivery_intent": delivery_intent
        or {
            "delivery_kind": "local",
            "target": {"type": "runtime_console"},
            "format": "markdown",
        },
    }


def _write_proposal_action(sql="update orders set status = 'ready' where id = 1"):
    return {
        "kind": "write_proposal",
        "sql": sql,
        "capability_owner": "monitor_write_probe",
        "source_scope": ["orders"],
    }


async def _create_monitor(runtime, monitor_id, **kwargs):
    values = {
        "schedule": {"interval_seconds": 300},
        "trigger": {"force_trigger": False},
        "observation_plan": _metric_observation(),
        "metadata": {"validation": _validation()},
    }
    values.update(kwargs)
    monitor = DbMonitor(
        id=monitor_id,
        name=monitor_id.replace("_", " ").title(),
        **values,
    )
    return await runtime.create_monitor(monitor)


async def test_observation_materializes_validation_and_read_via_task_specs():
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 4}])
    runtime = TaskSpecSpyRuntime(
        runtime_id="db-monitor-observation-specs",
        plugins=(plugin,),
    )
    await _create_monitor(
        runtime,
        "observation_specs_monitor",
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    observation_batches = [
        batch
        for batch in runtime.spec_batches
        if [spec.reason for spec in batch]
        == ["monitor_observation_read_validation", "monitor_observation_read"]
    ]
    tasks = await runtime.store.list_tasks(run.operation_id)
    validation_task, read_task = tasks
    validation_dependency = next(
        dependency
        for dependency in read_task.dependencies
        if dependency.evidence_kind == "sql.validation"
    )

    assert observation_batches
    assert all(isinstance(spec, DbTaskSpec) for spec in observation_batches[0])
    assert [spec.capability_id for spec in observation_batches[0]] == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    assert validation_dependency.producer_task_id == validation_task.id
    assert read_task.input["validated_task_id"] == validation_task.id


async def test_monitor_write_governance_uses_task_from_plan_task_specs():
    read_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    write_plugin = MonitorWriteProbePlugin()
    runtime = TaskSpecSpyRuntime(
        runtime_id="db-monitor-write-governance-specs",
        plugins=(read_plugin, write_plugin),
    )
    await _create_monitor(
        runtime,
        "write_governance_specs_monitor",
        schedule={"interval_seconds": 0},
        source_scope=("orders",),
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan=_write_proposal_action(),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    write_task = next(
        task for task in snapshot.tasks if task.capability_id == "db.sql.execute_write"
    )
    planned_write_task_ids = {
        task.id
        for batch in runtime.planned_task_batches
        for task in batch
        if task.capability_id == "db.sql.execute_write"
    }

    assert write_task.id in planned_write_task_ids
    assert runtime.write_governance_task_ids == [write_task.id]
    assert any(
        batch
        and batch[-1].capability_id == "db.sql.execute_write"
        and batch[-1].reason == "monitor_write_execution"
        for batch in runtime.spec_batches
    )


async def test_scheduler_records_due_observation_and_not_due_decisions():
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 4}])
    runtime = DbRuntime(runtime_id="db-monitor-scheduler-due", plugins=(plugin,))
    await _create_monitor(
        runtime,
        "orders_backlog",
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
    )

    first = await runtime.tick_monitors(now=NOW)
    assert len(first) == 1
    assert first[0].status == "succeeded"
    assert first[0].triggered is False
    assert first[0].summary["reason"] == "no_match"
    assert first[0].summary["observation_task_ids"]
    assert plugin.validate_executor.calls == 1
    assert plugin.read_executor.calls == 1

    state_after_first = await runtime.monitor_store.load_monitor_state("orders_backlog")
    assert state_after_first.last_tick_at == NOW.isoformat()
    assert state_after_first.cursor["last_observation_fingerprint"]

    second = await runtime.tick_monitors(now=NOW + timedelta(seconds=60))
    assert second[0].status == "skipped"
    assert second[0].summary["reason"] == "not_due"

    state_after_second = await runtime.monitor_store.load_monitor_state(
        "orders_backlog"
    )
    assert state_after_second.last_tick_at == NOW.isoformat()

    runs = await runtime.monitor_store.list_monitor_runs("orders_backlog")
    assert [run.summary["reason"] for run in runs] == ["no_match", "not_due"]
    assert [
        operation.operation_type for operation in await runtime.store.list_operations()
    ] == [
        "monitor.create",
        "monitor.tick",
        "monitor.tick",
    ]
    tasks = await runtime.store.list_tasks(first[0].operation_id)
    assert [task.capability_id for task in tasks] == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    assert tasks[1].input["sql_ref"] == "sql.validation"
    assert tasks[1].input["validated_evidence_id"]
    assert tasks[1].input["validated_task_id"] == tasks[0].id


async def test_tick_monitors_runs_exactly_one_scheduler_pass(monkeypatch):
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 4}])
    runtime = DbRuntime(runtime_id="db-monitor-one-shot", plugins=(plugin,))
    await _create_monitor(
        runtime,
        "one_shot_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
    )
    original_run_once = DbMonitorScheduler.run_once
    pass_scheduler_ids = []

    async def count_run_once(scheduler, *, now=None):
        pass_scheduler_ids.append(scheduler.scheduler_id)
        return await original_run_once(scheduler, now=now)

    monkeypatch.setattr(DbMonitorScheduler, "run_once", count_run_once)

    runs = await runtime.tick_monitors(now=NOW)

    assert len(runs) == 1
    assert len(pass_scheduler_ids) == 1
    assert plugin.validate_executor.calls == 1
    assert plugin.read_executor.calls == 1


async def test_two_schedulers_share_one_lease_and_only_one_triggers(monkeypatch):
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    runtime = DbRuntime(runtime_id="db-monitor-multi-host", plugins=(plugin,))
    monitor = await _create_monitor(
        runtime,
        "multi_host_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
    )
    scheduler_a = DbMonitorScheduler(
        runtime=runtime,
        monitor_store=runtime.monitor_store,
        scheduler_id="monitor-host-a",
    )
    scheduler_b = DbMonitorScheduler(
        runtime=runtime,
        monitor_store=runtime.monitor_store,
        scheduler_id="monitor-host-b",
    )
    first_tick_started = asyncio.Event()
    release_first_tick = asyncio.Event()
    original_tick_monitor = scheduler_a.runner.tick_monitor

    async def block_after_claim(monitor_id, *, now=None, lease_id=None):
        first_tick_started.set()
        await release_first_tick.wait()
        return await original_tick_monitor(
            monitor_id,
            now=now,
            lease_id=lease_id,
        )

    monkeypatch.setattr(scheduler_a.runner, "tick_monitor", block_after_claim)
    first_pass = asyncio.create_task(scheduler_a.run_once(now=NOW))
    try:
        await first_tick_started.wait()
        losing_results = await scheduler_b.run_once(now=NOW)
    finally:
        release_first_tick.set()
    winning_results = await first_pass

    all_results = (*winning_results, *losing_results)
    persisted_runs = await runtime.monitor_store.list_monitor_runs(monitor.id)

    assert sum(result.claimed for result in all_results) == 1
    assert sum(result.run.triggered for result in all_results) == 1
    assert sum(run.triggered for run in persisted_runs) == 1
    assert losing_results[0].claimed is False
    assert losing_results[0].run.status == "skipped"
    assert losing_results[0].run.summary["reason"] == "lease_lost"
    assert plugin.validate_executor.calls == 1
    assert plugin.read_executor.calls == 1


async def test_scheduler_identity_and_utc_clock_stay_stable_across_passes():
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 4}])
    runtime = DbRuntime(runtime_id="db-monitor-stable-host", plugins=(plugin,))
    monitor = await _create_monitor(
        runtime,
        "stable_host_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
    )
    scheduler = DbMonitorScheduler(
        runtime=runtime,
        scheduler_id="stable-monitor-host",
    )
    second_tick = NOW + timedelta(seconds=1)

    first_results = await scheduler.run_once(now=NOW)
    second_results = await scheduler.run_once(now=second_tick)
    state = await runtime.monitor_store.load_monitor_state(monitor.id)

    assert scheduler.scheduler_id == "stable-monitor-host"
    assert first_results[0].claimed is True
    assert second_results[0].claimed is True
    assert state.last_tick_at == second_tick.isoformat()
    assert datetime.fromisoformat(state.last_tick_at).utcoffset() == timedelta(0)
    assert plugin.validate_executor.calls == 2
    assert plugin.read_executor.calls == 2


async def test_scheduler_releases_lease_after_success_and_runner_failure(
    monkeypatch,
):
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 4}])
    runtime = DbRuntime(runtime_id="db-monitor-lease-release", plugins=(plugin,))
    monitor = await _create_monitor(
        runtime,
        "lease_release_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
    )
    scheduler = DbMonitorScheduler(
        runtime=runtime,
        scheduler_id="lease-release-success",
    )

    results = await scheduler.run_once(now=NOW)
    assert results[0].claimed is True
    assert await runtime.monitor_store.claim_monitor_tick_lease(
        monitor.id,
        lease_id="after-success",
        now=(NOW + timedelta(seconds=1)).isoformat(),
        expires_at=(NOW + timedelta(minutes=1)).isoformat(),
    )
    await runtime.monitor_store.release_monitor_tick_lease(
        monitor.id,
        lease_id="after-success",
    )

    failing_scheduler = DbMonitorScheduler(
        runtime=runtime,
        scheduler_id="lease-release-failure",
    )

    async def fail_tick(*args, **kwargs):
        raise RuntimeError("scheduler runner failed")

    monkeypatch.setattr(failing_scheduler.runner, "tick_monitor", fail_tick)

    with pytest.raises(RuntimeError, match="scheduler runner failed"):
        await failing_scheduler.run_once(now=NOW + timedelta(seconds=2))

    assert await runtime.monitor_store.claim_monitor_tick_lease(
        monitor.id,
        lease_id="after-failure",
        now=(NOW + timedelta(seconds=3)).isoformat(),
        expires_at=(NOW + timedelta(minutes=1)).isoformat(),
    )


async def test_scheduler_prevents_tick_lease_collisions():
    runtime = DbRuntime(runtime_id="db-monitor-scheduler-lease")
    await _create_monitor(runtime, "leased_monitor")
    claimed = await runtime.monitor_store.claim_monitor_tick_lease(
        "leased_monitor",
        lease_id="external-holder",
        now=NOW.isoformat(),
        expires_at=(NOW + timedelta(minutes=5)).isoformat(),
    )
    assert claimed is True

    runs = await runtime.tick_monitors(now=NOW + timedelta(seconds=1))

    assert runs[0].status == "skipped"
    assert runs[0].summary["reason"] == "lease_lost"
    state = await runtime.monitor_store.load_monitor_state("leased_monitor")
    assert state.last_tick_at is None
    assert await runtime.store.list_tasks() == []


async def test_in_memory_delete_clears_tick_lease_for_recreated_monitor():
    runtime = DbRuntime(runtime_id="db-monitor-scheduler-delete-lease")
    monitor = await _create_monitor(runtime, "recreated_monitor")
    assert await runtime.monitor_store.claim_monitor_tick_lease(
        monitor.id,
        lease_id="old-lease",
        now=NOW.isoformat(),
        expires_at=(NOW + timedelta(minutes=5)).isoformat(),
    )

    await runtime.delete_monitor(monitor.id)
    await _create_monitor(runtime, monitor.id)

    assert await runtime.monitor_store.claim_monitor_tick_lease(
        monitor.id,
        lease_id="new-lease",
        now=(NOW + timedelta(seconds=1)).isoformat(),
        expires_at=(NOW + timedelta(minutes=5)).isoformat(),
    )


async def test_sqlite_tick_lease_prevents_cross_store_collisions(tmp_path):
    path = tmp_path / "runtime.sqlite"
    runtime = DbRuntime(
        runtime_id="db-monitor-scheduler-sqlite-lease",
        store=SQLiteRuntimeStore(path),
    )
    await _create_monitor(runtime, "sqlite_leased_monitor")
    other_store = SQLiteDbMonitorStore(path)

    assert await runtime.monitor_store.claim_monitor_tick_lease(
        "sqlite_leased_monitor",
        lease_id="process-a",
        now=NOW.isoformat(),
        expires_at=(NOW + timedelta(minutes=5)).isoformat(),
    )
    assert not await other_store.claim_monitor_tick_lease(
        "sqlite_leased_monitor",
        lease_id="process-b",
        now=(NOW + timedelta(seconds=1)).isoformat(),
        expires_at=(NOW + timedelta(minutes=5)).isoformat(),
    )

    runs = await runtime.tick_monitors(now=NOW + timedelta(seconds=2))

    assert runs[0].status == "skipped"
    assert runs[0].summary["reason"] == "lease_lost"
    assert await runtime.store.list_tasks() == []


async def test_run_commit_rejects_stale_monitor_or_state_snapshot():
    runtime = DbRuntime(runtime_id="db-monitor-scheduler-stale-run")
    monitor = await _create_monitor(runtime, "stale_monitor")
    state = await runtime.monitor_store.load_monitor_state(monitor.id)
    assert state is not None

    await runtime.update_monitor(monitor.id, {"description": "changed"})

    try:
        await runtime.monitor_store.commit_monitor_mutation(
            _run_mutation(monitor=monitor, state=state, suffix="stale-monitor")
        )
    except ValueError as exc:
        assert "changed during mutation" in str(exc)
    else:
        raise AssertionError("stale monitor run commit should fail")

    current_monitor = await runtime.monitor_store.load_monitor(monitor.id)
    current_state = await runtime.monitor_store.load_monitor_state(monitor.id)
    assert current_monitor is not None
    assert current_state is not None
    await runtime.monitor_store.save_monitor_state(
        DbMonitorState.from_dict(
            {
                **current_state.to_dict(),
                "cursor": {"advanced": True},
            }
        )
    )

    try:
        await runtime.monitor_store.commit_monitor_mutation(
            _run_mutation(
                monitor=current_monitor,
                state=current_state,
                suffix="stale-state",
            )
        )
    except ValueError as exc:
        assert "state" in str(exc)
    else:
        raise AssertionError("stale state run commit should fail")


async def test_scheduler_respects_pause_cooldown_and_backoff_gates():
    runtime = DbRuntime(runtime_id="db-monitor-scheduler-gates")
    paused = await _create_monitor(runtime, "paused_monitor")
    cooling = await _create_monitor(runtime, "cooling_monitor")
    backing_off = await _create_monitor(runtime, "backoff_monitor")
    await runtime.pause_monitor(
        paused.id, paused_until=(NOW + timedelta(hours=1)).isoformat()
    )
    await runtime.monitor_store.save_monitor_state(
        DbMonitorState(
            monitor_id=cooling.id,
            cooldown_until=(NOW + timedelta(minutes=10)).isoformat(),
        )
    )
    await runtime.monitor_store.save_monitor_state(
        DbMonitorState(
            monitor_id=backing_off.id,
            error={"backoff_until": (NOW + timedelta(minutes=10)).isoformat()},
        )
    )

    runs = await runtime.tick_monitors(now=NOW)

    assert {run.monitor_id: run.summary["reason"] for run in runs} == {
        "paused_monitor": "paused",
        "cooling_monitor": "cooldown",
        "backoff_monitor": "backoff",
    }
    assert all(run.status == "skipped" for run in runs)
    assert [task.capability_id for task in await runtime.store.list_tasks()] == [
        "db.monitor.plan_lifecycle",
        "db.monitor.commit_lifecycle",
    ]


def _run_mutation(*, monitor, state, suffix):
    operation_id = f"monitor-tick-{suffix}"
    return DbMonitorMutation(
        action="run",
        operation=Operation(
            id=operation_id,
            operation_type="monitor.tick",
            status=OperationStatus.SUCCEEDED,
        ),
        monitor_before=monitor,
        state_before=state,
        state_after=state,
        run_after=DbMonitorRun(
            id=f"monitor-run-{suffix}",
            monitor_id=monitor.id,
            operation_id=operation_id,
            tick_started_at=NOW.isoformat(),
            tick_finished_at=NOW.isoformat(),
            status="succeeded",
        ),
    )


async def test_validation_blocked_monitor_persists_blocked_run_and_audit():
    runtime = DbRuntime(runtime_id="db-monitor-scheduler-validation")
    await _create_monitor(
        runtime,
        "invalid_monitor",
        metadata={
            "validation": _validation(
                False,
                errors=("missing delivery capability",),
                missing_capabilities=("slack.message.send",),
            )
        },
    )

    runs = await runtime.tick_monitors(now=NOW)

    assert runs[0].status == "blocked"
    assert runs[0].summary["reason"] == "validation_blocked"
    state = await runtime.monitor_store.load_monitor_state("invalid_monitor")
    assert state.consecutive_failures == 1
    assert state.error["reason"] == "validation_blocked"
    evidence = await runtime.store.list_evidence(runs[0].operation_id)
    assert evidence[0].kind == "monitor.trigger_decision"
    assert evidence[0].accepted is False
    assert evidence[0].payload["validation"]["missing_capabilities"] == [
        "slack.message.send"
    ]
    assert await runtime.store.list_tasks() == []


async def test_due_monitor_observation_runs_through_db_runtime_execute_task():
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 7}])
    runtime = DbRuntime(
        runtime_id="db-monitor-scheduler-execute-task",
        plugins=(plugin,),
    )
    await _create_monitor(
        runtime,
        "observed_monitor",
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
    )
    calls = []
    original_execute_task = runtime.execute_task

    async def spy_execute_task(task, operation, context=None):
        calls.append((task, operation, context))
        return await original_execute_task(task, operation, context=context)

    runtime.execute_task = spy_execute_task

    runs = await runtime.tick_monitors(now=NOW)

    assert runs[0].status == "succeeded"
    assert [task.capability_id for task, _, _ in calls] == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    assert all(
        operation.id == runs[0].operation_id
        and operation.operation_type == "monitor.tick"
        for _, operation, _ in calls
    )
    tasks = await runtime.store.list_tasks(runs[0].operation_id)
    assert [task.id for task in tasks] == [task.id for task, _, _ in calls]
    assert calls[1][0].input["sql_ref"] == "sql.validation"


async def test_planned_read_observation_resolves_cursor_parameters_before_read():
    plugin = MonitorReadProbePlugin(rows=[{"id": 42, "email": "new@example.test"}])
    runtime = DbRuntime(
        runtime_id="db-monitor-scheduler-cursor-params",
        plugins=(plugin,),
    )
    await _create_monitor(
        runtime,
        "new_orders_monitor",
        observation_plan={
            "kind": "planned_read",
            "sql": "select * from orders where id > ? order by id asc limit 100",
            "parameters": [
                {
                    "ref": "monitor.state.cursor.last_id",
                    "source": "monitor_state",
                    "path": ["cursor", "last_id"],
                }
            ],
            "cursor": {"field": "id", "initialization": "zero"},
            "cursor_update": {"last_id": "max(rows.id)"},
            "value_path": "rows",
            "source_scope": ["orders"],
            "validation_owner": "monitor_read_probe",
            "execution_owner": "monitor_read_probe",
            "sql_dialect": "sqlite",
        },
        trigger={
            "type": "new_rows",
            "path": "rows",
            "operator": "count_gt",
            "value": 0,
        },
    )
    state = await runtime.monitor_store.load_monitor_state("new_orders_monitor")
    assert state is not None
    await runtime.monitor_store.save_monitor_state(
        replace(state, cursor={"last_id": 17})
    )
    calls = []
    original_execute_task = runtime.execute_task

    async def spy_execute_task(task, operation, context=None):
        calls.append((task, operation, context))
        return await original_execute_task(task, operation, context=context)

    runtime.execute_task = spy_execute_task

    runs = await runtime.tick_monitors(now=NOW)

    assert runs[0].status == "triggered"
    read_task = next(
        task for task, _, _ in calls if task.capability_id == "db.sql.execute_read"
    )
    assert read_task.input["params"] == [17]
    assert read_task.input["param_specs"] == [
        {
            "ref": "monitor.state.cursor.last_id",
            "source": "monitor_state",
            "path": ["cursor", "last_id"],
        }
    ]
    assert "focus" not in read_task.input
    updated = await runtime.monitor_store.load_monitor_state("new_orders_monitor")
    assert updated.cursor["last_id"] == 42


async def test_planned_read_observation_preserves_typed_param_specs_before_read():
    plugin = MonitorReadProbePlugin(
        rows=[{"id": 1, "created_at": "2026-06-24T18:30:00+00:00"}]
    )
    runtime = DbRuntime(
        runtime_id="db-monitor-scheduler-typed-cursor-params",
        plugins=(plugin,),
    )
    typed_param = {
        "ref": "monitor.state.cursor.last_created_at",
        "source": "monitor_state",
        "path": ["cursor", "last_created_at"],
        "table": "runtime_operations",
        "column": "created_at",
        "db_type": "timestamp with time zone",
        "native_type": "datetime",
        "dialect": "postgresql",
        "nullable": False,
    }
    await _create_monitor(
        runtime,
        "runtime_operations_new_rows",
        observation_plan={
            "kind": "planned_read",
            "sql": (
                "select * from runtime_operations "
                "where created_at > $1 order by created_at asc limit 100"
            ),
            "parameters": [typed_param],
            "cursor": {"field": "created_at", "initialization": "monitor_created_at"},
            "cursor_update": {"last_created_at": "max(rows.created_at)"},
            "value_path": "rows",
            "source_scope": ["runtime_operations"],
            "validation_owner": "monitor_read_probe",
            "execution_owner": "monitor_read_probe",
            "sql_dialect": "postgresql",
        },
        trigger={
            "type": "new_rows",
            "path": "rows",
            "operator": "count_gt",
            "value": 0,
        },
    )
    state = await runtime.monitor_store.load_monitor_state(
        "runtime_operations_new_rows"
    )
    assert state is not None
    await runtime.monitor_store.save_monitor_state(
        replace(
            state,
            cursor={"last_created_at": "2026-06-24T18:22:26.382861+00:00"},
        )
    )
    calls = []
    original_execute_task = runtime.execute_task

    async def spy_execute_task(task, operation, context=None):
        calls.append((task, operation, context))
        return await original_execute_task(task, operation, context=context)

    runtime.execute_task = spy_execute_task

    runs = await runtime.tick_monitors(now=NOW)

    assert runs[0].status == "triggered"
    read_task = next(
        task for task, _, _ in calls if task.capability_id == "db.sql.execute_read"
    )
    assert read_task.input["params"] == ["2026-06-24T18:22:26.382861+00:00"]
    assert read_task.input["param_specs"] == [typed_param]


async def test_rest_source_observation_executes_through_runtime_and_cites_plugin_evidence():
    plugin = MonitorCapabilityProbePlugin(
        plugin_id="rest_source_probe",
        capability_id="rest.http.get",
        evidence_kind="api.http.response",
        role="source",
        kind="rest",
        access=AccessMode.READ,
        side_effecting=False,
    )
    runtime = DbRuntime(
        runtime_id="db-monitor-plugin-source",
        plugins=(plugin,),
    )
    await _create_monitor(
        runtime,
        "rest_source_monitor",
        observation_plan=_plugin_source_observation(),
        trigger={"gt": 1},
    )
    calls = []
    original_execute_task = runtime.execute_task

    async def spy_execute_task(task, operation, context=None):
        calls.append((task, operation, context))
        return await original_execute_task(task, operation, context=context)

    runtime.execute_task = spy_execute_task

    runs = await runtime.tick_monitors(now=NOW)
    evidence = await runtime.store.list_evidence(runs[0].operation_id)
    observation = next(item for item in evidence if item.kind == "monitor.observation")
    decision = next(
        item for item in evidence if item.kind == "monitor.trigger_decision"
    )
    plugin_result = next(item for item in evidence if item.kind == "api.http.response")

    assert runs[0].status == "triggered"
    assert [task.capability_id for task, _, _ in calls] == ["rest.http.get"]
    assert calls[0][1].operation_type == "monitor.tick"
    assert plugin.executor.calls == 1
    assert observation.payload["observed_value"] == 3
    assert plugin_result.id in observation.payload["evidence_ids"]
    assert plugin_result.id in decision.payload["observation_source_evidence_ids"]
    assert decision.payload["observation_evidence_id"] == observation.id


@pytest.mark.parametrize(
    ("observation_plan", "plugins", "reason"),
    [
        (
            _metric_observation("select * from"),
            (MonitorReadProbePlugin(),),
            "observation_sql_validation_failed",
        ),
        (
            _metric_observation("delete from orders"),
            (MonitorReadProbePlugin(),),
            "unsafe_observation_sql",
        ),
        (
            _metric_observation("drop table orders"),
            (MonitorReadProbePlugin(),),
            "unsafe_observation_sql",
        ),
        (
            {
                **_metric_observation(),
                "source_scope": ["customers"],
            },
            (MonitorReadProbePlugin(),),
            "observation_source_scope_blocked",
        ),
        (
            _metric_observation(),
            (),
            "missing_observation_capability",
        ),
        (
            _metric_observation(),
            (MonitorReadProbePlugin(validation_valid=False),),
            "observation_sql_validation_failed",
        ),
    ],
)
async def test_unsafe_or_unavailable_observation_blocks_with_audit(
    observation_plan,
    plugins,
    reason,
):
    runtime = DbRuntime(
        runtime_id=f"db-monitor-scheduler-blocked-{reason}",
        plugins=plugins,
    )
    await _create_monitor(
        runtime,
        "blocked_observation_monitor",
        observation_plan=observation_plan,
        trigger={"path": "pending_count", "gt": 10},
    )

    runs = await runtime.tick_monitors(now=NOW)

    assert runs[0].status == "blocked"
    assert runs[0].summary["reason"] == reason
    state = await runtime.monitor_store.load_monitor_state(
        "blocked_observation_monitor"
    )
    assert state.consecutive_failures == 1
    assert state.error["reason"] == reason
    evidence = await runtime.store.list_evidence(runs[0].operation_id)
    observation = next(item for item in evidence if item.kind == "monitor.observation")
    decision = next(
        item for item in evidence if item.kind == "monitor.trigger_decision"
    )
    assert observation.accepted is False
    assert observation.payload["reason"] == reason
    assert decision.accepted is False
    assert decision.payload["observation_evidence_id"] == observation.id


def test_monitor_scheduler_does_not_bypass_runtime_execution_boundaries():
    source = inspect.getsource(monitor_scheduler_module)

    assert ".kernel.execute_task(" not in source
    assert "RuntimeKernel.execute_task" not in source
    assert "analyze_sql" not in source
    assert "_fallback_sql_facts" not in source
    assert "WorkerRuntime" not in source
    assert "approval_channel" not in source
    assert "report_generate" not in source
    assert "slack" not in source.lower()
    assert "email" not in source.lower()
    assert "webhook" not in source.lower()
    assert "httpx" not in source.lower()
    assert "requests." not in source.lower()
    assert ".query(" not in source
    assert ".execute(" not in source


async def test_tick_monitors_sets_up_plugins_before_observation_execution():
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 4}])
    runtime = DbRuntime(
        runtime_id="db-monitor-scheduler-setup",
        plugins=(plugin,),
    )
    await _create_monitor(
        runtime,
        "setup_monitor",
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
    )

    assert plugin.setup_called is False

    runs = await runtime.tick_monitors(now=NOW)

    assert runs[0].status == "succeeded"
    assert plugin.setup_called is True


async def test_observation_executor_failure_persists_failed_observation_evidence():
    plugin = MonitorReadProbePlugin(fail_read=True)
    runtime = DbRuntime(
        runtime_id="db-monitor-scheduler-read-failure",
        plugins=(plugin,),
    )
    await _create_monitor(
        runtime,
        "failing_observation_monitor",
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
    )

    runs = await runtime.tick_monitors(now=NOW)

    assert runs[0].status == "failed"
    assert runs[0].summary["reason"] == "observation_execution_failed"
    evidence = await runtime.store.list_evidence(runs[0].operation_id)
    observation = next(item for item in evidence if item.kind == "monitor.observation")
    decision = next(
        item for item in evidence if item.kind == "monitor.trigger_decision"
    )
    assert observation.accepted is False
    assert observation.payload["status"] == "failed"
    assert observation.payload["reason"] == "observation_execution_failed"
    assert observation.payload["task_ids"]
    assert observation.payload["evidence_ids"]
    assert decision.payload["observation_evidence_id"] == observation.id


async def test_non_trigger_tick_persists_run_state_and_trigger_decision_evidence():
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 4}])
    runtime = DbRuntime(
        runtime_id="db-monitor-scheduler-nontrigger",
        plugins=(plugin,),
    )
    await _create_monitor(
        runtime,
        "healthy_monitor",
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
    )

    runs = await runtime.tick_monitors(now=NOW)

    assert runs[0].status == "succeeded"
    assert runs[0].triggered is False
    assert runs[0].trigger_decision_evidence_id is not None
    assert runs[0].summary["reason"] == "no_match"
    state = await runtime.monitor_store.load_monitor_state("healthy_monitor")
    assert state.last_tick_at == NOW.isoformat()
    assert state.consecutive_matches == 0
    evidence = await runtime.store.list_evidence(runs[0].operation_id)
    assert [item.kind for item in evidence][-2:] == [
        "monitor.observation",
        "monitor.trigger_decision",
    ]
    observation = next(item for item in evidence if item.kind == "monitor.observation")
    decision = next(
        item for item in evidence if item.kind == "monitor.trigger_decision"
    )
    assert observation.payload["observed_value"] == {"pending_count": 4}
    assert decision.payload["observation_evidence_id"] == observation.id
    assert decision.payload["summary"]["value_summary"] == {
        "keys": ["pending_count"],
        "type": "object",
    }
    assert await runtime.store.list_tasks(runs[0].operation_id)


async def test_triggered_tick_creates_generic_operation_and_counts_consecutive_matches():
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    runtime = DbRuntime(
        runtime_id="db-monitor-scheduler-triggered",
        plugins=(plugin,),
    )
    await _create_monitor(
        runtime,
        "triggering_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10, "consecutive_matches": 2},
        action_plan={
            "steps": [
                {"kind": "report_generate"},
                {"kind": "deliver", "capability_id": "slack.message.send"},
            ]
        },
    )

    first = await runtime.tick_monitors(now=NOW)
    second = await runtime.tick_monitors(now=NOW + timedelta(seconds=1))

    assert first[0].status == "succeeded"
    assert first[0].triggered is False
    assert first[0].summary["consecutive_matches"] == 1
    assert first[0].summary["trigger_ready"] is False
    assert second[0].status == "triggered"
    assert second[0].triggered is True
    assert second[0].summary["consecutive_matches"] == 2
    assert second[0].summary["triggered_operation_id"] is not None

    state = await runtime.monitor_store.load_monitor_state("triggering_monitor")
    assert state.consecutive_matches == 2
    assert state.last_triggered_at == (NOW + timedelta(seconds=1)).isoformat()
    assert state.last_operation_id == second[0].operation_id
    assert state.last_tick_operation_id == second[0].operation_id
    assert (
        state.last_triggered_operation_id == second[0].summary["triggered_operation_id"]
    )

    operations = await runtime.store.list_operations()
    operation_types = [operation.operation_type for operation in operations]
    assert operation_types == [
        "monitor.create",
        "monitor.tick",
        "monitor.tick",
        "monitor.triggered",
    ]
    triggered_operation = operations[-1]
    assert triggered_operation.metadata["parent_operation_id"] == second[0].operation_id
    assert triggered_operation.metadata["tick_operation_id"] == second[0].operation_id
    assert triggered_operation.metadata["monitor_run_id"] == second[0].id
    assert {task.capability_id for task in await runtime.store.list_tasks()} == {
        "db.sql.validate",
        "db.sql.execute_read",
    }


async def test_triggered_investigation_action_executes_analysis_through_execute_task():
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    runtime = DbRuntime(
        runtime_id="db-monitor-action-investigation",
        plugins=(plugin,),
    )
    calls = []
    original_execute_task = runtime.execute_task

    async def spy_execute_task(task, operation, context=None):
        calls.append((task.capability_id, operation.id, dict(context or {})))
        return await original_execute_task(task, operation, context=context)

    runtime.execute_task = spy_execute_task
    await _create_monitor(
        runtime,
        "investigating_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan={
            "kind": "investigation",
            "goal": "Explain why pending orders breached",
            "steps": [
                {
                    "id": "final_synthesis",
                    "kind": "synthesis",
                    "purpose": "Summarize the monitor trigger",
                    "expected_evidence": ["analysis.synthesis"],
                }
            ],
        },
    )

    runs = await runtime.tick_monitors(now=NOW)
    run = runs[0]
    child_id = run.summary["triggered_operation_id"]
    snapshot = await runtime.inspect_operation(child_id)

    assert run.triggered is True
    assert run.summary["action_status"] == "succeeded"
    assert "db.analysis.plan.validate" in {item[0] for item in calls}
    assert "db.analysis.summarize" in {item[0] for item in calls}
    assert all(item[1] in {run.operation_id, child_id} for item in calls)
    assert any(item.kind == "analysis.plan" for item in snapshot.evidence)
    assert any(item.kind == "analysis.synthesis" for item in snapshot.evidence)
    action_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.action_result"
    )
    cited = action_result.payload["cited_tick_evidence_refs"]
    assert {item["kind"] for item in cited} == {
        "monitor.observation",
        "monitor.trigger_decision",
    }
    assert action_result.payload["task_ids"]


async def test_scheduled_report_action_uses_validated_reads_and_blocks_vague_delivery():
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12, "revenue": 42}])
    runtime = DbRuntime(
        runtime_id="db-monitor-action-report",
        plugins=(plugin,),
    )
    await _create_monitor(
        runtime,
        "reporting_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"type": "scheduled"},
        action_plan={
            "kind": "scheduled_report",
            "title": "Daily order health",
            "steps": [
                {
                    "id": "freshness",
                    "kind": "freshness_sql",
                    "metric": "orders_freshness",
                    "sql": "select max(updated_at) as latest_updated_at from orders",
                    "source_scope": ["orders"],
                    "capability_owner": "monitor_read_probe",
                },
                {
                    "id": "revenue",
                    "kind": "metric_sql",
                    "metric": "revenue",
                    "sql": "select sum(total) as revenue from orders",
                    "source_scope": ["orders"],
                    "capability_owner": "monitor_read_probe",
                },
                {
                    "id": "report_summary",
                    "kind": "synthesis",
                    "purpose": "Generate the report narrative",
                    "expected_evidence": ["analysis.synthesis"],
                },
            ],
            "output": {"kind": "report", "format": "markdown"},
            "delivery": {"mode": "slack"},
        },
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    task_capabilities = [task.capability_id for task in snapshot.tasks]

    assert run.summary["action_status"] == "succeeded"
    assert task_capabilities.count("db.sql.validate") == 2
    assert task_capabilities.count("db.sql.execute_read") == 2
    assert all(
        "focus" not in task.input
        for task in snapshot.tasks
        if task.capability_id == "db.sql.execute_read"
    )
    assert not any("delivery" in task.capability_id for task in snapshot.tasks)
    assert not any("slack" in task.capability_id for task in snapshot.tasks)
    assert any(item.kind == "analysis.synthesis" for item in snapshot.evidence)
    report = next(item for item in snapshot.evidence if item.kind == "monitor.report")
    assert report.payload["delivery_status"] == "deferred"
    assert report.payload["delivery_phase"] == 6
    assert report.payload["delivery_intent"] == {"mode": "slack"}
    action_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.action_result"
    )
    assert report.id in {
        item["id"] for item in action_result.payload["produced_evidence_refs"]
    }
    delivery_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_result"
    )
    assert delivery_result.accepted is False
    assert delivery_result.payload["status"] == "blocked"
    assert delivery_result.payload["block_reason"] == "missing_capability"
    assert run.summary["delivery_status"] == "blocked"


async def test_slack_scheduled_report_delivery_executes_through_runtime_and_persists_result():
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12, "revenue": 42}])
    delivery_plugin = MonitorCapabilityProbePlugin(
        plugin_id="slack_delivery_probe",
        capability_id="slack.summary.send",
        evidence_kind="slack.operation.result",
        role="delivery",
        kind="slack",
    )
    runtime = DbRuntime(
        runtime_id="db-monitor-delivery-slack",
        plugins=(db_plugin, delivery_plugin),
    )
    calls = []
    original_execute_task = runtime.execute_task

    async def spy_execute_task(task, operation, context=None):
        calls.append((task, operation, dict(context or {})))
        return await original_execute_task(task, operation, context=context)

    runtime.execute_task = spy_execute_task
    await _create_monitor(
        runtime,
        "slack_delivery_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"type": "scheduled"},
        action_plan=_scheduled_report_action(
            {
                "delivery_kind": "slack",
                "capability_id": "slack.summary.send",
                "capability_owner": "slack_delivery_probe",
                "target": {"channel": "#revops"},
                "format": "markdown",
            }
        ),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    delivery_task_calls = [
        item for item in calls if item[0].capability_id == "slack.summary.send"
    ]
    delivery_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_result"
    )
    plugin_result = next(
        item for item in snapshot.evidence if item.kind == "slack.operation.result"
    )

    assert run.summary["delivery_status"] == "succeeded"
    assert len(delivery_task_calls) == 1
    assert delivery_task_calls[0][1].id == run.summary["triggered_operation_id"]
    assert delivery_task_calls[0][2]["monitor_action_role"] == "delivery"
    assert delivery_plugin.executor.calls == 1
    assert delivery_result.accepted is True
    assert plugin_result.id in {
        item["id"] for item in delivery_result.payload["plugin_result_evidence_refs"]
    }
    assert run.summary["delivery_result_evidence_id"] == delivery_result.id
    assert (
        run.summary["delivery_plugin_result_evidence_refs"][0]["id"] == plugin_result.id
    )


async def test_notification_action_uses_builtin_local_delivery_capability():
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    runtime = DbRuntime(
        runtime_id="db-monitor-delivery-local",
        plugins=(db_plugin,),
    )
    await runtime.setup()
    assert runtime.registry.get_capability(
        "monitor.delivery.local",
        owner="db_runtime",
    )
    calls = []
    original_execute_task = runtime.execute_task

    async def spy_execute_task(task, operation, context=None):
        calls.append((task, operation, dict(context or {})))
        return await original_execute_task(task, operation, context=context)

    runtime.execute_task = spy_execute_task
    await _create_monitor(
        runtime,
        "local_notification_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan=_notification_action(),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    delivery_task = next(
        task
        for task in snapshot.tasks
        if task.capability_id == "monitor.delivery.local"
    )
    delivery_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_result"
    )
    local_delivery = next(
        item for item in snapshot.evidence if item.kind == "local.notification.delivery"
    )
    report = next(item for item in snapshot.evidence if item.kind == "monitor.report")
    decision_ref = next(
        ref
        for ref in delivery_result.payload["source_evidence_refs"]
        if ref["kind"] == "monitor.trigger_decision"
    )

    assert run.summary["action_kind"] == "notification"
    assert run.summary["delivery_status"] == "succeeded"
    assert run.summary["delivery_target"] == {"type": "runtime_console"}
    assert run.summary["delivery_channel"] == "runtime_console"
    assert [item[0].capability_id for item in calls].count(
        "monitor.delivery.local"
    ) == 1
    assert delivery_task.metadata["monitor_id"] == "local_notification_monitor"
    assert delivery_task.metadata["monitor_run_id"] == run.id
    assert delivery_task.metadata["tick_operation_id"] == run.operation_id
    assert delivery_task.metadata["monitor_delivery_target"] == {
        "type": "runtime_console"
    }
    assert local_delivery.task_id == delivery_task.id
    assert local_delivery.payload["status"] == "delivered"
    assert local_delivery.payload["target_channel"] == "runtime_console"
    assert local_delivery.payload["monitor_run_id"] == run.id
    assert local_delivery.payload["tick_operation_id"] == run.operation_id
    assert delivery_result.payload["capability_id"] == "monitor.delivery.local"
    assert delivery_result.payload["capability_owner"] == "db_runtime"
    assert delivery_result.payload["delivery_target"] == {"type": "runtime_console"}
    assert delivery_result.payload["delivery_channel"] == "runtime_console"
    assert delivery_result.metadata["monitor_delivery_channel"] == "runtime_console"
    assert delivery_result.payload["action_plan_fingerprint"]
    assert delivery_result.payload["report_fingerprint"]
    assert delivery_result.payload["delivery_result_evidence_id"] == delivery_result.id
    assert report.payload["action_kind"] == "notification"
    assert decision_ref["id"] == run.trigger_decision_evidence_id


async def test_local_delivery_replay_reuses_existing_result_without_reexecuting_task():
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    runtime = DbRuntime(
        runtime_id="db-monitor-delivery-local-replay",
        plugins=(db_plugin,),
    )
    await _create_monitor(
        runtime,
        "local_replay_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan=_notification_action(),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    child_id = run.summary["triggered_operation_id"]
    before = await runtime.inspect_operation(child_id)
    existing_result = next(
        item for item in before.evidence if item.kind == "monitor.delivery_result"
    )
    calls = []
    original_execute_task = runtime.execute_task

    async def spy_execute_task(task, operation, context=None):
        calls.append(task.capability_id)
        return await original_execute_task(task, operation, context=context)

    runtime.execute_task = spy_execute_task
    replayed = await runtime.execute_monitor_delivery(
        child_id,
        monitor_id="local_replay_monitor",
        monitor_name="Local Replay Monitor",
        monitor_run_id=run.id,
        tick_operation_id=run.operation_id,
        report_evidence_id=run.summary["report_evidence_id"],
    )
    after = await runtime.inspect_operation(child_id)

    assert replayed["delivery_result_evidence_id"] == existing_result.id
    assert calls == []
    assert [
        item.kind
        for item in after.evidence
        if item.kind == "local.notification.delivery"
    ] == ["local.notification.delivery"]


async def test_local_delivery_unsupported_target_blocks_before_task_execution():
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    runtime = DbRuntime(
        runtime_id="db-monitor-delivery-local-unsupported",
        plugins=(db_plugin,),
    )
    await _create_monitor(
        runtime,
        "unsupported_local_delivery_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan=_notification_action(
            {
                "delivery_kind": "local",
                "target": {"type": "pagerduty"},
                "format": "markdown",
            }
        ),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    delivery_plan = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_plan"
    )
    delivery_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_result"
    )

    assert run.summary["delivery_status"] == "blocked"
    assert delivery_plan.accepted is False
    assert delivery_plan.payload["block_reason"] == "unsupported_delivery_target"
    assert delivery_result.payload["block_reason"] == "unsupported_delivery_target"
    assert delivery_result.payload["delivery_target"] == {"type": "pagerduty"}
    assert delivery_result.payload["delivery_channel"] == "pagerduty"
    assert not any(
        task.capability_id == "monitor.delivery.local" for task in snapshot.tasks
    )
    assert not any(
        item.kind == "local.notification.delivery" for item in snapshot.evidence
    )


@pytest.mark.parametrize(
    ("kind", "plugin_id", "capability_id", "evidence_kind", "target"),
    [
        (
            "email",
            "email_delivery_probe",
            "email.message.send",
            "email.operation.result",
            {"recipient": "ops@example.com"},
        ),
        (
            "webhook",
            "webhook_delivery_probe",
            "webhook.post",
            "webhook.operation.result",
            {"url": "https://hooks.example.test/revops"},
        ),
        (
            "rest",
            "rest_delivery_probe",
            "rest.http.post",
            "rest.operation.result",
            {"path": "/callbacks/report"},
        ),
    ],
)
async def test_delivery_selection_is_generic_for_email_webhook_and_rest(
    kind,
    plugin_id,
    capability_id,
    evidence_kind,
    target,
):
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12, "revenue": 42}])
    delivery_plugin = MonitorCapabilityProbePlugin(
        plugin_id=plugin_id,
        capability_id=capability_id,
        evidence_kind=evidence_kind,
        role="delivery",
        kind=kind,
    )
    runtime = DbRuntime(
        runtime_id=f"db-monitor-delivery-{kind}",
        plugins=(db_plugin, delivery_plugin),
    )
    await _create_monitor(
        runtime,
        f"{kind}_delivery_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"type": "scheduled"},
        action_plan=_scheduled_report_action(
            {"delivery_kind": kind, "target": target, "format": "markdown"}
        ),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    delivery_task = next(
        task for task in snapshot.tasks if task.capability_id == capability_id
    )
    delivery_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_result"
    )

    assert delivery_plugin.executor.calls == 1
    assert delivery_task.metadata["owner"] == plugin_id
    assert delivery_task.input["delivery_kind"] == kind
    assert delivery_task.input["target"] == target
    assert delivery_result.payload["capability_id"] == capability_id
    assert delivery_result.payload["capability_owner"] == plugin_id


@pytest.mark.parametrize(
    ("explicit_target", "expected_target", "expected_reason"),
    [
        (None, {"type": "team_inbox", "name": "default"}, None),
        (
            {"type": "team_inbox", "name": "explicit"},
            {"type": "team_inbox", "name": "explicit"},
            None,
        ),
        (
            {"type": "pager", "name": "explicit"},
            {"type": "pager", "name": "explicit"},
            "unsupported_delivery_target",
        ),
        ({}, {}, "missing_delivery_target"),
    ],
)
async def test_delivery_default_target_normalizes_runtime_contract_without_overrides(
    explicit_target,
    expected_target,
    expected_reason,
):
    declared_default = {"type": "team_inbox", "name": "default"}
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    delivery_plugin = MonitorCapabilityProbePlugin(
        plugin_id="team_delivery_probe",
        capability_id="team.inbox.send",
        evidence_kind="team.delivery.result",
        role="delivery",
        kind="team",
        accepted_target_types=("team_inbox",),
        default_target=declared_default,
    )
    runtime = DbRuntime(
        runtime_id=f"db-monitor-delivery-default-{expected_reason or 'accepted'}-"
        f"{(explicit_target or {}).get('name', 'omitted')}",
        plugins=(db_plugin, delivery_plugin),
    )
    intent = {"delivery_kind": "team", "format": "markdown"}
    if explicit_target is not None:
        intent["target"] = explicit_target
    await _create_monitor(
        runtime,
        "default_target_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan=_notification_action(intent),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    delivery_plan = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_plan"
    )
    delivery_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_result"
    )

    assert delivery_plan.payload["delivery_intent"]["target"] == expected_target
    assert delivery_result.payload["delivery_target"] == expected_target
    assert declared_default == {"type": "team_inbox", "name": "default"}
    if expected_reason is None:
        delivery_task = next(
            task for task in snapshot.tasks if task.capability_id == "team.inbox.send"
        )
        assert delivery_plugin.executor.calls == 1
        assert delivery_task.input["target"] == expected_target
        assert delivery_task.metadata["monitor_delivery_target"] == expected_target
        assert delivery_result.accepted is True
    else:
        assert delivery_plugin.executor.calls == 0
        assert delivery_plan.payload["block_reason"] == expected_reason
        assert delivery_plan.payload["capability_id"] == "team.inbox.send"
        assert delivery_result.payload["block_reason"] == expected_reason
        assert not any(
            task.capability_id == "team.inbox.send" for task in snapshot.tasks
        )


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("missing", "missing_delivery_target"),
        ("ambiguous", "ambiguous_capability"),
        ("malformed", "invalid_delivery_target_default"),
        ("unsupported", "unsupported_delivery_target"),
    ],
)
async def test_delivery_default_target_blocks_before_plugin_execution(case, reason):
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    common = {
        "capability_id": "team.inbox.send",
        "evidence_kind": "team.delivery.result",
        "role": "delivery",
        "kind": "team",
        "accepted_target_types": ("team_inbox",),
    }
    first = MonitorCapabilityProbePlugin(
        plugin_id="team_delivery_probe_a",
        default_target=(
            None
            if case == "missing"
            else (
                []
                if case == "malformed"
                else (
                    {"type": "pager"}
                    if case == "unsupported"
                    else {"type": "team_inbox", "name": "a"}
                )
            )
        ),
        **common,
    )
    delivery_plugins = [first]
    if case == "ambiguous":
        delivery_plugins.append(
            MonitorCapabilityProbePlugin(
                plugin_id="team_delivery_probe_b",
                default_target={"type": "team_inbox", "name": "b"},
                **common,
            )
        )
    runtime = DbRuntime(
        runtime_id=f"db-monitor-delivery-default-block-{case}",
        plugins=(db_plugin, *delivery_plugins),
    )
    await _create_monitor(
        runtime,
        f"default_target_{case}_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan=_notification_action(
            {"delivery_kind": "team", "format": "markdown"}
        ),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    delivery_plan = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_plan"
    )
    delivery_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_result"
    )

    assert delivery_plan.accepted is False
    assert delivery_plan.payload["block_reason"] == reason
    assert delivery_result.payload["block_reason"] == reason
    assert all(plugin.executor.calls == 0 for plugin in delivery_plugins)
    assert not any(
        task.metadata.get("reason") == "monitor_delivery" for task in snapshot.tasks
    )


@pytest.mark.parametrize(
    ("plugins", "intent", "reason"),
    [
        (
            (),
            {
                "delivery_kind": "slack",
                "target": {"channel": "#revops"},
                "format": "markdown",
            },
            "missing_capability",
        ),
        (
            (
                MonitorCapabilityProbePlugin(
                    plugin_id="slack_delivery_probe_a",
                    capability_id="slack.summary.send",
                    evidence_kind="slack.operation.result",
                    role="delivery",
                    kind="slack",
                ),
                MonitorCapabilityProbePlugin(
                    plugin_id="slack_delivery_probe_b",
                    capability_id="slack.summary.send",
                    evidence_kind="slack.operation.result",
                    role="delivery",
                    kind="slack",
                ),
            ),
            {
                "delivery_kind": "slack",
                "target": {"channel": "#revops"},
                "format": "markdown",
            },
            "ambiguous_capability",
        ),
        (
            (
                MonitorCapabilityProbePlugin(
                    plugin_id="slack_delivery_probe",
                    capability_id="slack.summary.send",
                    evidence_kind="slack.operation.result",
                    role="delivery",
                    kind="slack",
                    accepted_formats=("plain",),
                ),
            ),
            {
                "delivery_kind": "slack",
                "target": {"channel": "#revops"},
                "format": "markdown",
            },
            "unsupported_format",
        ),
        (
            (
                MonitorCapabilityProbePlugin(
                    plugin_id="slack_delivery_probe",
                    capability_id="slack.summary.send",
                    evidence_kind="slack.operation.result",
                    role="delivery",
                    kind="slack",
                    input_schema={
                        "type": "object",
                        "required": ["target", "payload_source"],
                        "properties": {
                            "target": {
                                "type": "object",
                                "required": ["workspace"],
                                "properties": {"workspace": {"type": "string"}},
                            },
                            "payload_source": {
                                "type": "object",
                                "required": ["report_evidence_id"],
                            },
                        },
                    },
                ),
            ),
            {
                "delivery_kind": "slack",
                "target": {"channel": "#revops"},
                "format": "markdown",
                "capability_id": "slack.summary.send",
                "capability_owner": "slack_delivery_probe",
            },
            "invalid_capability_input",
        ),
    ],
)
async def test_delivery_capability_blocks_are_audited(plugins, intent, reason):
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12, "revenue": 42}])
    runtime = DbRuntime(
        runtime_id=f"db-monitor-delivery-block-{reason}",
        plugins=(db_plugin, *plugins),
    )
    await _create_monitor(
        runtime,
        f"blocked_delivery_{reason}",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"type": "scheduled"},
        action_plan=_scheduled_report_action(intent),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    delivery_plan = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_plan"
    )
    delivery_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_result"
    )

    assert run.summary["delivery_status"] == "blocked"
    assert delivery_plan.accepted is False
    assert delivery_plan.payload["block_reason"] == reason
    assert delivery_result.accepted is False
    assert delivery_result.payload["block_reason"] == reason
    assert not any(
        task.metadata.get("reason") == "monitor_delivery" for task in snapshot.tasks
    )


async def test_plugin_source_missing_or_unsafe_capability_blocks_with_audit():
    unsafe = MonitorCapabilityProbePlugin(
        plugin_id="rest_source_probe",
        capability_id="rest.http.get",
        evidence_kind="api.http.response",
        role="source",
        kind="rest",
        access=AccessMode.WRITE,
        side_effecting=True,
    )
    runtime = DbRuntime(
        runtime_id="db-monitor-source-unsafe",
        plugins=(unsafe,),
    )
    await _create_monitor(
        runtime,
        "unsafe_source_monitor",
        observation_plan=_plugin_source_observation(),
        trigger={"gt": 1},
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    evidence = await runtime.store.list_evidence(run.operation_id)
    observation = next(item for item in evidence if item.kind == "monitor.observation")

    assert run.status == "blocked"
    assert observation.accepted is False
    assert observation.payload["reason"] in {
        "capability_shape_unsupported",
        "missing_capability",
    }
    assert await runtime.store.list_tasks(run.operation_id) == []


async def test_delivery_input_cites_phase5_evidence_without_prompt_or_query_payloads():
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12, "revenue": 42}])
    delivery_plugin = MonitorCapabilityProbePlugin(
        plugin_id="slack_delivery_probe",
        capability_id="slack.summary.send",
        evidence_kind="slack.operation.result",
        role="delivery",
        kind="slack",
    )
    runtime = DbRuntime(
        runtime_id="db-monitor-delivery-provenance",
        plugins=(db_plugin, delivery_plugin),
    )
    await _create_monitor(
        runtime,
        "delivery_provenance_monitor",
        description="do not copy this prompt text",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"type": "scheduled"},
        action_plan=_scheduled_report_action(
            {
                "delivery_kind": "slack",
                "target": {"channel": "#revops"},
                "format": "markdown",
            }
        ),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    delivery_task = next(
        task for task in snapshot.tasks if task.capability_id == "slack.summary.send"
    )
    report = next(item for item in snapshot.evidence if item.kind == "monitor.report")
    synthesis = next(
        item for item in snapshot.evidence if item.kind == "analysis.synthesis"
    )
    serialized_input = str(delivery_task.input)

    assert delivery_task.input["payload_source"]["report_evidence_id"] == report.id
    assert delivery_task.input["payload_source"]["report_fingerprint"]
    assert any(
        ref["id"] == synthesis.id
        for ref in delivery_task.input["payload_source"]["source_evidence_refs"]
    )
    assert "do not copy this prompt text" not in serialized_input
    assert "pending_count" not in serialized_input
    assert "revenue': 42" not in serialized_input


async def test_governance_required_delivery_blocks_without_approval_mutation():
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12, "revenue": 42}])
    delivery_plugin = MonitorCapabilityProbePlugin(
        plugin_id="slack_delivery_probe",
        capability_id="slack.summary.send",
        evidence_kind="slack.operation.result",
        role="delivery",
        kind="slack",
        requires_approval=True,
    )
    runtime = DbRuntime(
        runtime_id="db-monitor-delivery-approval-block",
        plugins=(db_plugin, delivery_plugin),
    )
    await _create_monitor(
        runtime,
        "approval_delivery_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"type": "scheduled"},
        action_plan=_scheduled_report_action(
            {
                "delivery_kind": "slack",
                "target": {"channel": "#revops"},
                "format": "markdown",
            }
        ),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    delivery_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_result"
    )

    assert run.summary["delivery_status"] == "blocked"
    assert delivery_result.payload["block_reason"] == "governance_approval_required"
    assert delivery_plugin.executor.calls == 0
    assert await runtime.store.list_approval_requests() == []


async def test_governed_delivery_requests_approval_without_sending():
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12, "revenue": 42}])
    delivery_plugin = MonitorCapabilityProbePlugin(
        plugin_id="slack_delivery_probe",
        capability_id="slack.summary.send",
        evidence_kind="slack.operation.result",
        role="delivery",
        kind="slack",
        requires_approval=True,
    )
    runtime = DbRuntime(
        runtime_id="db-monitor-delivery-governed-approval",
        plugins=(db_plugin, delivery_plugin),
    )
    await _create_monitor(
        runtime,
        "governed_delivery_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"type": "scheduled"},
        policy={"governed_delivery": True},
        action_plan=_scheduled_report_action(
            {
                "delivery_kind": "slack",
                "target": {"channel": "#revops"},
                "format": "markdown",
                "requires_approval": True,
            }
        ),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    child_id = run.summary["triggered_operation_id"]
    snapshot = await runtime.inspect_operation(child_id)
    approvals = await runtime.store.list_approval_requests(child_id)
    delivery_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.delivery_result"
    )
    delivery_task = next(
        task for task in snapshot.tasks if task.capability_id == "slack.summary.send"
    )

    assert delivery_plugin.executor.calls == 0
    assert len(approvals) == 1
    assert approvals[0].status is ApprovalStatus.PENDING
    assert delivery_task.status is TaskStatus.PENDING
    assert delivery_result.payload["block_reason"] == "governance_approval_required"
    assert await runtime.store.list_governance_audit_records(child_id)
    inbox = await runtime.list_monitor_approvals(monitor_id="governed_delivery_monitor")
    assert [item["approval_id"] for item in inbox] == [approvals[0].approval_id]
    assert inbox[0]["context"]["kind"] == "monitor.delivery"
    assert inbox[0]["context"]["monitor_run_id"] == run.id


async def test_approved_governed_delivery_resumes_and_sends_once():
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12, "revenue": 42}])
    delivery_plugin = MonitorCapabilityProbePlugin(
        plugin_id="slack_delivery_probe",
        capability_id="slack.summary.send",
        evidence_kind="slack.operation.result",
        role="delivery",
        kind="slack",
        requires_approval=True,
    )
    runtime = DbRuntime(
        runtime_id="db-monitor-delivery-governed-resume",
        plugins=(db_plugin, delivery_plugin),
    )
    await _create_monitor(
        runtime,
        "approved_delivery_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"type": "scheduled"},
        policy={"governed_delivery": True},
        action_plan=_scheduled_report_action(
            {
                "delivery_kind": "slack",
                "target": {"channel": "#revops"},
                "format": "markdown",
                "requires_approval": True,
            }
        ),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    child_id = run.summary["triggered_operation_id"]
    approval = (await runtime.store.list_approval_requests(child_id))[0]
    await runtime.approve_monitor_approval(approval.approval_id)
    calls = []
    original_execute_task = runtime.execute_task

    async def spy_execute_task(task, operation, context=None):
        calls.append((task.capability_id, task.id, operation.id))
        return await original_execute_task(task, operation, context=context)

    runtime.execute_task = spy_execute_task
    resumed = await runtime.resume_operation(child_id)

    delivery_calls = [item for item in calls if item[0] == "slack.summary.send"]
    delivery_result = [
        item for item in resumed.evidence if item.kind == "monitor.delivery_result"
    ][-1]
    plugin_result = next(
        item for item in resumed.evidence if item.kind == "slack.operation.result"
    )

    assert len(delivery_calls) == 1
    assert delivery_plugin.executor.calls == 1
    assert delivery_result.accepted is True
    assert plugin_result.id in {
        item["id"] for item in delivery_result.payload["plugin_result_evidence_refs"]
    }
    inspection = await runtime.inspect_monitor("approved_delivery_monitor")
    finalized_run = next(item for item in inspection.runs if item.id == run.id)
    assert finalized_run.summary["delivery_status"] == "succeeded"


@pytest.mark.parametrize(
    ("method", "expected_status", "expected_operation_status"),
    [
        ("reject_monitor_approval", "approval_rejected", OperationStatus.FAILED),
        ("cancel_monitor_approval", "approval_cancelled", OperationStatus.CANCELLED),
        ("expire", "approval_expired", OperationStatus.BLOCKED),
    ],
)
async def test_terminal_delivery_approvals_do_not_send(
    method, expected_status, expected_operation_status
):
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12, "revenue": 42}])
    delivery_plugin = MonitorCapabilityProbePlugin(
        plugin_id="slack_delivery_probe",
        capability_id="slack.summary.send",
        evidence_kind="slack.operation.result",
        role="delivery",
        kind="slack",
        requires_approval=True,
    )
    runtime = DbRuntime(
        runtime_id=f"db-monitor-delivery-terminal-{method}",
        plugins=(db_plugin, delivery_plugin),
    )
    await _create_monitor(
        runtime,
        f"terminal_delivery_{method}",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"type": "scheduled"},
        policy={"governed_delivery": True},
        action_plan=_scheduled_report_action(
            {
                "delivery_kind": "slack",
                "target": {"channel": "#revops"},
                "format": "markdown",
                "requires_approval": True,
            }
        ),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    child_id = run.summary["triggered_operation_id"]
    approval = (await runtime.store.list_approval_requests(child_id))[0]
    if method == "expire":
        await runtime.approval_channel.expire(approval.approval_id)
    else:
        await getattr(runtime, method)(approval.approval_id)

    resumed = await runtime.resume_operation(child_id)
    delivery_result = [
        item for item in resumed.evidence if item.kind == "monitor.delivery_result"
    ][-1]

    assert delivery_plugin.executor.calls == 0
    assert resumed.operation.status is expected_operation_status
    assert delivery_result.payload["block_reason"] == expected_status


async def test_monitor_approval_helpers_delegate_to_channel(monkeypatch):
    runtime = DbRuntime(runtime_id="db-monitor-approval-helper")
    approval = ApprovalRequest(
        approval_id="monitor-op:policy:human",
        operation_id="monitor-op",
        reason="approve monitor action",
        proposed_action={"approval": "human"},
        risk=RiskLevel.LOW,
        metadata={"version": "1"},
    )
    await runtime.store.save_operation(
        Operation(
            id="monitor-op",
            operation_type="monitor.triggered",
            metadata={
                "monitor_action_context": {
                    "monitor_id": "helper_monitor",
                    "monitor_run_id": "run-1",
                    "tick_operation_id": "tick-1",
                    "action_kind": "write_proposal",
                }
            },
        )
    )
    await runtime.approval_channel.request(approval)
    delegated = []
    original_approve = runtime.approval_channel.approve

    async def approve_spy(approval_id):
        delegated.append(approval_id)
        return await original_approve(approval_id)

    monkeypatch.setattr(runtime.approval_channel, "approve", approve_spy)

    listed = await runtime.list_monitor_approvals(
        monitor_id="helper_monitor",
        monitor_run_id="run-1",
    )
    approved = await runtime.approve_monitor_approval(approval.approval_id)

    assert [item["approval_id"] for item in listed] == [approval.approval_id]
    assert delegated == [approval.approval_id]
    assert approved.status is ApprovalStatus.APPROVED
    assert await runtime.list_monitor_approvals(monitor_id="helper_monitor") == ()


async def test_resume_skips_completed_delivery_task_and_materializes_result():
    db_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12, "revenue": 42}])
    delivery_plugin = MonitorCapabilityProbePlugin(
        plugin_id="slack_delivery_probe",
        capability_id="slack.summary.send",
        evidence_kind="slack.operation.result",
        role="delivery",
        kind="slack",
    )
    runtime = DbRuntime(
        runtime_id="db-monitor-delivery-resume",
        plugins=(db_plugin, delivery_plugin),
    )
    original_persist_delivery_result = runtime._persist_monitor_delivery_result

    async def skip_delivery_result(operation, **kwargs):
        return {
            "monitor_id": kwargs["monitor_id"],
            "monitor_run_id": kwargs["monitor_run_id"],
            "tick_operation_id": kwargs["tick_operation_id"],
            "delivery_operation_id": operation.id,
            "delivery_kind": kwargs["delivery_kind"],
            "status": kwargs["status"],
            "block_reason": kwargs.get("block_reason"),
            "task_ids": list(kwargs.get("task_ids") or ()),
            "plugin_result_evidence_refs": [],
            "idempotency_key": kwargs["idempotency_key"],
            "delivery_result_evidence_id": None,
        }

    runtime._persist_monitor_delivery_result = skip_delivery_result
    await _create_monitor(
        runtime,
        "resumable_delivery_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"type": "scheduled"},
        action_plan=_scheduled_report_action(
            {
                "delivery_kind": "slack",
                "target": {"channel": "#revops"},
                "format": "markdown",
            }
        ),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    child_id = run.summary["triggered_operation_id"]
    before = await runtime.inspect_operation(child_id)
    assert delivery_plugin.executor.calls == 1
    assert not any(item.kind == "monitor.delivery_result" for item in before.evidence)

    runtime._persist_monitor_delivery_result = original_persist_delivery_result
    resumed = await runtime.resume_operation(child_id)

    assert delivery_plugin.executor.calls == 1
    assert any(item.kind == "monitor.delivery_result" for item in resumed.evidence)
    delivery_tasks = [
        task
        for task in resumed.tasks
        if task.metadata.get("reason") == "monitor_delivery"
    ]
    assert len(delivery_tasks) == 1
    assert delivery_tasks[0].status is TaskStatus.SUCCEEDED
    inspection = await runtime.inspect_monitor("resumable_delivery_monitor")
    finalized_run = next(item for item in inspection.runs if item.id == run.id)
    assert finalized_run.summary["delivery_status"] == "succeeded"


async def test_monitor_action_resume_finalizes_missing_terminal_evidence():
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    runtime = DbRuntime(
        runtime_id="db-monitor-action-resume-finalize",
        plugins=(plugin,),
    )
    original_persist_result = runtime._persist_monitor_action_result

    async def skip_terminal_result(operation, **kwargs):
        return {
            "monitor_id": kwargs["monitor_id"],
            "monitor_run_id": kwargs["monitor_run_id"],
            "tick_operation_id": kwargs["tick_operation_id"],
            "action_kind": kwargs["action_kind"],
            "action_plan_fingerprint": kwargs["action_plan_fingerprint"],
            "status": kwargs["status"],
            "block_reason": kwargs.get("block_reason"),
            "task_ids": [
                task.id for task in await runtime.store.list_tasks(operation.id)
            ],
            "produced_evidence_refs": [],
            "budget_usage": {},
        }

    runtime._persist_monitor_action_result = skip_terminal_result
    await _create_monitor(
        runtime,
        "resumable_action_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan={
            "kind": "investigation",
            "goal": "Explain the monitor trigger",
            "steps": [
                {
                    "id": "final_synthesis",
                    "kind": "synthesis",
                    "purpose": "Summarize the monitor trigger",
                    "expected_evidence": ["analysis.synthesis"],
                }
            ],
        },
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    child_id = run.summary["triggered_operation_id"]
    before = await runtime.inspect_operation(child_id)
    assert before.operation.status is OperationStatus.SUCCEEDED
    assert not any(item.kind == "monitor.action_result" for item in before.evidence)

    runtime._persist_monitor_action_result = original_persist_result
    resumed = await runtime.resume_operation(child_id)

    assert any(item.kind == "monitor.action_result" for item in resumed.evidence)
    assert all(task.id in resumed.completed_task_ids for task in before.tasks)
    inspection = await runtime.inspect_monitor("resumable_action_monitor")
    finalized_run = next(item for item in inspection.runs if item.id == run.id)
    assert finalized_run.summary["action_evidence_id"]
    assert finalized_run.summary["action_status"] == "succeeded"


async def test_monitor_action_blocks_missing_capability_with_audit_evidence():
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    runtime = DbRuntime(
        runtime_id="db-monitor-action-missing-capability",
        plugins=(plugin,),
    )
    await _create_monitor(
        runtime,
        "missing_capability_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan={
            "kind": "investigation",
            "goal": "Search the catalog",
            "steps": [
                {
                    "id": "catalog_step",
                    "kind": "catalog_search",
                    "purpose": "Find order assets",
                    "capability_id": "catalog.schema.search",
                    "capability_owner": "catalog",
                    "input": {"query": "orders"},
                    "expected_evidence": ["schema.search_result"],
                }
            ],
        },
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    action_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.action_result"
    )
    validation = next(
        item for item in snapshot.evidence if item.kind == "analysis.plan.validation"
    )

    assert run.summary["action_status"] == "blocked"
    assert snapshot.operation.status is OperationStatus.BLOCKED
    assert action_result.accepted is False
    assert validation.accepted is False
    assert "missing_capability:catalog.schema.search" in validation.payload["errors"]
    assert not any(
        task.capability_id == "catalog.schema.search" for task in snapshot.tasks
    )


async def test_scheduled_report_analysis_step_missing_capability_is_not_ignored():
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12, "revenue": 42}])
    runtime = DbRuntime(
        runtime_id="db-monitor-report-quality-missing",
        plugins=(plugin,),
    )
    await _create_monitor(
        runtime,
        "quality_report_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"type": "scheduled"},
        action_plan={
            "kind": "scheduled_report",
            "title": "Daily quality report",
            "steps": [
                {
                    "id": "revenue",
                    "kind": "metric_sql",
                    "metric": "revenue",
                    "sql": "select sum(total) as revenue from orders",
                    "source_scope": ["orders"],
                    "capability_owner": "monitor_read_probe",
                },
                {
                    "id": "quality",
                    "kind": "quality_profile",
                    "purpose": "Profile report inputs",
                    "capability_id": "quality.profile",
                    "capability_owner": "data_quality",
                    "input": {"asset": "orders"},
                    "expected_evidence": ["quality.profile"],
                },
            ],
        },
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    validation = next(
        item for item in snapshot.evidence if item.kind == "analysis.plan.validation"
    )

    assert run.summary["action_status"] == "blocked"
    assert snapshot.operation.status is OperationStatus.BLOCKED
    assert "missing_capability:quality.profile" in validation.payload["errors"]
    assert not any(item.kind == "monitor.report" for item in snapshot.evidence)


async def test_report_action_unsafe_sql_fails_before_read_execution():
    plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    runtime = DbRuntime(
        runtime_id="db-monitor-action-unsafe-sql",
        plugins=(plugin,),
    )
    await _create_monitor(
        runtime,
        "unsafe_report_monitor",
        schedule={"interval_seconds": 0},
        observation_plan=_metric_observation(),
        trigger={"type": "scheduled"},
        action_plan={
            "kind": "scheduled_report",
            "title": "Unsafe report",
            "steps": [
                {
                    "id": "unsafe",
                    "kind": "metric_sql",
                    "metric": "unsafe",
                    "sql": "delete from orders",
                    "source_scope": ["orders"],
                    "capability_owner": "monitor_read_probe",
                }
            ],
        },
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    action_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.action_result"
    )
    report_read_tasks = [
        task
        for task in snapshot.tasks
        if task.metadata.get("monitor_action_role") == "report_read"
        or task.metadata.get("reason") == "monitor_report_read"
    ]

    assert run.summary["action_status"] == "failed"
    assert snapshot.operation.status is OperationStatus.FAILED
    assert action_result.accepted is False
    assert action_result.payload["block_reason"] == "unsafe_report_sql"
    assert all(task.status is not TaskStatus.SUCCEEDED for task in report_read_tasks)


async def test_write_proposal_validates_sql_and_does_not_execute_write():
    read_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    write_plugin = MonitorWriteProbePlugin()
    runtime = DbRuntime(
        runtime_id="db-monitor-write-proposal",
        plugins=(read_plugin, write_plugin),
    )
    await _create_monitor(
        runtime,
        "write_proposal_monitor",
        schedule={"interval_seconds": 0},
        source_scope=("orders",),
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan=_write_proposal_action(),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    proposal = [
        item for item in snapshot.evidence if item.kind == "monitor.write_proposal"
    ][-1]
    action_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.action_result"
    )

    assert write_plugin.validate_executor.calls == 1
    assert write_plugin.write_executor.calls == 0
    assert proposal.payload["status"] == "approval_required"
    assert proposal.payload["validation_evidence_id"]
    assert {item["kind"] for item in proposal.payload["source_evidence_refs"]} == {
        "monitor.observation",
        "monitor.trigger_decision",
    }
    assert action_result.payload["status"] == "approval_required"
    assert any(task.capability_id == "db.sql.execute_write" for task in snapshot.tasks)
    approvals = await runtime.store.list_approval_requests(
        run.summary["triggered_operation_id"]
    )
    assert len(approvals) == 1


async def test_low_risk_write_proposal_still_requires_approval():
    read_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    write_plugin = MonitorWriteProbePlugin(write_risk=RiskLevel.LOW)
    runtime = DbRuntime(
        runtime_id="db-monitor-write-low-risk-approval",
        plugins=(read_plugin, write_plugin),
    )
    await _create_monitor(
        runtime,
        "low_risk_write_proposal_monitor",
        schedule={"interval_seconds": 0},
        source_scope=("orders",),
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan=_write_proposal_action(),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    child_id = run.summary["triggered_operation_id"]
    snapshot = await runtime.inspect_operation(child_id)
    proposal = [
        item for item in snapshot.evidence if item.kind == "monitor.write_proposal"
    ][-1]
    action_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.action_result"
    )
    approvals = await runtime.store.list_approval_requests(child_id)

    assert write_plugin.write_executor.calls == 0
    assert proposal.payload["status"] == "approval_required"
    assert action_result.payload["status"] == "approval_required"
    assert len(approvals) == 1
    assert approvals[0].requested_by_policy_id == "approval_required_for_writes"


async def test_approved_write_execution_resumes_through_execute_task():
    read_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    write_plugin = MonitorWriteProbePlugin()
    runtime = DbRuntime(
        runtime_id="db-monitor-write-approved",
        plugins=(read_plugin, write_plugin),
    )
    await _create_monitor(
        runtime,
        "approved_write_monitor",
        schedule={"interval_seconds": 0},
        source_scope=("orders",),
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan=_write_proposal_action(),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    child_id = run.summary["triggered_operation_id"]
    approval = (await runtime.store.list_approval_requests(child_id))[0]
    await runtime.approve_monitor_approval(approval.approval_id)
    calls = []
    original_execute_task = runtime.execute_task

    async def spy_execute_task(task, operation, context=None):
        calls.append((task.capability_id, task.id, operation.id))
        return await original_execute_task(task, operation, context=context)

    runtime.execute_task = spy_execute_task
    resumed = await runtime.resume_operation(child_id)

    write_calls = [item for item in calls if item[0] == "db.sql.execute_write"]
    write_result = next(
        item for item in resumed.evidence if item.kind == "write.execution"
    )
    monitor_write = next(
        item for item in resumed.evidence if item.kind == "monitor.write_execution"
    )
    action_result = [
        item for item in resumed.evidence if item.kind == "monitor.action_result"
    ][-1]

    assert len(write_calls) == 1
    assert write_plugin.write_executor.calls == 1
    assert write_result.id in {
        item["id"] for item in monitor_write.payload["write_evidence_refs"]
    }
    assert action_result.payload["status"] == "succeeded"
    inspection = await runtime.inspect_monitor("approved_write_monitor")
    finalized_run = next(item for item in inspection.runs if item.id == run.id)
    assert finalized_run.summary["action_status"] == "succeeded"


async def test_stale_write_proposal_blocks_without_write_execution():
    read_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    write_plugin = MonitorWriteProbePlugin()
    runtime = DbRuntime(
        runtime_id="db-monitor-write-stale",
        plugins=(read_plugin, write_plugin),
    )
    await _create_monitor(
        runtime,
        "stale_write_monitor",
        schedule={"interval_seconds": 0},
        source_scope=("orders",),
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan=_write_proposal_action(),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    child_id = run.summary["triggered_operation_id"]
    approval = (await runtime.store.list_approval_requests(child_id))[0]
    for proposal in [
        item
        for item in await runtime.store.list_evidence(child_id)
        if item.kind == "monitor.write_proposal"
    ]:
        await runtime.store.save_evidence(
            replace(
                proposal,
                payload={
                    **proposal.payload,
                    "proposal_fingerprint": "tampered-proposal",
                },
            )
        )
    await runtime.approve_monitor_approval(approval.approval_id)

    resumed = await runtime.resume_operation(child_id)
    action_result = [
        item for item in resumed.evidence if item.kind == "monitor.action_result"
    ][-1]

    assert write_plugin.write_executor.calls == 0
    assert action_result.payload["status"] == "blocked"
    assert action_result.payload["block_reason"] in {
        "monitor_write_proposal_stale",
        "write_execution_not_completed",
    }


async def test_destructive_write_proposal_is_denied_before_write_execution():
    read_plugin = MonitorReadProbePlugin(rows=[{"pending_count": 12}])
    write_plugin = MonitorWriteProbePlugin()
    runtime = DbRuntime(
        runtime_id="db-monitor-write-destructive",
        plugins=(read_plugin, write_plugin),
    )
    await _create_monitor(
        runtime,
        "destructive_write_monitor",
        schedule={"interval_seconds": 0},
        source_scope=("orders",),
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan=_write_proposal_action("delete from orders"),
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    action_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.action_result"
    )

    assert write_plugin.write_executor.calls == 0
    assert action_result.payload["status"] == "blocked"
    assert action_result.payload["block_reason"] == "deny_destructive_operations"
    assert any(
        decision.policy_id == "deny_destructive_operations"
        for decision in snapshot.policy_decisions
    )
    assert any(
        "runtime:deny_destructive_operations@2"
        in record.evaluation_trace["decision_policy_identities"]
        for record in snapshot.governance_audit_records
    )


@pytest.mark.parametrize(
    ("plugins", "action_plan", "source_scope", "reason"),
    [
        (
            (MonitorReadProbePlugin(rows=[{"pending_count": 12}]),),
            _write_proposal_action(),
            ("orders",),
            "missing_write_capability",
        ),
        (
            (
                MonitorReadProbePlugin(rows=[{"pending_count": 12}]),
                MonitorWriteProbePlugin(plugin_id="monitor_write_probe"),
                MonitorWriteProbePlugin(plugin_id="monitor_write_probe_b"),
            ),
            {
                "kind": "write_proposal",
                "sql": "update orders set status = 'ready' where id = 1",
            },
            ("orders",),
            "ambiguous_write_capability",
        ),
        (
            (
                MonitorReadProbePlugin(rows=[{"pending_count": 12}]),
                MonitorWriteProbePlugin(),
            ),
            {
                "kind": "write_proposal",
                "sql": "update orders set status = 'ready' where id = 1",
                "capability_owner": "monitor_write_probe",
            },
            ("customers",),
            "write_source_scope_blocked",
        ),
    ],
)
async def test_write_proposal_blocks_missing_ambiguous_or_scope_drift(
    plugins,
    action_plan,
    source_scope,
    reason,
):
    runtime = DbRuntime(
        runtime_id=f"db-monitor-write-block-{reason}",
        plugins=plugins,
    )
    await _create_monitor(
        runtime,
        f"blocked_write_{reason}",
        schedule={"interval_seconds": 0},
        source_scope=source_scope,
        observation_plan=_metric_observation(),
        trigger={"path": "pending_count", "gt": 10},
        action_plan=action_plan,
    )

    run = (await runtime.tick_monitors(now=NOW))[0]
    snapshot = await runtime.inspect_operation(run.summary["triggered_operation_id"])
    action_result = next(
        item for item in snapshot.evidence if item.kind == "monitor.action_result"
    )

    assert action_result.payload["status"] == "blocked"
    assert action_result.payload["block_reason"] == reason
    has_write_task = any(
        task.capability_id == "db.sql.execute_write" for task in snapshot.tasks
    )
    if reason == "write_source_scope_blocked":
        assert has_write_task
    else:
        assert not has_write_task


def test_phase7_monitor_runtime_boundaries_do_not_add_direct_paths():
    source = "\n".join(
        inspect.getsource(item)
        for item in (
            DbRuntime.execute_monitor_delivery,
            DbRuntime.execute_monitor_action,
            DbRuntime._execute_monitor_write_proposal_action,
            DbRuntime._finalize_resumed_monitor_delivery,
            DbRuntime._finalize_resumed_monitor_write_action,
        )
    )

    assert ".kernel.execute_task(" not in source
    assert "RuntimeKernel.execute_task" not in source
    assert "approval_channel." not in source
    assert ".execute_write(" not in source
    assert "slack_sdk" not in source
    assert "httpx" not in source
    assert "requests." not in source
    assert "regenerate" not in source.lower()
