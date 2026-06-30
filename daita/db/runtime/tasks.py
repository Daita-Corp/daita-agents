"""Task execution helpers for ``DbRuntime``."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable, Mapping
from uuid import uuid4

from daita.runtime import (
    ApprovalStatus,
    Capability,
    Evidence,
    GovernanceResult,
    Operation,
    RuntimeKernelExecutorFailed,
    RuntimeKernelGovernanceBlocked,
    RuntimeKernelLeaseLost,
    RuntimeKernelTaskAlreadyTerminal,
    RuntimeKernelTaskNotRunnable,
    Task,
    TaskDependency,
    TaskStatus,
)

from ..models import DbIntent, DbIntentKind, DbOperationContract
from ..sql_evidence import blocked_scope_resources, sql_validation_facts_from_evidence
from .types import (
    _DEFAULT_TASK_LEASE_SECONDS,
    DbRuntimeGovernanceBlocked,
    DbRuntimeTaskNotRunnable,
)


@dataclass(frozen=True)
class DbTaskSpec:
    """Runtime-owned description of DB work before a persisted task exists."""

    capability_id: str
    owner: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    reason: str = "planner"
    sequence: int = 1
    dependencies: tuple[TaskDependency, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    deterministic_key: str | None = None
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        if not self.capability_id:
            raise ValueError("capability_id is required")
        if self.sequence < 1:
            raise ValueError("sequence must be at least 1")
        object.__setattr__(self, "input", _json_dict(self.input))
        object.__setattr__(
            self,
            "dependencies",
            tuple(
                (
                    dependency
                    if isinstance(dependency, TaskDependency)
                    else TaskDependency.from_dict(dependency)
                )
                for dependency in self.dependencies
            ),
        )
        object.__setattr__(self, "metadata", _json_dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "owner": self.owner,
            "input": self.input,
            "reason": self.reason,
            "sequence": self.sequence,
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
            "metadata": self.metadata,
            "deterministic_key": self.deterministic_key,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DbTaskSpec":
        values = dict(data)
        values["dependencies"] = tuple(
            TaskDependency.from_dict(item) for item in values.get("dependencies", ())
        )
        return cls(**values)


@dataclass(frozen=True)
class DbTaskPlan:
    """Materialization result for one or more DB task specs."""

    tasks: tuple[Task, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "diagnostics", _json_dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": [task.to_dict() for task in self.tasks],
            "diagnostics": self.diagnostics,
        }


class DbRuntimeTasksMixin:
    async def execute_task(
        self,
        task: Task,
        operation: Operation,
        context: dict[str, Any] | None = None,
    ) -> tuple[Evidence, ...]:
        """Execute one runtime task through the shared runtime kernel."""
        capability = self._capability_for_task(task)
        if capability.executor != task.executor_id:
            raise ValueError(
                f"task executor {task.executor_id!r} does not match capability "
                f"{task.capability_id!r} executor {capability.executor!r}"
            )
        stored_task = await self.store.load_task(task.id)
        if stored_task is None:
            task = replace(
                task,
                dependencies=task.dependencies
                or _task_dependencies_for_capability(operation, capability),
            )
            task = await self._plan_kernel_task(task)
        elif (
            stored_task.status is TaskStatus.PENDING and stored_task.input != task.input
        ):
            task = replace(
                stored_task,
                input=task.input,
                dependencies=task.dependencies or stored_task.dependencies,
                metadata={
                    **stored_task.metadata,
                    **{
                        key: value
                        for key, value in task.metadata.items()
                        if key in {"owner", "reason"}
                    },
                },
            )
            await self.store.save_task(task)
        else:
            task = stored_task
        default_dependencies = _task_dependencies_for_capability(operation, capability)
        if not task.dependencies and default_dependencies:
            task = replace(task, dependencies=default_dependencies)
            await self.store.save_task(task)
        try:
            result = await self.kernel.execute_task(
                task.id,
                context={
                    "capability_owner": capability.owner,
                    **(context or {}),
                },
                lease_owner=self.runtime_id,
                lease_seconds=_DEFAULT_TASK_LEASE_SECONDS,
            )
        except RuntimeKernelGovernanceBlocked as exc:
            result = exc.result
            raise DbRuntimeGovernanceBlocked(
                operation=result.operation if result is not None else operation,
                task=result.task if result is not None else task,
                governance=(
                    result.governance
                    if result is not None and result.governance is not None
                    else GovernanceResult(False, True, False)
                ),
            ) from exc
        except RuntimeKernelTaskAlreadyTerminal as exc:
            result = exc.result
            blocked_task = result.task if result is not None else task
            raise DbRuntimeTaskNotRunnable(
                blocked_task,
                f"Task {blocked_task.id} is already {blocked_task.status.value}; "
                "completed tasks are not replayed without explicit invalidation.",
            ) from exc
        except RuntimeKernelLeaseLost as exc:
            result = exc.result
            blocked_task = result.task if result is not None else task
            raise DbRuntimeTaskNotRunnable(
                blocked_task,
                f"Task {blocked_task.id} lease was lost before commit.",
            ) from exc
        except RuntimeKernelTaskNotRunnable as exc:
            result = exc.result
            blocked_task = result.task if result is not None else task
            readiness = (
                result.events[-1].payload.get("readiness", {})
                if result is not None and result.events
                else {}
            )
            raise DbRuntimeTaskNotRunnable(
                blocked_task,
                str(exc),
                readiness=readiness,
            ) from exc
        except RuntimeKernelExecutorFailed as exc:
            raise (exc.__cause__ or exc) from exc
        return result.evidence

    async def execute_capability(
        self,
        capability_id: str,
        *,
        owner: str | None = None,
        operation_type: str,
        input: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> tuple[Evidence, ...]:
        """Create and execute a single task for one registered capability."""
        if not self._is_setup:
            await self.setup()
        capability = self.registry.get_capability(capability_id, owner=owner)
        output_evidence = capability.output_evidence
        validation_capability = self._validation_capability_for_sql_execute(capability)
        if (
            capability.id
            in {
                "db.sql.execute_read",
                "db.sql.execute_write",
            }
            and validation_capability is not None
        ):
            output_evidence = frozenset(
                (
                    *sorted(validation_capability.output_evidence),
                    *sorted(output_evidence),
                )
            )
        try:
            operation = await self.kernel.create_operation(
                operation_id=operation_id,
                operation_type=operation_type,
                request={
                    "prompt": _prompt_from_direct_input(input or {}),
                    "input": input or {},
                    "capability_id": capability.id,
                    "capability_owner": capability.owner,
                },
                required_evidence=output_evidence,
                metadata={
                    "direct_capability_id": capability.id,
                    "direct_capability_owner": capability.owner,
                    "access": capability.access.value,
                },
                evaluate_governance=False,
            )
        except RuntimeKernelGovernanceBlocked as exc:
            raise DbRuntimeGovernanceBlocked(
                operation=exc.operation
                or await self.store.load_operation(operation_id or ""),
                task=None,
                governance=exc.governance or GovernanceResult(False, True, False),
            ) from exc
        task_plans = self._direct_capability_tasks(
            operation,
            capability,
            input or {},
            validation_capability=validation_capability,
        )
        tasks = []
        for task in task_plans:
            tasks.append(await self._plan_kernel_task(task))
        primary_task = tasks[-1]
        if (
            capability.id == "db.sql.execute_write"
            and validation_capability is None
            and not (input or {}).get("validated_evidence_id")
        ):
            blocked_task = await self.kernel.block_task(
                primary_task.id,
                message=(
                    "Direct write execution requires db.sql.validate "
                    "or a validated_evidence_id."
                ),
            )
            await self.kernel.block_operation(
                operation.id,
                message=(
                    "Direct write execution requires db.sql.validate "
                    "or a validated_evidence_id."
                ),
            )
            raise DbRuntimeTaskNotRunnable(
                blocked_task,
                "Direct write execution requires db.sql.validate or validated_evidence_id.",
            )
        try:
            await self.kernel.evaluate_operation_governance(
                operation.id,
                capability=capability,
            )
        except RuntimeKernelGovernanceBlocked as exc:
            await self.kernel.block_task(
                primary_task.id,
                message=f"Task {primary_task.id} blocked by operation governance.",
                payload={
                    "governance": (
                        exc.governance.to_dict() if exc.governance is not None else {}
                    )
                },
            )
            raise DbRuntimeGovernanceBlocked(
                operation=exc.operation or operation,
                task=replace(primary_task, status=TaskStatus.BLOCKED),
                governance=exc.governance or GovernanceResult(False, True, False),
            ) from exc
        try:
            collected: list[Evidence] = []
            for task in tasks:
                collected.extend(await self.execute_task(task, operation))
            evidence = tuple(collected)
        except DbRuntimeGovernanceBlocked:
            raise
        except Exception as exc:
            await self.kernel.fail_operation_if_active(operation.id, exc)
            raise
        await self.kernel.complete_operation(operation.id)
        return evidence

    async def _plan_kernel_task(self, task: Task) -> Task:
        """Persist a DB-planned task through the shared kernel planner."""
        return await self.kernel.plan_task(
            task_id=task.id,
            operation_id=task.operation_id,
            capability_id=task.capability_id,
            owner=str(task.metadata["owner"]) if task.metadata.get("owner") else None,
            input=task.input,
            metadata=task.metadata,
            dependencies=task.dependencies,
        )

    async def plan_task_specs(
        self,
        operation: Operation,
        specs: Iterable[DbTaskSpec],
        *,
        contract: DbOperationContract | Mapping[str, Any] | None = None,
    ) -> DbTaskPlan:
        """Materialize DB task specs through the shared runtime kernel."""
        planned: list[Task] = []
        diagnostics: dict[str, Any] = {
            "spec_count": 0,
            "reused_task_count": 0,
            "planned_task_count": 0,
        }
        prior_by_capability_owner: dict[tuple[str, str], Task] = {}
        for spec in specs:
            diagnostics["spec_count"] += 1
            capability = self.registry.get_capability(
                spec.capability_id,
                owner=spec.owner,
            )
            validation_task = prior_by_capability_owner.get(
                ("db.sql.validate", capability.owner)
            )
            task = self._task_for_spec(
                operation,
                capability,
                spec,
                contract=contract,
                validation_task=validation_task,
            )
            existing = await self.store.load_task(task.id)
            if existing is not None:
                planned.append(existing)
                diagnostics["reused_task_count"] += 1
            else:
                planned.append(await self._plan_kernel_task(task))
                diagnostics["planned_task_count"] += 1
            prior_by_capability_owner[(capability.id, capability.owner)] = planned[-1]
        return DbTaskPlan(tasks=tuple(planned), diagnostics=diagnostics)

    async def plan_validated_read_spec(
        self,
        operation: Operation,
        *,
        sql: str,
        params: tuple[Any, ...] | list[Any] = (),
        param_specs: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
        owner: str | None = None,
        reason: str = "validated_read",
        sequence: int = 1,
        focus: Any = None,
        metadata: dict[str, Any] | None = None,
        contract: DbOperationContract | Mapping[str, Any] | None = None,
    ) -> DbTaskPlan:
        """Persist SQL validation followed by a read task under one operation."""
        read_capability = self.registry.get_capability(
            "db.sql.execute_read",
            owner=owner,
        )
        validation_capability = self._validation_capability_for_sql_execute(
            read_capability
        )
        if validation_capability is None:
            raise KeyError("db.sql.validate")
        validation_spec = DbTaskSpec(
            capability_id=validation_capability.id,
            owner=validation_capability.owner,
            input={"sql": sql, "operation": "query"},
            reason=f"{reason}_validation",
            sequence=sequence,
            metadata=metadata or {},
        )
        execute_input: dict[str, Any] = {
            "sql_ref": "sql.validation",
            "params": list(params),
        }
        if param_specs:
            execute_input["param_specs"] = list(param_specs)
        if focus is not None:
            execute_input["focus"] = focus
        read_spec = DbTaskSpec(
            capability_id=read_capability.id,
            owner=read_capability.owner,
            input=execute_input,
            reason=reason,
            sequence=sequence + 1,
            metadata=metadata or {},
        )
        return await self.plan_task_specs(
            operation,
            (validation_spec, read_spec),
            contract=contract,
        )

    async def plan_validated_write_spec(
        self,
        operation: Operation,
        *,
        sql: str,
        params: tuple[Any, ...] | list[Any] = (),
        param_specs: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
        owner: str | None = None,
        reason: str = "validated_write",
        sequence: int = 1,
        metadata: dict[str, Any] | None = None,
        contract: DbOperationContract | Mapping[str, Any] | None = None,
    ) -> DbTaskPlan:
        """Persist SQL validation followed by an approval-gated write task."""
        write_capability = self.registry.get_capability(
            "db.sql.execute_write",
            owner=owner,
        )
        validation_capability = self._validation_capability_for_sql_execute(
            write_capability
        )
        if validation_capability is None:
            raise KeyError("db.sql.validate")
        validation_spec = DbTaskSpec(
            capability_id=validation_capability.id,
            owner=validation_capability.owner,
            input={"sql": sql, "operation": operation.operation_type},
            reason=f"{reason}_validation",
            sequence=sequence,
            metadata=metadata or {},
        )
        execute_input: dict[str, Any] = {
            "sql_ref": "sql.validation",
            "params": list(params),
        }
        if param_specs:
            execute_input["param_specs"] = list(param_specs)
        write_spec = DbTaskSpec(
            capability_id=write_capability.id,
            owner=write_capability.owner,
            input=execute_input,
            reason=reason,
            sequence=sequence + 1,
            metadata=metadata or {},
        )
        return await self.plan_task_specs(
            operation,
            (validation_spec, write_spec),
            contract=contract,
        )

    async def _persist_contract_tasks(
        self,
        operation: Operation,
        contract: DbOperationContract,
    ) -> None:
        existing = {
            (task.capability_id, task.executor_id, task.metadata.get("owner"))
            for task in await self.store.list_tasks(operation.id)
        }
        planned_by_capability: dict[tuple[str, str], Task] = {
            (task.capability_id, str(task.metadata.get("owner") or "")): task
            for task in await self.store.list_tasks(operation.id)
        }
        for sequence, selected in enumerate(
            contract.metadata.get("selected_capabilities", ()), start=1
        ):
            capability = self.registry.get_capability(
                str(selected["id"]),
                owner=str(selected["owner"]),
            )
            key = (capability.id, capability.executor, capability.owner)
            if key in existing:
                continue
            task = Task(
                id=f"db-task-{uuid4()}",
                operation_id=operation.id,
                capability_id=capability.id,
                executor_id=capability.executor,
                input=_planned_task_input(operation, capability),
                required_evidence=capability.output_evidence,
                metadata={
                    "owner": capability.owner,
                    "reason": str(selected.get("reason") or "contract"),
                    "sequence": sequence,
                },
            )
            validation_task = planned_by_capability.get(
                ("db.sql.validate", capability.owner)
            )
            task = replace(
                task,
                input={
                    **task.input,
                    "input_hash": _stable_hash(task.input),
                },
                dependencies=_task_dependencies_for_capability(
                    operation,
                    capability,
                    validation_task=validation_task,
                ),
                metadata={
                    **task.metadata,
                    "input_hash": _stable_hash(task.input),
                    "idempotency_key": _stable_hash(
                        {
                            "operation_id": operation.id,
                            "task_id": task.id,
                            "capability_id": task.capability_id,
                            "input": task.input,
                        }
                    ),
                    "idempotent": capability.idempotent,
                    "replay_safe": capability.replay_safe,
                    "side_effecting": capability.side_effecting,
                },
            )
            await self._plan_kernel_task(task)
            existing.add(key)
            planned_by_capability[(capability.id, capability.owner)] = task

    async def _planned_task_for_capability(
        self,
        operation_id: str,
        capability: Capability,
        *,
        metadata_match: dict[str, Any] | None = None,
    ) -> Task | None:
        for task in await self.store.list_tasks(operation_id):
            if task.status is not TaskStatus.PENDING:
                continue
            if task.capability_id != capability.id:
                continue
            if task.executor_id != capability.executor:
                continue
            if task.metadata.get("owner") != capability.owner:
                continue
            if metadata_match and any(
                task.metadata.get(key) != value for key, value in metadata_match.items()
            ):
                continue
            return task
        return None

    async def _persist_verification_result_evidence(
        self,
        operation: Operation,
        verification: Any,
        evidence: tuple[Evidence, ...],
    ) -> Evidence:
        existing = await self._latest_accepted_evidence(
            operation.id,
            "verification.result",
            payload={"passed": True},
        )
        if existing is not None:
            return existing
        accepted = tuple(item for item in evidence if item.accepted and item.id)
        evidence_details = [
            {
                "id": item.id,
                "kind": item.kind,
                "owner": item.owner,
                "task_id": item.task_id,
                "payload_fingerprint": item.metadata.get("payload_fingerprint")
                or _payload_fingerprint(item.payload),
            }
            for item in accepted
        ]
        payload = {
            "passed": bool(verification.passed),
            "evidence_refs": [item["id"] for item in evidence_details],
            "evidence_details": evidence_details,
            "warnings": list(verification.warnings),
            "missing_evidence": list(verification.missing_evidence),
            "diagnostics": verification.diagnostics,
            "input_fingerprint": _stable_hash(
                {
                    "operation_id": operation.id,
                    "evidence": evidence_details,
                    "warnings": list(verification.warnings),
                    "missing_evidence": list(verification.missing_evidence),
                }
            ),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        verification_evidence = Evidence(
            id=f"evidence-{uuid4()}",
            kind="verification.result",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=True,
            payload=payload,
            metadata={
                "payload_fingerprint": _payload_fingerprint(payload),
                "input_fingerprint": payload["input_fingerprint"],
            },
        )
        await self.store.save_evidence(verification_evidence)
        return verification_evidence

    async def _execute_answer_synthesis(
        self,
        *,
        operation: Operation,
        intent: DbIntent,
        outcome_evidence: tuple[Evidence, ...],
    ) -> tuple[Evidence, Task]:
        existing = await self._latest_accepted_evidence(
            operation.id,
            "answer.synthesis",
        )
        if existing is not None:
            task = next(
                (
                    item
                    for item in await self.store.list_tasks(operation.id)
                    if item.id == existing.task_id
                ),
                Task(
                    id=str(existing.task_id or f"db-task-{uuid4()}"),
                    operation_id=operation.id,
                    capability_id="db.answer.synthesize",
                    executor_id="db.answer.synthesize.runtime",
                    required_evidence=frozenset({"answer.synthesis"}),
                    metadata={"owner": "db_runtime", "reason": "answer_synthesis"},
                ),
            )
            return existing, task

        capability = self.registry.get_capability(
            "db.answer.synthesize", owner="db_runtime"
        )
        dependencies = _synthesis_dependencies(operation, intent, outcome_evidence)
        task_input = {
            "evidence_refs": [
                {
                    "id": dependency.evidence_id,
                    "kind": dependency.evidence_kind,
                    "payload_fingerprint": dependency.payload_fingerprint,
                }
                for dependency in dependencies
            ],
            "row_budget": _synthesis_context_option(
                self.config.metadata, "synthesis_row_budget", 25
            ),
            "char_budget": _synthesis_context_option(
                self.config.metadata, "synthesis_context_char_budget", 16000
            ),
        }
        input_hash = _stable_hash(task_input)
        task = Task(
            id=f"db-task-{uuid4()}",
            operation_id=operation.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input={**task_input, "input_hash": input_hash},
            required_evidence=capability.output_evidence,
            dependencies=dependencies,
            metadata={
                "owner": capability.owner,
                "reason": "answer_synthesis",
                "sequence": 10_000,
                "input_hash": input_hash,
                "idempotency_key": _stable_hash(
                    {
                        "operation_id": operation.id,
                        "capability_id": capability.id,
                        "evidence_refs": task_input["evidence_refs"],
                    }
                ),
                "idempotent": capability.idempotent,
                "replay_safe": capability.replay_safe,
                "side_effecting": capability.side_effecting,
            },
        )
        evidence = await self.execute_task(
            task,
            operation,
            context={"capability_owner": capability.owner},
        )
        synthesis = next(
            (
                item
                for item in evidence
                if item.kind == "answer.synthesis" and item.accepted
            ),
            None,
        )
        if synthesis is None:
            raise RuntimeError("answer.synthesis evidence was not produced")
        stored_task = await self.store.load_task(task.id)
        return synthesis, stored_task or task

    def _validation_capability_for_sql_execute(
        self,
        capability: Capability,
    ) -> Capability | None:
        if capability.id not in {"db.sql.execute_read", "db.sql.execute_write"}:
            return None
        try:
            return self.registry.get_capability(
                "db.sql.validate", owner=capability.owner
            )
        except KeyError:
            return None

    def _direct_capability_tasks(
        self,
        operation: Operation,
        capability: Capability,
        input: dict[str, Any],
        *,
        validation_capability: Capability | None,
    ) -> tuple[Task, ...]:
        if (
            capability.id not in {"db.sql.execute_read", "db.sql.execute_write"}
            or validation_capability is None
        ):
            task = self._task_for_capability(
                operation,
                capability,
                input=input,
                reason="direct",
                sequence=1,
            )
            return (task,)
        validation_task = self._task_for_capability(
            operation,
            validation_capability,
            input={
                "sql": str(input.get("sql") or ""),
                "operation": (
                    "query"
                    if capability.id == "db.sql.execute_read"
                    else operation.operation_type
                ),
            },
            reason="direct_validation",
            sequence=1,
        )
        execute_input = (
            {
                "sql_ref": "sql.validation",
                "params": list(input.get("params") or []),
                **(
                    {"param_specs": list(input.get("param_specs") or [])}
                    if input.get("param_specs")
                    else {}
                ),
            }
            if capability.id == "db.sql.execute_read"
            else {
                "sql_ref": "sql.validation",
                "params": list(input.get("params") or []),
                **(
                    {"param_specs": list(input.get("param_specs") or [])}
                    if input.get("param_specs")
                    else {}
                ),
            }
        )
        execute_task = self._task_for_capability(
            operation,
            capability,
            input=execute_input,
            reason="direct",
            sequence=2,
            validation_task=validation_task,
        )
        return (validation_task, execute_task)

    def plan_validated_read_tasks(
        self,
        operation: Operation,
        *,
        sql: str,
        params: tuple[Any, ...] | list[Any] = (),
        param_specs: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
        owner: str | None = None,
        reason: str = "validated_read",
        sequence: int = 1,
        focus: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Task, Task]:
        """Plan a DB SQL validation/read task pair under an existing operation."""

        read_capability = self.registry.get_capability(
            "db.sql.execute_read",
            owner=owner,
        )
        validation_capability = self._validation_capability_for_sql_execute(
            read_capability
        )
        if validation_capability is None:
            raise KeyError("db.sql.validate")
        validation_task = self._task_for_capability(
            operation,
            validation_capability,
            input={"sql": sql, "operation": "query"},
            reason=f"{reason}_validation",
            sequence=sequence,
            metadata=metadata,
        )
        execute_input: dict[str, Any] = {
            "sql_ref": "sql.validation",
            "params": list(params),
        }
        if param_specs:
            execute_input["param_specs"] = list(param_specs)
        if focus is not None:
            execute_input["focus"] = focus
        read_task = self._task_for_capability(
            operation,
            read_capability,
            input=execute_input,
            reason=reason,
            sequence=sequence + 1,
            validation_task=validation_task,
            metadata=metadata,
        )
        return validation_task, read_task

    def _task_for_capability(
        self,
        operation: Operation,
        capability: Capability,
        *,
        input: dict[str, Any],
        reason: str,
        sequence: int,
        validation_task: Task | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        input_hash = _stable_hash(input)
        task = Task(
            id=f"db-task-{uuid4()}",
            operation_id=operation.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input={**input, "input_hash": input_hash},
            required_evidence=capability.output_evidence,
            metadata={
                **dict(metadata or {}),
                "owner": capability.owner,
                "reason": reason,
                "sequence": sequence,
                "input_hash": input_hash,
                "idempotency_key": _stable_hash(
                    {
                        "operation_id": operation.id,
                        "capability_id": capability.id,
                        "executor_id": capability.executor,
                        "input": input,
                    }
                ),
                "idempotent": capability.idempotent,
                "replay_safe": capability.replay_safe,
                "side_effecting": capability.side_effecting,
            },
        )
        return replace(
            task,
            dependencies=_task_dependencies_for_capability(
                operation,
                capability,
                validation_task=validation_task,
            ),
        )

    def _task_for_spec(
        self,
        operation: Operation,
        capability: Capability,
        spec: DbTaskSpec,
        *,
        contract: DbOperationContract | Mapping[str, Any] | None = None,
        validation_task: Task | None = None,
    ) -> Task:
        input_hash = _stable_hash(spec.input)
        idempotency_key = spec.idempotency_key or _stable_hash(
            {
                "operation_id": operation.id,
                "capability_id": capability.id,
                "owner": capability.owner,
                "input_hash": input_hash,
                "sequence": spec.sequence,
                "deterministic_key": spec.deterministic_key,
            }
        )
        task_fingerprint = _stable_hash(
            {
                "operation_id": operation.id,
                "idempotency_key": idempotency_key,
            }
        )
        task_id = f"db-task-{task_fingerprint[:32]}"
        dependencies = _combine_dependencies(
            _task_dependencies_for_capability(
                operation,
                capability,
                validation_task=validation_task,
            ),
            spec.dependencies,
        )
        metadata = {
            **spec.metadata,
            "owner": capability.owner,
            "reason": spec.reason,
            "sequence": spec.sequence,
            "input_hash": input_hash,
            "idempotency_key": idempotency_key,
            "deterministic_key": spec.deterministic_key,
            "idempotent": capability.idempotent,
            "replay_safe": capability.replay_safe,
            "side_effecting": capability.side_effecting,
        }
        contract_snapshot = _contract_snapshot(contract)
        if contract_snapshot is not None:
            metadata["contract"] = contract_snapshot
        return Task(
            id=task_id,
            operation_id=operation.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input={**spec.input, "input_hash": input_hash},
            required_evidence=capability.output_evidence,
            dependencies=dependencies,
            metadata=metadata,
        )

    async def _task_readiness(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        unsatisfied: list[dict[str, Any]] = []
        for dependency in task.dependencies:
            if dependency.kind.value == "evidence":
                if not await self._evidence_dependency_satisfied(dependency, operation):
                    unsatisfied.append(dependency.to_dict())
            elif dependency.kind.value == "approval":
                if not await self._approval_dependency_satisfied(dependency, operation):
                    unsatisfied.append(dependency.to_dict())
        return {
            "ready": not unsatisfied,
            "unsatisfied_dependencies": unsatisfied,
            "dependency_count": len(task.dependencies),
        }

    async def _evidence_dependency_satisfied(
        self,
        dependency: TaskDependency,
        operation: Operation,
    ) -> bool:
        operation_id = dependency.operation_id or operation.id
        for evidence in await self.store.list_evidence(operation_id):
            if evidence.kind != dependency.evidence_kind:
                continue
            if (
                dependency.evidence_id is not None
                and evidence.id != dependency.evidence_id
            ):
                continue
            if (
                dependency.evidence_owner is not None
                and evidence.owner != dependency.evidence_owner
            ):
                continue
            if (
                dependency.producer_task_id is not None
                and evidence.task_id != dependency.producer_task_id
            ):
                continue
            if evidence.accepted is not dependency.evidence_accepted:
                continue
            if (
                dependency.input_hash is not None
                and evidence.metadata.get("task_input_hash") != dependency.input_hash
            ):
                continue
            if _payload_contains(evidence.payload, dependency.evidence_payload):
                if (
                    dependency.payload_fingerprint is not None
                    and dependency.payload_fingerprint
                    != _payload_fingerprint(evidence.payload)
                ):
                    continue
                return True
        return False

    async def _approval_dependency_satisfied(
        self,
        dependency: TaskDependency,
        operation: Operation,
    ) -> bool:
        operation_id = dependency.operation_id or operation.id
        for approval in await self.store.list_approval_requests(operation_id):
            if (
                dependency.approval_id is not None
                and approval.approval_id != dependency.approval_id
            ):
                continue
            if (
                dependency.approval_policy_id is not None
                and approval.requested_by_policy_id != dependency.approval_policy_id
            ):
                continue
            if (
                dependency.approval_name is not None
                and approval.proposed_action.get("approval") != dependency.approval_name
            ):
                continue
            if (
                dependency.approval_version is not None
                and approval.metadata.get("version") != dependency.approval_version
            ):
                continue
            if approval.status is dependency.approval_status:
                return True
        return False

    async def _executable_input_for_task(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        if task.capability_id not in {
            "db.sql.execute_read",
            "db.sql.execute_write",
        }:
            return task.input
        validation_dependency = next(
            (
                dependency
                for dependency in task.dependencies
                if dependency.kind.value == "evidence"
                and dependency.evidence_kind == "sql.validation"
            ),
            None,
        )
        validation = (
            await self._accepted_evidence_for_dependency(
                operation.id,
                validation_dependency,
            )
            if validation_dependency is not None
            else await self._latest_accepted_evidence(
                operation.id,
                "sql.validation",
                payload={"valid": True},
            )
        )
        if validation is None:
            return task.input
        if task.metadata.get("monitor_action_role") == "write_execution":
            proposal_fingerprint = str(task.metadata.get("proposal_fingerprint") or "")
            proposal_evidence_id = str(task.metadata.get("proposal_evidence_id") or "")
            proposal_matches = [
                item
                for item in await self.store.list_evidence(operation.id)
                if item.kind == "monitor.write_proposal"
                and item.id == proposal_evidence_id
            ]
            proposal = proposal_matches[-1] if proposal_matches else None
            expected_validation_id = str(
                task.metadata.get("validation_evidence_id") or ""
            )
            expected_validation_fingerprint = str(
                task.metadata.get("validation_payload_fingerprint") or ""
            )
            actual_validation_fingerprint = validation.metadata.get(
                "payload_fingerprint"
            ) or _payload_fingerprint(validation.payload)
            if (
                proposal is None
                or proposal.payload.get("status")
                not in {"approval_required", "approved"}
                or proposal.payload.get("proposal_fingerprint") != proposal_fingerprint
                or validation.id != expected_validation_id
                or actual_validation_fingerprint != expected_validation_fingerprint
                or proposal.payload.get("validation_payload_fingerprint")
                != expected_validation_fingerprint
            ):
                raise RuntimeError("monitor_write_proposal_stale")
            facts = sql_validation_facts_from_evidence(validation)
            blocked_resources = blocked_scope_resources(
                facts.target_resources,
                tuple(task.metadata.get("source_scope") or ()),
            )
            if facts.valid is not True or blocked_resources:
                raise RuntimeError("monitor_write_validation_stale")
        sql = validation.payload.get("sql")
        if not sql:
            return task.input
        return {
            **task.input,
            "sql": sql,
            "validated_evidence_id": validation.id,
            "validated_task_id": validation.task_id,
        }

    async def _accepted_evidence_for_dependency(
        self,
        operation_id: str,
        dependency: TaskDependency,
    ) -> Evidence | None:
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation_id)
            if evidence.kind == dependency.evidence_kind
            and evidence.accepted is dependency.evidence_accepted
            and (
                dependency.evidence_id is None or evidence.id == dependency.evidence_id
            )
            and (
                dependency.evidence_owner is None
                or evidence.owner == dependency.evidence_owner
            )
            and (
                dependency.producer_task_id is None
                or evidence.task_id == dependency.producer_task_id
            )
            and (
                dependency.input_hash is None
                or evidence.metadata.get("task_input_hash") == dependency.input_hash
            )
            and _payload_contains(evidence.payload, dependency.evidence_payload)
            and (
                dependency.payload_fingerprint is None
                or dependency.payload_fingerprint
                == _payload_fingerprint(evidence.payload)
            )
        ]
        return matches[-1] if matches else None

    async def _authoritative_validation_evidence(
        self,
        operation: Operation,
        task: Task | None,
    ) -> tuple[Evidence, ...]:
        if task is None or task.capability_id not in {
            "db.sql.execute_read",
            "db.sql.execute_write",
        }:
            return ()
        if task.metadata.get("monitor_action_role") == "write_execution":
            expected_validation_id = str(
                task.metadata.get("validation_evidence_id") or ""
            )
            if expected_validation_id:
                matches = [
                    evidence
                    for evidence in await self.store.list_evidence(operation.id)
                    if evidence.kind == "sql.validation"
                    and evidence.id == expected_validation_id
                    and evidence.accepted
                ]
                if matches:
                    return (matches[-1],)
        validation_dependency = next(
            (
                dependency
                for dependency in task.dependencies
                if dependency.kind.value == "evidence"
                and dependency.evidence_kind == "sql.validation"
            ),
            None,
        )
        if validation_dependency is None:
            return ()
        evidence = await self._accepted_evidence_for_dependency(
            operation.id,
            validation_dependency,
        )
        return (evidence,) if evidence is not None else ()

    async def _latest_accepted_evidence(
        self,
        operation_id: str,
        kind: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Evidence | None:
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation_id)
            if evidence.kind == kind
            and evidence.accepted
            and _payload_contains(evidence.payload, payload or {})
        ]
        return matches[-1] if matches else None

    async def _latest_evidence(
        self,
        operation_id: str,
        kind: str,
        *,
        payload: dict[str, Any] | None = None,
        accepted: bool | None = None,
    ) -> Evidence | None:
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation_id)
            if evidence.kind == kind
            and (accepted is None or evidence.accepted is accepted)
            and _payload_contains(evidence.payload, payload or {})
        ]
        return matches[-1] if matches else None

    async def task_readiness(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        """Return DB-owned dependency readiness for kernel task execution."""
        return await self._task_readiness(task, operation)

    async def executable_input_for_task(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        """Hydrate DB task input from authoritative validation evidence."""
        return await self._executable_input_for_task(task, operation)

    def _capability_for_task(self, task: Task) -> Capability:
        owner = task.metadata.get("owner") if task.metadata else None
        if owner:
            return self.registry.get_capability(task.capability_id, owner=str(owner))
        try:
            return self.registry.get_capability(task.capability_id)
        except ValueError:
            for capability in self.registry.capabilities:
                if (
                    capability.id == task.capability_id
                    and capability.executor == task.executor_id
                ):
                    return capability
            raise


def _planned_task_input(operation: Operation, capability: Capability) -> dict[str, Any]:
    prompt = str(operation.request.get("prompt") or "")
    if capability.id in {"db.sql.execute_read", "db.sql.execute_write"}:
        return {"sql_ref": "sql.validation"}
    if capability.id == "db.sql.validate":
        return {"sql": prompt, "operation": operation.operation_type}
    return {"prompt": prompt}


def _synthesis_dependencies(
    operation: Operation,
    intent: DbIntent,
    evidence: tuple[Evidence, ...],
) -> tuple[TaskDependency, ...]:
    accepted = tuple(
        item
        for item in evidence
        if item.accepted and item.operation_id == operation.id and item.id
    )
    dependencies: list[TaskDependency] = []
    if intent.kind in {
        DbIntentKind.DATA_QUERY,
        DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
    }:
        _append_dependency_for_kind(dependencies, accepted, "planning.context")
        if not any(item.evidence_kind == "planning.context" for item in dependencies):
            _append_dependency_for_any(
                dependencies,
                accepted,
                ("schema.asset_profile", "catalog.source", "schema.search_result"),
            )
        for kind in (
            "query.result",
            "query.plan.proposal",
            "query.plan.validation",
            "sql.validation",
            "verification.result",
        ):
            _append_dependency_for_kind(dependencies, accepted, kind)
    elif intent.kind is DbIntentKind.SCHEMA_QUERY:
        _append_database_schema_dependency(dependencies, accepted)
        for kind in ("planning.context", "schema.search_result"):
            _append_dependency_for_kind(dependencies, accepted, kind)
        _append_schema_asset_dependencies(dependencies, accepted)
        _append_dependency_for_kind(dependencies, accepted, "verification.result")
    elif intent.kind is DbIntentKind.SCHEMA_RELATIONSHIP_QUERY:
        _append_database_schema_dependency(dependencies, accepted)
        for kind in (
            "planning.context",
            "schema.relationship_path",
            "schema.search_result",
        ):
            _append_dependency_for_kind(dependencies, accepted, kind)
        _append_dependency_for_kind(dependencies, accepted, "verification.result")
    else:
        _append_dependency_for_kind(dependencies, accepted, "verification.result")
    seen: set[tuple[str | None, str | None]] = set()
    unique: list[TaskDependency] = []
    for dependency in dependencies:
        key = (dependency.evidence_kind, dependency.evidence_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(dependency)
    return tuple(unique)


def _append_dependency_for_kind(
    dependencies: list[TaskDependency],
    evidence: tuple[Evidence, ...],
    kind: str,
) -> None:
    item = next(
        (candidate for candidate in reversed(evidence) if candidate.kind == kind),
        None,
    )
    if item is not None:
        dependencies.append(_dependency_for_evidence(item))


def _append_dependency_for_any(
    dependencies: list[TaskDependency],
    evidence: tuple[Evidence, ...],
    kinds: tuple[str, ...],
) -> None:
    for kind in kinds:
        item = next(
            (candidate for candidate in reversed(evidence) if candidate.kind == kind),
            None,
        )
        if item is not None:
            dependencies.append(_dependency_for_evidence(item))
            return
    catalog = next(
        (
            candidate
            for candidate in reversed(evidence)
            if candidate.kind.startswith("catalog.")
        ),
        None,
    )
    if catalog is not None:
        dependencies.append(_dependency_for_evidence(catalog))


def _append_database_schema_dependency(
    dependencies: list[TaskDependency],
    evidence: tuple[Evidence, ...],
) -> None:
    item = next(
        (
            candidate
            for candidate in reversed(evidence)
            if candidate.kind == "schema.asset_profile"
            and _schema_evidence_scope(candidate) == "database"
        ),
        None,
    )
    if item is None:
        item = next(
            (
                candidate
                for candidate in evidence
                if candidate.kind == "schema.asset_profile"
                and candidate.payload.get("tables")
            ),
            None,
        )
    if item is not None:
        dependencies.append(_dependency_for_evidence(item))


def _append_schema_asset_dependencies(
    dependencies: list[TaskDependency],
    evidence: tuple[Evidence, ...],
) -> None:
    scoped = [
        item
        for item in evidence
        if item.kind == "schema.asset_profile"
        and _schema_evidence_scope(item) == "asset"
        and item.id
    ]
    for item in scoped:
        dependencies.append(_dependency_for_evidence(item))
    if not scoped:
        _append_dependency_for_kind(dependencies, evidence, "schema.asset_profile")


def _schema_evidence_scope(evidence: Evidence) -> str | None:
    metadata_scope = evidence.metadata.get("scope")
    if metadata_scope:
        return str(metadata_scope)
    payload_metadata = evidence.payload.get("metadata")
    if isinstance(payload_metadata, dict) and payload_metadata.get("scope"):
        return str(payload_metadata["scope"])
    return None


def _dependency_for_evidence(evidence: Evidence) -> TaskDependency:
    return TaskDependency(
        kind="evidence",
        evidence_kind=evidence.kind,
        evidence_id=evidence.id,
        evidence_owner=evidence.owner,
        producer_task_id=evidence.task_id,
        evidence_accepted=True,
        operation_id=evidence.operation_id,
        payload_fingerprint=evidence.metadata.get("payload_fingerprint")
        or _payload_fingerprint(evidence.payload),
    )


def _synthesis_context_option(
    metadata: dict[str, Any],
    key: str,
    default: int,
) -> int:
    options = metadata.get("from_db_options")
    if isinstance(options, dict) and options.get(key) is not None:
        try:
            return int(options[key])
        except (TypeError, ValueError):
            return default
    return default


def _payload_contains(payload: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if payload.get(key) != value:
            return False
    return True


def _json_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    copied = dict(value or {})
    try:
        json.dumps(copied)
    except TypeError as exc:
        raise TypeError("DB runtime task mappings must be JSON serializable") from exc
    return copied


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    return _stable_hash(payload)


def _prompt_from_direct_input(input: dict[str, Any]) -> str:
    for key in ("prompt", "sql", "query", "content"):
        value = input.get(key)
        if value:
            return str(value)
    return ""


def _contract_snapshot(
    contract: DbOperationContract | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if contract is None:
        return None
    if isinstance(contract, Mapping):
        return _json_dict(contract)
    return {
        "operation_type": contract.operation_type,
        "required_capabilities": list(contract.required_capabilities),
        "required_evidence": list(contract.required_evidence),
        "access": contract.access.value,
        "limits": contract.limits.to_dict(),
        "policy_ids": list(contract.policy_ids),
        "metadata": contract.metadata,
    }


def _combine_dependencies(
    default_dependencies: tuple[TaskDependency, ...],
    spec_dependencies: tuple[TaskDependency, ...],
) -> tuple[TaskDependency, ...]:
    combined: list[TaskDependency] = []
    seen: set[str] = set()
    for dependency in (*default_dependencies, *spec_dependencies):
        fingerprint = _stable_hash(dependency.to_dict())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        combined.append(dependency)
    return tuple(combined)


def _task_dependencies_for_capability(
    operation: Operation,
    capability: Capability,
    *,
    validation_task: Task | None = None,
) -> tuple[TaskDependency, ...]:
    if capability.id not in {"db.sql.execute_read", "db.sql.execute_write"}:
        return ()
    if capability.id == "db.sql.execute_read" and validation_task is None:
        return ()
    validation_dependency = TaskDependency(
        kind="evidence",
        evidence_kind="sql.validation",
        evidence_owner=(
            validation_task.metadata.get("owner") if validation_task else None
        ),
        producer_task_id=validation_task.id if validation_task else None,
        producer_capability_id=(
            validation_task.capability_id if validation_task else "db.sql.validate"
        ),
        producer_executor_id=(validation_task.executor_id if validation_task else None),
        evidence_payload={"valid": True},
        operation_id=operation.id,
        input_hash=(
            validation_task.metadata.get("input_hash") if validation_task else None
        ),
    )
    if capability.id == "db.sql.execute_read":
        return (validation_dependency,)
    return (
        validation_dependency,
        TaskDependency(
            kind="approval",
            approval_status=ApprovalStatus.APPROVED,
            approval_policy_id="approval_required_for_writes",
            approval_name="human",
            operation_id=operation.id,
        ),
    )
