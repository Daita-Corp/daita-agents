"""Contract-bound DB planner loop."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Mapping, Protocol

from daita.runtime import Evidence, Operation, Task

from .models import DbOperationContract, DbRequest
from .planner_protocol import (
    CONTROL_DB_PLANNER_ACTION_KINDS,
    DbPlannerAction,
    DbPlannerDecision,
    DbPlannerObservation,
)
from .runtime.tasks import DbTaskSpec
from .safety import DbCapabilityLane


class DbPlanner(Protocol):
    """Planner object used by the loop."""

    def decide(
        self,
        context: Mapping[str, Any],
        observations: tuple[DbPlannerObservation, ...],
    ) -> DbPlannerDecision | Mapping[str, Any]: ...


@dataclass(frozen=True)
class DbAgentLoopResult:
    """Result of a contract-bound planner loop run."""

    status: str
    observations: tuple[DbPlannerObservation, ...]
    tasks: tuple[Task, ...]
    decision: DbPlannerDecision | None = None
    message: str | None = None


class DbAgentLoopBlocked(ValueError):
    """Raised when a planner action exceeds the operation contract."""


class DbAgentLoop:
    """Run planner actions inside a lane-based DB operation contract."""

    def __init__(self, runtime: Any, planner: DbPlanner, *, max_steps: int = 8) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.runtime = runtime
        self.planner = planner
        self.max_steps = max_steps

    async def run(
        self,
        *,
        request: DbRequest | str,
        operation: Operation,
        contract: DbOperationContract,
    ) -> DbAgentLoopResult:
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        observations: list[DbPlannerObservation] = []
        tasks: list[Task] = []

        for _ in range(self.max_steps):
            context = await self._planner_context(db_request, operation, contract)
            decision = await self._next_decision(context, tuple(observations))
            for action in decision.actions:
                if action.kind == "clarify":
                    return DbAgentLoopResult(
                        status="clarify",
                        observations=tuple(observations),
                        tasks=tuple(tasks),
                        decision=decision,
                        message=_action_message(action),
                    )
                if action.kind == "finish":
                    return DbAgentLoopResult(
                        status="finish",
                        observations=tuple(observations),
                        tasks=tuple(tasks),
                        decision=decision,
                        message=_action_message(action),
                    )
                action_tasks = self._tasks_for_action(action, operation, contract)
                for task in action_tasks:
                    evidence = await self.runtime.execute_task(
                        task,
                        operation,
                        context={
                            "db_planner_action": action.to_dict(),
                            "db_operation_contract": _contract_context(contract),
                        },
                    )
                    stored = await self.runtime.store.load_task(task.id)
                    tasks.append(stored or task)
                    observations.append(
                        DbPlannerObservation(
                            kind="task.executed",
                            action_id=action.action_id,
                            task_ids=(task.id,),
                            payload={
                                "capability_id": task.capability_id,
                                "evidence": [
                                    item.to_dict()
                                    for item in tuple(evidence or ())
                                    if isinstance(item, Evidence)
                                ],
                            },
                        )
                    )
            if any(action.kind == "synthesize" for action in decision.actions):
                return DbAgentLoopResult(
                    status="synthesize",
                    observations=tuple(observations),
                    tasks=tuple(tasks),
                    decision=decision,
                )
        return DbAgentLoopResult(
            status="blocked",
            observations=tuple(
                (
                    *observations,
                    DbPlannerObservation(
                        kind="loop.blocked",
                        blocked=True,
                        message="planner loop reached max_steps",
                    ),
                )
            ),
            tasks=tuple(tasks),
        )

    async def _next_decision(
        self,
        context: Mapping[str, Any],
        observations: tuple[DbPlannerObservation, ...],
    ) -> DbPlannerDecision:
        decision = self.planner.decide(context, observations)
        if inspect.isawaitable(decision):
            decision = await decision
        if isinstance(decision, DbPlannerDecision):
            return decision
        return DbPlannerDecision.from_dict(decision)

    async def _planner_context(
        self,
        request: DbRequest,
        operation: Operation,
        contract: DbOperationContract,
    ) -> dict[str, Any]:
        tasks = await self.runtime.store.list_tasks(operation.id)
        evidence = await self.runtime.store.list_evidence(operation.id)
        return {
            "request": {
                "prompt": request.prompt,
                "source_scope": list(request.source_scope),
                "constraints": request.constraints,
                "session_context": request.session_context,
            },
            "operation": operation.to_dict(),
            "contract": _contract_context(contract),
            "available_capabilities": [
                {
                    "id": capability.id,
                    "owner": capability.owner,
                    "executor": capability.executor,
                }
                for capability in self.runtime.registry.capabilities
            ],
            "task_observations": [task.to_dict() for task in tasks],
            "evidence_observations": [item.to_dict() for item in evidence],
        }

    def _tasks_for_action(
        self,
        action: DbPlannerAction,
        operation: Operation,
        contract: DbOperationContract,
    ) -> tuple[Task, ...]:
        if action.kind in CONTROL_DB_PLANNER_ACTION_KINDS:
            return ()
        specs = _action_task_specs(action)
        self._validate_action_specs(action, specs, contract)
        return self.runtime.materialize_task_specs(
            operation,
            _runtime_task_specs_for_action(action, specs, contract),
        )

    def _validate_action_specs(
        self,
        action: DbPlannerAction,
        specs: tuple["_TaskSpec", ...],
        contract: DbOperationContract,
    ) -> None:
        granted_lanes = _contract_lanes(contract)
        forbidden = frozenset(contract.metadata.get("forbidden_capabilities", ()))
        required = frozenset(contract.required_capabilities)
        selected = _selected_capability_ids(contract)
        for spec in specs:
            matched_lane = _matched_lane(spec, granted_lanes)
            if matched_lane is None:
                lane_values = ", ".join(lane.value for lane in spec.required_lanes)
                raise DbAgentLoopBlocked(
                    f"action {action.kind!r} requires one of lanes {lane_values!r}"
                )
            if spec.capability_id in forbidden:
                raise DbAgentLoopBlocked(
                    f"action {action.kind!r} uses forbidden capability "
                    f"{spec.capability_id!r}"
                )
            if (
                spec.must_be_contract_required
                and spec.capability_id not in required
                and spec.capability_id not in selected
            ):
                raise DbAgentLoopBlocked(
                    f"action {action.kind!r} uses capability "
                    f"{spec.capability_id!r} outside the operation contract"
                )
            if (
                contract.metadata.get("approval_required")
                and _spec_side_effecting(contract, spec)
                and not action.payload.get("approval_id")
            ):
                raise DbAgentLoopBlocked(
                    f"action {action.kind!r} requires approval before execution"
                )


@dataclass(frozen=True)
class _TaskSpec:
    kind: str
    capability_id: str
    required_lanes: tuple[DbCapabilityLane, ...]
    input_builder: Any
    owner: str | None = None
    must_be_contract_required: bool = True
    depends_on_validation: bool = False


def _action_task_specs(action: DbPlannerAction) -> tuple[_TaskSpec, ...]:
    if action.kind == "inspect_schema":
        specs: list[_TaskSpec] = []
        if action.payload.get("local_schema", True):
            specs.append(
                _TaskSpec(
                    kind="schema",
                    capability_id="db.schema.inspect",
                    required_lanes=(DbCapabilityLane.SCHEMA, DbCapabilityLane.READ),
                    input_builder=_schema_input,
                    must_be_contract_required=False,
                )
            )
        if action.payload.get("catalog_search_store_id"):
            specs.append(
                _TaskSpec(
                    kind="catalog_search",
                    capability_id="catalog.schema.search",
                    required_lanes=(DbCapabilityLane.SCHEMA, DbCapabilityLane.READ),
                    input_builder=_catalog_search_input,
                    owner="catalog",
                )
            )
        if action.payload.get("catalog_asset_store_id") and action.payload.get(
            "asset_ref"
        ):
            specs.append(
                _TaskSpec(
                    kind="catalog_asset",
                    capability_id="catalog.asset.inspect",
                    required_lanes=(DbCapabilityLane.SCHEMA, DbCapabilityLane.READ),
                    input_builder=_catalog_asset_input,
                    owner="catalog",
                )
            )
        if not specs:
            raise DbAgentLoopBlocked("inspect_schema action has no executable target")
        return tuple(specs)
    mapping: dict[str, tuple[_TaskSpec, ...]] = {
        "register_catalog_source": (
            _TaskSpec(
                kind="catalog_register",
                capability_id="catalog.source.register",
                required_lanes=(DbCapabilityLane.SCHEMA, DbCapabilityLane.READ),
                input_builder=_payload_input,
                owner="catalog",
                must_be_contract_required=False,
            ),
        ),
        "build_planning_context": (
            _TaskSpec(
                kind="context",
                capability_id="db.planning.context.build",
                required_lanes=(DbCapabilityLane.SCHEMA, DbCapabilityLane.READ),
                input_builder=_payload_input,
                owner="db_runtime",
                must_be_contract_required=False,
            ),
        ),
        "propose_sql_read": (
            _TaskSpec(
                kind="prepare",
                capability_id="db.query.prepare_read",
                required_lanes=(DbCapabilityLane.READ,),
                input_builder=_payload_input,
                owner="db_runtime",
                must_be_contract_required=False,
            ),
        ),
        "propose_sql_write": (
            _TaskSpec(
                kind="validate",
                capability_id="db.sql.validate",
                required_lanes=(
                    DbCapabilityLane.WRITE_PROPOSE,
                    DbCapabilityLane.WRITE_EXECUTE,
                ),
                input_builder=_sql_validation_input,
            ),
        ),
        "execute_validated_read": (
            _TaskSpec(
                kind="validate",
                capability_id="db.sql.validate",
                required_lanes=(DbCapabilityLane.READ,),
                input_builder=_sql_validation_input,
            ),
            _TaskSpec(
                kind="execute",
                capability_id="db.sql.execute_read",
                required_lanes=(DbCapabilityLane.READ,),
                input_builder=_sql_execute_input,
                depends_on_validation=True,
            ),
        ),
        "execute_validated_write": (
            _TaskSpec(
                kind="validate",
                capability_id="db.sql.validate",
                required_lanes=(DbCapabilityLane.WRITE_EXECUTE,),
                input_builder=_sql_validation_input,
            ),
            _TaskSpec(
                kind="execute",
                capability_id="db.sql.execute_write",
                required_lanes=(DbCapabilityLane.WRITE_EXECUTE,),
                input_builder=_sql_execute_input,
                depends_on_validation=True,
            ),
        ),
        "recall_memory": (
            _TaskSpec(
                kind="memory",
                capability_id="memory.semantic.recall",
                required_lanes=(DbCapabilityLane.MEMORY_ANSWER,),
                input_builder=_payload_input,
            ),
            _TaskSpec(
                kind="memory",
                capability_id="db.memory.answer_context.build",
                required_lanes=(DbCapabilityLane.MEMORY_ANSWER,),
                input_builder=_payload_input,
                owner="db_runtime",
            ),
        ),
        "write_memory": (
            _TaskSpec(
                kind="memory",
                capability_id="db.memory.plan_update",
                required_lanes=(DbCapabilityLane.MEMORY_WRITE,),
                input_builder=_payload_input,
                owner="db_runtime",
            ),
            _TaskSpec(
                kind="memory",
                capability_id="db.memory.commit_update",
                required_lanes=(DbCapabilityLane.MEMORY_WRITE,),
                input_builder=_payload_input,
                owner="db_runtime",
            ),
        ),
        "inspect_monitor": (
            _TaskSpec(
                kind="monitor",
                capability_id="db.monitor.inspect",
                required_lanes=(DbCapabilityLane.MONITOR_READ,),
                input_builder=_payload_input,
                must_be_contract_required=False,
            ),
        ),
        "update_monitor": (
            _TaskSpec(
                kind="monitor",
                capability_id="db.monitor.plan_lifecycle",
                required_lanes=(DbCapabilityLane.MONITOR_WRITE,),
                input_builder=_payload_input,
                owner="db_runtime",
            ),
        ),
        "execute_monitor": (
            _TaskSpec(
                kind="monitor",
                capability_id="db.monitor.execute",
                required_lanes=(DbCapabilityLane.MONITOR_EXECUTE,),
                input_builder=_payload_input,
                must_be_contract_required=False,
            ),
        ),
        "synthesize": (
            _TaskSpec(
                kind="answer",
                capability_id="db.answer.synthesize",
                required_lanes=(
                    DbCapabilityLane.SCHEMA,
                    DbCapabilityLane.MEMORY_ANSWER,
                    DbCapabilityLane.READ,
                    DbCapabilityLane.WRITE_PROPOSE,
                    DbCapabilityLane.WRITE_EXECUTE,
                    DbCapabilityLane.MONITOR_READ,
                    DbCapabilityLane.MONITOR_WRITE,
                    DbCapabilityLane.MONITOR_EXECUTE,
                ),
                input_builder=_payload_input,
                owner="db_runtime",
                must_be_contract_required=False,
            ),
        ),
    }
    try:
        return mapping[action.kind]
    except KeyError as exc:
        raise DbAgentLoopBlocked(
            f"unsupported executable action {action.kind!r}"
        ) from exc


def _runtime_task_specs_for_action(
    action: DbPlannerAction,
    specs: tuple[_TaskSpec, ...],
    contract: DbOperationContract,
) -> tuple[DbTaskSpec, ...]:
    return tuple(
        DbTaskSpec(
            capability_id=spec.capability_id,
            owner=_owner_for_spec(contract, spec),
            input=spec.input_builder(action),
            reason=str(action.payload.get("reason") or f"planner_action:{action.kind}"),
            sequence=index,
            metadata={
                "planner_action_kind": action.kind,
                "planner_action_id": action.action_id,
                "required_lane": _matched_lane(spec, _contract_lanes(contract)).value,
            },
            depends_on_validation=spec.depends_on_validation,
        )
        for index, spec in enumerate(specs, start=1)
    )


def _owner_for_spec(contract: DbOperationContract, spec: _TaskSpec) -> str | None:
    selected = _selected_capability_by_id(contract, spec.capability_id)
    if spec.owner is not None:
        return spec.owner
    owner = selected.get("owner")
    return str(owner) if owner else None


def _spec_side_effecting(contract: DbOperationContract, spec: _TaskSpec) -> bool:
    selected = _selected_capability_by_id(contract, spec.capability_id)
    if "side_effecting" in selected:
        return bool(selected["side_effecting"])
    return spec.capability_id in {
        "db.sql.execute_write",
        "db.memory.commit_update",
        "db.monitor.execute",
    }


def _matched_lane(
    spec: _TaskSpec,
    granted_lanes: frozenset[DbCapabilityLane],
) -> DbCapabilityLane | None:
    return next((lane for lane in spec.required_lanes if lane in granted_lanes), None)


def _selected_capability_by_id(
    contract: DbOperationContract,
    capability_id: str,
) -> dict[str, Any]:
    for item in contract.metadata.get("selected_capabilities", ()):
        if item.get("id") == capability_id:
            return dict(item)
    return {}


def _selected_capability_ids(contract: DbOperationContract) -> frozenset[str]:
    return frozenset(
        str(item.get("id"))
        for item in contract.metadata.get("selected_capabilities", ())
        if item.get("id")
    )


def _contract_lanes(contract: DbOperationContract) -> frozenset[DbCapabilityLane]:
    return frozenset(
        DbCapabilityLane(value) for value in contract.metadata.get("granted_lanes", ())
    )


def _contract_context(contract: DbOperationContract) -> dict[str, Any]:
    return {
        "operation_type": contract.operation_type,
        "required_capabilities": list(contract.required_capabilities),
        "required_evidence": list(contract.required_evidence),
        "access": contract.access.value,
        "policy_ids": list(contract.policy_ids),
        "granted_lanes": list(contract.metadata.get("granted_lanes", ())),
        "forbidden_capabilities": list(
            contract.metadata.get("forbidden_capabilities", ())
        ),
        "approval_required": bool(contract.metadata.get("approval_required", False)),
    }


def _payload_input(action: DbPlannerAction) -> dict[str, Any]:
    return dict(action.payload)


def _schema_input(action: DbPlannerAction) -> dict[str, Any]:
    return {
        "focus": action.payload.get("focus"),
        "source_scope": list(action.payload.get("source_scope") or ()),
    }


def _catalog_search_input(action: DbPlannerAction) -> dict[str, Any]:
    return {
        "store_id": str(action.payload.get("catalog_search_store_id") or ""),
        "query": str(action.payload.get("query") or ""),
        "limit": int(action.payload.get("limit") or 10),
    }


def _catalog_asset_input(action: DbPlannerAction) -> dict[str, Any]:
    return {
        "store_id": str(action.payload.get("catalog_asset_store_id") or ""),
        "asset_ref": str(action.payload.get("asset_ref") or ""),
        "limit": int(action.payload.get("limit") or 100),
    }


def _sql_validation_input(action: DbPlannerAction) -> dict[str, Any]:
    return {
        "sql": str(action.payload.get("sql") or ""),
        "operation": str(action.payload.get("operation") or "query"),
    }


def _sql_execute_input(action: DbPlannerAction) -> dict[str, Any]:
    output: dict[str, Any] = {
        "sql_ref": str(action.payload.get("sql_ref") or "sql.validation"),
        "params": list(action.payload.get("params") or ()),
    }
    if action.payload.get("param_specs"):
        output["param_specs"] = list(action.payload.get("param_specs") or ())
    if action.payload.get("focus") is not None:
        output["focus"] = action.payload["focus"]
    return output


def _action_message(action: DbPlannerAction) -> str | None:
    value = action.payload.get("message") or action.payload.get("answer")
    return str(value) if value is not None else None
