"""Strict current codecs for revision-2 adaptive task-graph records."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from ..._json import FrozenJsonObject
from ...distribution.models import (
    DeliveryState,
    GraphJobDelivery,
)
from ...jobs.graph.models import (
    AttemptState,
    BudgetAmount,
    BudgetLimit,
    ControlKind,
    ControlState,
    EdgeKind,
    GraphAuthority,
    GraphDesiredState,
    GraphEvent,
    GraphJob,
    GraphJobSpecification,
    GraphLimits,
    GraphMutation,
    GraphState,
    GraphTask,
    GraphTaskSpecification,
    JobGraph,
    MutationDecision,
    TaskAttempt,
    TaskCheckpoint,
    TaskComment,
    TaskControl,
    TaskDependency,
    TaskExecutionKind,
    TaskResult,
    TaskRole,
    TaskState,
)
from ...llm.models import ModelSensitivity
from .common import (
    JsonValue,
    datetime_decode,
    datetime_encode,
    dump_payload,
    integer,
    load_payload,
    optional_datetime_decode,
    optional_datetime_encode,
    optional_integer,
    optional_text,
    plain_decode,
    plain_encode,
    record,
    record_fields,
    sequence,
    text,
)
from .distribution import (
    decode_conversation_inbox_target,
    decode_outcome_reference,
    encode_conversation_inbox_target,
    encode_outcome_reference,
)

_GRAPH_LIMIT_FIELDS = (
    "max_tasks",
    "max_edges",
    "max_depth",
    "max_direct_parents",
    "max_fan_out",
    "max_attempts_per_task",
    "max_parallelism",
    "max_mutations",
    "max_events",
    "max_controls_per_task",
    "max_checkpoints_per_attempt",
    "max_comments_per_task",
    "max_artifacts",
    "max_artifact_bytes",
    "max_total_artifact_bytes",
)


def _mapping(value: JsonValue, label: str) -> dict[str, object]:
    decoded = plain_decode(value)
    if not isinstance(decoded, dict):
        raise TypeError(f"stored {label} must be an object")
    return decoded


def _enum(value: JsonValue, enum_type, label: str):
    try:
        return enum_type(text(value, label))
    except ValueError:
        raise ValueError(f"stored {label} enum is invalid") from None


def _text_tuple(value: JsonValue, label: str) -> tuple[str, ...]:
    return tuple(text(item, label) for item in sequence(value, label))


def _pair_tuple(value: JsonValue, label: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for item in sequence(value, label):
        raw = sequence(item, label)
        if len(raw) != 2:
            raise ValueError(f"stored {label} pair is invalid")
        pairs.append((text(raw[0], label), text(raw[1], label)))
    return tuple(pairs)


def _encode_authority(value: GraphAuthority) -> JsonValue:
    return record(
        "GraphAuthority",
        {
            "source_ids": list(value.source_ids),
            "resource_ids": list(value.resource_ids),
            "connector_ids": list(value.connector_ids),
            "capability_ids": list(value.capability_ids),
            "access_modes": list(value.access_modes),
            "operational_effects": list(value.operational_effects),
            "model_route_ids": list(value.model_route_ids),
            "sensitivity": value.sensitivity.value,
            "contract_bindings": plain_encode(value.contract_bindings),
        },
    )


def _decode_authority(value: JsonValue) -> GraphAuthority:
    fields = record_fields(
        value,
        "GraphAuthority",
        (
            "source_ids",
            "resource_ids",
            "connector_ids",
            "capability_ids",
            "access_modes",
            "operational_effects",
            "model_route_ids",
            "sensitivity",
            "contract_bindings",
        ),
    )
    return GraphAuthority(
        source_ids=_text_tuple(fields["source_ids"], "authority source IDs"),
        resource_ids=_text_tuple(fields["resource_ids"], "authority resource IDs"),
        connector_ids=_text_tuple(fields["connector_ids"], "authority connector IDs"),
        capability_ids=_text_tuple(
            fields["capability_ids"], "authority capability IDs"
        ),
        access_modes=_text_tuple(fields["access_modes"], "authority access modes"),
        operational_effects=_text_tuple(
            fields["operational_effects"], "authority operational effects"
        ),
        model_route_ids=_text_tuple(
            fields["model_route_ids"], "authority model routes"
        ),
        sensitivity=_enum(
            fields["sensitivity"], ModelSensitivity, "authority sensitivity"
        ),
        contract_bindings=_mapping(
            fields["contract_bindings"], "authority contract bindings"
        ),
    )


def _encode_limits(value: GraphLimits) -> JsonValue:
    return record(
        "GraphLimits",
        {name: getattr(value, name) for name in _GRAPH_LIMIT_FIELDS},
    )


def _decode_limits(value: JsonValue) -> GraphLimits:
    fields = record_fields(value, "GraphLimits", _GRAPH_LIMIT_FIELDS)
    return GraphLimits(
        **{
            name: integer(fields[name], f"graph limit {name}")
            for name in _GRAPH_LIMIT_FIELDS
        }
    )


def _encode_budget_limit(value: BudgetLimit) -> JsonValue:
    return record(
        "BudgetLimit",
        {
            "dimension": value.dimension,
            "ceiling": value.ceiling,
            "control_reserved": value.control_reserved,
        },
    )


def _decode_budget_limit(value: JsonValue) -> BudgetLimit:
    fields = record_fields(
        value, "BudgetLimit", ("dimension", "ceiling", "control_reserved")
    )
    return BudgetLimit(
        dimension=text(fields["dimension"], "budget dimension"),
        ceiling=integer(fields["ceiling"], "budget ceiling"),
        control_reserved=integer(fields["control_reserved"], "budget control reserve"),
    )


def _encode_budget_amount(value: BudgetAmount) -> JsonValue:
    return record(
        "BudgetAmount", {"dimension": value.dimension, "amount": value.amount}
    )


def _decode_budget_amount(value: JsonValue) -> BudgetAmount:
    fields = record_fields(value, "BudgetAmount", ("dimension", "amount"))
    return BudgetAmount(
        dimension=text(fields["dimension"], "budget dimension"),
        amount=integer(fields["amount"], "budget amount"),
    )


def _encode_job_specification(value: GraphJobSpecification) -> JsonValue:
    return record(
        "GraphJobSpecification",
        {
            "principal_id": value.principal_id,
            "objective": value.objective,
            "outcome_contract": plain_encode(value.outcome_contract),
            "authority": _encode_authority(value.authority),
            "distribution_plan_digest": value.distribution_plan_digest,
            "budgets": [_encode_budget_limit(item) for item in value.budgets],
            "deadline_at": datetime_encode(value.deadline_at),
            "limits": _encode_limits(value.limits),
            "retry_policy": plain_encode(value.retry_policy),
            "cancellation_policy": plain_encode(value.cancellation_policy),
            "effect_mode": value.effect_mode,
            "planner_task_template": (
                None
                if value.planner_task_template is None
                else plain_encode(value.planner_task_template)
            ),
            "finalizer_task_template": plain_encode(value.finalizer_task_template),
        },
    )


def _decode_job_specification(value: JsonValue) -> GraphJobSpecification:
    fields = record_fields(
        value,
        "GraphJobSpecification",
        (
            "principal_id",
            "objective",
            "outcome_contract",
            "authority",
            "distribution_plan_digest",
            "budgets",
            "deadline_at",
            "limits",
            "retry_policy",
            "cancellation_policy",
            "effect_mode",
            "planner_task_template",
            "finalizer_task_template",
        ),
    )
    planner = fields["planner_task_template"]
    return GraphJobSpecification(
        principal_id=text(fields["principal_id"], "graph principal ID"),
        objective=text(fields["objective"], "graph objective"),
        outcome_contract=_mapping(fields["outcome_contract"], "outcome contract"),
        authority=_decode_authority(fields["authority"]),
        distribution_plan_digest=text(
            fields["distribution_plan_digest"], "distribution plan digest"
        ),
        budgets=tuple(
            _decode_budget_limit(item)
            for item in sequence(fields["budgets"], "graph budgets")
        ),
        deadline_at=datetime_decode(fields["deadline_at"]),
        limits=_decode_limits(fields["limits"]),
        retry_policy=_mapping(fields["retry_policy"], "graph retry policy"),
        cancellation_policy=_mapping(
            fields["cancellation_policy"], "graph cancellation policy"
        ),
        effect_mode=text(fields["effect_mode"], "graph effect mode"),
        planner_task_template=(
            None if planner is None else _mapping(planner, "planner task template")
        ),
        finalizer_task_template=_mapping(
            fields["finalizer_task_template"], "finalizer task template"
        ),
    )


def _encode_task_specification(value: GraphTaskSpecification) -> JsonValue:
    return record(
        "GraphTaskSpecification",
        {
            "title": value.title,
            "description": value.description,
            "expected_result_contract": plain_encode(value.expected_result_contract),
            "authority": _encode_authority(value.authority),
            "budgets": [_encode_budget_amount(item) for item in value.budgets],
            "max_steps": value.max_steps,
            "max_wall_time_seconds": value.max_wall_time_seconds,
            "created_by": value.created_by,
        },
    )


def _decode_task_specification(value: JsonValue) -> GraphTaskSpecification:
    fields = record_fields(
        value,
        "GraphTaskSpecification",
        (
            "title",
            "description",
            "expected_result_contract",
            "authority",
            "budgets",
            "max_steps",
            "max_wall_time_seconds",
            "created_by",
        ),
    )
    return GraphTaskSpecification(
        title=text(fields["title"], "task title"),
        description=text(fields["description"], "task description"),
        expected_result_contract=_mapping(
            fields["expected_result_contract"], "task result contract"
        ),
        authority=_decode_authority(fields["authority"]),
        budgets=tuple(
            _decode_budget_amount(item)
            for item in sequence(fields["budgets"], "task budgets")
        ),
        max_steps=integer(fields["max_steps"], "task max steps"),
        max_wall_time_seconds=integer(
            fields["max_wall_time_seconds"], "task wall time"
        ),
        created_by=text(fields["created_by"], "task creator"),
    )


def encode_graph_job(value: GraphJob) -> str:
    return dump_payload(
        record(
            "GraphJob",
            {
                "agent_id": value.agent_id,
                "job_id": value.job_id,
                "conversation_id": value.conversation_id,
                "origin_run_id": value.origin_run_id,
                "origin_call_id": value.origin_call_id,
                "state": value.state.value,
                "desired_state": value.desired_state.value,
                "created_at": datetime_encode(value.created_at),
                "updated_at": datetime_encode(value.updated_at),
                "deadline_at": datetime_encode(value.deadline_at),
                "specification": _encode_job_specification(value.specification),
                "specification_digest": value.specification_digest,
                "finalizer_task_id": value.finalizer_task_id,
                "terminal_at": optional_datetime_encode(value.terminal_at),
                "terminal_result_id": value.terminal_result_id,
                "failure_code": value.failure_code,
                "migration_provenance": plain_encode(value.migration_provenance),
            },
        )
    )


def decode_graph_job(value: str) -> GraphJob:
    fields = record_fields(
        load_payload(value),
        "GraphJob",
        (
            "agent_id",
            "job_id",
            "conversation_id",
            "origin_run_id",
            "origin_call_id",
            "state",
            "desired_state",
            "created_at",
            "updated_at",
            "deadline_at",
            "specification",
            "specification_digest",
            "finalizer_task_id",
            "terminal_at",
            "terminal_result_id",
            "failure_code",
            "migration_provenance",
        ),
    )
    return GraphJob(
        agent_id=text(fields["agent_id"], "graph agent ID"),
        job_id=text(fields["job_id"], "graph job ID"),
        conversation_id=text(fields["conversation_id"], "graph conversation ID"),
        origin_run_id=text(fields["origin_run_id"], "graph origin run ID"),
        origin_call_id=text(fields["origin_call_id"], "graph origin call ID"),
        state=_enum(fields["state"], GraphState, "graph state"),
        desired_state=_enum(
            fields["desired_state"], GraphDesiredState, "graph desired state"
        ),
        created_at=datetime_decode(fields["created_at"]),
        updated_at=datetime_decode(fields["updated_at"]),
        deadline_at=datetime_decode(fields["deadline_at"]),
        specification=_decode_job_specification(fields["specification"]),
        specification_digest=text(
            fields["specification_digest"], "graph specification digest"
        ),
        finalizer_task_id=text(fields["finalizer_task_id"], "graph finalizer task ID"),
        terminal_at=optional_datetime_decode(fields["terminal_at"]),
        terminal_result_id=optional_text(
            fields["terminal_result_id"], "graph terminal result ID"
        ),
        failure_code=optional_text(fields["failure_code"], "graph failure code"),
        migration_provenance=_mapping(
            fields["migration_provenance"], "graph migration provenance"
        ),
    )


def encode_job_graph(value: JobGraph) -> str:
    return dump_payload(
        record(
            "JobGraph",
            {
                "agent_id": value.agent_id,
                "job_id": value.job_id,
                "revision": value.revision,
                "task_count": value.task_count,
                "edge_count": value.edge_count,
                "mutation_count": value.mutation_count,
                "active_attempt_count": value.active_attempt_count,
                "next_ready_at": optional_datetime_encode(value.next_ready_at),
                "finalization_attempt_id": value.finalization_attempt_id,
                "finalization_started_revision": value.finalization_started_revision,
                "created_at": datetime_encode(value.created_at),
                "updated_at": datetime_encode(value.updated_at),
                "topology_digest": value.topology_digest,
            },
        )
    )


def decode_job_graph(value: str) -> JobGraph:
    fields = record_fields(
        load_payload(value),
        "JobGraph",
        (
            "agent_id",
            "job_id",
            "revision",
            "task_count",
            "edge_count",
            "mutation_count",
            "active_attempt_count",
            "next_ready_at",
            "finalization_attempt_id",
            "finalization_started_revision",
            "created_at",
            "updated_at",
            "topology_digest",
        ),
    )
    return JobGraph(
        agent_id=text(fields["agent_id"], "job graph agent ID"),
        job_id=text(fields["job_id"], "job graph job ID"),
        revision=integer(fields["revision"], "job graph revision"),
        task_count=integer(fields["task_count"], "job graph task count"),
        edge_count=integer(fields["edge_count"], "job graph edge count"),
        mutation_count=integer(fields["mutation_count"], "job graph mutation count"),
        active_attempt_count=integer(
            fields["active_attempt_count"], "job graph active attempt count"
        ),
        next_ready_at=optional_datetime_decode(fields["next_ready_at"]),
        finalization_attempt_id=optional_text(
            fields["finalization_attempt_id"], "finalization attempt ID"
        ),
        finalization_started_revision=optional_integer(
            fields["finalization_started_revision"],
            "finalization started revision",
        ),
        created_at=datetime_decode(fields["created_at"]),
        updated_at=datetime_decode(fields["updated_at"]),
        topology_digest=text(fields["topology_digest"], "topology digest"),
    )


def encode_graph_task(value: GraphTask) -> str:
    return dump_payload(
        record(
            "GraphTask",
            {
                "agent_id": value.agent_id,
                "job_id": value.job_id,
                "task_id": value.task_id,
                "state": value.state.value,
                "role": value.role.value,
                "execution_kind": value.execution_kind.value,
                "priority": value.priority,
                "not_before": optional_datetime_encode(value.not_before),
                "current_attempt_id": value.current_attempt_id,
                "task_revision": value.task_revision,
                "specification": _encode_task_specification(value.specification),
                "task_spec_digest": value.task_spec_digest,
                "task_scope_digest": value.task_scope_digest,
                "attempt_count": value.attempt_count,
                "failure_streak": value.failure_streak,
                "fencing_epoch": value.fencing_epoch,
                "created_at": datetime_encode(value.created_at),
                "updated_at": datetime_encode(value.updated_at),
                "terminal_at": optional_datetime_encode(value.terminal_at),
                "supersedes_task_id": value.supersedes_task_id,
                "superseded_by_task_id": value.superseded_by_task_id,
                "latest_result_id": value.latest_result_id,
                "latest_control_id": value.latest_control_id,
                "latest_checkpoint_id": value.latest_checkpoint_id,
            },
        )
    )


def decode_graph_task(value: str) -> GraphTask:
    fields = record_fields(
        load_payload(value),
        "GraphTask",
        (
            "agent_id",
            "job_id",
            "task_id",
            "state",
            "role",
            "execution_kind",
            "priority",
            "not_before",
            "current_attempt_id",
            "task_revision",
            "specification",
            "task_spec_digest",
            "task_scope_digest",
            "attempt_count",
            "failure_streak",
            "fencing_epoch",
            "created_at",
            "updated_at",
            "terminal_at",
            "supersedes_task_id",
            "superseded_by_task_id",
            "latest_result_id",
            "latest_control_id",
            "latest_checkpoint_id",
        ),
    )
    return GraphTask(
        agent_id=text(fields["agent_id"], "task agent ID"),
        job_id=text(fields["job_id"], "task job ID"),
        task_id=text(fields["task_id"], "task ID"),
        state=_enum(fields["state"], TaskState, "task state"),
        role=_enum(fields["role"], TaskRole, "task role"),
        execution_kind=_enum(
            fields["execution_kind"], TaskExecutionKind, "task execution kind"
        ),
        priority=integer(fields["priority"], "task priority"),
        not_before=optional_datetime_decode(fields["not_before"]),
        current_attempt_id=optional_text(
            fields["current_attempt_id"], "task current attempt ID"
        ),
        task_revision=integer(fields["task_revision"], "task revision"),
        specification=_decode_task_specification(fields["specification"]),
        task_spec_digest=text(fields["task_spec_digest"], "task digest"),
        task_scope_digest=text(fields["task_scope_digest"], "task scope digest"),
        attempt_count=integer(fields["attempt_count"], "task attempt count"),
        failure_streak=integer(fields["failure_streak"], "task failure streak"),
        fencing_epoch=integer(fields["fencing_epoch"], "task fencing epoch"),
        created_at=datetime_decode(fields["created_at"]),
        updated_at=datetime_decode(fields["updated_at"]),
        terminal_at=optional_datetime_decode(fields["terminal_at"]),
        supersedes_task_id=optional_text(
            fields["supersedes_task_id"], "task supersedes ID"
        ),
        superseded_by_task_id=optional_text(
            fields["superseded_by_task_id"], "task superseded-by ID"
        ),
        latest_result_id=optional_text(
            fields["latest_result_id"], "task latest result ID"
        ),
        latest_control_id=optional_text(
            fields["latest_control_id"], "task latest control ID"
        ),
        latest_checkpoint_id=optional_text(
            fields["latest_checkpoint_id"], "task latest checkpoint ID"
        ),
    )


def encode_task_dependency(value: TaskDependency) -> str:
    return dump_payload(
        record(
            "TaskDependency",
            {
                "agent_id": value.agent_id,
                "job_id": value.job_id,
                "upstream_task_id": value.upstream_task_id,
                "downstream_task_id": value.downstream_task_id,
                "edge_kind": value.edge_kind.value,
                "created_at": datetime_encode(value.created_at),
                "creator_key": value.creator_key,
                "mutation_id": value.mutation_id,
            },
        )
    )


def decode_task_dependency(value: str) -> TaskDependency:
    fields = record_fields(
        load_payload(value),
        "TaskDependency",
        (
            "agent_id",
            "job_id",
            "upstream_task_id",
            "downstream_task_id",
            "edge_kind",
            "created_at",
            "creator_key",
            "mutation_id",
        ),
    )
    return TaskDependency(
        agent_id=text(fields["agent_id"], "edge agent ID"),
        job_id=text(fields["job_id"], "edge job ID"),
        upstream_task_id=text(fields["upstream_task_id"], "edge upstream ID"),
        downstream_task_id=text(fields["downstream_task_id"], "edge downstream ID"),
        edge_kind=_enum(fields["edge_kind"], EdgeKind, "edge kind"),
        created_at=datetime_decode(fields["created_at"]),
        creator_key=text(fields["creator_key"], "edge creator key"),
        mutation_id=optional_text(fields["mutation_id"], "edge mutation ID"),
    )


def encode_task_attempt(value: TaskAttempt) -> str:
    return dump_payload(
        record(
            "TaskAttempt",
            {
                "agent_id": value.agent_id,
                "job_id": value.job_id,
                "task_id": value.task_id,
                "attempt_id": value.attempt_id,
                "ordinal": value.ordinal,
                "fencing_epoch": value.fencing_epoch,
                "state": value.state.value,
                "claim_token": value.claim_token,
                "run_id": value.run_id,
                "lease_expires_at": optional_datetime_encode(value.lease_expires_at),
                "absolute_deadline_at": datetime_encode(value.absolute_deadline_at),
                "started_at": optional_datetime_encode(value.started_at),
                "heartbeat_at": optional_datetime_encode(value.heartbeat_at),
                "ended_at": optional_datetime_encode(value.ended_at),
                "execution_scope_digest": value.execution_scope_digest,
                "executor_id": value.executor_id,
                "error_code": value.error_code,
                "diagnostic": value.diagnostic,
                "reserved_budgets": [
                    _encode_budget_amount(item) for item in value.reserved_budgets
                ],
                "measured_usage": [
                    _encode_budget_amount(item) for item in value.measured_usage
                ],
                "loop_exit_id": value.loop_exit_id,
                "checkpoint_ids": list(value.checkpoint_ids),
                "result_id": value.result_id,
                "control_ids": list(value.control_ids),
                "artifact_ids": list(value.artifact_ids),
                "effect_receipt_ids": list(value.effect_receipt_ids),
            },
        )
    )


def decode_task_attempt(value: str) -> TaskAttempt:
    fields = record_fields(
        load_payload(value),
        "TaskAttempt",
        (
            "agent_id",
            "job_id",
            "task_id",
            "attempt_id",
            "ordinal",
            "fencing_epoch",
            "state",
            "claim_token",
            "run_id",
            "lease_expires_at",
            "absolute_deadline_at",
            "started_at",
            "heartbeat_at",
            "ended_at",
            "execution_scope_digest",
            "executor_id",
            "error_code",
            "diagnostic",
            "reserved_budgets",
            "measured_usage",
            "loop_exit_id",
            "checkpoint_ids",
            "result_id",
            "control_ids",
            "artifact_ids",
            "effect_receipt_ids",
        ),
    )
    return TaskAttempt(
        agent_id=text(fields["agent_id"], "attempt agent ID"),
        job_id=text(fields["job_id"], "attempt job ID"),
        task_id=text(fields["task_id"], "attempt task ID"),
        attempt_id=text(fields["attempt_id"], "attempt ID"),
        ordinal=integer(fields["ordinal"], "attempt ordinal"),
        fencing_epoch=integer(fields["fencing_epoch"], "attempt fence"),
        state=_enum(fields["state"], AttemptState, "attempt state"),
        claim_token=text(fields["claim_token"], "attempt claim token"),
        run_id=text(fields["run_id"], "attempt run ID"),
        lease_expires_at=optional_datetime_decode(fields["lease_expires_at"]),
        absolute_deadline_at=datetime_decode(fields["absolute_deadline_at"]),
        started_at=optional_datetime_decode(fields["started_at"]),
        heartbeat_at=optional_datetime_decode(fields["heartbeat_at"]),
        ended_at=optional_datetime_decode(fields["ended_at"]),
        execution_scope_digest=text(
            fields["execution_scope_digest"], "attempt scope digest"
        ),
        executor_id=text(fields["executor_id"], "attempt executor ID"),
        error_code=optional_text(fields["error_code"], "attempt error code"),
        diagnostic=optional_text(fields["diagnostic"], "attempt diagnostic"),
        reserved_budgets=tuple(
            _decode_budget_amount(item)
            for item in sequence(fields["reserved_budgets"], "attempt reserved budgets")
        ),
        measured_usage=tuple(
            _decode_budget_amount(item)
            for item in sequence(fields["measured_usage"], "attempt measured usage")
        ),
        loop_exit_id=optional_text(fields["loop_exit_id"], "attempt LoopExit ID"),
        checkpoint_ids=_text_tuple(fields["checkpoint_ids"], "attempt checkpoint IDs"),
        result_id=optional_text(fields["result_id"], "attempt result ID"),
        control_ids=_text_tuple(fields["control_ids"], "attempt control IDs"),
        artifact_ids=_text_tuple(fields["artifact_ids"], "attempt artifact IDs"),
        effect_receipt_ids=_text_tuple(
            fields["effect_receipt_ids"], "attempt receipt IDs"
        ),
    )


def encode_task_result(value: TaskResult) -> str:
    return dump_payload(
        record(
            "TaskResult",
            {
                "agent_id": value.agent_id,
                "job_id": value.job_id,
                "task_id": value.task_id,
                "result_id": value.result_id,
                "attempt_id": value.attempt_id,
                "run_id": value.run_id,
                "result_kind": value.result_kind,
                "schema_digest": value.schema_digest,
                "payload": plain_encode(value.payload),
                "summary": value.summary,
                "sensitivity": value.sensitivity.value,
                "provenance": plain_encode(value.provenance),
                "artifact_ids": list(value.artifact_ids),
                "effect_receipt_ids": list(value.effect_receipt_ids),
                "verification": plain_encode(value.verification),
                "residual_risk": value.residual_risk,
                "downstream_constraints": plain_encode(value.downstream_constraints),
                "completed_at": datetime_encode(value.completed_at),
                "result_digest": value.result_digest,
            },
        )
    )


def decode_task_result(value: str) -> TaskResult:
    fields = record_fields(
        load_payload(value),
        "TaskResult",
        (
            "agent_id",
            "job_id",
            "task_id",
            "result_id",
            "attempt_id",
            "run_id",
            "result_kind",
            "schema_digest",
            "payload",
            "summary",
            "sensitivity",
            "provenance",
            "artifact_ids",
            "effect_receipt_ids",
            "verification",
            "residual_risk",
            "downstream_constraints",
            "completed_at",
            "result_digest",
        ),
    )
    return TaskResult(
        agent_id=text(fields["agent_id"], "result agent ID"),
        job_id=text(fields["job_id"], "result job ID"),
        task_id=text(fields["task_id"], "result task ID"),
        result_id=text(fields["result_id"], "result ID"),
        attempt_id=text(fields["attempt_id"], "result attempt ID"),
        run_id=text(fields["run_id"], "result run ID"),
        result_kind=text(fields["result_kind"], "result kind"),
        schema_digest=text(fields["schema_digest"], "result schema digest"),
        payload=_mapping(fields["payload"], "result payload"),
        summary=text(fields["summary"], "result summary"),
        sensitivity=_enum(
            fields["sensitivity"], ModelSensitivity, "result sensitivity"
        ),
        provenance=_mapping(fields["provenance"], "result provenance"),
        artifact_ids=_text_tuple(fields["artifact_ids"], "result artifact IDs"),
        effect_receipt_ids=_text_tuple(
            fields["effect_receipt_ids"], "result receipt IDs"
        ),
        verification=_mapping(fields["verification"], "result verification"),
        residual_risk=optional_text(fields["residual_risk"], "result residual risk"),
        downstream_constraints=_mapping(
            fields["downstream_constraints"], "result downstream constraints"
        ),
        completed_at=datetime_decode(fields["completed_at"]),
        result_digest=text(fields["result_digest"], "result digest"),
    )


def encode_task_checkpoint(value: TaskCheckpoint) -> str:
    return dump_payload(
        record(
            "TaskCheckpoint",
            {
                "agent_id": value.agent_id,
                "job_id": value.job_id,
                "task_id": value.task_id,
                "attempt_id": value.attempt_id,
                "checkpoint_id": value.checkpoint_id,
                "fencing_epoch": value.fencing_epoch,
                "ordinal": value.ordinal,
                "milestone": value.milestone,
                "payload": plain_encode(value.payload),
                "created_at": datetime_encode(value.created_at),
                "payload_digest": value.payload_digest,
            },
        )
    )


def decode_task_checkpoint(value: str) -> TaskCheckpoint:
    fields = record_fields(
        load_payload(value),
        "TaskCheckpoint",
        (
            "agent_id",
            "job_id",
            "task_id",
            "attempt_id",
            "checkpoint_id",
            "fencing_epoch",
            "ordinal",
            "milestone",
            "payload",
            "created_at",
            "payload_digest",
        ),
    )
    return TaskCheckpoint(
        agent_id=text(fields["agent_id"], "checkpoint agent ID"),
        job_id=text(fields["job_id"], "checkpoint job ID"),
        task_id=text(fields["task_id"], "checkpoint task ID"),
        attempt_id=text(fields["attempt_id"], "checkpoint attempt ID"),
        checkpoint_id=text(fields["checkpoint_id"], "checkpoint ID"),
        fencing_epoch=integer(fields["fencing_epoch"], "checkpoint fence"),
        ordinal=integer(fields["ordinal"], "checkpoint ordinal"),
        milestone=text(fields["milestone"], "checkpoint milestone"),
        payload=_mapping(fields["payload"], "checkpoint payload"),
        created_at=datetime_decode(fields["created_at"]),
        payload_digest=text(fields["payload_digest"], "checkpoint digest"),
    )


def encode_task_comment(value: TaskComment) -> str:
    return dump_payload(
        record(
            "TaskComment",
            {
                "agent_id": value.agent_id,
                "job_id": value.job_id,
                "task_id": value.task_id,
                "comment_id": value.comment_id,
                "author_kind": value.author_kind,
                "author_id": value.author_id,
                "sensitivity": value.sensitivity.value,
                "body": value.body,
                "created_at": datetime_encode(value.created_at),
                "body_digest": value.body_digest,
            },
        )
    )


def decode_task_comment(value: str) -> TaskComment:
    fields = record_fields(
        load_payload(value),
        "TaskComment",
        (
            "agent_id",
            "job_id",
            "task_id",
            "comment_id",
            "author_kind",
            "author_id",
            "sensitivity",
            "body",
            "created_at",
            "body_digest",
        ),
    )
    return TaskComment(
        agent_id=text(fields["agent_id"], "comment agent ID"),
        job_id=text(fields["job_id"], "comment job ID"),
        task_id=text(fields["task_id"], "comment task ID"),
        comment_id=text(fields["comment_id"], "comment ID"),
        author_kind=text(fields["author_kind"], "comment author kind"),
        author_id=text(fields["author_id"], "comment author ID"),
        sensitivity=_enum(
            fields["sensitivity"], ModelSensitivity, "comment sensitivity"
        ),
        body=text(fields["body"], "comment body"),
        created_at=datetime_decode(fields["created_at"]),
        body_digest=text(fields["body_digest"], "comment body digest"),
    )


def encode_task_control(value: TaskControl) -> str:
    return dump_payload(
        record(
            "TaskControl",
            {
                "agent_id": value.agent_id,
                "job_id": value.job_id,
                "task_id": value.task_id,
                "control_id": value.control_id,
                "kind": value.kind.value,
                "state": value.state.value,
                "requesting_attempt_id": value.requesting_attempt_id,
                "payload": plain_encode(value.payload),
                "created_at": datetime_encode(value.created_at),
                "payload_digest": value.payload_digest,
                "resolved_at": optional_datetime_encode(value.resolved_at),
                "resolved_by_kind": value.resolved_by_kind,
                "resolved_by_id": value.resolved_by_id,
                "resolution": (
                    None if value.resolution is None else plain_encode(value.resolution)
                ),
            },
        )
    )


def decode_task_control(value: str) -> TaskControl:
    fields = record_fields(
        load_payload(value),
        "TaskControl",
        (
            "agent_id",
            "job_id",
            "task_id",
            "control_id",
            "kind",
            "state",
            "requesting_attempt_id",
            "payload",
            "created_at",
            "payload_digest",
            "resolved_at",
            "resolved_by_kind",
            "resolved_by_id",
            "resolution",
        ),
    )
    resolution = fields["resolution"]
    return TaskControl(
        agent_id=text(fields["agent_id"], "control agent ID"),
        job_id=text(fields["job_id"], "control job ID"),
        task_id=text(fields["task_id"], "control task ID"),
        control_id=text(fields["control_id"], "control ID"),
        kind=_enum(fields["kind"], ControlKind, "control kind"),
        state=_enum(fields["state"], ControlState, "control state"),
        requesting_attempt_id=optional_text(
            fields["requesting_attempt_id"], "control requesting attempt ID"
        ),
        payload=_mapping(fields["payload"], "control payload"),
        created_at=datetime_decode(fields["created_at"]),
        payload_digest=text(fields["payload_digest"], "control payload digest"),
        resolved_at=optional_datetime_decode(fields["resolved_at"]),
        resolved_by_kind=optional_text(
            fields["resolved_by_kind"], "control resolver kind"
        ),
        resolved_by_id=optional_text(fields["resolved_by_id"], "control resolver ID"),
        resolution=(
            None if resolution is None else _mapping(resolution, "control resolution")
        ),
    )


def encode_graph_mutation(value: GraphMutation) -> str:
    return dump_payload(
        record(
            "GraphMutation",
            {
                "agent_id": value.agent_id,
                "job_id": value.job_id,
                "mutation_id": value.mutation_id,
                "actor_kind": value.actor_kind,
                "actor_key": value.actor_key,
                "idempotency_key": value.idempotency_key,
                "payload_digest": value.payload_digest,
                "expected_revision": value.expected_revision,
                "committed_revision": value.committed_revision,
                "decision": value.decision.value,
                "created_at": datetime_encode(value.created_at),
                "resulting_task_ids": list(value.resulting_task_ids),
                "resulting_edges": [list(item) for item in value.resulting_edges],
                "failure_code": value.failure_code,
                "creator_task_id": value.creator_task_id,
                "creator_attempt_id": value.creator_attempt_id,
            },
        )
    )


def decode_graph_mutation(value: str) -> GraphMutation:
    fields = record_fields(
        load_payload(value),
        "GraphMutation",
        (
            "agent_id",
            "job_id",
            "mutation_id",
            "actor_kind",
            "actor_key",
            "idempotency_key",
            "payload_digest",
            "expected_revision",
            "committed_revision",
            "decision",
            "created_at",
            "resulting_task_ids",
            "resulting_edges",
            "failure_code",
            "creator_task_id",
            "creator_attempt_id",
        ),
    )
    return GraphMutation(
        agent_id=text(fields["agent_id"], "mutation agent ID"),
        job_id=text(fields["job_id"], "mutation job ID"),
        mutation_id=text(fields["mutation_id"], "mutation ID"),
        actor_kind=text(fields["actor_kind"], "mutation actor kind"),
        actor_key=text(fields["actor_key"], "mutation actor key"),
        idempotency_key=text(fields["idempotency_key"], "mutation idempotency key"),
        payload_digest=text(fields["payload_digest"], "mutation payload digest"),
        expected_revision=integer(
            fields["expected_revision"], "mutation expected revision"
        ),
        committed_revision=integer(
            fields["committed_revision"], "mutation committed revision"
        ),
        decision=_enum(fields["decision"], MutationDecision, "mutation decision"),
        created_at=datetime_decode(fields["created_at"]),
        resulting_task_ids=_text_tuple(
            fields["resulting_task_ids"], "mutation resulting task IDs"
        ),
        resulting_edges=_pair_tuple(
            fields["resulting_edges"], "mutation resulting edges"
        ),
        failure_code=optional_text(fields["failure_code"], "mutation failure code"),
        creator_task_id=optional_text(
            fields["creator_task_id"], "mutation creator task ID"
        ),
        creator_attempt_id=optional_text(
            fields["creator_attempt_id"], "mutation creator attempt ID"
        ),
    )


def encode_graph_event_payload(*, kind: str, payload: Mapping[str, object]) -> str:
    return dump_payload(
        record(
            "GraphEventPayload",
            {"kind": kind, "payload": plain_encode(payload)},
        )
    )


def decode_graph_event(
    value: str,
    *,
    event_id: int,
    agent_id: str,
    job_id: str,
    task_id: str | None,
    attempt_id: str | None,
    kind: str,
    created_at: datetime,
) -> GraphEvent:
    fields = record_fields(
        load_payload(value), "GraphEventPayload", ("kind", "payload")
    )
    encoded_kind = text(fields["kind"], "event kind")
    if encoded_kind != kind:
        raise ValueError("stored event kind projection is inconsistent")
    return GraphEvent(
        event_id=event_id,
        agent_id=agent_id,
        job_id=job_id,
        task_id=task_id,
        attempt_id=attempt_id,
        kind=kind,
        created_at=created_at,
        payload=_mapping(fields["payload"], "event payload"),
    )


def encode_graph_job_delivery(value: GraphJobDelivery) -> str:
    if not isinstance(value, GraphJobDelivery):
        raise TypeError("graph delivery codec requires GraphJobDelivery")
    return dump_payload(
        record(
            "Delivery",
            {
                "conversation_id": value.conversation_id,
                "subject_kind": "graph_job",
                "subject_id": value.job_id,
                "logical_key": value.logical_key,
                "target": encode_conversation_inbox_target(value.target),
                "outcome": encode_outcome_reference(value.outcome),
                "visibility_state": value.visibility_state.value,
                "acknowledged_at": None,
                "blocked_reason_code": value.blocked_reason_code,
                "created_at": datetime_encode(value.created_at),
                "updated_at": datetime_encode(value.updated_at),
                "migration_provenance": plain_encode(value.migration_provenance),
            },
        )
    )


def decode_graph_job_delivery(
    value: str,
    *,
    agent_id: str,
    delivery_id: str,
    conversation_id: str,
    subject_kind: str,
    subject_id: str,
    logical_key: str,
    state: str,
) -> GraphJobDelivery | dict[str, object]:
    """Strictly validate the current revision-2 delivery record shape."""

    fields = record_fields(
        load_payload(value),
        "Delivery",
        (
            "conversation_id",
            "subject_kind",
            "subject_id",
            "logical_key",
            "target",
            "outcome",
            "visibility_state",
            "acknowledged_at",
            "blocked_reason_code",
            "created_at",
            "updated_at",
            "migration_provenance",
        ),
    )
    decoded_conversation = text(fields["conversation_id"], "delivery conversation")
    decoded_subject_kind = text(fields["subject_kind"], "delivery subject kind")
    decoded_subject_id = text(fields["subject_id"], "delivery subject ID")
    decoded_logical_key = text(fields["logical_key"], "delivery logical key")
    decoded_state = _enum(fields["visibility_state"], DeliveryState, "delivery state")
    if (
        decoded_conversation != conversation_id
        or decoded_subject_kind != subject_kind
        or decoded_subject_id != subject_id
        or decoded_logical_key != logical_key
        or decoded_state.value != state
    ):
        raise ValueError("graph delivery projection differs from its payload")
    if decoded_subject_kind not in {
        "routine_occurrence",
        "graph_job",
        "graph_attention",
    }:
        raise ValueError("graph delivery subject kind is invalid")
    target = decode_conversation_inbox_target(fields["target"])
    if target.conversation_id != conversation_id:
        raise ValueError("graph delivery target conversation differs")
    outcome = decode_outcome_reference(fields["outcome"])
    optional_datetime_decode(fields["acknowledged_at"])
    optional_text(fields["blocked_reason_code"], "delivery blocked reason")
    created_at = datetime_decode(fields["created_at"])
    updated_at = datetime_decode(fields["updated_at"])
    provenance = _mapping(
        fields["migration_provenance"], "delivery migration provenance"
    )
    if decoded_subject_kind == "graph_job":
        return GraphJobDelivery(
            delivery_id=delivery_id,
            agent_id=agent_id,
            conversation_id=decoded_conversation,
            job_id=decoded_subject_id,
            logical_key=decoded_logical_key,
            target=target,
            outcome=outcome,
            visibility_state=decoded_state,
            blocked_reason_code=optional_text(
                fields["blocked_reason_code"], "delivery blocked reason"
            ),
            created_at=created_at,
            updated_at=updated_at,
            migration_provenance=FrozenJsonObject.from_mapping(provenance),
        )
    return {
        "agent_id": agent_id,
        "delivery_id": delivery_id,
        "conversation_id": decoded_conversation,
        "subject_kind": decoded_subject_kind,
        "subject_id": decoded_subject_id,
        "logical_key": decoded_logical_key,
        "migration_provenance": provenance,
    }


__all__ = [
    "decode_graph_event",
    "decode_graph_job_delivery",
    "decode_graph_job",
    "decode_graph_mutation",
    "decode_graph_task",
    "decode_job_graph",
    "decode_task_attempt",
    "decode_task_checkpoint",
    "decode_task_comment",
    "decode_task_control",
    "decode_task_dependency",
    "decode_task_result",
    "encode_graph_event_payload",
    "encode_graph_job",
    "encode_graph_job_delivery",
    "encode_graph_mutation",
    "encode_graph_task",
    "encode_job_graph",
    "encode_task_attempt",
    "encode_task_checkpoint",
    "encode_task_comment",
    "encode_task_control",
    "encode_task_dependency",
    "encode_task_result",
]
