"""Encode values crossing the host socket as tagged JSON with exact round trips.

Every JSON object on the wire is one ``{"$type": tag, "v": value}`` envelope, so
decoding never guesses: ``str | Path`` unions and ``Mapping[str, object]``
fields decode to the type that was sent. Records with a strict SQLite codec
reuse it; the remaining frozen dataclasses use one field-driven codec whose
decode re-runs ``__post_init__``. There is no fallback for unknown types.
"""

from __future__ import annotations

import base64
import dataclasses
import functools
import json
import math
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Final, TypeVar, overload

from .._json import FrozenJsonObject
from ..adapters.folder import FolderSource
from ..adapters.mcp import (
    MCPAuthentication,
    MCPAuthenticationMode,
    MCPBindingStatus,
    MCPCompletionSemantics,
    MCPInspectedTool,
    MCPServerBinding,
    MCPServerInspection,
    MCPToolSelection,
)
from ..adapters.models import SourceRegistration
from ..adapters.postgresql import (
    PostgreSQLProbeResult,
    PostgreSQLProbeSchema,
    PostgreSQLSource,
)
from ..adapters.postgresql_write import RelationalUpdateReadiness
from ..adapters.sqlite import SQLiteSource
from ..artifacts.models import (
    ArtifactAuthorship,
    ArtifactDeliveryReceipt,
    ArtifactDestination,
    ArtifactDestinationKind,
    ArtifactPayload,
    ArtifactRef,
    DestinationAuthorization,
    DestinationAvailability,
)
from ..capabilities import AccessMode, AutomationEligibility, OperationalEffect
from ..catalog.models import CatalogResource, CatalogSummary, ResourceKind, Sensitivity
from ..distribution.models import (
    Delivery,
    DeliveryInspection,
    DeliveryState,
    DeliverySubjectKind,
    DeliveryView,
    DistributionDestination,
    DistributionDestinationState,
    OutcomeArtifactReference,
    OutcomeConclusionKind,
    OutcomeContract,
    OutcomeState,
)
from ..errors import ErrorRetryability, StateCompatibilityCode
from ..hosting.home_upgrade import AgentHomeStatus
from ..hosting.sources import SourceEditResult
from ..jobs.graph.models import (
    BudgetLedger,
    GraphEvent,
    GraphInspection,
    GraphJob,
    GraphMutation,
    GraphState,
    GraphTask,
    JobGraph,
    TaskAttempt,
    TaskCheckpoint,
    TaskComment,
    TaskControl,
    TaskDependency,
    TaskResult,
    TaskState,
)
from ..jobs.owner import GraphBlockerProjection
from ..jobs.projections import (
    DependencyProjection,
    GraphBoardProjection,
    GraphDiagnostics,
    GraphTimelinePage,
    KanbanColumn,
)
from ..llm.models import (
    CanonicalMessage,
    ModelProfile,
    ModelSensitivity,
    ToolCall,
    ToolResultBlock,
)
from ..llm.routing import ModelRoute, ModelRouteCandidate, RetryPolicy
from ..loop.models import ConversationRun, LoopExit, RunInput, Transcript
from ..observation import AgentEvent, AgentEventKind
from ..routines.models import (
    AmbiguousTimePolicy,
    CalendarDaySelector,
    CalendarSchedule,
    IntervalSchedule,
    MisfirePolicy,
    NonexistentTimePolicy,
    OnceSchedule,
    ReportingMode,
    RequestedCapabilityGrant,
    ResourceRevisionPrecheck,
    RoutineOccurrence,
    RoutineState,
    ScheduledRoutine,
    ScheduledRoutineDraft,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
    ScheduleKind,
)
from ..security.secrets import SecretReference
from ..semantics import (
    SemanticAnnotation,
    SemanticAnnotationState,
    SemanticAnnotationView,
    SemanticKind,
)
from ..skills.store import Skill, SkillSummary
from ..storage.sqlite_codecs import (
    decode_delivery,
    decode_loop_exit,
    decode_mcp_binding,
    decode_message,
    decode_outcome_contract,
    decode_receipt,
    decode_relational_write_scope,
    decode_routine_occurrence,
    decode_run_input,
    decode_scheduled_routine,
    decode_semantic_annotation,
    decode_source,
    decode_source_read_scope,
    encode_delivery,
    encode_loop_exit,
    encode_mcp_binding,
    encode_message,
    encode_outcome_contract,
    encode_receipt,
    encode_relational_write_scope,
    encode_routine_occurrence,
    encode_run_input,
    encode_scheduled_routine,
    encode_semantic_annotation,
    encode_source,
    encode_source_read_scope,
)
from ..storage.sqlite_codecs.artifacts import (
    decode_artifact_ref,
    decode_delivery_receipt,
    encode_artifact_ref,
    encode_delivery_receipt,
)
from ..storage.sqlite_codecs.common import JsonValue, dump_payload, load_payload
from ..storage.sqlite_codecs.graph import (
    decode_graph_job,
    decode_graph_mutation,
    decode_graph_task,
    decode_job_graph,
    decode_task_attempt,
    decode_task_checkpoint,
    decode_task_comment,
    decode_task_control,
    decode_task_dependency,
    decode_task_result,
    encode_graph_job,
    encode_graph_mutation,
    encode_graph_task,
    encode_job_graph,
    encode_task_attempt,
    encode_task_checkpoint,
    encode_task_comment,
    encode_task_control,
    encode_task_dependency,
    encode_task_result,
)
from ..storage.sqlite_records import (
    EffectReceipt,
    EffectResolutionDecision,
    RelationalWriteScope,
    SourcePermissionResource,
    SourcePermissionsInspection,
    SourcePermissionsPreview,
    SourcePermissionState,
    SourcePermissionSummary,
    SourceReadMode,
    SourceReadScope,
)
from ..work_board import WorkBoard, WorkCard, WorkColumn, WorkKind
from ..workspace import LocalFileAccess, LocalWorkspace

_T = TypeVar("_T")
_TAG: Final = "$type"
_VALUE: Final = "v"


@dataclass(frozen=True, slots=True)
class _Codec:
    cls: type
    tag: str
    encode: Callable[[Any], JsonValue]
    decode: Callable[[JsonValue], Any]
    # Wire field names for a generic dataclass; None for every other codec.
    fields: tuple[str, ...] | None = None


_BY_TYPE: dict[type, _Codec] = {}
_BY_TAG: dict[str, _Codec] = {}


def encode(value: object) -> JsonValue:
    """Return the JSON wire form of one value, or raise TypeError if it has none."""

    if value is None or type(value) in (bool, int, str):
        return value  # type: ignore[return-value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("a non-finite float cannot cross the host socket")
        return value
    if type(value) is tuple:
        return [encode(item) for item in value]
    codec = _BY_TYPE.get(type(value))
    if codec is None:
        raise TypeError(f"no host wire codec for {_name(type(value))}")
    return {_TAG: codec.tag, _VALUE: codec.encode(value)}


@overload
def decode(payload: JsonValue, expected: type[_T]) -> _T: ...


@overload
def decode(payload: JsonValue, expected: object = ...) -> Any: ...


def decode(payload: JsonValue, expected: object = object) -> Any:
    """Decode one wire value and require it to match the expected annotation."""

    value = _from_wire(payload)
    if not _conforms(value, expected):
        raise TypeError(f"host wire value {_name(type(value))} is not {expected!r}")
    return value


def supports(annotation: object) -> bool:
    """Return whether every value admitted by one annotation has a wire codec."""

    return _supports(annotation, set())


def _from_wire(payload: JsonValue) -> object:
    if payload is None or type(payload) in (bool, int, str):
        return payload
    if type(payload) is float:
        if not math.isfinite(payload):
            raise ValueError("a non-finite float cannot cross the host socket")
        return payload
    if type(payload) is list:
        return tuple(_from_wire(item) for item in payload)
    if type(payload) is dict:
        if set(payload) != {_TAG, _VALUE} or type(payload[_TAG]) is not str:
            raise ValueError("a host wire object must be one tagged value")
        codec = _BY_TAG.get(payload[_TAG])
        if codec is None:
            raise TypeError(f"unknown host wire type: {payload[_TAG]}")
        return codec.decode(payload[_VALUE])
    raise TypeError(f"{type(payload).__name__} is not a JSON value")


def _conforms(value: object, expected: object) -> bool:
    if expected is object or expected is Any:
        return True
    if expected is None or expected is type(None):
        return value is None
    origin = typing.get_origin(expected)
    arguments = typing.get_args(expected)
    if origin in (typing.Union, types.UnionType):
        return any(_conforms(value, member) for member in arguments)
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return all(_conforms(item, arguments[0]) for item in value)
        return len(value) == len(arguments) and all(
            _conforms(item, member) for item, member in zip(value, arguments)
        )
    if origin is frozenset:
        return isinstance(value, frozenset) and all(
            _conforms(item, arguments[0]) for item in value
        )
    if origin in (dict, Mapping):
        return isinstance(value, origin) and all(
            _conforms(key, arguments[0]) and _conforms(item, arguments[1])
            for key, item in value.items()
        )
    if expected is float:
        return type(value) in (int, float)
    if expected in (bool, int, str, bytes):
        return type(value) is expected
    return isinstance(expected, type) and isinstance(value, expected)


def _supports(annotation: object, seen: set[type]) -> bool:
    if annotation in (object, None, type(None), bool, int, float, str, Path):
        return True
    origin = typing.get_origin(annotation)
    if origin is not None:
        return origin in (
            tuple,
            frozenset,
            dict,
            Mapping,
            typing.Union,
            types.UnionType,
        ) and all(
            _supports(member, seen)
            for member in typing.get_args(annotation)
            if member is not Ellipsis
        )
    if not isinstance(annotation, type):
        return False
    if getattr(annotation, "_is_protocol", False):
        return any(issubclass(cls, annotation) for cls in _BY_TYPE)
    codec = _BY_TYPE.get(annotation)
    if codec is None:
        return False
    if codec.fields is None or annotation in seen:
        return True
    seen.add(annotation)
    hints = _type_hints(annotation)
    return all(_supports(hints[name], seen) for name in codec.fields)


def _register(
    cls: type,
    encode_value: Callable[[Any], JsonValue],
    decode_value: Callable[[JsonValue], Any],
    *,
    tag: str | None = None,
    fields: tuple[str, ...] | None = None,
) -> None:
    codec = _Codec(
        cls, tag or f"daita.{cls.__name__}", encode_value, decode_value, fields
    )
    if cls in _BY_TYPE or codec.tag in _BY_TAG:
        raise RuntimeError(f"duplicate host wire codec: {codec.tag}")
    _BY_TYPE[cls] = codec
    _BY_TAG[codec.tag] = codec


def _register_enum(cls: type[Enum]) -> None:
    def decode_member(payload: JsonValue) -> Enum:
        return cls(_text(payload, cls.__name__))

    _register(cls, lambda member: _text(member.value, cls.__name__), decode_member)


def _register_dataclass(
    cls: type, *, process_local: frozenset[str] = frozenset()
) -> None:
    params = getattr(cls, "__dataclass_params__", None)
    if params is None or not params.frozen:
        raise TypeError(f"{_name(cls)} is not a frozen dataclass")
    names = tuple(
        item.name for item in dataclasses.fields(cls) if item.name not in process_local
    )

    def encode_fields(value: object) -> JsonValue:
        return {name: encode(getattr(value, name)) for name in names}

    def decode_fields(payload: JsonValue) -> object:
        values = _object(payload, names, cls.__name__)
        hints = _type_hints(cls)
        # Construction re-runs the record's own __post_init__ validation.
        return cls(**{name: decode(values[name], hints[name]) for name in names})

    _register(cls, encode_fields, decode_fields, fields=names)


def _register_stored(
    cls: type,
    encode_record: Callable[[Any], str],
    decode_record: Callable[..., Any],
    *keys: str,
) -> None:
    """Reuse one SQLite record codec; its key columns travel beside the record."""

    names = ("record", *keys)

    def encode_value(value: object) -> JsonValue:
        return {
            "record": load_payload(encode_record(value)),
            **{key: _text(getattr(value, key), key) for key in keys},
        }

    def decode_value(payload: JsonValue) -> object:
        values = _object(payload, names, cls.__name__)
        return decode_record(
            dump_payload(values["record"]),
            **{key: _text(values[key], key) for key in keys},
        )

    _register(cls, encode_value, decode_value)


def _encode_datetime(value: datetime) -> JsonValue:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("a naive datetime cannot cross the host socket")
    return value.isoformat()


def _decode_datetime(payload: JsonValue) -> datetime:
    value = datetime.fromisoformat(_text(payload, "datetime"))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("a naive datetime cannot cross the host socket")
    return value


def _encode_decimal(value: Decimal) -> JsonValue:
    if not value.is_finite():
        raise ValueError("a non-finite decimal cannot cross the host socket")
    return str(value)


def _decode_decimal(payload: JsonValue) -> Decimal:
    try:
        value = Decimal(_text(payload, "decimal"))
    except InvalidOperation:
        raise ValueError("host wire decimal is invalid") from None
    if not value.is_finite():
        raise ValueError("a non-finite decimal cannot cross the host socket")
    return value


def _decode_timedelta(payload: JsonValue) -> timedelta:
    if type(payload) is not int:
        raise ValueError("host wire timedelta must be integer microseconds")
    return timedelta(microseconds=payload)


def _encode_frozenset(value: frozenset[object]) -> JsonValue:
    items = [encode(item) for item in value]
    return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))


def _encode_dict(value: dict[object, object]) -> JsonValue:
    if any(type(key) is not str for key in value):
        raise TypeError("host wire mapping keys must be text")
    return {key: encode(item) for key, item in value.items()}  # type: ignore[misc]


def _decode_dict(payload: JsonValue) -> dict[str, object]:
    if type(payload) is not dict:
        raise ValueError("host wire mapping must be an object")
    return {key: _from_wire(item) for key, item in payload.items()}


def _encode_mapping_proxy(value: types.MappingProxyType[object, object]) -> JsonValue:
    # Pairs, because read-only record mappings may be keyed by enums.
    return [[encode(key), encode(item)] for key, item in value.items()]


def _decode_mapping_proxy(payload: JsonValue) -> types.MappingProxyType[object, object]:
    if type(payload) is not list or any(
        type(pair) is not list or len(pair) != 2 for pair in payload
    ):
        raise ValueError("host wire read-only mapping must be an array of pairs")
    return types.MappingProxyType(
        {_from_wire(pair[0]): _from_wire(pair[1]) for pair in payload}  # type: ignore[index]
    )


def _decode_frozen_json(payload: JsonValue) -> FrozenJsonObject:
    if type(payload) is not dict:
        raise ValueError("host wire JSON object must be an object")
    return FrozenJsonObject.from_mapping(payload)


def _decode_frozenset(payload: JsonValue) -> frozenset[object]:
    if type(payload) is not list:
        raise ValueError("host wire set must be an array")
    return frozenset(_from_wire(item) for item in payload)


def _object(
    payload: JsonValue, names: tuple[str, ...], label: str
) -> dict[str, JsonValue]:
    if type(payload) is not dict or set(payload) != set(names):
        raise ValueError(f"host wire {label} fields are invalid")
    return payload


def _text(value: object, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"host wire {label} must be text")
    return value


@functools.cache
def _type_hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def _name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


_register(datetime, _encode_datetime, _decode_datetime, tag="datetime")
_register(Decimal, _encode_decimal, _decode_decimal, tag="decimal")
_register(type(Path()), str, lambda payload: Path(_text(payload, "path")), tag="path")
_register(
    timedelta,
    lambda value: value // timedelta(microseconds=1),
    _decode_timedelta,
    tag="timedelta",
)
_register(
    bytes,
    lambda value: base64.b64encode(value).decode("ascii"),
    lambda payload: base64.b64decode(_text(payload, "bytes"), validate=True),
    tag="bytes",
)
_register(frozenset, _encode_frozenset, _decode_frozenset, tag="frozenset")
_register(dict, _encode_dict, _decode_dict, tag="dict")
_register(
    types.MappingProxyType, _encode_mapping_proxy, _decode_mapping_proxy, tag="mapping"
)
_register(FrozenJsonObject, lambda value: value.to_dict(), _decode_frozen_json)

for _enum in (
    AccessMode,
    AgentEventKind,
    AmbiguousTimePolicy,
    ArtifactAuthorship,
    ArtifactDestinationKind,
    AutomationEligibility,
    CalendarDaySelector,
    DeliveryState,
    DeliverySubjectKind,
    DestinationAuthorization,
    DestinationAvailability,
    DistributionDestinationState,
    EffectResolutionDecision,
    ErrorRetryability,
    GraphState,
    LocalFileAccess,
    MCPAuthenticationMode,
    MCPCompletionSemantics,
    MisfirePolicy,
    ModelSensitivity,
    NonexistentTimePolicy,
    OperationalEffect,
    OutcomeConclusionKind,
    OutcomeState,
    ReportingMode,
    ResourceKind,
    RoutineState,
    ScheduleKind,
    SemanticAnnotationState,
    SemanticKind,
    Sensitivity,
    SourceReadMode,
    StateCompatibilityCode,
    TaskState,
    WorkColumn,
    WorkKind,
):
    _register_enum(_enum)

_register_stored(SourceRegistration, encode_source, decode_source)
_register_stored(LoopExit, encode_loop_exit, decode_loop_exit)
_register_stored(CanonicalMessage, encode_message, decode_message)
_register_stored(RunInput, encode_run_input, decode_run_input)
_register_stored(EffectReceipt, encode_receipt, decode_receipt)
_register_stored(
    SemanticAnnotation, encode_semantic_annotation, decode_semantic_annotation
)
_register_stored(GraphJob, encode_graph_job, decode_graph_job)
_register_stored(JobGraph, encode_job_graph, decode_job_graph)
_register_stored(GraphTask, encode_graph_task, decode_graph_task)
_register_stored(TaskDependency, encode_task_dependency, decode_task_dependency)
_register_stored(TaskAttempt, encode_task_attempt, decode_task_attempt)
_register_stored(TaskResult, encode_task_result, decode_task_result)
_register_stored(TaskCheckpoint, encode_task_checkpoint, decode_task_checkpoint)
_register_stored(TaskComment, encode_task_comment, decode_task_comment)
_register_stored(TaskControl, encode_task_control, decode_task_control)
_register_stored(GraphMutation, encode_graph_mutation, decode_graph_mutation)
_register_stored(
    ScheduledRoutine,
    encode_scheduled_routine,
    decode_scheduled_routine,
    "agent_id",
    "routine_id",
)
_register_stored(
    RoutineOccurrence,
    encode_routine_occurrence,
    decode_routine_occurrence,
    "agent_id",
    "occurrence_id",
)
_register_stored(Delivery, encode_delivery, decode_delivery, "agent_id", "delivery_id")
_register_stored(
    MCPServerBinding, encode_mcp_binding, decode_mcp_binding, "agent_id", "binding_id"
)
_register_stored(
    SourceReadScope,
    encode_source_read_scope,
    decode_source_read_scope,
    "agent_id",
    "source_id",
)
_register_stored(
    RelationalWriteScope,
    encode_relational_write_scope,
    decode_relational_write_scope,
    "agent_id",
    "source_id",
    "resource_id",
    "authorization_fingerprint",
)
_register(OutcomeContract, encode_outcome_contract, decode_outcome_contract)
_register(ArtifactRef, encode_artifact_ref, decode_artifact_ref)
_register(ArtifactDeliveryReceipt, encode_delivery_receipt, decode_delivery_receipt)

for _record in (
    AgentEvent,
    AgentHomeStatus,
    ArtifactDestination,
    ArtifactPayload,
    BudgetLedger,
    CalendarSchedule,
    CatalogResource,
    CatalogSummary,
    ConversationRun,
    DeliveryInspection,
    DeliveryView,
    DependencyProjection,
    DistributionDestination,
    FolderSource,
    GraphBlockerProjection,
    GraphBoardProjection,
    GraphDiagnostics,
    GraphEvent,
    GraphInspection,
    GraphTimelinePage,
    IntervalSchedule,
    KanbanColumn,
    LocalWorkspace,
    MCPAuthentication,
    MCPBindingStatus,
    MCPInspectedTool,
    MCPServerInspection,
    MCPToolSelection,
    ModelProfile,
    ModelRoute,
    ModelRouteCandidate,
    OnceSchedule,
    OutcomeArtifactReference,
    PostgreSQLProbeResult,
    PostgreSQLProbeSchema,
    RelationalUpdateReadiness,
    RequestedCapabilityGrant,
    ResourceRevisionPrecheck,
    RetryPolicy,
    SQLiteSource,
    ScheduledRoutineDraft,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
    SecretReference,
    SemanticAnnotationView,
    Skill,
    SkillSummary,
    SourceEditResult,
    SourcePermissionResource,
    SourcePermissionState,
    SourcePermissionSummary,
    SourcePermissionsInspection,
    SourcePermissionsPreview,
    ToolCall,
    ToolResultBlock,
    Transcript,
    WorkBoard,
    WorkCard,
):
    _register_dataclass(_record)

# The host resolves PostgreSQL credentials with its own provider; a client's
# in-process provider object never crosses the socket.
_register_dataclass(PostgreSQLSource, process_local=frozenset({"secret_provider"}))

__all__ = ["JsonValue", "decode", "encode", "supports"]
