"""
Typed records for the new database runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from os import PathLike
import json
from typing import Any, Literal, Mapping

from daita.runtime import AccessMode, Evidence, OperationStatus, RuntimeStore


class DbIntentKind(str, Enum):
    """Semantic categories the DB runtime can plan around."""

    CONVERSATIONAL = "conversational"
    SCHEMA_QUERY = "schema.query"
    SCHEMA_RELATIONSHIP_QUERY = "schema.relationship_query"
    DATA_QUERY = "data.query"
    CATALOG_ASSISTED_DATA_QUERY = "data.query.catalog_assisted"
    METRIC_QUERY = "metric.query"
    REPORT_GENERATE = "report.generate"
    QUALITY_CHECK = "quality.check"
    LINEAGE_TRACE = "lineage.trace"
    ANOMALY_INVESTIGATE = "anomaly.investigate"
    MEMORY_UPDATE = "memory.update"
    WRITE_PROPOSE = "write.propose"
    WRITE_EXECUTE = "write.execute"
    ADMIN = "admin"


def _tuple_strings(values: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    items = tuple(values)
    for value in items:
        if not isinstance(value, str):
            raise TypeError("values must be strings")
    return items


def _json_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    copied = dict(value or {})
    try:
        json.dumps(copied)
    except TypeError as exc:
        raise TypeError("DB runtime mappings must be JSON serializable") from exc
    return copied


@dataclass(frozen=True)
class DbLimits:
    """Execution limits used while planning and running DB operations."""

    max_rows: int = 500
    timeout_seconds: float = 30.0
    max_tasks: int = 20
    max_evidence_items: int = 50

    def __post_init__(self) -> None:
        if self.max_rows < 1:
            raise ValueError("max_rows must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_tasks < 1:
            raise ValueError("max_tasks must be at least 1")
        if self.max_evidence_items < 1:
            raise ValueError("max_evidence_items must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_rows": self.max_rows,
            "timeout_seconds": self.timeout_seconds,
            "max_tasks": self.max_tasks,
            "max_evidence_items": self.max_evidence_items,
        }


@dataclass(frozen=True)
class DbRuntimeConfig:
    """Configuration for a `DbRuntime` instance."""

    profile: str = "analyst"
    limits: DbLimits = field(default_factory=DbLimits)
    plugins: tuple[Any, ...] = ()
    policies: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.profile:
            raise ValueError("profile is required")
        object.__setattr__(self, "plugins", tuple(self.plugins))
        object.__setattr__(self, "policies", tuple(self.policies))
        object.__setattr__(self, "metadata", _json_dict(self.metadata))


@dataclass(frozen=True)
class DbRuntimeOptions:
    """Construction-time runtime options for `Agent.from_db()`."""

    store: RuntimeStore | Literal["sqlite"] | None = None
    store_path: str | PathLike[str] | None = None

    def __post_init__(self) -> None:
        if self.store is None:
            if self.store_path is not None:
                raise ValueError("store_path requires store='sqlite'")
            return
        if self.store == "sqlite":
            if self.store_path is None:
                raise ValueError("store_path is required when store='sqlite'")
            return
        if self.store_path is not None:
            raise ValueError("store_path is only supported when store='sqlite'")
        if not isinstance(self.store, RuntimeStore):
            raise TypeError("store must be 'sqlite' or a RuntimeStore instance")


@dataclass(frozen=True)
class DbRequest:
    """Normalized user request for the DB runtime."""

    prompt: str
    user_id: str | None = None
    session_id: str | None = None
    source_scope: tuple[str, ...] = ()
    mode: str | None = None
    requested_capabilities: tuple[str, ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    session_context: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt is required")
        object.__setattr__(self, "source_scope", _tuple_strings(self.source_scope))
        object.__setattr__(
            self,
            "requested_capabilities",
            _tuple_strings(self.requested_capabilities),
        )
        object.__setattr__(self, "constraints", _json_dict(self.constraints))
        object.__setattr__(self, "metadata", _json_dict(self.metadata))
        if self.session_context is not None:
            object.__setattr__(
                self,
                "session_context",
                _json_dict(self.session_context),
            )


@dataclass(frozen=True)
class DbIntent:
    """Classified intent used by contract builders."""

    kind: DbIntentKind
    confidence: float = 1.0
    access: AccessMode = AccessMode.NONE
    evidence_mode: str = "none"
    requested_outputs: tuple[str, ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", DbIntentKind(self.kind))
        object.__setattr__(self, "access", AccessMode(self.access))
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(
            self, "requested_outputs", _tuple_strings(self.requested_outputs)
        )
        object.__setattr__(self, "constraints", _json_dict(self.constraints))
        object.__setattr__(self, "diagnostics", _json_dict(self.diagnostics))


@dataclass(frozen=True)
class DbOperationContract:
    """Structured operation contract enforced by `DbRuntime`."""

    operation_type: str
    required_capabilities: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    access: AccessMode = AccessMode.NONE
    limits: DbLimits = field(default_factory=DbLimits)
    policy_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operation_type:
            raise ValueError("operation_type is required")
        object.__setattr__(
            self, "required_capabilities", _tuple_strings(self.required_capabilities)
        )
        object.__setattr__(
            self, "required_evidence", _tuple_strings(self.required_evidence)
        )
        object.__setattr__(self, "access", AccessMode(self.access))
        object.__setattr__(self, "policy_ids", _tuple_strings(self.policy_ids))
        object.__setattr__(self, "metadata", _json_dict(self.metadata))


@dataclass(frozen=True)
class DbOperationResult:
    """Result returned by `DbRuntime.run()`."""

    operation_id: str
    request: DbRequest
    intent: DbIntent
    contract: DbOperationContract
    status: OperationStatus
    answer: str | None = None
    evidence: tuple[Evidence, ...] = ()
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", OperationStatus(self.status))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "warnings", _tuple_strings(self.warnings))
        object.__setattr__(self, "diagnostics", _json_dict(self.diagnostics))

    @property
    def telemetry(self) -> dict[str, Any]:
        """Normalized DB operation telemetry for framework and hosted consumers."""
        return db_operation_telemetry(self)


def db_operation_telemetry(result: DbOperationResult) -> dict[str, Any]:
    """Return normalized model/token/cost telemetry for a DB operation result."""
    diagnostics = _telemetry_source_diagnostics(result)
    if _is_deterministic_telemetry(diagnostics):
        return _deterministic_telemetry(diagnostics)
    if diagnostics:
        return normalize_db_telemetry_diagnostics(diagnostics)
    return _unknown_telemetry()


def _telemetry_source_diagnostics(result: DbOperationResult) -> dict[str, Any]:
    answer = _latest_synthesis_diagnostics(result.evidence, kind="answer.synthesis")
    if answer:
        return answer
    analysis = _latest_synthesis_diagnostics(
        result.evidence,
        kind="analysis.synthesis",
        require_final=True,
    )
    if analysis:
        return analysis
    synthesis = result.diagnostics.get("synthesis")
    if isinstance(synthesis, dict):
        diagnostics = synthesis.get("diagnostics")
        if isinstance(diagnostics, dict) and diagnostics:
            return dict(diagnostics)
    analysis = result.diagnostics.get("analysis")
    if isinstance(analysis, dict):
        if not bool(analysis.get("partial")):
            diagnostics = analysis.get("diagnostics")
            if isinstance(diagnostics, dict) and diagnostics:
                return dict(diagnostics)
            synthesis = analysis.get("synthesis")
            if isinstance(synthesis, dict) and not bool(synthesis.get("partial")):
                diagnostics = synthesis.get("diagnostics")
                if isinstance(diagnostics, dict) and diagnostics:
                    return dict(diagnostics)
    return {}


def _latest_synthesis_diagnostics(
    evidence: tuple[Evidence, ...],
    *,
    kind: str,
    require_final: bool = False,
) -> dict[str, Any]:
    for item in reversed(evidence):
        if item.kind != kind or not item.accepted:
            continue
        payload = item.payload if isinstance(item.payload, dict) else {}
        if require_final and bool(payload.get("partial")):
            continue
        diagnostics = payload.get("diagnostics")
        if isinstance(diagnostics, dict) and diagnostics:
            return dict(diagnostics)
    return {}


def _is_deterministic_telemetry(diagnostics: Mapping[str, Any]) -> bool:
    mode = str(diagnostics.get("mode") or "")
    return (
        mode in {"deterministic", "deterministic_fallback"}
        or diagnostics.get("provider") == "daita.db"
        or diagnostics.get("model") == "deterministic"
    )


def _deterministic_telemetry(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "provider": "daita.db",
        "model": "deterministic",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
        "estimated_cost_usd": 0.0,
        "latency_ms": db_optional_float(diagnostics.get("latency_ms")),
        "mode": "deterministic",
    }


def _unknown_telemetry() -> dict[str, Any]:
    return {
        "provider": "unknown",
        "model": "unknown",
        "input_tokens": "unknown",
        "output_tokens": "unknown",
        "total_tokens": "unknown",
        "llm_calls": "unknown",
        "estimated_cost_usd": "unknown",
        "latency_ms": None,
        "mode": "unknown",
    }


def normalize_db_telemetry_diagnostics(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one synthesis diagnostics payload into DB telemetry fields."""
    tokens = (
        diagnostics.get("tokens") if isinstance(diagnostics.get("tokens"), dict) else {}
    )
    input_tokens = db_optional_int(
        diagnostics.get("input_tokens")
        if diagnostics.get("input_tokens") is not None
        else tokens.get("input_tokens", tokens.get("prompt_tokens"))
    )
    output_tokens = db_optional_int(
        diagnostics.get("output_tokens")
        if diagnostics.get("output_tokens") is not None
        else tokens.get("output_tokens", tokens.get("completion_tokens"))
    )
    total_tokens = db_optional_int(
        diagnostics.get("total_tokens")
        if diagnostics.get("total_tokens") is not None
        else tokens.get("total_tokens")
    )
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    mode = str(diagnostics.get("mode") or "unknown")
    llm_calls = db_optional_int(diagnostics.get("llm_calls"))
    if llm_calls is None and mode == "llm":
        llm_calls = 1
    cost = diagnostics.get("estimated_cost_usd")
    if cost is None:
        cost = diagnostics.get("estimated_cost")
    return {
        "provider": str(diagnostics.get("provider") or "unknown"),
        "model": str(diagnostics.get("model") or "unknown"),
        "input_tokens": input_tokens if input_tokens is not None else "unknown",
        "output_tokens": output_tokens if output_tokens is not None else "unknown",
        "total_tokens": total_tokens if total_tokens is not None else "unknown",
        "llm_calls": llm_calls if llm_calls is not None else "unknown",
        "estimated_cost_usd": (
            db_optional_float(cost) if cost is not None else "unknown"
        ),
        "latency_ms": db_optional_float(diagnostics.get("latency_ms")),
        "mode": mode,
    }


def db_optional_int(value: Any) -> int | None:
    if value is None or value == "unknown":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def db_optional_float(value: Any) -> float | None:
    if value is None or value == "unknown":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class DbRuntimeInspection:
    """Diagnostics snapshot for a `DbRuntime`."""

    runtime_id: str
    runtime_kind: str
    source_type: str
    source_repr: str
    profile: str
    plugin_ids: tuple[str, ...]
    capability_count: int
    executor_count: int
    evidence_schema_count: int
    policy_count: int
    context_provider_count: int
    tool_view_count: int
    worker_count: int
    capability_ids: tuple[str, ...] = ()
    executor_ids: tuple[str, ...] = ()
    evidence_schema_kinds: tuple[str, ...] = ()
    policy_ids: tuple[str, ...] = ()
    context_provider_ids: tuple[str, ...] = ()
    tool_view_names: tuple[str, ...] = ()
    worker_ids: tuple[str, ...] = ()
    diagnostics: tuple[dict[str, str], ...] = ()
    operation_count: int = 0
    last_operation_id: str | None = None
    limits: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "limits", _json_dict(self.limits))
        object.__setattr__(self, "metadata", _json_dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_id": self.runtime_id,
            "runtime_kind": self.runtime_kind,
            "source_type": self.source_type,
            "source_repr": self.source_repr,
            "profile": self.profile,
            "plugin_ids": list(self.plugin_ids),
            "capability_count": self.capability_count,
            "executor_count": self.executor_count,
            "evidence_schema_count": self.evidence_schema_count,
            "policy_count": self.policy_count,
            "context_provider_count": self.context_provider_count,
            "tool_view_count": self.tool_view_count,
            "worker_count": self.worker_count,
            "capability_ids": list(self.capability_ids),
            "executor_ids": list(self.executor_ids),
            "evidence_schema_kinds": list(self.evidence_schema_kinds),
            "policy_ids": list(self.policy_ids),
            "context_provider_ids": list(self.context_provider_ids),
            "tool_view_names": list(self.tool_view_names),
            "worker_ids": list(self.worker_ids),
            "diagnostics": list(self.diagnostics),
            "operation_count": self.operation_count,
            "last_operation_id": self.last_operation_id,
            "limits": dict(self.limits),
            "metadata": dict(self.metadata),
        }
