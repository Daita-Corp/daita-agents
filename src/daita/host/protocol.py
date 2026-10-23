"""Define the private host protocol: version, headers, surface and envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Final

from ..adapters.mcp import (
    MCPAdmissionError,
    MCPAuthenticationError,
    MCPError,
    MCPProtocolError,
    MCPRemoteToolError,
    MCPTransportError,
)
from ..adapters.protocols import ResourceAdapterError
from ..artifacts.models import ArtifactError
from ..errors import (
    AgentError,
    AuthenticationError,
    ConfigError,
    DaitaError,
    ErrorRetryability,
    LLMError,
    PermanentError,
    RateLimitError,
    StateCompatibilityError,
    TransientError,
)
from ..hosting.errors import (
    AgentAlreadyExistsError,
    AgentHomeError,
    AgentIdentityMismatchError,
    AgentModelConfigurationError,
    AgentNameError,
    AgentNotConfiguredError,
    AgentNotFoundError,
    HostActiveError,
)
from .codec import JsonValue, decode, encode

PROTOCOL_VERSION: Final = 1
PROTOCOL_HEADER: Final = "X-Daita-Protocol"
VERSION_HEADER: Final = "X-Daita-Version"
# The client's working directory, percent-encoded, for relative path arguments.
CWD_HEADER: Final = "X-Daita-Cwd"
MAX_REQUEST_BYTES: Final = 4 * 1024 * 1024
MAX_RESPONSE_BYTES: Final = 32 * 1024 * 1024

# Every Agent member that src/daita/tui/ and src/daita/cli.py use, plus the
# workspace properties RemoteAgent mirrors from the agent status.
REMOTE_AGENT_SURFACE: Final = frozenset(
    {
        # Lifecycle and properties
        "open",
        "create",
        "delete",
        "list",
        "inspect_home",
        "close",
        "name",
        "id",
        "home",
        "model_profile",
        "model_route",
        "workspace",
        "session_source_id",
        "model_requires_explicit_limits",
        # Runs and conversations
        "run",
        "learn",
        "conversation_exists",
        "conversation_runs",
        "transcript",
        "tool_evidence",
        "analysis_usage",
        "clear_conversations",
        # Model
        "configure_model",
        "authenticate_model_subscription",
        # Sources and catalog
        "list_sources",
        "attach",
        "attach_sqlite",
        "attach_postgresql",
        "probe_postgresql",
        "store_postgresql_password",
        "delete_postgresql_password",
        "edit_sqlite_source",
        "edit_postgresql_source",
        "detach",
        "refresh_source",
        "resolve_source",
        "inspect_source_permissions",
        "preview_source_permissions",
        "apply_source_permissions",
        "relational_update_readiness",
        "catalog_summary",
        "list_catalog_resources",
        "inspect_catalog_resource",
        # Artifacts, deliveries and work
        "list_artifacts",
        "read_artifact",
        "save_artifact",
        "delete_artifact",
        "export_destination",
        "set_export_destination",
        "reset_export_destination",
        "inspect_delivery",
        "acknowledge_delivery",
        "work_board",
        # Routines
        "list_routines",
        "inspect_routine",
        "pause_routine",
        "resume_routine",
        "disable_routine",
        "run_routine_now",
        "create_routine",
        "update_routine",
        # Jobs
        "list_jobs",
        "inspect_job",
        "inspect_job_task",
        "job_board",
        "job_timeline",
        "graph_blockers",
        "read_job_result",
        "list_job_tasks",
        "list_job_dependencies",
        "list_task_attempts",
        "list_task_checkpoints",
        "list_task_artifacts",
        "read_task_result",
        "cancel_job",
        "cancel_graph_job",
        "answer_task_input",
        "accept_task_review",
        "request_task_review_changes",
        "retry_task_control",
        "reject_task_control",
        "replace_graph_task_by_policy",
        # Effect receipts
        "list_effects",
        "inspect_effect",
        "resolve_effect",
        # Memory, skills and semantics
        "read_memory",
        "set_memory",
        "read_user_profile",
        "set_user_profile",
        "list_skills",
        "read_skill",
        "save_skill",
        "delete_skill",
        "list_semantic_annotations",
        "read_semantic_annotation",
        "delete_semantic_annotation",
        # MCP
        "list_mcp_servers",
        "inspect_mcp_server",
        "attach_mcp_server",
        "refresh_mcp_server",
        "revoke_mcp_server",
        "store_mcp_bearer",
        "delete_mcp_bearer",
    }
)
# RemoteAgent members with no Agent counterpart: facts only a host client has.
CLIENT_ONLY_MEMBERS: Final = frozenset(
    {
        "active_runs",
        "follow_run",
        "connection",
        "model_configuration_requires_replacement",
    }
)

# In-process composition the host owns; "return" marks the RemoteAgent the
# client builds itself instead of receiving one.
_COMPOSITION = frozenset(
    {
        "return",
        "hosted",
        "root",
        "config",
        "model",
        "model_profile",
        "limits",
        "clock",
        "id_factory",
        "secret_provider",
        "mcp_client_factory",
        "mcp_connection_provider",
        "keychain",
        "model_validator",
        "observer",
        "approval_handler",
        "downloads_directory",
        "diagnostics_path",
    }
)

# Annotations of surface members that never cross the socket. Callbacks stay on
# the client and are answered through host interactions.
CLIENT_SIDE_ANNOTATIONS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "open": _COMPOSITION,
        "create": _COMPOSITION,
        "delete": frozenset({"root", "keychain", "config"}),
        "list": frozenset({"root"}),
        "inspect_home": frozenset({"root"}),
        "authenticate_model_subscription": frozenset(
            {"on_verification", "on_progress"}
        ),
        "edit_sqlite_source": frozenset({"confirmation_handler"}),
        "edit_postgresql_source": frozenset({"confirmation_handler"}),
        "create_routine": frozenset({"confirmation_handler"}),
        "update_routine": frozenset({"confirmation_handler"}),
    }
)


class HostVersionMismatch(RuntimeError):
    """The client and host run different protocol or package versions."""

    def __init__(self, message: str, *, protocol: int, version: str) -> None:
        super().__init__(message)
        self.protocol = protocol
        self.version = version


class ConversationBusyError(RuntimeError):
    """A foreground run is already active in this conversation."""

    def __init__(self, message: str, *, conversation_id: str) -> None:
        super().__init__(message)
        self.conversation_id = conversation_id


class WorkspaceBusyError(RuntimeError):
    """The agent is busy in another folder, so it cannot switch to this one."""

    def __init__(self, message: str, *, folder: Path) -> None:
        super().__init__(message)
        self.folder = folder


class InteractionResolvedError(RuntimeError):
    """The interaction was already answered, dismissed or expired."""


class AgentLockedError(HostActiveError):
    """Another process holds the agent's writer lock; the host never fights it."""


class HostShuttingDownError(RuntimeError):
    """The host is draining and accepts no new runs."""


# One stable wire code per exception type. A raised subclass travels as its
# nearest listed ancestor.
ERROR_CODES: Final[Mapping[type[BaseException], str]] = MappingProxyType(
    {
        HostVersionMismatch: "version_mismatch",
        ConversationBusyError: "conversation_busy",
        WorkspaceBusyError: "workspace_busy",
        InteractionResolvedError: "already_resolved",
        AgentLockedError: "agent_locked",
        HostShuttingDownError: "shutting_down",
        AgentNotFoundError: "agent_not_found",
        AgentNameError: "agent_name_invalid",
        AgentAlreadyExistsError: "agent_already_exists",
        AgentIdentityMismatchError: "agent_identity_mismatch",
        AgentModelConfigurationError: "agent_model_configuration_invalid",
        AgentNotConfiguredError: "agent_not_configured",
        HostActiveError: "host_active",
        AgentHomeError: "agent_home_error",
        AgentError: "agent_error",
        StateCompatibilityError: "state_incompatible",
        ResourceAdapterError: "source_error",
        ArtifactError: "artifact_error",
        MCPTransportError: "mcp_transport_error",
        MCPProtocolError: "mcp_protocol_error",
        MCPAuthenticationError: "mcp_authentication_error",
        MCPAdmissionError: "mcp_admission_error",
        MCPRemoteToolError: "mcp_remote_tool_error",
        MCPError: "mcp_error",
        ConfigError: "config_error",
        RateLimitError: "rate_limited",
        TransientError: "transient_error",
        AuthenticationError: "authentication_failed",
        PermanentError: "permanent_error",
        LLMError: "llm_error",
        DaitaError: "daita_error",
        PermissionError: "permission_denied",
        FileNotFoundError: "file_not_found",
        OSError: "os_error",
        KeyError: "key_not_found",
        ImportError: "dependency_missing",
        TypeError: "invalid_type",
        ValueError: "invalid_value",
        RuntimeError: "runtime_error",
    }
)
_ERROR_TYPES: Final = {code: error_type for error_type, code in ERROR_CODES.items()}
_RETRYABLE: Final = frozenset(
    {ErrorRetryability.TRANSIENT, ErrorRetryability.RETRYABLE}
)


class EventType(str, Enum):
    """The kinds of record on an agent's event stream."""

    AGENT_EVENT = "agent_event"
    RUN_STATE = "run_state"
    INTERACTION = "interaction"
    WORK_CHANGED = "work_changed"
    AGENT_STATE = "agent_state"
    HEARTBEAT = "heartbeat"
    RESET = "reset"


def error_envelope(error: BaseException) -> dict[str, JsonValue]:
    """Return the one error envelope that lets a client raise the same type."""

    error_type = next(
        (item for item in type(error).__mro__ if item in ERROR_CODES), None
    )
    if error_type is None:
        raise TypeError(f"no host error code for {type(error).__qualname__}")
    # Exceptions rebuild from args and public attributes because their
    # constructors differ; DaitaError attributes are safe facts by contract.
    attributes = {
        name: encode(value)
        for name, value in vars(error).items()
        if not name.startswith("_")
    }
    return {
        "error": {
            "code": ERROR_CODES[error_type],
            "message": str(error),
            "retryable": getattr(error, "retryability", None) in _RETRYABLE,
            "details": {"args": encode(error.args), "attributes": attributes},
        }
    }


def error_from_envelope(payload: JsonValue) -> BaseException:
    """Rebuild the exception one error envelope describes, ready to raise."""

    body = _fields(payload, {"error"})["error"]
    fields = _fields(body, {"code", "message", "retryable", "details"})
    error_type = _ERROR_TYPES.get(fields["code"])  # type: ignore[arg-type]
    if error_type is None:
        raise ValueError(f"unknown host error code: {fields['code']!r}")
    details = _fields(fields["details"], {"args", "attributes"})
    attributes = details["attributes"]
    if type(attributes) is not dict:
        raise ValueError("host error attributes must be an object")
    error = error_type.__new__(error_type, *decode(details["args"], tuple[object, ...]))
    for name, value in attributes.items():
        if not name.isidentifier() or name.startswith("_"):
            raise ValueError("host error attribute name is invalid")
        setattr(error, name, decode(value))
    return error


def event_envelope(seq: int, event: EventType, data: object) -> dict[str, JsonValue]:
    """Return one sequenced event-stream record."""

    if type(seq) is not int or seq < 0:
        raise ValueError("event sequence must be a non-negative integer")
    return {"seq": seq, "event": EventType(event).value, "data": encode(data)}


def parse_event(payload: JsonValue) -> tuple[int, EventType, object]:
    """Decode one event-stream record into its sequence, type and data."""

    fields = _fields(payload, {"seq", "event", "data"})
    seq = fields["seq"]
    if type(seq) is not int or seq < 0:
        raise ValueError("event sequence must be a non-negative integer")
    return seq, EventType(fields["event"]), decode(fields["data"])


def _fields(payload: JsonValue, names: set[str]) -> dict[str, JsonValue]:
    if type(payload) is not dict or set(payload) != names:
        raise ValueError(f"host protocol record must have exactly {sorted(names)}")
    return payload


__all__ = [
    "CLIENT_ONLY_MEMBERS",
    "CLIENT_SIDE_ANNOTATIONS",
    "CWD_HEADER",
    "ERROR_CODES",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "PROTOCOL_HEADER",
    "PROTOCOL_VERSION",
    "REMOTE_AGENT_SURFACE",
    "VERSION_HEADER",
    "AgentLockedError",
    "ConversationBusyError",
    "EventType",
    "HostShuttingDownError",
    "HostVersionMismatch",
    "InteractionResolvedError",
    "WorkspaceBusyError",
    "error_envelope",
    "error_from_envelope",
    "event_envelope",
    "parse_event",
]
