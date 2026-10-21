"""Host errors rebuild as the same exception type; events keep their sequence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from daita._json import FrozenJsonObject
from daita.adapters.mcp import (
    MCPAdmissionError,
    MCPAuthenticationError,
    MCPError,
    MCPProtocolError,
    MCPRemoteToolError,
    MCPTransportError,
)
from daita.adapters.protocols import ResourceAdapterError
from daita.adapters.sqlite import SQLiteSourceError
from daita.artifacts.models import ArtifactError
from daita.errors import (
    AgentError,
    AuthenticationError,
    ConfigError,
    DaitaError,
    ErrorRetryability,
    LLMError,
    PermanentError,
    RateLimitError,
    StateCompatibilityCode,
    StateCompatibilityError,
    TransientError,
)
from daita.host.protocol import (
    ERROR_CODES,
    AgentLockedError,
    ConversationBusyError,
    EventType,
    HostShuttingDownError,
    HostVersionMismatch,
    InteractionResolvedError,
    WorkspaceBusyError,
    error_envelope,
    error_from_envelope,
    event_envelope,
    parse_event,
)
from daita.hosting.errors import (
    AgentAlreadyExistsError,
    AgentHomeError,
    AgentIdentityMismatchError,
    AgentModelConfigurationError,
    AgentNameError,
    AgentNotConfiguredError,
    AgentNotFoundError,
    HostActiveError,
)
from daita.observation import AgentEvent, AgentEventKind

pytestmark = pytest.mark.unit

ERRORS: tuple[BaseException, ...] = (
    HostVersionMismatch("the Daita host runs 1.3.0", protocol=1, version="1.3.0"),
    ConversationBusyError(
        "a run is already active in this conversation", conversation_id="c-1"
    ),
    WorkspaceBusyError("Daita is busy in /Users/x/work.", folder=Path("/Users/x/work")),
    InteractionResolvedError("interaction is not pending: interaction-1"),
    AgentLockedError(
        "agent is in use by another process: sales",
        holder={"role": "script", "pid": 5530, "started_at": "2026-10-20T09:00:00Z"},
    ),
    HostShuttingDownError("the Daita host is shutting down"),
    AgentNotFoundError("agent does not exist: sales"),
    AgentNameError("agent names must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}"),
    AgentAlreadyExistsError("agent already exists: sales", agent_id="agent-1"),
    AgentIdentityMismatchError("agent home identity does not match"),
    AgentModelConfigurationError("saved model configuration must be replaced"),
    AgentNotConfiguredError("agent has no model configured"),
    HostActiveError("host_active: another writer owns /Users/x/.daita/agents/sales"),
    AgentHomeError("agent writer lock must be a regular file"),
    AgentError("Agent operation failed.", error_code="agent_closed"),
    StateCompatibilityError(
        StateCompatibilityCode.NEWER_REVISION,
        Path("/Users/x/.daita/agents/sales/state.db"),
        "This agent was saved by a newer Daita release.",
        current_revision="4",
        found_revision="5",
    ),
    ResourceAdapterError("source-1", "source_unavailable", "The source is offline."),
    ArtifactError("artifact_not_found", "Artifact is unavailable.", {"id": "a-1"}),
    MCPTransportError("mcp_transport", "The server did not answer."),
    MCPProtocolError("mcp_protocol", "The server sent an invalid response."),
    MCPAuthenticationError("mcp_auth", "The bearer token was refused."),
    MCPAdmissionError("mcp_tool_unadmitted", "The tool is not admitted."),
    MCPRemoteToolError("mcp_remote_tool", "The remote tool failed."),
    MCPError("mcp_error", "The MCP operation failed.", {"binding_id": "b-1"}),
    ConfigError("Model route is invalid.", section="model"),
    RateLimitError(retry_after_seconds=2.5),
    TransientError("The provider is temporarily unavailable."),
    AuthenticationError(provider_id="openai"),
    PermanentError("The request cannot be retried."),
    LLMError("Model request failed.", provider_id="anthropic"),
    DaitaError("Daita operation failed.", retryability=ErrorRetryability.RETRYABLE),
    PermissionError(13, "Permission denied"),
    FileNotFoundError("export directory is missing"),
    OSError("disk is full"),
    KeyError("run-404"),
    ImportError("pipx reinstall daita-agents"),
    TypeError("workspace must be LocalWorkspace"),
    ValueError("conversation is unavailable to this caller"),
    RuntimeError("no agent is open"),
)


def test_every_mapped_error_type_has_a_case() -> None:
    assert {type(error) for error in ERRORS} == set(ERROR_CODES)


@pytest.mark.parametrize("error", ERRORS, ids=lambda error: type(error).__name__)
def test_errors_rebuild_as_the_same_type_code_and_state(error: BaseException) -> None:
    envelope = json.loads(json.dumps(error_envelope(error)))
    rebuilt = error_from_envelope(envelope)

    assert type(rebuilt) is type(error)
    assert envelope["error"]["code"] == ERROR_CODES[type(error)]
    assert rebuilt.args == error.args
    assert str(rebuilt) == envelope["error"]["message"] == str(error)
    assert vars(rebuilt) == vars(error)
    assert envelope["error"]["retryable"] is (
        getattr(error, "retryability", None)
        in {ErrorRetryability.TRANSIENT, ErrorRetryability.RETRYABLE}
    )


def test_an_unlisted_subclass_travels_as_its_nearest_listed_ancestor() -> None:
    error = SQLiteSourceError(
        "sqlite_locked", "The database is locked.", source_id="source-1"
    )

    rebuilt = error_from_envelope(error_envelope(error))

    assert type(rebuilt) is ResourceAdapterError
    assert (rebuilt.source_id, rebuilt.code, str(rebuilt)) == (
        "source-1",
        "sqlite_locked",
        "The database is locked.",
    )


def test_unmapped_errors_and_unknown_codes_are_refused() -> None:
    class Unmapped(Exception):
        pass

    with pytest.raises(TypeError, match="no host error code"):
        error_envelope(Unmapped("x"))
    envelope = json.loads(json.dumps(error_envelope(ValueError("x"))))
    envelope["error"]["code"] = "pickled"
    with pytest.raises(ValueError, match="unknown host error code"):
        error_from_envelope(envelope)


def test_event_records_round_trip_their_sequence_type_and_data() -> None:
    event = AgentEvent(
        kind=AgentEventKind.MODEL_TEXT_DELTA,
        occurred_at=datetime(2026, 10, 20, 9, tzinfo=UTC),
        run_id="run-1",
        conversation_id="conversation-1",
        data=FrozenJsonObject.from_mapping({"text": "EMEA "}),
    )

    record = json.loads(
        json.dumps(event_envelope(18_342, EventType.AGENT_EVENT, event))
    )

    assert parse_event(record) == (18_342, EventType.AGENT_EVENT, event)
    with pytest.raises(ValueError, match="sequence"):
        event_envelope(-1, EventType.HEARTBEAT, None)
