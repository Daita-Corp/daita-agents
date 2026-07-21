"""Immutable provider-neutral context projection vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from ..llm.models import CanonicalMessage, MessageRole, ToolResultBlock

_CONTEXT_ID = re.compile(r"[a-z0-9][a-z0-9._:-]{0,127}\Z")


def _bounded_identity(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _CONTEXT_ID.fullmatch(value):
        raise ValueError(f"{field_name} must be a bounded lowercase context identity")


def _bounded_text(
    value: str,
    field_name: str,
    *,
    maximum: int,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(value) > maximum:
        raise ValueError(f"{field_name} must contain at most {maximum} characters")


class ContextKind(str, Enum):
    """Stable semantic categories used by projection policy."""

    SYSTEM = "system"
    INTENT = "intent"
    OPERATION = "operation"
    SESSION_RECENT = "session_recent"
    EVIDENCE = "evidence"
    SKILL = "skill"
    CATALOG = "catalog"
    SOURCE_ROUTING = "source_routing"
    MEMORY = "memory"
    SESSION_SUMMARY = "session_summary"
    POLICY_NOTICE = "policy_notice"


class ContextTrust(str, Enum):
    """Whether projected content is authoritative or untrusted data."""

    TRUSTED_SYSTEM = "trusted_system"
    TRUSTED_RUNTIME = "trusted_runtime"
    UNTRUSTED_EXTERNAL = "untrusted_external"


@dataclass(frozen=True, slots=True)
class ContextProvenance:
    """One stable reference explaining where a context block came from."""

    kind: str
    reference_id: str
    revision: str | None = None

    def __post_init__(self) -> None:
        _bounded_identity(self.kind, "context provenance kind")
        _bounded_text(
            self.reference_id,
            "context provenance reference_id",
            maximum=512,
        )
        if self.revision is not None:
            _bounded_text(
                self.revision,
                "context provenance revision",
                maximum=256,
            )


@dataclass(frozen=True, slots=True)
class ContextMessageGroup:
    """Messages that context budgeting must retain or omit as one unit."""

    id: str
    messages: tuple[CanonicalMessage, ...]

    def __post_init__(self) -> None:
        _bounded_identity(self.id, "context message-group id")
        if isinstance(self.messages, (str, bytes)):
            raise TypeError("context message-group messages must be a sequence")
        messages = tuple(self.messages)
        if not messages:
            raise ValueError("context message group requires at least one message")
        if any(not isinstance(message, CanonicalMessage) for message in messages):
            raise TypeError("context message-group messages must be canonical messages")
        if len({message.agent_id for message in messages}) != 1:
            raise ValueError("context message group must belong to one agent")
        if len({message.operation_id for message in messages}) != 1:
            raise ValueError("context message group must belong to one operation")

        tool_call_ids: set[str] = set()
        tool_result_ids: set[str] = set()
        pending_result_ids: set[str] = set()
        for message in messages:
            if message.role is MessageRole.ASSISTANT:
                if pending_result_ids:
                    raise ValueError(
                        "context message group cannot interrupt pending tool results"
                    )
                message_call_ids = {call.id for call in message.tool_calls}
                if message_call_ids & tool_call_ids:
                    raise ValueError(
                        "context message group has duplicate assistant tool-call IDs"
                    )
                tool_call_ids.update(message_call_ids)
                pending_result_ids.update(message_call_ids)
                continue
            if message.role is not MessageRole.TOOL:
                if pending_result_ids:
                    raise ValueError(
                        "context message group cannot interrupt pending tool results"
                    )
                continue
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    continue
                if block.call_id in tool_result_ids:
                    raise ValueError("context message group has duplicate tool results")
                if block.call_id not in pending_result_ids:
                    raise ValueError("context message group has an orphan tool result")
                tool_result_ids.add(block.call_id)
                pending_result_ids.remove(block.call_id)
        if pending_result_ids or tool_call_ids != tool_result_ids:
            raise ValueError(
                "context message group must keep tool calls and results together"
            )
        object.__setattr__(self, "messages", messages)


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """One attributable, trusted, indivisible context selection candidate.

    Higher numeric priority wins among optional blocks. Required blocks always
    win or the complete selection fails closed.
    """

    id: str
    owner: str
    kind: ContextKind
    trust: ContextTrust
    provenance: tuple[ContextProvenance, ...]
    groups: tuple[ContextMessageGroup, ...]
    priority: int = 0
    required: bool = False

    def __post_init__(self) -> None:
        _bounded_identity(self.id, "context block id")
        _bounded_identity(self.owner, "context block owner")
        if not isinstance(self.kind, ContextKind):
            raise TypeError("context block kind must be a ContextKind")
        if not isinstance(self.trust, ContextTrust):
            raise TypeError("context block trust must be a ContextTrust")
        if isinstance(self.provenance, (str, bytes)):
            raise TypeError("context block provenance must be a sequence")
        provenance = tuple(self.provenance)
        if not provenance:
            raise ValueError("context block requires at least one provenance record")
        if any(not isinstance(item, ContextProvenance) for item in provenance):
            raise TypeError(
                "context block provenance must contain ContextProvenance records"
            )
        if len(provenance) != len(set(provenance)):
            raise ValueError("context block provenance records must be unique")
        if isinstance(self.groups, (str, bytes)):
            raise TypeError("context block groups must be a sequence")
        groups = tuple(self.groups)
        if not groups:
            raise ValueError("context block requires at least one message group")
        if any(not isinstance(group, ContextMessageGroup) for group in groups):
            raise TypeError(
                "context block groups must contain ContextMessageGroup records"
            )
        group_ids = [group.id for group in groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("context block message-group IDs must be unique")
        agent_ids = {message.agent_id for group in groups for message in group.messages}
        operation_ids = {
            message.operation_id for group in groups for message in group.messages
        }
        if len(agent_ids) != 1 or len(operation_ids) != 1:
            raise ValueError("context block groups must share one agent and operation")
        if (
            not isinstance(self.priority, int)
            or isinstance(self.priority, bool)
            or not 0 <= self.priority <= 1_000_000
        ):
            raise ValueError(
                "context block priority must be an integer from 0 to 1000000"
            )
        if not isinstance(self.required, bool):
            raise TypeError("context block required must be a boolean")
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "groups", groups)

    @property
    def messages(self) -> tuple[CanonicalMessage, ...]:
        """Flatten groups without weakening their budgeting boundary."""

        return tuple(message for group in self.groups for message in group.messages)


__all__ = [
    "ContextBlock",
    "ContextKind",
    "ContextMessageGroup",
    "ContextProvenance",
    "ContextTrust",
]
