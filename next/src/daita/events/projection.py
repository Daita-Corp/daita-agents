"""Audience-specific projections of canonical committed runtime events."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timezone
from enum import Enum
import hashlib
import re

from .._json import FrozenJsonObject
from .models import CommittedEvent


class EventAudience(str, Enum):
    PUBLIC = "public"
    AUDIT = "audit"
    TELEMETRY = "telemetry"


_REDACTED = "[redacted]"
_MAX_AUDIT_DEPTH = 6
_MAX_AUDIT_ITEMS = 64
_MAX_AUDIT_STRING_BYTES = 512
_MAX_AUDIT_KEY_BYTES = 128
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "arguments",
        "authorization",
        "client_secret",
        "connection_string",
        "connector_error",
        "content",
        "context_selection",
        "cookie",
        "credentials",
        "dsn",
        "error_message",
        "file_content",
        "headers",
        "input",
        "output",
        "password",
        "passwd",
        "private_key",
        "prompt",
        "provider_request",
        "provider_response",
        "query_params",
        "query_parameters",
        "request",
        "response",
        "rows",
        "secret",
        "set_cookie",
        "stacktrace",
        "token",
        "url",
        "uri",
    }
)
_TELEMETRY_NUMBERS = frozenset(
    {
        "action_count",
        "attempt",
        "count",
        "duration_ms",
        "evidence_bytes",
        "input_tokens",
        "latency_ms",
        "limit",
        "output_tokens",
        "retry_count",
        "total_tokens",
        "used",
    }
)
_TELEMETRY_DECIMALS = frozenset({"estimated_cost_usd"})
_TELEMETRY_BOOLEANS = frozenset({"allowed", "success", "truncated"})
_TELEMETRY_CODES = frozenset(
    {
        "budget",
        "code",
        "error_code",
        "finish_reason",
        "model_id",
        "provider_id",
        "reason_code",
        "selected_provider_id",
        "status",
    }
)
_SAFE_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")
_DECIMAL_RE = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?$")


def project_committed_event(
    committed: CommittedEvent,
    *,
    audience: EventAudience,
) -> FrozenJsonObject:
    """Project one durably committed event for a specific consumer audience."""

    if not isinstance(committed, CommittedEvent):
        raise TypeError("event projection requires a CommittedEvent")
    if not isinstance(audience, EventAudience):
        raise TypeError("event projection audience must be an EventAudience")
    event = committed.event
    if audience is EventAudience.PUBLIC:
        payload: Mapping[str, object] = {}
    elif audience is EventAudience.AUDIT:
        redacted = _redact_audit_value(event.payload, depth=0)
        if not isinstance(redacted, Mapping):
            raise TypeError("audit event payload must project to an object")
        payload = redacted
    else:
        payload = _telemetry_payload(event.payload)
    return FrozenJsonObject.from_mapping(
        {
            "sequence": committed.cursor.sequence,
            "id": event.id,
            "type": event.type,
            "agent_id": event.agent_id,
            "created_at": event.created_at.astimezone(timezone.utc).isoformat(),
            "operation_id": event.operation_id,
            "session_id": event.session_id,
            "turn_id": event.turn_id,
            "model_call_id": event.model_call_id,
            "call_id": event.call_id,
            "task_id": event.task_id,
            "evidence_id": event.evidence_id,
            "approval_id": event.approval_id,
            "monitor_id": event.monitor_id,
            "capability_id": event.capability_id,
            "executor_id": event.executor_id,
            "payload": payload,
        }
    )


def _redact_audit_value(value: object, *, depth: int) -> object:
    if depth >= _MAX_AUDIT_DEPTH:
        return "[omitted]"
    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        for index, (raw_key, item) in enumerate(value.items()):
            if index >= _MAX_AUDIT_ITEMS:
                projected["_omitted"] = True
                break
            key = _bounded_key(raw_key)
            if _sensitive_key(raw_key):
                projected[key] = _REDACTED
            else:
                projected[key] = _redact_audit_value(item, depth=depth + 1)
        return projected
    if isinstance(value, (tuple, list)):
        values = tuple(value)
        projected_items = [
            _redact_audit_value(item, depth=depth + 1)
            for item in values[:_MAX_AUDIT_ITEMS]
        ]
        if len(values) > _MAX_AUDIT_ITEMS:
            projected_items.append("[omitted]")
        return projected_items
    if isinstance(value, str):
        if _sensitive_text(value):
            return _REDACTED
        encoded = value.encode("utf-8")
        if len(encoded) > _MAX_AUDIT_STRING_BYTES:
            return "[omitted:sha256:" + hashlib.sha256(encoded).hexdigest() + "]"
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError("audit event payload contains a non-JSON value")


def _bounded_key(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("audit event payload keys must be strings")
    encoded = value.encode("utf-8")
    if len(encoded) <= _MAX_AUDIT_KEY_BYTES and value.isprintable():
        return value
    return "field_" + hashlib.sha256(encoded).hexdigest()[:24]


def _sensitive_key(value: object) -> bool:
    if not isinstance(value, str):
        return True
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return any(
        normalized == key
        or normalized.startswith(key + "_")
        or normalized.endswith("_" + key)
        for key in _SENSITIVE_KEYS
    )


def _sensitive_text(value: str) -> bool:
    lowered = value.lower().strip()
    if lowered.startswith(("bearer ", "basic ")):
        return True
    if "://" in lowered and ("@" in lowered or "?" in lowered):
        return True
    return any(
        marker in lowered
        for marker in (
            "api_key=",
            "apikey=",
            "authorization=",
            "password=",
            "secret=",
            "token=",
        )
    )


def _telemetry_payload(payload: Mapping[str, object]) -> dict[str, object]:
    projected: dict[str, object] = {}
    for key, value in payload.items():
        if key in _TELEMETRY_NUMBERS and (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            projected[key] = value
        elif key in _TELEMETRY_DECIMALS and (
            (isinstance(value, (int, float)) and not isinstance(value, bool))
            or (isinstance(value, str) and _DECIMAL_RE.fullmatch(value) is not None)
        ):
            projected[key] = value
        elif key in _TELEMETRY_BOOLEANS and isinstance(value, bool):
            projected[key] = value
        elif (
            key in _TELEMETRY_CODES
            and isinstance(value, str)
            and _SAFE_CODE_RE.fullmatch(value) is not None
        ):
            projected[key] = value
    return projected


__all__ = ["EventAudience", "project_committed_event"]
