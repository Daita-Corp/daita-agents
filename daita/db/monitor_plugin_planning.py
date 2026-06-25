"""Generic monitor plugin source and delivery planning.

This module owns monitor-aware capability selection and deterministic plugin
task input mapping. Runtime execution remains owned by ``DbRuntime``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from daita.runtime import AccessMode, Capability, Evidence

from .analysis import evidence_ref


@dataclass(frozen=True)
class MonitorPluginPlanningBlocked(RuntimeError):
    """Structured block returned when a monitor plugin step is not executable."""

    reason: str
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", dict(self.details or {}))
        RuntimeError.__init__(self, self.reason)


@dataclass(frozen=True)
class MonitorSourceIntent:
    """Typed source observation intent compiled from a persisted monitor plan."""

    raw: dict[str, Any]
    source_kind: str
    request: dict[str, Any]
    correlation: dict[str, Any]
    expected_evidence: tuple[str, ...]
    value_path: str | None = None
    sequence: int = 100


@dataclass(frozen=True)
class MonitorDeliveryIntent:
    """Typed delivery intent compiled from Phase 5 report evidence."""

    raw: dict[str, Any]
    delivery_kind: str
    target: dict[str, Any]
    format: str | None = None
    subject: str | None = None


@dataclass(frozen=True)
class MonitorPluginTaskPlan:
    """Executable monitor plugin task plan for the runtime to persist."""

    role: str
    capability: Capability
    input_payload: dict[str, Any]
    idempotency_key: str
    reason: str
    sequence: int
    metadata: dict[str, Any]
    intent_payload: dict[str, Any]
    source_evidence_refs: tuple[dict[str, Any], ...] = ()

    @property
    def input_hash(self) -> str:
        return stable_monitor_fingerprint(self.input_payload)


class MonitorPluginPlanner:
    """Plan source/delivery plugin tasks from persisted monitor evidence."""

    def __init__(self, capabilities: tuple[Capability, ...]) -> None:
        self.capabilities = tuple(capabilities)

    def plan_source(
        self,
        source_step: Mapping[str, Any],
        *,
        cursor: Mapping[str, Any],
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
    ) -> MonitorPluginTaskPlan:
        intent = _source_intent(source_step, cursor=cursor)
        input_payload = {
            "monitor_source": {
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "source_kind": intent.source_kind,
                "request": intent.request,
                "correlation": intent.correlation,
            },
            "request": intent.request,
            "correlation": intent.correlation,
        }
        capability = self._select_capability(
            intent.raw,
            role="source",
            kind=intent.source_kind,
            target=intent.request,
            expected_evidence=intent.expected_evidence,
            input_payload=input_payload,
        )
        return MonitorPluginTaskPlan(
            role="source",
            capability=capability,
            input_payload=input_payload,
            idempotency_key=stable_monitor_fingerprint(
                {
                    "role": "source",
                    "monitor_id": monitor_id,
                    "monitor_run_id": monitor_run_id,
                    "tick_operation_id": tick_operation_id,
                    "capability_id": capability.id,
                    "capability_owner": capability.owner,
                    "request": intent.request,
                }
            ),
            reason="monitor_plugin_source",
            sequence=intent.sequence,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_source_kind": intent.source_kind,
                "monitor_role": "source",
            },
            intent_payload={
                "source_kind": intent.source_kind,
                "request": intent.request,
                "correlation": intent.correlation,
                "expected_evidence": list(intent.expected_evidence),
                "value_path": intent.value_path,
            },
        )

    def plan_delivery(
        self,
        delivery_intent: Mapping[str, Any],
        *,
        report: Evidence,
        source_evidence_refs: tuple[dict[str, Any], ...],
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
    ) -> MonitorPluginTaskPlan:
        intent = _delivery_intent(delivery_intent, report=report)
        report_fingerprint = monitor_report_fingerprint(report)
        action_fingerprint = str(report.payload.get("action_plan_fingerprint") or "")
        capability = self._select_capability(
            intent.raw,
            role="delivery",
            kind=intent.delivery_kind,
            target=intent.target,
            payload_kind=_payload_kind_from_delivery_intent(intent),
            expected_evidence=(),
            input_payload=None,
        )
        _validate_delivery_format(intent, capability)
        idempotency_key = stable_monitor_fingerprint(
            {
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "action_plan_fingerprint": action_fingerprint,
                "report_fingerprint": report_fingerprint,
                "target": intent.target,
                "capability_id": capability.id,
                "capability_owner": capability.owner,
            }
        )
        input_payload = {
            "delivery_kind": intent.delivery_kind,
            "target": intent.target,
            "format": intent.format or report.payload.get("format"),
            "subject": intent.subject or report.payload.get("title"),
            "idempotency_key": idempotency_key,
            "payload_source": {
                "kind": "monitor.report",
                "report_evidence_id": report.id,
                "report_fingerprint": report_fingerprint,
                "action_plan_fingerprint": action_fingerprint,
                "source_evidence_refs": [dict(item) for item in source_evidence_refs],
            },
        }
        _validate_capability_input_schema(capability, input_payload)
        return MonitorPluginTaskPlan(
            role="delivery",
            capability=capability,
            input_payload=input_payload,
            idempotency_key=idempotency_key,
            reason="monitor_delivery",
            sequence=9000,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_role": "delivery",
                "monitor_delivery_kind": intent.delivery_kind,
                "monitor_delivery_target": intent.target,
                "monitor_report_evidence_id": report.id,
                "monitor_report_fingerprint": report_fingerprint,
                "monitor_action_fingerprint": action_fingerprint,
                "source_evidence_refs": [dict(item) for item in source_evidence_refs],
            },
            intent_payload={
                "delivery_kind": intent.delivery_kind,
                "target": intent.target,
                "format": intent.format,
                "subject": intent.subject,
            },
            source_evidence_refs=source_evidence_refs,
        )

    def select_delivery_capability(
        self,
        delivery_intent: Mapping[str, Any],
    ) -> Capability:
        """Select and validate the monitor delivery capability for an intent."""
        intent = _delivery_intent_from_mapping(delivery_intent)
        capability = self._select_capability(
            intent.raw,
            role="delivery",
            kind=intent.delivery_kind,
            target=intent.target,
            payload_kind=_payload_kind_from_delivery_intent(intent),
            expected_evidence=(),
            input_payload=None,
        )
        _validate_delivery_format(intent, capability)
        return capability

    def _select_capability(
        self,
        intent: Mapping[str, Any],
        *,
        role: str,
        kind: str,
        target: Mapping[str, Any],
        payload_kind: str | None = None,
        expected_evidence: tuple[str, ...],
        input_payload: Mapping[str, Any] | None,
    ) -> Capability:
        capability_id = intent.get("capability_id")
        owner = intent.get("capability_owner") or intent.get("owner")
        if capability_id:
            matches = [
                capability
                for capability in self.capabilities
                if capability.id == str(capability_id)
                and (owner is None or capability.owner == str(owner))
            ]
            if not matches:
                raise MonitorPluginPlanningBlocked(
                    "missing_capability",
                    {"capability_id": str(capability_id), "owner": owner},
                )
            if len(matches) > 1:
                raise MonitorPluginPlanningBlocked(
                    "ambiguous_capability",
                    {
                        "capability_id": str(capability_id),
                        "owners": [capability.owner for capability in matches],
                    },
                )
            capability = matches[0]
            if not _capability_supports_monitor_role(
                capability,
                role=role,
                kind=kind,
                expected_evidence=expected_evidence,
            ):
                raise MonitorPluginPlanningBlocked(
                    "capability_shape_unsupported",
                    {
                        "capability_id": capability.id,
                        "owner": capability.owner,
                        "role": role,
                    },
                )
            if input_payload is not None:
                _validate_capability_input_schema(capability, input_payload)
            _validate_delivery_payload_kind(
                capability,
                payload_kind=payload_kind,
                role=role,
            )
            return capability

        candidates = [
            capability
            for capability in self.capabilities
            if _capability_supports_monitor_role(
                capability,
                role=role,
                kind=kind,
                expected_evidence=expected_evidence,
            )
        ]
        payload_supported_candidates = candidates
        if role == "delivery" and payload_kind:
            payload_supported_candidates = [
                capability
                for capability in candidates
                if _capability_accepts_delivery_payload_kind(
                    capability,
                    payload_kind=payload_kind,
                )
            ]
            if candidates and not payload_supported_candidates:
                raise MonitorPluginPlanningBlocked(
                    "unsupported_payload_kind",
                    {
                        "payload_kind": payload_kind,
                        "matches": [
                            {"id": capability.id, "owner": capability.owner}
                            for capability in candidates
                        ],
                    },
                )
        candidates = payload_supported_candidates
        if input_payload is not None:
            candidates = [
                capability
                for capability in candidates
                if not _capability_input_schema_errors(capability, input_payload)
            ]
        candidates = [
            capability
            for capability in candidates
            if _capability_accepts_target_shape(capability, target)
        ]
        if not candidates:
            raise MonitorPluginPlanningBlocked(
                "missing_capability",
                {"role": role, "kind": kind},
            )
        if len(candidates) > 1:
            raise MonitorPluginPlanningBlocked(
                "ambiguous_capability",
                {
                    "role": role,
                    "kind": kind,
                    "matches": [
                        {"id": capability.id, "owner": capability.owner}
                        for capability in candidates
                    ],
                },
            )
        return candidates[0]


def monitor_report_fingerprint(report: Evidence) -> str:
    return str(
        report.metadata.get("payload_fingerprint") or ""
    ) or stable_monitor_fingerprint(report.payload)


def monitor_delivery_source_refs(
    report: Evidence,
    operation_evidence: tuple[Evidence, ...],
) -> tuple[dict[str, Any], ...]:
    refs: list[dict[str, Any]] = [evidence_ref(report)]
    for ref in report.payload.get("cited_tick_evidence_refs") or ():
        if isinstance(ref, Mapping):
            refs.append(dict(ref))
    for item in operation_evidence:
        if item.id == report.id:
            continue
        if item.kind in {"monitor.action_result", "analysis.synthesis"}:
            refs.append(evidence_ref(item))
    for ref in report.payload.get("produced_evidence_refs") or ():
        if isinstance(ref, Mapping) and ref.get("kind") == "analysis.synthesis":
            refs.append(dict(ref))
    return tuple(_dedupe_evidence_refs(refs))


def monitor_source_observed_value(
    source_step: Mapping[str, Any],
    evidence: tuple[Evidence, ...],
) -> Any:
    payloads = [item.payload for item in evidence]
    value: Any = payloads[0] if len(payloads) == 1 else payloads
    path = str(source_step.get("value_path") or "")
    if path:
        return _extract_path(value, path)
    if isinstance(value, Mapping) and "body" in value:
        return value.get("body")
    return value


def stable_monitor_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_intent(
    source_step: Mapping[str, Any],
    *,
    cursor: Mapping[str, Any],
) -> MonitorSourceIntent:
    raw = dict(source_step)
    request = dict(raw.get("request") or {})
    for key in ("endpoint", "resource", "path", "method", "query", "params"):
        if key in raw and key not in request:
            request[key] = raw[key]
    request = _resolve_cursor_refs(request, cursor=cursor)
    method = str(request.get("method") or "GET").upper()
    if method not in {"GET", "HEAD", "OPTIONS", "READ"}:
        raise MonitorPluginPlanningBlocked("unsafe_source_call", {"method": method})
    if not any(request.get(key) for key in ("endpoint", "resource", "path", "url")):
        raise MonitorPluginPlanningBlocked(
            "missing_source_target",
            {"required_any": ["endpoint", "resource", "path", "url"]},
        )
    correlation = dict(raw.get("correlation") or {})
    if not correlation and raw.get("require_correlation", True):
        raise MonitorPluginPlanningBlocked("missing_source_correlation")
    sequence = raw.get("sequence") or 100
    try:
        sequence_value = int(sequence)
    except (TypeError, ValueError):
        sequence_value = 100
    return MonitorSourceIntent(
        raw=raw,
        source_kind=str(raw.get("source_kind") or raw.get("kind") or ""),
        request=request,
        correlation=correlation,
        expected_evidence=tuple(
            str(item) for item in raw.get("expected_evidence") or ()
        ),
        value_path=str(raw.get("value_path")) if raw.get("value_path") else None,
        sequence=sequence_value,
    )


def _delivery_intent(
    delivery_intent: Mapping[str, Any],
    *,
    report: Evidence,
) -> MonitorDeliveryIntent:
    return _delivery_intent_from_mapping(
        delivery_intent,
        fallback_subject=str(report.payload.get("title") or ""),
    )


def _delivery_intent_from_mapping(
    delivery_intent: Mapping[str, Any],
    *,
    fallback_subject: str = "",
) -> MonitorDeliveryIntent:
    raw = dict(delivery_intent)
    target = raw.get("target")
    normalized = (
        {str(key): value for key, value in target.items() if value}
        if isinstance(target, Mapping)
        else {}
    )
    for key in (
        "channel",
        "recipient",
        "recipients",
        "address",
        "url",
        "endpoint",
        "path",
    ):
        if raw.get(key) and key not in normalized:
            normalized[key] = raw[key]
    if not normalized:
        raise MonitorPluginPlanningBlocked("missing_delivery_target")
    return MonitorDeliveryIntent(
        raw=raw,
        delivery_kind=str(raw.get("delivery_kind") or raw.get("mode") or ""),
        target=normalized,
        format=str(raw.get("format")) if raw.get("format") else None,
        subject=(
            str(raw.get("subject") or raw.get("title"))
            if raw.get("subject") or raw.get("title")
            else fallback_subject
        ),
    )


def _payload_kind_from_delivery_intent(intent: MonitorDeliveryIntent) -> str:
    payload_source = intent.raw.get("payload_source")
    if isinstance(payload_source, Mapping):
        return str(payload_source.get("type") or payload_source.get("kind") or "")
    return ""


def _capability_supports_monitor_role(
    capability: Capability,
    *,
    role: str,
    kind: str,
    expected_evidence: tuple[str, ...],
) -> bool:
    metadata = capability.metadata
    roles = {str(item) for item in metadata.get("monitor_roles") or ()}
    role_kind = metadata.get(f"{role}_kind")
    role_kinds = {str(item) for item in metadata.get(f"{role}_kinds") or ()}
    role_supported = (
        role in roles
        or (role_kind is not None and (not kind or str(role_kind) == kind))
        or (kind and kind in role_kinds)
        or f"monitor.{role}" in capability.operation_types
    )
    if not role_supported:
        return False
    if kind:
        if role_kind is not None and str(role_kind) != kind:
            return False
        if role_kinds and kind not in role_kinds:
            return False
        if (
            capability.domains
            and kind not in capability.domains
            and "monitor" not in capability.domains
        ):
            return False
    if expected_evidence and not set(expected_evidence).issubset(
        capability.output_evidence
    ):
        return False
    if role == "source":
        return not capability.side_effecting and capability.access in {
            AccessMode.NONE,
            AccessMode.METADATA_READ,
            AccessMode.READ,
        }
    if role == "delivery":
        if capability.access is AccessMode.ADMIN:
            return False
        accepted = {
            str(item)
            for item in metadata.get("accepted_payload_kinds")
            or metadata.get("accepted_evidence_kinds")
            or ()
        }
        if accepted and not (
            {"monitor.report", "analysis.synthesis", "monitor.action_result"} & accepted
        ):
            return False
    return True


def _validate_delivery_payload_kind(
    capability: Capability,
    *,
    payload_kind: str | None,
    role: str,
) -> None:
    if role != "delivery" or not payload_kind:
        return
    if not _capability_accepts_delivery_payload_kind(
        capability,
        payload_kind=payload_kind,
    ):
        raise MonitorPluginPlanningBlocked(
            "unsupported_payload_kind",
            {
                "payload_kind": payload_kind,
                "capability_id": capability.id,
                "owner": capability.owner,
                "accepted_payload_kinds": sorted(
                    _accepted_delivery_payload_kinds(capability)
                ),
            },
        )


def _capability_accepts_delivery_payload_kind(
    capability: Capability,
    *,
    payload_kind: str,
) -> bool:
    accepted = _accepted_delivery_payload_kinds(capability)
    return not accepted or payload_kind in accepted


def _accepted_delivery_payload_kinds(capability: Capability) -> set[str]:
    return {
        str(item)
        for item in capability.metadata.get("accepted_payload_kinds")
        or capability.metadata.get("accepted_evidence_kinds")
        or ()
    }


def _capability_accepts_target_shape(
    capability: Capability,
    target: Mapping[str, Any],
) -> bool:
    schema = capability.input_schema
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return True
    property_names = set(properties)
    target_keys = set(target)
    return bool(target_keys & property_names) or bool(
        {"target", "request", "payload_source", "monitor_source"} & property_names
    )


def _validate_delivery_format(
    intent: MonitorDeliveryIntent,
    capability: Capability,
) -> None:
    accepted_target_types = {
        str(item)
        for item in capability.metadata.get("accepted_target_types")
        or capability.metadata.get("accepted_channels")
        or ()
    }
    if accepted_target_types:
        target_type = str(
            intent.target.get("type")
            or intent.target.get("channel")
            or intent.target.get("mode")
            or ""
        )
        if target_type not in accepted_target_types:
            raise MonitorPluginPlanningBlocked(
                "unsupported_delivery_target",
                {
                    "target_type": target_type,
                    "accepted_target_types": sorted(accepted_target_types),
                    "capability_id": capability.id,
                    "owner": capability.owner,
                },
            )
    if not intent.format:
        return
    accepted = {
        str(item)
        for item in capability.metadata.get("accepted_formats")
        or capability.metadata.get("accepted_payload_formats")
        or ()
    }
    if accepted and intent.format not in accepted:
        raise MonitorPluginPlanningBlocked(
            "unsupported_format",
            {
                "format": intent.format,
                "accepted_formats": sorted(accepted),
                "capability_id": capability.id,
                "owner": capability.owner,
            },
        )


def _validate_capability_input_schema(
    capability: Capability,
    input_payload: Mapping[str, Any],
) -> None:
    errors = _capability_input_schema_errors(capability, input_payload)
    if errors:
        raise MonitorPluginPlanningBlocked(
            "invalid_capability_input",
            {
                "capability_id": capability.id,
                "owner": capability.owner,
                "errors": errors,
            },
        )


def _capability_input_schema_errors(
    capability: Capability,
    input_payload: Mapping[str, Any],
) -> tuple[str, ...]:
    return tuple(
        _json_schema_subset_errors(input_payload, capability.input_schema, "$")
    )


def _json_schema_subset_errors(
    value: Any,
    schema: Mapping[str, Any],
    path: str,
) -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, Mapping):
            return [f"{path}:expected_object"]
        required = [str(item) for item in schema.get("required") or ()]
        for key in required:
            if key not in value or value.get(key) in (None, "", [], {}):
                errors.append(f"{path}.{key}:missing_required")
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, Mapping):
                    errors.extend(
                        _json_schema_subset_errors(
                            value[key], child_schema, f"{path}.{key}"
                        )
                    )
        return errors
    if expected_type == "array" and not isinstance(value, list):
        errors.append(f"{path}:expected_array")
    elif expected_type == "string" and value is not None and not isinstance(value, str):
        errors.append(f"{path}:expected_string")
    elif (
        expected_type == "number"
        and value is not None
        and not isinstance(value, (int, float))
    ):
        errors.append(f"{path}:expected_number")
    elif (
        expected_type == "boolean" and value is not None and not isinstance(value, bool)
    ):
        errors.append(f"{path}:expected_boolean")
    return errors


def _resolve_cursor_refs(value: Any, *, cursor: Mapping[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("monitor.state.cursor."):
        current: Any = cursor
        for part in value.removeprefix("monitor.state.cursor.").split("."):
            if isinstance(current, Mapping):
                current = current.get(part)
            else:
                return None
        return current
    if isinstance(value, Mapping):
        return {
            str(key): _resolve_cursor_refs(item, cursor=cursor)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_cursor_refs(item, cursor=cursor) for item in value]
    return value


def _dedupe_evidence_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any]] = set()
    deduped: list[dict[str, Any]] = []
    for ref in refs:
        key = (ref.get("id"), ref.get("kind"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def _extract_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current
