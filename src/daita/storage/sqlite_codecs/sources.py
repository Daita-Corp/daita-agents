"""Encode and decode current source registration records."""

from __future__ import annotations

from ...adapters.models import SourceRegistration
from ...llm.models import ModelSensitivity
from .common import (
    JsonValue,
    datetime_decode,
    datetime_encode,
    dump_payload,
    load_payload,
    mapping,
    optional_datetime_decode,
    optional_datetime_encode,
    plain_decode,
    plain_encode,
    record,
    record_fields,
    sequence,
    text,
)

_CURRENT_SOURCE_ADAPTER_IDS = frozenset({"sqlite", "postgresql"})


class CurrentSourceAdapterError(ValueError):
    """Reject a source adapter outside the one canonical runtime shape."""


def _require_current_adapter(adapter_id: str) -> None:
    if adapter_id not in _CURRENT_SOURCE_ADAPTER_IDS:
        raise CurrentSourceAdapterError(
            "stored source registration uses an unsupported current adapter"
        )


def _require_current_source(value: SourceRegistration) -> SourceRegistration:
    if not isinstance(value, SourceRegistration):
        raise TypeError("source codec requires SourceRegistration")
    _require_current_adapter(value.adapter_id)
    if value.adapter_id == "postgresql" and "write_access" in value.configuration:
        raise ValueError("PostgreSQL source contains embedded write admission")
    return value


def encode_source(value: SourceRegistration) -> str:
    return dump_payload(_encode_source(_require_current_source(value)))


def decode_source(value: str) -> SourceRegistration:
    decoded = _decode_source(load_payload(value))
    if decoded.adapter_id == "postgresql" and "write_access" in decoded.configuration:
        raise ValueError("stored PostgreSQL source contains embedded write admission")
    return decoded


def decode_source_credential_reference_for_deletion(
    value: str,
    *,
    agent_id: str,
    source_id: str,
) -> str | None:
    """Read only deletion-critical fields across pre-production source shapes."""

    decoded = load_payload(value)
    if not isinstance(decoded, dict) or set(decoded) != {"__record__", "fields"}:
        raise ValueError("stored SourceRegistration record envelope is invalid")
    if decoded["__record__"] != "SourceRegistration":
        raise ValueError("stored record is not SourceRegistration")
    fields = mapping(decoded["fields"], "SourceRegistration fields")
    stored_agent_id = text(fields.get("agent_id"), "source agent_id")
    stored_source_id = text(fields.get("id"), "source id")
    if stored_agent_id != agent_id or stored_source_id != source_id:
        raise ValueError("stored source row identity is invalid")
    adapter_id = text(fields.get("adapter_id"), "source adapter_id")
    configuration = mapping(fields.get("configuration"), "source configuration")
    if adapter_id != "postgresql":
        return None
    reference = configuration.get("credential_ref")
    return None if reference is None else text(reference, "source credential_ref")


def _encode_source(value: SourceRegistration) -> dict[str, JsonValue]:
    return record(
        "SourceRegistration",
        {
            "id": value.id,
            "agent_id": value.agent_id,
            "adapter_id": value.adapter_id,
            "native_identity": value.native_identity,
            "display_name": value.display_name,
            "configuration": plain_encode(value.configuration),
            "attached_at": datetime_encode(value.attached_at),
            "detached_at": optional_datetime_encode(value.detached_at),
            "summary": value.summary,
            "when_to_use": value.when_to_use,
            "keywords": list(value.keywords),
            "presentation_sensitivity": value.presentation_sensitivity.value,
        },
    )


def _decode_source(value: JsonValue) -> SourceRegistration:
    fields = record_fields(
        value,
        "SourceRegistration",
        (
            "id",
            "agent_id",
            "adapter_id",
            "native_identity",
            "display_name",
            "configuration",
            "attached_at",
            "detached_at",
            "summary",
            "when_to_use",
            "keywords",
            "presentation_sensitivity",
        ),
    )
    raw_configuration = mapping(fields["configuration"], "source configuration")
    configuration = plain_decode(raw_configuration)
    if not isinstance(configuration, dict):
        raise ValueError("stored source configuration is invalid")
    adapter_id = text(fields["adapter_id"], "source adapter_id")
    _require_current_adapter(adapter_id)
    if adapter_id == "postgresql" and "write_access" in configuration:
        raise ValueError("stored PostgreSQL source contains embedded write admission")
    return SourceRegistration(
        id=text(fields["id"], "source id"),
        agent_id=text(fields["agent_id"], "source agent_id"),
        adapter_id=adapter_id,
        native_identity=text(fields["native_identity"], "source native_identity"),
        display_name=text(fields["display_name"], "source display_name"),
        configuration=configuration,
        attached_at=datetime_decode(fields["attached_at"]),
        detached_at=optional_datetime_decode(fields["detached_at"]),
        summary=text(fields["summary"], "source summary"),
        presentation_sensitivity=ModelSensitivity(
            text(fields["presentation_sensitivity"], "source presentation sensitivity")
        ),
        when_to_use=text(fields["when_to_use"], "source when_to_use"),
        keywords=tuple(
            text(item, "source keyword")
            for item in sequence(fields["keywords"], "source keywords")
        ),
    )


__all__ = [
    "CurrentSourceAdapterError",
    "decode_source",
    "decode_source_credential_reference_for_deletion",
    "encode_source",
]
