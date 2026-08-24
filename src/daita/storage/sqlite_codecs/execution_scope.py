"""Encode and decode the one current immutable execution-scope value."""

from __future__ import annotations

from ...capabilities import AccessMode, ExecutionScope, OperationalEffect
from ...llm.models import ModelSensitivity
from .common import (
    decimal_decode,
    decimal_encode,
    integer,
    optional_text,
    plain_encode,
    record,
    record_fields,
    sequence,
    text,
)


def encode_execution_scope(value: ExecutionScope):
    if not isinstance(value, ExecutionScope):
        raise TypeError("execution scope codec requires ExecutionScope")
    return record(
        "ExecutionScope",
        {
            "scope_id": value.scope_id,
            "revision": value.revision,
            "agent_id": value.agent_id,
            "principal_id": value.principal_id,
            "grant_id": value.grant_id,
            "job_id": value.job_id,
            "job_revision": value.job_revision,
            "allowed_source_ids": list(value.allowed_source_ids),
            "allowed_resource_ids": list(value.allowed_resource_ids),
            "allowed_capability_ids": list(value.allowed_capability_ids),
            "allowed_access_modes": plain_encode(
                tuple(sorted(item.value for item in value.allowed_access_modes))
            ),
            "allowed_operational_effects": plain_encode(
                tuple(sorted(item.value for item in value.allowed_operational_effects))
            ),
            "sensitivity_ceiling": value.sensitivity_ceiling.value,
            "eligible_model_routes": list(value.eligible_model_routes),
            "per_run_max_cost_usd": decimal_encode(value.per_run_max_cost_usd),
            "per_run_max_tokens": value.per_run_max_tokens,
            "delivery_destination": value.delivery_destination,
        },
    )


def decode_execution_scope(value) -> ExecutionScope:
    fields = record_fields(
        value,
        "ExecutionScope",
        (
            "scope_id",
            "revision",
            "agent_id",
            "principal_id",
            "grant_id",
            "job_id",
            "job_revision",
            "allowed_source_ids",
            "allowed_resource_ids",
            "allowed_capability_ids",
            "allowed_access_modes",
            "allowed_operational_effects",
            "sensitivity_ceiling",
            "eligible_model_routes",
            "per_run_max_cost_usd",
            "per_run_max_tokens",
            "delivery_destination",
        ),
    )
    try:
        access_modes = frozenset(
            AccessMode(text(item, "execution scope access mode"))
            for item in sequence(
                fields["allowed_access_modes"],
                "execution scope access modes",
            )
        )
        effects = frozenset(
            OperationalEffect(text(item, "execution scope operational effect"))
            for item in sequence(
                fields["allowed_operational_effects"],
                "execution scope operational effects",
            )
        )
        sensitivity = ModelSensitivity(
            text(fields["sensitivity_ceiling"], "execution scope sensitivity")
        )
    except ValueError:
        raise ValueError("stored execution scope enum is invalid") from None
    return ExecutionScope(
        scope_id=text(fields["scope_id"], "execution scope id"),
        revision=integer(fields["revision"], "execution scope revision"),
        agent_id=text(fields["agent_id"], "execution scope agent id"),
        principal_id=text(fields["principal_id"], "execution scope principal id"),
        grant_id=text(fields["grant_id"], "execution scope grant id"),
        job_id=optional_text(fields["job_id"], "execution scope job id"),
        job_revision=(
            None
            if fields["job_revision"] is None
            else integer(fields["job_revision"], "execution scope job revision")
        ),
        allowed_source_ids=tuple(
            text(item, "execution scope source id")
            for item in sequence(
                fields["allowed_source_ids"], "execution scope source ids"
            )
        ),
        allowed_resource_ids=tuple(
            text(item, "execution scope resource id")
            for item in sequence(
                fields["allowed_resource_ids"], "execution scope resource ids"
            )
        ),
        allowed_capability_ids=tuple(
            text(item, "execution scope capability id")
            for item in sequence(
                fields["allowed_capability_ids"],
                "execution scope capability ids",
            )
        ),
        allowed_access_modes=access_modes,
        allowed_operational_effects=effects,
        sensitivity_ceiling=sensitivity,
        eligible_model_routes=tuple(
            text(item, "execution scope model route")
            for item in sequence(
                fields["eligible_model_routes"],
                "execution scope model routes",
            )
        ),
        per_run_max_cost_usd=decimal_decode(fields["per_run_max_cost_usd"]),
        per_run_max_tokens=integer(
            fields["per_run_max_tokens"],
            "execution scope per-run tokens",
        ),
        delivery_destination=text(
            fields["delivery_destination"],
            "execution scope delivery destination",
        ),
    )


__all__ = ["decode_execution_scope", "encode_execution_scope"]
