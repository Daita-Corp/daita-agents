"""Encode and decode the one current immutable execution-scope value."""

from __future__ import annotations

from ..._json import FrozenJsonObject
from ...capabilities import (
    AccessMode,
    CapabilityGrant,
    ExecutionContractBindings,
    ExecutionScope,
    OperationalEffect,
)
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
            "routine_id": value.routine_id,
            "routine_revision": value.routine_revision,
            "occurrence_id": value.occurrence_id,
            "allowed_source_ids": list(value.allowed_source_ids),
            "allowed_connector_binding_ids": list(value.allowed_connector_binding_ids),
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
            "distribution_plan_digest": value.distribution_plan_digest,
            "contract_bindings": encode_execution_contract_bindings(
                value.contract_bindings
            ),
            "capability_grants": [
                encode_capability_grant(grant) for grant in value.capability_grants
            ],
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
            "routine_id",
            "routine_revision",
            "occurrence_id",
            "allowed_source_ids",
            "allowed_connector_binding_ids",
            "allowed_resource_ids",
            "allowed_capability_ids",
            "allowed_access_modes",
            "allowed_operational_effects",
            "sensitivity_ceiling",
            "eligible_model_routes",
            "per_run_max_cost_usd",
            "per_run_max_tokens",
            "distribution_plan_digest",
            "contract_bindings",
            "capability_grants",
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
        contract_bindings=decode_execution_contract_bindings(
            fields["contract_bindings"]
        ),
        capability_grants=tuple(
            decode_capability_grant(item)
            for item in sequence(fields["capability_grants"], "scope capability grants")
        ),
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
        routine_id=optional_text(fields["routine_id"], "execution scope routine id"),
        routine_revision=(
            None
            if fields["routine_revision"] is None
            else integer(fields["routine_revision"], "execution scope routine revision")
        ),
        occurrence_id=optional_text(
            fields["occurrence_id"],
            "execution scope occurrence id",
        ),
        allowed_source_ids=tuple(
            text(item, "execution scope source id")
            for item in sequence(
                fields["allowed_source_ids"], "execution scope source ids"
            )
        ),
        allowed_connector_binding_ids=tuple(
            text(item, "execution scope connector binding id")
            for item in sequence(
                fields["allowed_connector_binding_ids"],
                "execution scope connector binding ids",
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
        distribution_plan_digest=text(
            fields["distribution_plan_digest"],
            "execution scope distribution plan digest",
        ),
    )


def encode_execution_contract_bindings(value: ExecutionContractBindings):
    return record(
        "ExecutionContractBindings",
        {key: plain_encode(item) for key, item in value.material().items()},
    )


def decode_execution_contract_bindings(value) -> ExecutionContractBindings:
    fields = record_fields(
        value,
        "ExecutionContractBindings",
        (
            "capability_contracts",
            "tool_origins",
            "resource_revisions",
            "model_routes",
        ),
    )
    maps: dict[str, dict[str, str]] = {}
    for name, raw in fields.items():
        if not isinstance(raw, dict):
            raise ValueError(
                "execution contract bindings must contain fixed-shape maps"
            )
        maps[name] = {
            text(key, "binding identity"): text(digest, "binding digest")
            for key, digest in raw.items()
        }
    return ExecutionContractBindings(**maps)


def encode_capability_grant(value: CapabilityGrant):
    return record(
        "CapabilityGrant",
        {
            **{key: plain_encode(item) for key, item in value.material().items()},
            "grant_digest": value.grant_digest,
        },
    )


def decode_capability_grant(value) -> CapabilityGrant:
    fields = record_fields(
        value,
        "CapabilityGrant",
        (
            "grant_id",
            "domain_owner_id",
            "capability_id",
            "capability_contract_digest",
            "constraints_kind",
            "constraints",
            "max_calls_per_occurrence",
            "grant_digest",
        ),
    )
    constraints = fields["constraints"]
    if not isinstance(constraints, dict):
        raise ValueError("grant constraints must be an object")
    grant = CapabilityGrant(
        grant_id=text(fields["grant_id"], "grant id"),
        domain_owner_id=text(fields["domain_owner_id"], "grant owner"),
        capability_id=text(fields["capability_id"], "grant capability"),
        capability_contract_digest=text(
            fields["capability_contract_digest"], "grant contract"
        ),
        constraints_kind=text(fields["constraints_kind"], "grant constraints kind"),
        constraints=FrozenJsonObject.from_mapping(constraints),
        max_calls_per_occurrence=integer(
            fields["max_calls_per_occurrence"], "grant call ceiling"
        ),
    )
    if fields["grant_digest"] != grant.grant_digest:
        raise ValueError("grant digest does not match its authority")
    return grant


__all__ = ["decode_execution_scope", "encode_execution_scope"]
