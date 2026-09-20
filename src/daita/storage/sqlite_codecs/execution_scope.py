"""Encode and decode the one current immutable execution-scope value."""

from __future__ import annotations

from ..._json import FrozenJsonObject
from ...capabilities import (
    AccessMode,
    CapabilityGrant,
    ExecutionContractBindings,
    ExecutionScope,
    ExecutionScopeKind,
    GraphTaskBinding,
    OperationalEffect,
)
from ...llm.models import ModelSensitivity
from .common import (
    datetime_decode,
    datetime_encode,
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
            "scope_kind": value.scope_kind.value,
            "graph_task_binding": (
                None
                if value.graph_task_binding is None
                else _encode_graph_task_binding(value.graph_task_binding)
            ),
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
            "scope_kind",
            "graph_task_binding",
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
        scope_kind = ExecutionScopeKind(
            text(fields["scope_kind"], "execution scope kind")
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
        scope_kind=scope_kind,
        graph_task_binding=(
            None
            if fields["graph_task_binding"] is None
            else _decode_graph_task_binding(fields["graph_task_binding"])
        ),
    )


def _encode_graph_task_binding(value: GraphTaskBinding):
    if not isinstance(value, GraphTaskBinding):
        raise TypeError("graph task binding codec requires GraphTaskBinding")
    return record(
        "GraphTaskBinding",
        {
            "agent_id": value.agent_id,
            "job_id": value.job_id,
            "root_authority_digest": value.root_authority_digest,
            "task_id": value.task_id,
            "task_revision": value.task_revision,
            "task_spec_digest": value.task_spec_digest,
            "task_scope_digest": value.task_scope_digest,
            "attempt_id": value.attempt_id,
            "claim_token_digest": value.claim_token_digest,
            "fencing_epoch": value.fencing_epoch,
            "graph_revision_at_claim": value.graph_revision_at_claim,
            "task_role": value.task_role,
            "task_deadline_at": datetime_encode(value.task_deadline_at),
            "job_deadline_at": datetime_encode(value.job_deadline_at),
            "budget_reservation_identity": value.budget_reservation_identity,
        },
    )


def _decode_graph_task_binding(value) -> GraphTaskBinding:
    fields = record_fields(
        value,
        "GraphTaskBinding",
        (
            "agent_id",
            "job_id",
            "root_authority_digest",
            "task_id",
            "task_revision",
            "task_spec_digest",
            "task_scope_digest",
            "attempt_id",
            "claim_token_digest",
            "fencing_epoch",
            "graph_revision_at_claim",
            "task_role",
            "task_deadline_at",
            "job_deadline_at",
            "budget_reservation_identity",
        ),
    )
    return GraphTaskBinding(
        agent_id=text(fields["agent_id"], "graph binding agent id"),
        job_id=text(fields["job_id"], "graph binding job id"),
        root_authority_digest=text(
            fields["root_authority_digest"], "graph binding root authority digest"
        ),
        task_id=text(fields["task_id"], "graph binding task id"),
        task_revision=integer(fields["task_revision"], "graph binding task revision"),
        task_spec_digest=text(
            fields["task_spec_digest"], "graph binding task spec digest"
        ),
        task_scope_digest=text(
            fields["task_scope_digest"], "graph binding task scope digest"
        ),
        attempt_id=text(fields["attempt_id"], "graph binding attempt id"),
        claim_token_digest=text(
            fields["claim_token_digest"], "graph binding claim digest"
        ),
        fencing_epoch=integer(fields["fencing_epoch"], "graph binding fencing epoch"),
        graph_revision_at_claim=integer(
            fields["graph_revision_at_claim"], "graph binding graph revision"
        ),
        task_role=text(fields["task_role"], "graph binding task role"),
        task_deadline_at=datetime_decode(fields["task_deadline_at"]),
        job_deadline_at=datetime_decode(fields["job_deadline_at"]),
        budget_reservation_identity=text(
            fields["budget_reservation_identity"],
            "graph binding budget reservation identity",
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
