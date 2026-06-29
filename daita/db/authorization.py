"""Plain authorization metadata helpers for DB runtime governance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from daita.runtime import Operation

AUTHORIZATION_MODES = frozenset({"interactive", "preauthorized", "deny"})


def normalize_authorization(
    metadata: Mapping[str, Any] | None,
    *,
    default_mode: str = "interactive",
) -> dict[str, Any]:
    """Normalize caller-provided authorization metadata into plain JSON data."""
    metadata = metadata if isinstance(metadata, Mapping) else {}
    raw = metadata.get("authorization")
    raw_authorization = raw if isinstance(raw, Mapping) else {}
    mode = str(raw_authorization.get("mode") or "").strip().lower()
    if mode not in AUTHORIZATION_MODES:
        mode = (
            "deny"
            if _metadata_identifies_automation(metadata)
            else _authorization_mode(default_mode)
        )

    grants = _normalize_grants(raw_authorization.get("grants"))
    normalized = {
        "mode": mode,
        "principal": optional_string(raw_authorization.get("principal")),
        "actor_type": optional_string(raw_authorization.get("actor_type")),
        "grant_ids": list(safe_strings(raw_authorization.get("grant_ids"))),
        "grants": grants,
    }
    if mode == "preauthorized" and not grants:
        normalized["mode"] = "deny"
    if normalized["mode"] == "deny" and not normalized["actor_type"]:
        normalized["actor_type"] = optional_string(
            metadata.get("actor_type")
            or metadata.get("caller_type")
            or metadata.get("request_origin")
        )
    return normalized


def authorization_from_operation(operation: Operation) -> dict[str, Any]:
    raw = operation.metadata.get("authorization")
    if not isinstance(raw, Mapping) and isinstance(
        operation.request.get("metadata"), Mapping
    ):
        raw = operation.request["metadata"].get("authorization")
    if isinstance(raw, Mapping):
        return normalize_authorization({"authorization": raw})
    return normalize_authorization({})


def authorization_mode(operation: Operation) -> str:
    return str(authorization_from_operation(operation).get("mode") or "interactive")


def authorization_from_governance_request(request: Any) -> dict[str, Any]:
    governance_facts = request_governance_facts(request)
    authorization = governance_facts.get("authorization")
    if not isinstance(authorization, Mapping):
        authoritative = governance_facts.get("authoritative")
        if isinstance(authoritative, Mapping):
            authorization = authoritative.get("authorization")
    if isinstance(authorization, Mapping):
        return normalize_authorization({"authorization": authorization})
    return normalize_authorization({})


def request_governance_facts(request: Any) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        return {}
    facts = request.get("governance_facts")
    return dict(facts) if isinstance(facts, Mapping) else {}


def match_preauthorization_grant(
    request: Any,
    *,
    require_destructive: bool = False,
    require_admin: bool = False,
) -> dict[str, Any]:
    authorization = authorization_from_governance_request(request)
    concrete = authorization_concrete_facts(
        request,
        require_destructive=require_destructive,
        require_admin=require_admin,
    )
    first_mismatch: str | None = None
    for grant in authorization.get("grants") or ():
        mismatch = grant_mismatch(grant, authorization, concrete)
        if mismatch is None:
            return {
                "matched": True,
                "grant_id": grant.get("id"),
                "facts": concrete,
            }
        if first_mismatch is None:
            first_mismatch = mismatch
    return {
        "matched": False,
        "reason": first_mismatch or "no_matching_grant",
        "facts": concrete,
    }


def grant_mismatch(
    grant: Mapping[str, Any],
    authorization: Mapping[str, Any],
    concrete: Mapping[str, Any],
) -> str | None:
    grant_principal = optional_string(grant.get("principal"))
    if grant_principal and grant_principal != authorization.get("principal"):
        return "principal_mismatch"
    grant_lanes = frozenset(safe_strings(grant.get("lanes")))
    if not grant_lanes or not set(concrete["lanes"]).issubset(grant_lanes):
        return "lane_not_granted"
    grant_capabilities = frozenset(safe_strings(grant.get("capabilities")))
    if not grant_capabilities or not set(concrete["capabilities"]).issubset(
        grant_capabilities
    ):
        return "capability_not_granted"
    contract_capabilities = set(concrete["contract_capabilities"])
    if contract_capabilities and not set(concrete["capabilities"]).issubset(
        contract_capabilities
    ):
        return "capability_outside_contract"
    grant_sources = frozenset(safe_strings(grant.get("source_scope")))
    if grant_sources:
        sources = tuple(concrete["source_scope"])
        if not sources or any(
            not source_in_scope(source, grant_sources) for source in sources
        ):
            return "source_scope_not_granted"
    if access_rank(concrete["access"]) > access_rank(
        str(grant.get("max_access") or "read")
    ):
        return "access_exceeds_grant"
    if concrete["destructive"] and not bool(grant.get("allow_destructive")):
        return "destructive_not_granted"
    if concrete["admin"] and not bool(grant.get("allow_admin")):
        return "admin_not_granted"
    if concrete["requires_sql_validation"] and not concrete["sql_validation_ok"]:
        return "sql_validation_required"
    if concrete["requires_guardrail_pass"] and not concrete["guardrail_ok"]:
        return "connector_guardrail_required"
    if (
        bool(grant.get("requires_idempotency_key"))
        and not concrete["idempotency_key_present"]
    ):
        return "idempotency_key_required"
    return None


def authorization_concrete_facts(
    request: Any,
    *,
    require_destructive: bool,
    require_admin: bool,
) -> dict[str, Any]:
    governance_facts = request_governance_facts(request)
    authoritative = governance_facts.get("authoritative")
    authoritative = authoritative if isinstance(authoritative, Mapping) else {}
    validation = authoritative.get("validation")
    validation = validation if isinstance(validation, Mapping) else {}
    operation = authoritative.get("operation") or governance_facts.get("operation")
    operation = operation if isinstance(operation, Mapping) else {}
    contract = authoritative.get("contract") or governance_facts.get("contract")
    contract = contract if isinstance(contract, Mapping) else {}
    task = authoritative.get("task") or governance_facts.get("task")
    task = task if isinstance(task, Mapping) else {}
    capability = authoritative.get("capability") or governance_facts.get("capability")
    capability = capability if isinstance(capability, Mapping) else {}

    capabilities = concrete_capabilities(task, capability, operation, contract)
    access = concrete_access(
        operation,
        capability,
        validation,
        require_destructive=require_destructive,
        require_admin=require_admin,
    )
    task_present = bool(task.get("id"))
    requires_sql_validation = task_present and "db.sql.execute_write" in capabilities
    validation_valid = any(
        bool(item.get("valid")) for item in validation_statements(validation)
    )
    guardrail_ok = validation_guardrail_ok(validation)
    destructive = require_destructive or bool(
        safe_strings(validation.get("destructive_statement_classes"))
    )
    admin = require_admin or bool(
        safe_strings(validation.get("admin_statement_classes"))
    )
    return {
        "authorization_decision_source": "authorization_facts",
        "lanes": concrete_lanes(task, operation, contract),
        "capabilities": list(capabilities),
        "contract_capabilities": safe_strings(
            contract.get("required_capabilities")
            or contract.get("selected_capability_ids")
            or operation.get("capability_ids")
        ),
        "source_scope": concrete_source_scope(governance_facts, validation),
        "access": access,
        "destructive": destructive,
        "admin": admin,
        "requires_sql_validation": requires_sql_validation,
        "sql_validation_ok": (not requires_sql_validation) or validation_valid,
        "requires_guardrail_pass": requires_sql_validation,
        "guardrail_ok": (not requires_sql_validation) or guardrail_ok,
        "idempotency_key_present": has_idempotency_key(request, task),
    }


def concrete_lanes(
    task: Mapping[str, Any],
    operation: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[str, ...]:
    lanes = safe_strings((task.get("required_lane"), task.get("requested_lane")))
    if lanes:
        return lanes
    return safe_strings(contract.get("granted_lanes") or operation.get("granted_lanes"))


def concrete_capabilities(
    task: Mapping[str, Any],
    capability: Mapping[str, Any],
    operation: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[str, ...]:
    task_capability = task.get("capability_id") or capability.get("id")
    if task_capability:
        return (str(task_capability),)
    return safe_strings(
        operation.get("capability_ids")
        or contract.get("required_capabilities")
        or contract.get("selected_capability_ids")
    )


def concrete_access(
    operation: Mapping[str, Any],
    capability: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    require_destructive: bool,
    require_admin: bool,
) -> str:
    capability_access = str(
        capability.get("access") or operation.get("access") or ""
    ).lower()
    if (
        require_admin
        or safe_strings(validation.get("admin_statement_classes"))
        or capability_access == "admin"
    ):
        return "admin"
    if (
        require_destructive
        or safe_strings(validation.get("mutating_statement_classes"))
        or capability_access == "write"
    ):
        return "write"
    return "read"


def concrete_source_scope(
    governance_facts: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> tuple[str, ...]:
    return safe_strings(
        (
            *safe_strings(governance_facts.get("source_scope")),
            *safe_strings(validation.get("target_resources")),
        )
    )


def validation_statements(
    validation: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        item for item in validation.get("statements") or () if isinstance(item, Mapping)
    )


def validation_guardrail_ok(validation: Mapping[str, Any]) -> bool:
    results = safe_strings(validation.get("guardrail_results"))
    return bool(results) and all(
        str(result).lower() in {"passed", "allow", "allowed", "ok"}
        for result in results
    )


def source_in_scope(source: str, granted_sources: frozenset[str]) -> bool:
    if source in granted_sources:
        return True
    return any(source.endswith(f".{granted}") for granted in granted_sources) or any(
        granted.endswith(f".{source}") for granted in granted_sources
    )


def access_rank(access: str) -> int:
    return {"read": 1, "write": 2, "admin": 3}.get(
        str(access or "read").strip().lower(),
        1,
    )


def has_idempotency_key(request: Any, task: Mapping[str, Any]) -> bool:
    if bool(task.get("idempotency_key_present")):
        return True
    if isinstance(task.get("metadata"), Mapping) and task["metadata"].get(
        "idempotency_key"
    ):
        return True
    task_input = task.get("input")
    if isinstance(task_input, Mapping) and task_input.get("idempotency_key"):
        return True
    if isinstance(request, Mapping):
        if request.get("idempotency_key"):
            return True
        metadata = request.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("idempotency_key"):
            return True
    return False


def safe_strings(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    if isinstance(values, (list, tuple, set, frozenset)):
        return tuple(str(value) for value in values if value is not None)
    return (str(values),)


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_grants(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    grants: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        grants.append(
            {
                "id": optional_string(item.get("id")),
                "principal": optional_string(item.get("principal")),
                "lanes": list(safe_strings(item.get("lanes"))),
                "capabilities": list(safe_strings(item.get("capabilities"))),
                "source_scope": list(safe_strings(item.get("source_scope"))),
                "max_access": _authorization_access(item.get("max_access")),
                "allow_destructive": bool(item.get("allow_destructive")),
                "allow_admin": bool(item.get("allow_admin")),
                "requires_idempotency_key": bool(item.get("requires_idempotency_key")),
            }
        )
    return grants


def _authorization_access(value: Any) -> str:
    normalized = str(value or "read").strip().lower()
    return normalized if normalized in {"read", "write", "admin"} else "read"


def _authorization_mode(value: Any) -> str:
    mode = str(value or "interactive").strip().lower()
    return mode if mode in AUTHORIZATION_MODES else "interactive"


def _metadata_identifies_automation(metadata: Mapping[str, Any]) -> bool:
    if bool(metadata.get("automation") or metadata.get("automated")):
        return True
    for key in ("actor_type", "caller_type", "request_origin", "runtime_path"):
        value = str(metadata.get(key) or "").strip().lower()
        if value in {"automation", "api", "service", "monitor", "worker"}:
            return True
    return False
