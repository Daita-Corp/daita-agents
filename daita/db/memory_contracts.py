"""Pure helpers for DB memory semantic contracts.

This module is deliberately subordinate to the existing DB runtime owners: it
does not recall, store, plan, validate, repair, or execute SQL.
"""

from __future__ import annotations

import json
import re
from typing import Any

DB_MEMORY_SEMANTIC_CONTRACT_KEY = "semantic_contract"
DB_MEMORY_SEMANTIC_CONTRACT_VERSION = 1
DB_MEMORY_CONTRACT_KINDS = frozenset(
    {"metric_definition", "unit_convention", "value_alias"}
)
DB_MEMORY_ENFORCEABLE_CONTRACT_KINDS = frozenset(
    {"metric_definition", "unit_convention"}
)


def normalize_db_memory_semantic_contract(value: Any) -> dict[str, Any] | None:
    """Return a normalized versioned DB semantic contract, or None."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("semantic_contract must be an object")
    version = int(value.get("version") or DB_MEMORY_SEMANTIC_CONTRACT_VERSION)
    if version != DB_MEMORY_SEMANTIC_CONTRACT_VERSION:
        raise ValueError(f"unsupported semantic_contract version {version!r}")
    contract_kind = str(value.get("contract_kind") or "").strip()
    if contract_kind not in DB_MEMORY_CONTRACT_KINDS:
        raise ValueError(f"unsupported semantic_contract kind {contract_kind!r}")
    subject = value.get("subject") if isinstance(value.get("subject"), dict) else {}
    requirements = (
        value.get("requirements") if isinstance(value.get("requirements"), dict) else {}
    )
    grounding = (
        value.get("grounding") if isinstance(value.get("grounding"), dict) else {}
    )
    enforcement = (
        value.get("enforcement") if isinstance(value.get("enforcement"), dict) else {}
    )
    normalized = {
        "version": version,
        "contract_kind": contract_kind,
        "subject": {
            "type": str(subject.get("type") or contract_kind).strip(),
            "key": str(subject.get("key") or "").strip(),
            "aliases": _string_list(subject.get("aliases")),
        },
        "requirements": {
            "refs": _dict_list(requirements.get("refs")),
            "relationships": _dict_list(requirements.get("relationships")),
            "filters": _dict_list(requirements.get("filters")),
            "aggregations": _dict_list(requirements.get("aggregations")),
            "result_shape": (
                dict(requirements.get("result_shape"))
                if isinstance(requirements.get("result_shape"), dict)
                else {}
            ),
            "unit_conversion": (
                dict(requirements.get("unit_conversion"))
                if isinstance(requirements.get("unit_conversion"), dict)
                else {}
            ),
        },
        "grounding": {
            "source_identity": grounding.get("source_identity"),
            "schema_fingerprint": grounding.get("schema_fingerprint"),
            "catalog_refs": _string_list(grounding.get("catalog_refs")),
            "evidence_refs": _string_list(grounding.get("evidence_refs")),
        },
        "enforcement": {
            "mode": str(enforcement.get("mode") or "required_when_recalled").strip(),
            "min_confidence": confidence_value(
                enforcement.get("min_confidence"), default=0.8
            ),
        },
    }
    json.dumps(normalized, sort_keys=True, default=str)
    return normalized


def db_memory_contract_refs(
    contract: dict[str, Any] | None,
) -> tuple[dict[str, str], ...]:
    """Extract schema refs declared by a normalized semantic contract."""
    if not isinstance(contract, dict):
        return ()
    requirements = contract.get("requirements") or {}
    refs: list[dict[str, str]] = []
    for item in requirements.get("refs", []) or []:
        ref = _schema_ref_from_any(item.get("ref") if isinstance(item, dict) else item)
        if ref:
            refs.append(ref)
    for relationship in requirements.get("relationships", []) or []:
        if not isinstance(relationship, dict):
            continue
        for key in ("from", "to"):
            ref = _schema_ref_from_any(relationship.get(key))
            if ref:
                refs.append(ref)
    for item in requirements.get("filters", []) or []:
        ref = _schema_ref_from_any(item.get("ref") if isinstance(item, dict) else None)
        if ref:
            refs.append(ref)
    for item in requirements.get("aggregations", []) or []:
        ref = _schema_ref_from_any(item.get("ref") if isinstance(item, dict) else None)
        if ref:
            refs.append(ref)
    return tuple(_dedupe_schema_refs(refs))


def extract_db_memory_semantic_contract(
    record: Any,
    *,
    schema: dict[str, Any] | None = None,
    source_identity: str | None = None,
    schema_fingerprint: str | None = None,
    evidence_refs: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Extract a deterministic schema-grounded contract from a DB memory record."""
    metadata = dict(getattr(record, "metadata", {}) or {})
    existing = normalize_db_memory_semantic_contract(
        metadata.get(DB_MEMORY_SEMANTIC_CONTRACT_KEY)
    )
    if existing is not None:
        return existing

    kind = str(getattr(record, "kind", "") or "")
    if kind == "unit_convention":
        return _unit_convention_contract(
            record,
            schema=schema or {},
            source_identity=source_identity,
            schema_fingerprint=schema_fingerprint,
            evidence_refs=evidence_refs,
        )
    if kind == "metric_definition":
        return _metric_definition_contract(
            record,
            schema=schema or {},
            source_identity=source_identity,
            schema_fingerprint=schema_fingerprint,
            evidence_refs=evidence_refs,
        )
    return None


def project_db_memory_semantic_contracts(
    db_memory_refs: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    prompt: str,
    schema: dict[str, Any],
    policy_summary: dict[str, Any] | None = None,
    source_identity: str | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Project recalled DB memory refs into compact validation contracts."""
    diagnostics: dict[str, Any] = {
        "candidate_count": 0,
        "enforced_count": 0,
        "advisory_count": 0,
        "omitted_count": 0,
        "omitted_reasons": {},
    }
    projected: list[dict[str, Any]] = []
    for ref in db_memory_refs:
        try:
            contract = normalize_db_memory_semantic_contract(
                ref.get(DB_MEMORY_SEMANTIC_CONTRACT_KEY)
            )
        except Exception:
            contract = None
            _bump_omitted(diagnostics, "invalid_contract")
        if contract is None:
            diagnostics["advisory_count"] += 1
            continue
        diagnostics["candidate_count"] += 1
        omit_reason = _contract_omit_reason(
            ref,
            contract,
            prompt=prompt,
            schema=schema,
            policy_summary=policy_summary or {},
            source_identity=source_identity,
        )
        if omit_reason:
            _bump_omitted(diagnostics, omit_reason)
            projected.append(
                _compact_semantic_contract(
                    ref,
                    contract,
                    enforceable=False,
                    omission_reason=omit_reason,
                )
            )
            diagnostics["advisory_count"] += 1
            continue
        projected.append(_compact_semantic_contract(ref, contract, enforceable=True))
        diagnostics["enforced_count"] += 1
    return tuple(projected), diagnostics


def db_memory_contracts_artifact_payload(
    *,
    source_identity: str | None,
    schema_fingerprint: str | None,
    recall_evidence_refs: tuple[str, ...] | list[str],
    selection_evidence_ref: dict[str, Any] | None,
    contracts: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Return first-class evidence payload for DB memory contracts."""

    projected_contracts = [dict(contract) for contract in contracts]
    omitted_counts = {
        str(reason): int(count)
        for reason, count in dict(diagnostics.get("omitted_reasons") or {}).items()
    }
    enforceable = [
        dict(contract)
        for contract in projected_contracts
        if contract.get("enforceable") is True
    ]
    advisory = [
        dict(contract)
        for contract in projected_contracts
        if contract.get("enforceable") is not True
    ]
    return {
        "artifact_kind": "db.memory.contracts",
        "source_identity": source_identity,
        "schema_fingerprint": schema_fingerprint,
        "recall_evidence_refs": [
            str(ref) for ref in recall_evidence_refs if str(ref).strip()
        ],
        "selection_evidence_ref": dict(selection_evidence_ref or {}),
        "contracts": projected_contracts,
        "enforceable_contracts": enforceable,
        "advisory_contracts": advisory,
        "contract_omission_reasons": omitted_counts,
        "source_schema_applicability": {
            "source_identity": source_identity,
            "schema_fingerprint": schema_fingerprint,
            "contract_candidate_count": int(diagnostics.get("candidate_count") or 0),
            "enforced_count": int(diagnostics.get("enforced_count") or 0),
            "advisory_count": int(diagnostics.get("advisory_count") or 0),
            "omitted_count": int(diagnostics.get("omitted_count") or 0),
        },
        "safe_diagnostic_summaries": safe_omission_summaries(omitted_counts),
    }


def _metric_definition_contract(
    record: Any,
    *,
    schema: dict[str, Any],
    source_identity: str | None,
    schema_fingerprint: str | None,
    evidence_refs: tuple[str, ...],
) -> dict[str, Any] | None:
    metadata = dict(getattr(record, "metadata", {}) or {})
    if not _has_explicit_metric_requirements(metadata):
        return None
    requirements = _requirements_from_metadata(metadata)
    if not requirements:
        return None
    refs = db_memory_contract_refs({"requirements": requirements})
    if not refs:
        return None
    if schema.get("tables") and not schema_refs_known_schema(refs, schema):
        return None
    if not schema.get("tables") and not _metadata_schema_refs_cover(metadata, refs):
        return None
    subject = _subject_from_metadata(record, metadata, default_type="metric")
    return normalize_db_memory_semantic_contract(
        {
            "version": DB_MEMORY_SEMANTIC_CONTRACT_VERSION,
            "contract_kind": "metric_definition",
            "subject": subject,
            "requirements": requirements,
            "grounding": _grounding(
                metadata,
                source_identity=source_identity,
                schema_fingerprint=schema_fingerprint,
                evidence_refs=evidence_refs,
            ),
            "enforcement": _enforcement(metadata, min_confidence=0.8),
        }
    )


def _has_explicit_metric_requirements(metadata: dict[str, Any]) -> bool:
    if isinstance(metadata.get("requirements"), dict):
        return True
    return any(
        metadata.get(key)
        for key in (
            "required_relationships",
            "relationships",
            "required_filters",
            "filters",
            "required_aggregations",
            "aggregations",
            "result_shape",
        )
    )


def _unit_convention_contract(
    record: Any,
    *,
    schema: dict[str, Any],
    source_identity: str | None,
    schema_fingerprint: str | None,
    evidence_refs: tuple[str, ...],
) -> dict[str, Any] | None:
    metadata = dict(getattr(record, "metadata", {}) or {})
    table = str(metadata.get("table") or "").strip()
    column = str(metadata.get("column") or "").strip()
    unit = str(metadata.get("unit") or "").strip().lower()
    if not table or not column or not unit:
        return None
    refs = ({"table": table, "column": column},)
    if schema.get("tables") and not schema_refs_known_schema(refs, schema):
        return None
    if not schema.get("tables") and not _metadata_schema_refs_cover(metadata, refs):
        return None
    conversion: dict[str, Any] = {"stored_unit": unit}
    if unit == "cents":
        conversion.update(
            {"display_unit": "dollars", "operator": "divide", "factor": 100}
        )
    elif unit == "basis_points":
        conversion.update(
            {"display_unit": "percent", "operator": "divide", "factor": 100}
        )
    elif unit == "percent":
        conversion.update({"display_unit": "percent", "operator": "identity"})
    else:
        return None
    return normalize_db_memory_semantic_contract(
        {
            "version": DB_MEMORY_SEMANTIC_CONTRACT_VERSION,
            "contract_kind": "unit_convention",
            "subject": {
                "type": "column",
                "key": f"{table}.{column}",
                "aliases": _string_list(metadata.get("aliases"))
                or [column.replace("_", " ")],
            },
            "requirements": {
                "refs": [
                    {"kind": "column", "ref": f"{table}.{column}", "role": "unit"}
                ],
                "unit_conversion": conversion,
            },
            "grounding": _grounding(
                metadata,
                source_identity=source_identity,
                schema_fingerprint=schema_fingerprint,
                evidence_refs=evidence_refs,
            ),
            "enforcement": _enforcement(metadata, min_confidence=0.75),
        }
    )


def _requirements_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("requirements")
    if isinstance(raw, dict):
        requirements = dict(raw)
    else:
        requirements = {}
        for source_key, target_key in (
            ("required_refs", "refs"),
            ("refs", "refs"),
            ("schema_refs", "refs"),
            ("required_relationships", "relationships"),
            ("relationships", "relationships"),
            ("required_filters", "filters"),
            ("filters", "filters"),
            ("required_aggregations", "aggregations"),
            ("aggregations", "aggregations"),
        ):
            if metadata.get(source_key):
                requirements.setdefault(target_key, metadata[source_key])
        for key in ("result_shape", "unit_conversion"):
            if isinstance(metadata.get(key), dict):
                requirements[key] = dict(metadata[key])
    return {
        "refs": _normalize_requirement_refs(requirements.get("refs")),
        "relationships": _normalize_relationships(requirements.get("relationships")),
        "filters": _dict_list(requirements.get("filters")),
        "aggregations": _dict_list(requirements.get("aggregations")),
        "result_shape": (
            dict(requirements.get("result_shape"))
            if isinstance(requirements.get("result_shape"), dict)
            else {}
        ),
        "unit_conversion": (
            dict(requirements.get("unit_conversion"))
            if isinstance(requirements.get("unit_conversion"), dict)
            else {}
        ),
    }


def _normalize_requirement_refs(value: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, dict) and item.get("ref"):
                refs.append(dict(item))
                continue
            ref = _schema_ref_from_any(item)
            if ref and ref.get("column"):
                refs.append(
                    {
                        "kind": "column",
                        "ref": f"{ref['table']}.{ref['column']}",
                    }
                )
    return refs


def _normalize_relationships(value: Any) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    if not isinstance(value, (list, tuple)):
        return relationships
    for item in value:
        if isinstance(item, dict):
            if item.get("from") and item.get("to"):
                relationships.append(dict(item))
            continue
        if isinstance(item, str) and "->" in item:
            left, right = [part.strip() for part in item.split("->", maxsplit=1)]
            if left and right:
                relationships.append({"from": left, "to": right})
    return relationships


def _subject_from_metadata(
    record: Any, metadata: dict[str, Any], *, default_type: str
) -> dict[str, Any]:
    subject = metadata.get("subject")
    if not isinstance(subject, dict):
        subject = metadata.get("semantic_subject")
    if isinstance(subject, dict):
        return {
            "type": str(subject.get("type") or default_type).strip(),
            "key": str(subject.get("key") or getattr(record, "key", "")).strip(),
            "aliases": _string_list(subject.get("aliases")),
        }
    return {
        "type": default_type,
        "key": str(metadata.get("metric") or getattr(record, "key", "")).strip(),
        "aliases": _string_list(metadata.get("aliases")),
    }


def _grounding(
    metadata: dict[str, Any],
    *,
    source_identity: str | None,
    schema_fingerprint: str | None,
    evidence_refs: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "source_identity": source_identity or metadata.get("source_identity"),
        "schema_fingerprint": schema_fingerprint
        or metadata.get("source_schema_fingerprint")
        or metadata.get("schema_fingerprint"),
        "catalog_refs": _string_list(metadata.get("catalog_refs")),
        "evidence_refs": list(evidence_refs)
        or _string_list(metadata.get("evidence_refs")),
    }


def _enforcement(metadata: dict[str, Any], *, min_confidence: float) -> dict[str, Any]:
    enforcement = (
        dict(metadata.get("enforcement"))
        if isinstance(metadata.get("enforcement"), dict)
        else {}
    )
    return {
        "mode": str(enforcement.get("mode") or "required_when_recalled").strip(),
        "min_confidence": confidence_value(
            enforcement.get("min_confidence"), default=min_confidence
        ),
    }


def _compact_semantic_contract(
    ref: dict[str, Any],
    contract: dict[str, Any],
    *,
    enforceable: bool,
    omission_reason: str | None = None,
) -> dict[str, Any]:
    requirements = contract.get("requirements") or {}
    subject = contract.get("subject") or {}
    grounding = contract.get("grounding") or {}
    enforcement = contract.get("enforcement") or {}
    relationships = []
    for item in requirements.get("relationships", []) or []:
        if isinstance(item, dict) and item.get("from") and item.get("to"):
            relationships.append(f"{item.get('from')} -> {item.get('to')}")
    evidence_refs = list(ref.get("evidence_refs") or [])
    evidence_refs.extend(grounding.get("evidence_refs") or [])
    compact = {
        "key": subject.get("key") or ref.get("key"),
        "memory_key": ref.get("key"),
        "kind": ref.get("kind"),
        "contract_kind": contract.get("contract_kind"),
        "subject_aliases": list(subject.get("aliases") or []),
        "required_refs": [
            item["ref"]
            for item in requirements.get("refs", []) or []
            if isinstance(item, dict) and item.get("ref")
        ],
        "required_relationships": relationships,
        "required_filters": [
            dict(item)
            for item in requirements.get("filters", []) or []
            if isinstance(item, dict)
        ],
        "required_aggregations": [
            dict(item)
            for item in requirements.get("aggregations", []) or []
            if isinstance(item, dict)
        ],
        "result_shape": dict(requirements.get("result_shape") or {}),
        "unit_conversion": dict(requirements.get("unit_conversion") or {}),
        "evidence_refs": list(dict.fromkeys(evidence_refs)),
        "confidence": ref.get("confidence"),
        "enforcement_mode": enforcement.get("mode") or "required_when_recalled",
        "enforceable": bool(enforceable),
    }
    if omission_reason:
        compact["omission_reason"] = omission_reason
    return compact


def _contract_omit_reason(
    ref: dict[str, Any],
    contract: dict[str, Any],
    *,
    prompt: str,
    schema: dict[str, Any],
    policy_summary: dict[str, Any],
    source_identity: str | None,
) -> str | None:
    contract_kind = str(contract.get("contract_kind") or "")
    if contract_kind not in DB_MEMORY_ENFORCEABLE_CONTRACT_KINDS:
        return "advisory_kind"
    if ref.get("active") is False:
        return "inactive"
    if ref.get("stale") is True:
        return "stale"
    if str(ref.get("semantic_contract_status") or "") != "validated":
        return "unvalidated_contract"
    if source_identity and ref.get("source_identity") not in {None, source_identity}:
        return "cross_source"
    enforcement = contract.get("enforcement") or {}
    if str(enforcement.get("mode") or "required_when_recalled") not in {
        "required_when_recalled",
        "required_when_relevant",
    }:
        return "advisory_mode"
    confidence = confidence_value(ref.get("confidence"), default=0.0)
    min_confidence = confidence_value(enforcement.get("min_confidence"), default=0.8)
    if confidence < min_confidence:
        return "low_confidence"
    if not _contract_refs_known_schema(contract, schema):
        return "schema_scope_mismatch"
    if _contract_blocked_by_policy(contract, policy_summary):
        return "blocked_by_policy"
    if not _contract_relevant_to_prompt(prompt, ref, contract):
        return "irrelevant"
    return None


def _contract_refs_known_schema(
    contract: dict[str, Any], schema: dict[str, Any]
) -> bool:
    refs = db_memory_contract_refs(contract)
    if not refs:
        return True
    if not schema.get("tables"):
        return False
    return schema_refs_known_schema(refs, schema)


def _contract_blocked_by_policy(
    contract: dict[str, Any], policy_summary: dict[str, Any]
) -> bool:
    blocked_tables = {
        _short_ref_key(table)
        for table in policy_summary.get("blocked_tables", []) or []
        if str(table).strip()
    }
    blocked_columns = {
        str(column).strip().lower()
        for column in policy_summary.get("blocked_columns", []) or []
        if str(column).strip()
    }
    for ref in db_memory_contract_refs(contract):
        table = _short_ref_key(ref.get("table"))
        column = str(ref.get("column") or "").lower()
        if table and table in blocked_tables:
            return True
        if table and column and f"{table}.{column}" in blocked_columns:
            return True
        if column and column in blocked_columns:
            return True
    return False


def _contract_relevant_to_prompt(
    prompt: str,
    ref: dict[str, Any],
    contract: dict[str, Any],
) -> bool:
    subject = contract.get("subject") or {}
    terms = [
        ref.get("key"),
        ref.get("text"),
        subject.get("key"),
        *(subject.get("aliases") or []),
    ]
    prompt_tokens = set(meaningful_tokens(prompt))
    contract_tokens = set(meaningful_tokens(" ".join(str(term) for term in terms)))
    return not prompt_tokens or bool(prompt_tokens & contract_tokens)


def schema_refs_known_schema(
    refs: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    schema: dict[str, Any],
) -> bool:
    tables = {
        str(table.get("name") or "").lower(): {
            str(column.get("name") or "").lower()
            for column in table.get("columns", []) or []
            if column.get("name")
        }
        for table in schema.get("tables", []) or []
        if table.get("name")
    }
    if not tables:
        return True
    for ref in refs:
        table = str(ref.get("table") or "").lower()
        column = str(ref.get("column") or "").lower()
        if table and table not in tables:
            return False
        if table and column and column not in tables.get(table, set()):
            return False
    return True


def _metadata_schema_refs_cover(
    metadata: dict[str, Any],
    required_refs: tuple[dict[str, Any], ...],
) -> bool:
    raw_refs = metadata.get("schema_refs")
    if not isinstance(raw_refs, (list, tuple, set)):
        return False
    refs = {
        f"{ref['table'].lower()}.{ref['column'].lower()}"
        for item in raw_refs
        for ref in [_schema_ref_from_any(item)]
        if ref and ref.get("table") and ref.get("column")
    }
    for required in required_refs:
        if not required.get("column"):
            continue
        key = f"{required.get('table', '').lower()}.{required['column'].lower()}"
        if key not in refs:
            return False
    return True


def _schema_ref_from_any(value: Any) -> dict[str, str] | None:
    if isinstance(value, dict):
        table = str(value.get("table") or "").strip()
        column = str(value.get("column") or "").strip()
    else:
        parts = [
            part.strip('"`[] ') for part in str(value or "").split(".") if part.strip()
        ]
        if len(parts) >= 2:
            table, column = parts[-2], parts[-1]
        elif len(parts) == 1:
            table, column = parts[0], ""
        else:
            table, column = "", ""
    if not table:
        return None
    ref = {"table": table}
    if column:
        ref["column"] = column
    return ref


def _dedupe_schema_refs(refs: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    deduped: list[dict[str, str]] = []
    for ref in refs:
        key = (
            str(ref.get("table") or "").lower(),
            str(ref.get("column") or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def confidence_value(value: Any, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "high":
            return 0.9
        if lowered == "medium":
            return 0.7
        if lowered == "low":
            return 0.4
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _short_ref_key(value: Any) -> str:
    return str(value or "").split(".")[-1].strip().lower()


def meaningful_tokens(text: Any) -> list[str]:
    stop = {
        "about",
        "after",
        "before",
        "calculate",
        "computed",
        "does",
        "from",
        "have",
        "many",
        "show",
        "that",
        "their",
        "there",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", str(text or "").lower())
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(part for part in token.split("_") if len(part) > 2)
        expanded.append(token)
    return [token for token in expanded if token not in stop]


def _bump_omitted(diagnostics: dict[str, Any], reason: str) -> None:
    diagnostics["omitted_count"] = int(diagnostics.get("omitted_count") or 0) + 1
    omitted = diagnostics.setdefault("omitted_reasons", {})
    omitted[reason] = int(omitted.get(reason) or 0) + 1


def safe_omission_summaries(
    omitted_counts: dict[str, int],
) -> list[dict[str, Any]]:
    return [
        {"reason": reason, "count": int(count)}
        for reason, count in sorted(omitted_counts.items())
        if int(count) > 0
    ]
