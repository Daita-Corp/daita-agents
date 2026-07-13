"""Planner-safe selection of accepted DB-memory recall evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from .contracts import (
    DB_MEMORY_SEMANTIC_CONTRACT_KEY,
    confidence_value,
    meaningful_tokens,
    normalize_db_memory_semantic_contract,
    safe_omission_summaries,
    schema_refs_known_schema,
)
from .records import DB_PLANNING_MEMORY_KINDS
from .safety import db_memory_pii_error


def db_memory_refs_from_recall_evidence(
    recall_evidence: tuple[Any, ...],
    *,
    prompt: str,
    schema: dict[str, Any],
    source_identity: str | None,
    schema_fingerprint: str | None,
    limit: int = 3,
    char_budget: int = 800,
    score_threshold: float = 0.45,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], dict[str, Any]]:
    """Project semantic recall evidence into compact, planner-safe DB refs."""
    diagnostics: dict[str, Any] = {
        "registered": bool(recall_evidence),
        "queried": bool(recall_evidence),
        "candidate_count": 0,
        "included_count": 0,
        "omitted_reasons": {},
    }
    candidates: list[tuple[dict[str, Any], Any]] = []
    for evidence in recall_evidence:
        payload = getattr(evidence, "payload", {}) or {}
        for result in payload.get("results", []) or []:
            if isinstance(result, dict):
                candidates.append((result, evidence))
    diagnostics["candidate_count"] = len(candidates)

    eligible: list[dict[str, Any]] = []
    for result, evidence in candidates:
        record = _db_memory_record_from_recall_result(result)
        reason = _planner_memory_omit_reason(
            record,
            result,
            prompt=prompt,
            schema=schema,
            source_identity=source_identity,
            schema_fingerprint=schema_fingerprint,
            score_threshold=score_threshold,
        )
        if reason:
            _bump_omitted(diagnostics, reason)
            continue
        raw_metadata = record.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        eligible.append(
            {
                "result": result,
                "record": record,
                "evidence_ids": (
                    [str(evidence.id)] if getattr(evidence, "id", None) else []
                ),
                "record_evidence_refs": _memory_evidence_ref_ids(metadata),
            }
        )

    selected_candidates, duplicate_count = _dedupe_planning_memory_candidates(eligible)
    diagnostics["deduplicated_candidate_count"] = len(selected_candidates)
    if duplicate_count:
        diagnostics["omitted_reasons"]["duplicate"] = duplicate_count

    refs: list[dict[str, Any]] = []
    evidence_refs: list[str] = []
    used_chars = 0
    valid_until: list[datetime] = []
    valid_until_raw: dict[datetime, str] = {}
    for candidate in selected_candidates:
        if len(refs) >= max(0, int(limit)):
            _bump_omitted(diagnostics, "limit")
            continue
        result = candidate["result"]
        record = candidate["record"]
        text = str(record.get("text") or "").strip()
        key = str(record.get("key") or "").strip()
        kind = str(record.get("kind") or "").strip()
        line = f"- {kind} {key}: {text}"
        if used_chars + len(line) > max(0, int(char_budget)):
            _bump_omitted(diagnostics, "budget")
            continue
        raw_metadata = record.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        ref_evidence = list(candidate["record_evidence_refs"])
        evidence_refs.extend(candidate["evidence_ids"])
        evidence_refs.extend(ref_evidence)
        ref = {
            "chunk_id": result.get("chunk_id"),
            "kind": kind,
            "key": key,
            "text": text,
            "confidence": confidence_value(metadata.get("confidence"), default=1.0),
            "importance": float(record.get("importance") or 0.0),
            "source_identity": metadata.get("source_identity"),
            "evidence_refs": ref_evidence,
            "schema_fingerprint": metadata.get("source_schema_fingerprint")
            or metadata.get("schema_fingerprint"),
        }
        if metadata.get("active") is False:
            ref["active"] = False
        if metadata.get("stale") is True:
            ref["stale"] = True
        if metadata.get("creation_path"):
            ref["creation_path"] = metadata.get("creation_path")
        if metadata.get("semantic_contract_status"):
            ref["semantic_contract_status"] = metadata.get("semantic_contract_status")
        try:
            contract = normalize_db_memory_semantic_contract(
                metadata.get(DB_MEMORY_SEMANTIC_CONTRACT_KEY)
            )
        except Exception:
            contract = None
        if contract is not None:
            ref[DB_MEMORY_SEMANTIC_CONTRACT_KEY] = contract
        refs.append(ref)
        used_chars += len(line)
        expires_at = _memory_expiry(metadata)
        if expires_at is not None:
            valid_until.append(expires_at)
            valid_until_raw[expires_at] = str(metadata.get("expires_at"))
    diagnostics["included_count"] = len(refs)
    diagnostics["char_budget"] = int(char_budget)
    diagnostics["used_chars"] = used_chars
    if valid_until:
        earliest = min(valid_until)
        diagnostics["valid_until"] = valid_until_raw[earliest]
    return tuple(refs), tuple(dict.fromkeys(evidence_refs)), diagnostics


def db_memory_selection_artifact_payload(
    *,
    source_identity: str | None,
    schema_fingerprint: str | None,
    recall_evidence_refs: tuple[str, ...] | list[str],
    memory_evidence_refs: tuple[str, ...] | list[str],
    included_refs: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    diagnostics: dict[str, Any],
    limit: int,
    char_budget: int,
    score_threshold: float,
) -> dict[str, Any]:
    """Return first-class evidence payload for DB memory selection."""
    omitted_counts = {
        str(reason): int(count)
        for reason, count in dict(diagnostics.get("omitted_reasons") or {}).items()
    }
    raw_candidate_count = int(diagnostics.get("candidate_count") or 0)
    included_count = int(diagnostics.get("included_count") or len(included_refs))
    return {
        "artifact_kind": "db.memory.selection",
        "source_identity": source_identity,
        "schema_fingerprint": schema_fingerprint,
        "recall_evidence_refs": [
            str(ref) for ref in recall_evidence_refs if str(ref).strip()
        ],
        "memory_evidence_refs": [
            str(ref) for ref in memory_evidence_refs if str(ref).strip()
        ],
        "raw_candidate_count": raw_candidate_count,
        "included_refs": [dict(ref) for ref in included_refs],
        "included_count": included_count,
        "omitted_counts_by_reason": omitted_counts,
        "safe_diagnostic_omission_summaries": safe_omission_summaries(omitted_counts),
        "freshness": {
            "checked_guards": ["active", "stale", "expires_at"],
            "valid_until": diagnostics.get("valid_until"),
        },
        "budget_usage": {
            "limit": max(0, int(limit)),
            "char_budget": max(0, int(char_budget)),
            "used_chars": int(diagnostics.get("used_chars") or 0),
            "score_threshold": float(score_threshold),
            "included_count": included_count,
            "raw_candidate_count": raw_candidate_count,
        },
    }


def db_memory_record_refs_known_schema(
    metadata: dict[str, Any],
    schema: dict[str, Any],
) -> bool:
    """Return whether DB memory metadata references known schema objects."""
    return _record_refs_known_schema(metadata, schema)


def _db_memory_record_from_recall_result(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata") or {}
    db_memory = metadata.get("db_memory") if isinstance(metadata, dict) else None
    if isinstance(db_memory, dict):
        return dict(db_memory)

    content = str(result.get("content", ""))
    try:
        marker = "DB memory record:\n"
        if content.startswith(marker):
            payload = json.loads(content[len(marker) :])
            if isinstance(payload, dict):
                return payload
    except Exception:
        return {}
    return {}


def _planner_memory_omit_reason(
    record: dict[str, Any],
    result: dict[str, Any],
    *,
    prompt: str,
    schema: dict[str, Any],
    source_identity: str | None,
    schema_fingerprint: str | None,
    score_threshold: float,
) -> str | None:
    kind = str(record.get("kind") or "")
    key = str(record.get("key") or "")
    text = str(record.get("text") or "")
    raw_metadata = record.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}

    if kind not in DB_PLANNING_MEMORY_KINDS:
        return "unsupported_kind"
    if not key or not text:
        return "malformed"
    if not _score_is_high_enough(result, score_threshold):
        return "low_score"
    record_source = metadata.get("source_identity")
    if source_identity and record_source != source_identity:
        return "cross_source" if record_source else "missing_source_identity"
    if metadata.get("active") is False:
        return "inactive"
    if metadata.get("stale") is True:
        return "stale"
    if _is_memory_expired(metadata):
        return "expired"
    record_schema = metadata.get("source_schema_fingerprint") or metadata.get(
        "schema_fingerprint"
    )
    if record_schema and schema_fingerprint and record_schema != schema_fingerprint:
        return "schema_mismatch"
    if confidence_value(metadata.get("confidence"), default=1.0) < 0.5:
        return "low_confidence"
    if kind == "value_alias" and not _value_alias_has_catalog_citation(metadata):
        return "missing_catalog_citation"
    if db_memory_pii_error(key=key, text=text, metadata=metadata):
        return "unsafe"
    if not _record_refs_known_schema(metadata, schema):
        if _has_valid_semantic_contract(metadata):
            return None
        return "schema_scope_mismatch"
    if not _memory_relevant_to_prompt(prompt, record):
        return "irrelevant"
    return None


def _has_valid_semantic_contract(metadata: dict[str, Any]) -> bool:
    try:
        return (
            normalize_db_memory_semantic_contract(
                metadata.get(DB_MEMORY_SEMANTIC_CONTRACT_KEY)
            )
            is not None
        )
    except Exception:
        return False


def _score_is_high_enough(result: dict[str, Any], threshold: float) -> bool:
    score = result.get("relevance_score", result.get("score"))
    if score is None:
        return True
    try:
        return float(score) >= float(threshold)
    except (TypeError, ValueError):
        return True


def _is_memory_expired(metadata: dict[str, Any]) -> bool:
    parsed = _memory_expiry(metadata)
    return parsed is not None and parsed <= datetime.now(timezone.utc)


def _memory_expiry(metadata: dict[str, Any]) -> datetime | None:
    expires_at = metadata.get("expires_at")
    if not expires_at:
        return None
    try:
        parsed = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _value_alias_has_catalog_citation(metadata: dict[str, Any]) -> bool:
    return bool(
        metadata.get("catalog_profile_ref")
        or metadata.get("catalog_evidence_id")
        or metadata.get("catalog_refs")
    )


def _record_refs_known_schema(metadata: dict[str, Any], schema: dict[str, Any]) -> bool:
    schema_refs = metadata.get("schema_refs")
    if isinstance(schema_refs, list) and schema_refs:
        return schema_refs_known_schema(_schema_refs_from_metadata(schema_refs), schema)
    tables = {
        str(table.get("name") or "").lower(): {
            str(column.get("name") or "").lower()
            for column in table.get("columns", []) or []
            if column.get("name")
        }
        for table in schema.get("tables", []) or []
        if table.get("name")
    }
    table = str(metadata.get("table") or "").lower()
    column = str(metadata.get("column") or "").lower()
    if table and table not in tables:
        return False
    if table and column and column not in tables.get(table, set()):
        return False
    return True


def _schema_refs_from_metadata(value: Any) -> tuple[dict[str, str], ...]:
    refs: list[dict[str, str]] = []
    if not isinstance(value, (list, tuple, set)):
        return ()
    for item in value:
        if isinstance(item, dict):
            table = str(item.get("table") or "").strip()
            column = str(item.get("column") or "").strip()
        else:
            parts = [
                part.strip('"`[] ')
                for part in str(item or "").split(".")
                if part.strip()
            ]
            if len(parts) >= 2:
                table, column = parts[-2], parts[-1]
            else:
                table, column = "", ""
        if table and column:
            refs.append({"table": table, "column": column})
        elif table:
            refs.append({"table": table})
    return tuple(refs)


def _memory_relevant_to_prompt(prompt: str, record: dict[str, Any]) -> bool:
    prompt_tokens = set(meaningful_tokens(prompt))
    if not prompt_tokens:
        return True
    raw_metadata = record.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    record_text = " ".join(
        str(value)
        for value in (
            record.get("kind"),
            record.get("key"),
            record.get("text"),
            metadata.get("metric"),
            metadata.get("table"),
            metadata.get("column"),
            metadata.get("alias"),
        )
        if value
    )
    record_tokens = set(meaningful_tokens(record_text))
    return bool(prompt_tokens & record_tokens)


def _memory_evidence_ref_ids(metadata: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    raw = metadata.get("evidence_refs")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("id"):
                refs.append(str(item["id"]))
            elif isinstance(item, str):
                refs.append(item)
    for key in ("catalog_evidence_id", "proposal_evidence_id"):
        if metadata.get(key):
            refs.append(str(metadata[key]))
    return list(dict.fromkeys(refs))


def _bump_omitted(diagnostics: dict[str, Any], reason: str) -> None:
    omitted = diagnostics.setdefault("omitted_reasons", {})
    omitted[reason] = int(omitted.get(reason) or 0) + 1


def _dedupe_planning_memory_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    deduped: list[dict[str, Any]] = []
    index_by_identity: dict[tuple[Any, ...], int] = {}
    duplicate_count = 0
    for candidate in candidates:
        identity = _planning_memory_candidate_identity(candidate)
        if identity is None:
            deduped.append(candidate)
            continue
        existing_index = index_by_identity.get(identity)
        if existing_index is None:
            index_by_identity[identity] = len(deduped)
            deduped.append(candidate)
            continue
        duplicate_count += 1
        existing = deduped[existing_index]
        merged_evidence_ids = sorted(
            {
                *existing.get("evidence_ids", ()),
                *candidate.get("evidence_ids", ()),
            }
        )
        merged_record_refs = sorted(
            {
                *existing.get("record_evidence_refs", ()),
                *candidate.get("record_evidence_refs", ()),
            }
        )
        selected = (
            candidate
            if _planning_memory_candidate_rank(candidate)
            > _planning_memory_candidate_rank(existing)
            else existing
        )
        deduped[existing_index] = {
            **selected,
            "evidence_ids": merged_evidence_ids,
            "record_evidence_refs": merged_record_refs,
        }
    return deduped, duplicate_count


def _planning_memory_candidate_identity(
    candidate: dict[str, Any],
) -> tuple[Any, ...] | None:
    result = candidate.get("result")
    result = result if isinstance(result, dict) else {}
    record_id = result.get("chunk_id") or result.get("record_id")
    if record_id:
        return ("record", str(record_id))
    record = candidate.get("record")
    record = record if isinstance(record, dict) else {}
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    source_identity = str(metadata.get("source_identity") or "").strip()
    kind = str(record.get("kind") or "").strip()
    key = str(record.get("key") or "").strip()
    if source_identity and kind and key:
        return ("semantic", source_identity, kind, key)
    return None


def _planning_memory_candidate_rank(candidate: dict[str, Any]) -> float:
    result = candidate.get("result")
    result = result if isinstance(result, dict) else {}
    value = result.get("relevance_score", result.get("score"))
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")
