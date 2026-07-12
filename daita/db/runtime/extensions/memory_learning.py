"""Runtime-owned executors for deterministic DB memory learning."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timezone
import re
from typing import Any, Mapping

from daita.runtime import Evidence, Operation, Task

from ...analysis import structural_schema_fingerprint
from ...fingerprints import persisted_fingerprint
from ...memory import (
    DB_MEMORY_SEMANTIC_CONTRACT_KEY,
    DBMemoryRecord,
    db_memory_pii_error,
    db_memory_record_chunk_ids_by_key,
    db_memory_record_refs_known_schema,
    normalize_db_memory_record,
    unit_records_from_schema,
)
from ...memory_contracts import extract_db_memory_semantic_contract
from ..memory_learning import _learner_task_id_from_operation
from ..tasks.models import DbTaskSpec

_LEARNING_SOURCE_EVIDENCE_KINDS = frozenset(
    {
        "planning.context",
        "query.plan.proposal",
        "query.plan.validation",
        "sql.validation",
        "query.result",
        "analysis.checkpoint",
        "analysis.synthesis",
        "schema.search_result",
        "schema.asset_profile",
        "schema.relationship_path",
        "schema.column_value_profile",
        "schema.column_value_search_result",
        "schema.column_value_hint",
        "catalog.source",
    }
)


@dataclass(frozen=True)
class DbMemoryLearningEnqueueExecutor:
    """Executor that plans the worker-owned memory learning task."""

    plugin: Any
    id: str = "db_runtime.memory.learning.enqueue"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.memory.learning.enqueue"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        source_operation_id = str(task.input.get("source_operation_id") or "").strip()
        source_identity = str(task.input.get("source_identity") or "").strip()
        learning_mode = str(task.input.get("learning_mode") or "safe").strip()
        source_evidence_ids = [
            str(item)
            for item in task.input.get("source_evidence_ids", []) or []
            if str(item).strip()
        ]
        run_capability = runtime.registry.get_capability(
            "db.memory.learning.run",
            owner="db_runtime",
        )
        existing = [
            item
            for item in await runtime.store.list_tasks(operation.id)
            if item.capability_id == run_capability.id
            and item.metadata.get("owner") == run_capability.owner
        ]
        if not existing:
            await runtime.plan_task_specs(
                operation,
                (
                    DbTaskSpec(
                        capability_id=run_capability.id,
                        task_id=_learner_task_id_from_operation(operation.id),
                        owner=run_capability.owner,
                        input={
                            "source_operation_id": source_operation_id,
                            "source_operation_type": task.input.get(
                                "source_operation_type"
                            ),
                            "source_identity": source_identity,
                            "source_schema_fingerprint": task.input.get(
                                "source_schema_fingerprint"
                            ),
                            "source_evidence_ids": source_evidence_ids,
                            "learning_mode": learning_mode,
                        },
                        reason="db_memory_learning_run",
                        sequence=1,
                        metadata={
                            "queue": "memory_learning",
                            "source_operation_id": source_operation_id,
                            "source_identity": source_identity,
                            "learning_mode": learning_mode,
                            "worker_id": "db.memory.learner",
                            "worker_owner": "db_runtime",
                        },
                        idempotency_key=persisted_fingerprint(
                            {
                                "source_operation_id": source_operation_id,
                                "source_identity": source_identity,
                                "learning_mode": learning_mode,
                            }
                        ),
                    ),
                ),
            )
        payload = {
            "enqueued": True,
            "source_operation_id": source_operation_id,
            "source_operation_type": task.input.get("source_operation_type"),
            "source_identity": source_identity,
            "source_schema_fingerprint": task.input.get("source_schema_fingerprint"),
            "learning_mode": learning_mode,
            "queue": "memory_learning",
            "worker_id": "db.memory.learner",
            "worker_owner": "db_runtime",
            "run_task_id": _learner_task_id_from_operation(operation.id),
            "source_evidence_ids": source_evidence_ids,
        }
        return [
            Evidence(
                kind="db.memory.learning.enqueue",
                owner="db_runtime",
                operation_id=operation.id,
                task_id=task.id,
                payload=payload,
                metadata={
                    "payload_fingerprint": persisted_fingerprint(payload),
                    "source_operation_id": source_operation_id,
                    "source_identity": source_identity,
                    "learning_mode": learning_mode,
                },
            )
        ]


@dataclass(frozen=True)
class DbMemoryLearningRunExecutor:
    """Worker executor that extracts and promotes safe DB memory candidates."""

    plugin: Any
    id: str = "db_runtime.memory.learning.run"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.memory.learning.run"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        source_operation_id = str(task.input.get("source_operation_id") or "").strip()
        source_identity = str(task.input.get("source_identity") or "").strip()
        schema_fingerprint = (
            str(task.input.get("source_schema_fingerprint") or "").strip() or None
        )
        source_evidence = await _load_source_evidence(runtime, task)
        learned_at = datetime.now(timezone.utc).isoformat()
        candidates = _extract_candidates(
            source_evidence,
            source_operation_id=source_operation_id,
            fallback_source_identity=source_identity,
            schema_fingerprint=schema_fingerprint,
            learning_mode=str(task.input.get("learning_mode") or "safe"),
            learned_at=learned_at,
        )
        schema = _schema_from_evidence(source_evidence)
        emitted: list[Evidence] = []
        for candidate in candidates:
            candidate_payload = _candidate_payload(candidate)
            emitted.append(
                Evidence(
                    kind="db.memory.candidate",
                    owner="db_runtime",
                    operation_id=operation.id,
                    task_id=task.id,
                    payload=candidate_payload,
                    metadata={
                        "candidate_key": candidate.key,
                        "candidate_kind": candidate.kind,
                        "source_operation_id": source_operation_id,
                        "source_identity": candidate.source_identity,
                        "payload_fingerprint": persisted_fingerprint(candidate_payload),
                    },
                )
            )
            rejection = await _promotion_rejection(
                runtime,
                candidate,
                expected_source_identity=source_identity,
                schema=schema,
            )
            if rejection is not None:
                emitted.append(
                    _rejection_evidence(
                        operation,
                        task,
                        candidate,
                        reason=rejection,
                    )
                )
                continue

            write_evidence = await _write_candidate(runtime, operation, candidate)
            write_success = bool(write_evidence) and all(
                item.payload.get("success") is not False for item in write_evidence
            )
            emitted.append(
                _promotion_evidence(
                    operation,
                    task,
                    candidate,
                    write_evidence=write_evidence,
                    accepted=write_success,
                )
            )
            if not write_success:
                emitted.append(
                    _rejection_evidence(
                        operation,
                        task,
                        candidate,
                        reason="memory_write_failed",
                        write_evidence=write_evidence,
                    )
                )
        return emitted


@dataclass(frozen=True)
class _Candidate:
    kind: str
    key: str
    text: str
    metadata: dict[str, Any]
    importance: float
    confidence: float
    candidate_type: str
    source_operation_id: str
    source_identity: str | None
    schema_fingerprint: str | None
    schema: dict[str, Any]
    evidence_refs: tuple[str, ...]
    learned_at: str

    def record(self) -> DBMemoryRecord:
        metadata = {
            **self.metadata,
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
            "source_operation_id": self.source_operation_id,
            "source_identity": self.source_identity,
            "workspace_scope": "source",
            "source_schema_fingerprint": self.schema_fingerprint,
            "created_by": "db_memory_learner:v1",
            "creation_path": "automatic_worker",
            "promotion_policy": "auto_high_confidence",
            "active": True,
            "last_verified_at": self.learned_at,
            "verification_count": 1,
            "negative_feedback_count": 0,
        }
        draft = DBMemoryRecord(
            kind=self.kind,
            key=self.key,
            text=self.text,
            metadata=metadata,
            importance=self.importance,
        )
        contract = extract_db_memory_semantic_contract(
            draft,
            schema=self.schema,
            source_identity=self.source_identity,
            schema_fingerprint=self.schema_fingerprint,
            evidence_refs=self.evidence_refs,
        )
        if contract is not None:
            metadata.setdefault(DB_MEMORY_SEMANTIC_CONTRACT_KEY, contract)
            metadata.setdefault("semantic_contract_status", "validated")
        return DBMemoryRecord(
            kind=self.kind,
            key=self.key,
            text=self.text,
            metadata=metadata,
            importance=self.importance,
        )


async def _load_source_evidence(runtime: Any, task: Task) -> tuple[Evidence, ...]:
    source_operation_id = str(task.input.get("source_operation_id") or "").strip()
    if not source_operation_id:
        return ()
    allowed_ids = {
        str(item)
        for item in task.input.get("source_evidence_ids", []) or []
        if str(item).strip()
    }
    evidence = [
        _safe_learning_evidence(item)
        for item in await runtime.store.list_evidence(source_operation_id)
        if item.accepted
        and item.id
        and item.kind in _LEARNING_SOURCE_EVIDENCE_KINDS
        and (not allowed_ids or item.id in allowed_ids)
    ]
    return tuple(evidence[:50])


def _safe_learning_evidence(evidence: Evidence) -> Evidence:
    if evidence.kind != "query.result":
        if evidence.kind == "planning.context":
            payload = dict(evidence.payload)
            payload["column_value_hints"] = [
                _safe_column_value_hint(hint)
                for hint in payload.get("column_value_hints", []) or []
                if isinstance(hint, dict)
            ]
            return replace(evidence, payload=payload)
        if evidence.kind == "schema.column_value_hint":
            payload = dict(evidence.payload)
            payload["hints"] = [
                _safe_column_value_hint(hint)
                for hint in payload.get("hints", []) or []
                if isinstance(hint, dict)
            ]
            return replace(evidence, payload=payload)
        return evidence
    payload = dict(evidence.payload)
    rows = payload.pop("rows", None)
    if isinstance(rows, list):
        payload.setdefault("row_count", len(rows))
    payload.pop("raw_rows", None)
    return replace(evidence, payload=payload)


def _safe_column_value_hint(hint: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "table",
        "column",
        "profile_ref",
        "distinct_count",
        "profile_status",
        "sampled",
        "truncated",
        "redacted",
        "stale",
        "stale_reason",
        "candidate_mapping",
    }
    safe_hint = {key: hint.get(key) for key in allowed if key in hint}
    mapping = safe_hint.get("candidate_mapping")
    if isinstance(mapping, dict):
        safe_hint["candidate_mapping"] = {
            key: mapping.get(key)
            for key in ("prompt_term", "confidence", "reason")
            if key in mapping
        }
    return safe_hint


def _extract_candidates(
    evidence: tuple[Evidence, ...],
    *,
    source_operation_id: str,
    fallback_source_identity: str | None,
    schema_fingerprint: str | None,
    learning_mode: str,
    learned_at: str,
) -> tuple[_Candidate, ...]:
    if learning_mode == "off":
        return ()
    schema = _schema_from_evidence(evidence)
    candidates: list[_Candidate] = []
    candidates.extend(
        _unit_candidates(
            evidence,
            schema=schema,
            source_operation_id=source_operation_id,
            fallback_source_identity=fallback_source_identity,
            schema_fingerprint=schema_fingerprint or _schema_fingerprint(evidence),
            learned_at=learned_at,
        )
    )
    candidates.extend(
        _value_alias_candidates(
            evidence,
            source_operation_id=source_operation_id,
            fallback_source_identity=fallback_source_identity,
            schema_fingerprint=schema_fingerprint or _schema_fingerprint(evidence),
            learned_at=learned_at,
        )
    )
    deduped: dict[str, _Candidate] = {}
    for candidate in candidates:
        deduped.setdefault(candidate.key, candidate)
    return tuple(deduped.values())


def _unit_candidates(
    evidence: tuple[Evidence, ...],
    *,
    schema: dict[str, Any],
    source_operation_id: str,
    fallback_source_identity: str | None,
    schema_fingerprint: str | None,
    learned_at: str,
) -> tuple[_Candidate, ...]:
    if not schema:
        return ()
    schema_evidence = _schema_evidence(evidence)
    source_identity = _source_identity_from_evidence(
        schema_evidence,
        fallback=fallback_source_identity,
    )
    evidence_refs = tuple(
        item.id
        for item in (schema_evidence,)
        if item is not None and item.id is not None
    )
    candidates: list[_Candidate] = []
    for record in unit_records_from_schema(schema):
        confidence = _confidence(record.metadata.get("confidence"), default=0.75)
        metadata = {
            **record.metadata,
            "schema_refs": [
                {
                    "table": record.metadata.get("table"),
                    "column": record.metadata.get("column"),
                }
            ],
        }
        candidates.append(
            _Candidate(
                kind=record.kind,
                key=record.key,
                text=record.text,
                metadata=metadata,
                importance=record.importance,
                confidence=confidence,
                candidate_type="unit_convention",
                source_operation_id=source_operation_id,
                source_identity=source_identity,
                schema_fingerprint=schema_fingerprint,
                schema=schema,
                evidence_refs=evidence_refs,
                learned_at=learned_at,
            )
        )
    return tuple(candidates)


def _value_alias_candidates(
    evidence: tuple[Evidence, ...],
    *,
    source_operation_id: str,
    fallback_source_identity: str | None,
    schema_fingerprint: str | None,
    learned_at: str,
) -> tuple[_Candidate, ...]:
    candidates: list[_Candidate] = []
    for evidence_item, hint in _column_value_hints_from_evidence(evidence):
        mapping = hint.get("candidate_mapping")
        if not isinstance(mapping, dict):
            continue
        alias = str(mapping.get("prompt_term") or "").strip()
        if not alias:
            continue
        table = str(hint.get("table") or "").strip()
        column = str(hint.get("column") or "").strip()
        if not table or not column:
            continue
        profile_ref = str(hint.get("profile_ref") or "").strip()
        confidence = _confidence(mapping.get("confidence"), default=0.0)
        key = f"value_alias:{table}.{column}:{_slug(alias)}"
        text = (
            f"{alias} is an alias hint for {table}.{column} values cited by "
            f"catalog profile {profile_ref}."
        )
        metadata = {
            "table": table,
            "column": column,
            "alias": alias,
            "catalog_profile_ref": profile_ref,
            "catalog_refs": [profile_ref] if profile_ref else [],
            "catalog_evidence_id": evidence_item.id,
            "schema_refs": [{"table": table, "column": column}],
            "mapping_reason": mapping.get("reason"),
        }
        candidates.append(
            _Candidate(
                kind="value_alias",
                key=key,
                text=text,
                metadata=metadata,
                importance=0.72,
                confidence=confidence,
                candidate_type="catalog_cited_value_alias",
                source_operation_id=source_operation_id,
                source_identity=_source_identity_from_evidence(
                    evidence_item,
                    fallback=fallback_source_identity,
                ),
                schema_fingerprint=schema_fingerprint,
                schema={},
                evidence_refs=(evidence_item.id,) if evidence_item.id else (),
                learned_at=learned_at,
            )
        )
    return tuple(candidates)


async def _promotion_rejection(
    runtime: Any,
    candidate: _Candidate,
    *,
    expected_source_identity: str | None,
    schema: dict[str, Any],
) -> str | None:
    if candidate.kind not in {"unit_convention", "value_alias"}:
        return "unsupported_kind"
    if not candidate.key or not candidate.text:
        return "malformed_candidate"
    threshold = 0.8 if candidate.kind == "value_alias" else 0.75
    if candidate.confidence < threshold:
        return "low_confidence"
    if not candidate.evidence_refs:
        return "missing_evidence"
    if not candidate.source_identity:
        return "missing_source_identity"
    if (
        expected_source_identity
        and candidate.source_identity != expected_source_identity
    ):
        return "cross_source_candidate"
    try:
        record = normalize_db_memory_record(candidate.record())
    except Exception as exc:
        return f"invalid_record:{exc}"
    pii_error = db_memory_pii_error(
        key=record.key,
        text=record.text,
        metadata=record.metadata,
    )
    if pii_error:
        return "pii_or_sensitive_candidate"
    if not db_memory_record_refs_known_schema(record.metadata, schema):
        return "schema_scope_mismatch"
    try:
        memory_plugin = runtime.registry.get_plugin("memory")
        existing = await db_memory_record_chunk_ids_by_key(memory_plugin, record)
    except KeyError:
        return "memory_plugin_missing"
    except Exception as exc:
        return f"duplicate_check_failed:{exc}"
    if existing:
        return "duplicate"
    try:
        runtime.registry.get_capability("memory.semantic.write", owner="memory")
    except Exception:
        return "memory_write_capability_missing"
    return None


async def _write_candidate(
    runtime: Any,
    operation: Operation,
    candidate: _Candidate,
) -> tuple[Evidence, ...]:
    record = normalize_db_memory_record(candidate.record())
    memory_capability = runtime.registry.get_capability(
        "memory.semantic.write",
        owner="memory",
    )
    write_plan = await runtime.plan_task_specs(
        operation,
        (
            DbTaskSpec(
                capability_id=memory_capability.id,
                owner=memory_capability.owner,
                input={
                    "db_memory_payload": record.to_dict(),
                    "db_memory_prompt": record.text,
                },
                reason="db_memory_learning_promotion",
                sequence=1,
                metadata={
                    "source_operation_id": candidate.source_operation_id,
                    "source_identity": candidate.source_identity,
                    "candidate_key": candidate.key,
                    "candidate_kind": candidate.kind,
                },
                deterministic_key=persisted_fingerprint(
                    {
                        "source_operation_id": candidate.source_operation_id,
                        "source_identity": candidate.source_identity,
                        "candidate_key": candidate.key,
                        "candidate_kind": candidate.kind,
                    }
                ),
            ),
        ),
    )
    write_task = write_plan.tasks[0]
    return await runtime.execute_task(
        write_task,
        operation,
        context={"capability_owner": memory_capability.owner},
    )


def _candidate_payload(candidate: _Candidate) -> dict[str, Any]:
    record = candidate.record()
    return {
        "candidate_key": candidate.key,
        "candidate_type": candidate.candidate_type,
        "kind": candidate.kind,
        "record": record.to_dict(),
        "confidence": candidate.confidence,
        "source_operation_id": candidate.source_operation_id,
        "source_identity": candidate.source_identity,
        "source_schema_fingerprint": candidate.schema_fingerprint,
        "evidence_refs": list(candidate.evidence_refs),
        "promotion_policy": "auto_high_confidence",
    }


def _promotion_evidence(
    operation: Operation,
    task: Task,
    candidate: _Candidate,
    *,
    write_evidence: tuple[Evidence, ...],
    accepted: bool,
) -> Evidence:
    payload = {
        "candidate_key": candidate.key,
        "kind": candidate.kind,
        "source_operation_id": candidate.source_operation_id,
        "source_identity": candidate.source_identity,
        "promoted": accepted,
        "write_evidence_ids": [item.id for item in write_evidence if item.id],
        "promotion_policy": "auto_high_confidence",
    }
    return Evidence(
        kind="db.memory.promotion",
        owner="db_runtime",
        operation_id=operation.id,
        task_id=task.id,
        accepted=accepted,
        payload=payload,
        metadata={
            "candidate_key": candidate.key,
            "source_operation_id": candidate.source_operation_id,
            "source_identity": candidate.source_identity,
            "payload_fingerprint": persisted_fingerprint(payload),
        },
    )


def _rejection_evidence(
    operation: Operation,
    task: Task,
    candidate: _Candidate,
    *,
    reason: str,
    write_evidence: tuple[Evidence, ...] = (),
) -> Evidence:
    payload = {
        "candidate_key": candidate.key,
        "kind": candidate.kind,
        "source_operation_id": candidate.source_operation_id,
        "source_identity": candidate.source_identity,
        "rejected": True,
        "reason": reason,
        "write_evidence_ids": [item.id for item in write_evidence if item.id],
    }
    return Evidence(
        kind="db.memory.rejection",
        owner="db_runtime",
        operation_id=operation.id,
        task_id=task.id,
        payload=payload,
        metadata={
            "candidate_key": candidate.key,
            "source_operation_id": candidate.source_operation_id,
            "source_identity": candidate.source_identity,
            "payload_fingerprint": persisted_fingerprint(payload),
        },
    )


def _schema_from_evidence(evidence: tuple[Evidence, ...]) -> dict[str, Any]:
    for item in reversed(evidence):
        if item.kind == "planning.context" and isinstance(
            item.payload.get("schema"), dict
        ):
            return dict(item.payload["schema"])
    for item in evidence:
        if item.kind in {"schema.asset_profile", "catalog.source"}:
            return dict(item.payload)
    return {}


def _schema_evidence(evidence: tuple[Evidence, ...]) -> Evidence | None:
    for item in reversed(evidence):
        if item.kind in {"planning.context", "schema.asset_profile", "catalog.source"}:
            return item
    return None


def _schema_fingerprint(evidence: tuple[Evidence, ...]) -> str | None:
    for item in reversed(evidence):
        if item.kind == "planning.context":
            value = item.payload.get("schema_fingerprint")
            if value:
                return str(value)
    schema = _schema_from_evidence(evidence)
    return structural_schema_fingerprint(schema)


def _column_value_hints_from_evidence(
    evidence: tuple[Evidence, ...],
) -> tuple[tuple[Evidence, dict[str, Any]], ...]:
    pairs: list[tuple[Evidence, dict[str, Any]]] = []
    for item in evidence:
        if item.kind == "planning.context":
            for hint in item.payload.get("column_value_hints", []) or []:
                if isinstance(hint, dict) and _learning_eligible_column_value_hint(
                    hint
                ):
                    pairs.append((item, hint))
        elif item.kind == "schema.column_value_hint":
            for hint in item.payload.get("hints", []) or []:
                if isinstance(hint, dict) and _learning_eligible_column_value_hint(
                    hint
                ):
                    pairs.append((item, hint))
    return tuple(pairs)


def _learning_eligible_column_value_hint(hint: dict[str, Any]) -> bool:
    if hint.get("profile_status") != "profiled":
        return False
    if hint.get("stale") or hint.get("redacted") or hint.get("sampled"):
        return False
    if hint.get("truncated"):
        return False
    if not str(hint.get("table") or "").strip():
        return False
    if not str(hint.get("column") or "").strip():
        return False
    if not str(hint.get("profile_ref") or "").strip():
        return False
    mapping = hint.get("candidate_mapping")
    return isinstance(mapping, dict) and bool(mapping.get("prompt_term"))


def _source_identity_from_evidence(
    evidence: Evidence | None,
    *,
    fallback: str | None,
) -> str | None:
    if evidence is not None:
        for value in (
            evidence.metadata.get("source_identity"),
            evidence.payload.get("source_identity"),
        ):
            if value:
                return str(value)
    return str(fallback).strip() if fallback else None


def _confidence(value: Any, *, default: float) -> float:
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


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:80] or "alias"
