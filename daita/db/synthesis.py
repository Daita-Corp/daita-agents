"""
Evidence-driven final answer synthesis for DB runtime operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from daita.runtime import Evidence, Operation, Task

from .context import DbContextRenderer
from .models import DbIntent, DbIntentKind, DbOperationContract, DbRequest
from .verification import DbVerificationResult

_ALLOWED_SUFFICIENCY = frozenset(
    {
        "answered",
        "partial",
        "needs_clarification",
        "insufficient_evidence",
    }
)
_DEFAULT_CONTEXT_ROW_BUDGET = 25
_DEFAULT_CONTEXT_CHAR_BUDGET = 16000
_DEFAULT_FIELD_CHAR_BUDGET = 500
_SENSITIVE_FIELD_RE = re.compile(
    r"(password|passcode|secret|token|api[_-]?key|auth|credential|ssn|social[_-]?security|email|phone)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s().]{7,}\d")


@dataclass(frozen=True)
class DbSynthesisResult:
    """Final answer and diagnostics derived from accepted evidence."""

    answer: str
    evidence_refs: tuple[dict[str, str | None], ...]
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence_refs": list(self.evidence_refs),
            "warnings": list(self.warnings),
            "diagnostics": self.diagnostics,
        }


class DbSynthesizer:
    """Create final answers only from accepted, verified evidence."""

    def __init__(self, context_renderer: DbContextRenderer | None = None) -> None:
        self.context_renderer = context_renderer or DbContextRenderer()

    def synthesize(
        self,
        request: DbRequest,
        intent: DbIntent,
        contract: DbOperationContract,
        evidence: tuple[Evidence, ...],
        verification: DbVerificationResult,
    ) -> DbSynthesisResult:
        """Return a deterministic answer from verified evidence."""
        if not verification.passed:
            raise ValueError(
                "cannot synthesize final answer before verification passes"
            )

        if intent.kind is DbIntentKind.SCHEMA_QUERY:
            answer = _schema_answer(evidence)
        elif intent.kind is DbIntentKind.SCHEMA_RELATIONSHIP_QUERY:
            answer = _schema_relationship_answer(evidence)
        elif intent.kind in {
            DbIntentKind.DATA_QUERY,
            DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
        }:
            answer = _data_answer(evidence)
        else:
            answer = "The DB operation completed with verified evidence."

        return DbSynthesisResult(
            answer=answer,
            evidence_refs=verification.evidence_refs,
            warnings=(),
            diagnostics={
                "synthesis": "deterministic",
                "operation_type": contract.operation_type,
                "context": self.context_renderer.render_evidence_summary(evidence),
                "prompt": request.prompt,
                "skill_synthesis_metadata": contract.metadata.get(
                    "skill_synthesis_metadata", {}
                ),
            },
        )


def _schema_answer(evidence: tuple[Evidence, ...]) -> str:
    schema = _database_schema_payload(evidence)
    tables = _tables_from_schema_payload(schema)
    parts = []
    for table in tables:
        columns = [
            str(column.get("name") or column.get("column_name"))
            for column in _columns_from_schema_table(table)
            if column.get("name") or column.get("column_name")
        ]
        parts.append(f"{table.get('name')}: {', '.join(columns)}")
    return f"Found {len(tables)} tables. " + "; ".join(parts)


def _schema_relationship_answer(evidence: tuple[Evidence, ...]) -> str:
    relationship = next(
        (item.payload for item in evidence if item.kind == "schema.relationship_path"),
        {},
    )
    if relationship.get("reachable") is False:
        return "No relationship path was found between the requested tables."
    paths = relationship.get("paths") or []
    if not paths:
        return "No relationship path evidence was produced."
    parts = []
    for path in paths[:3]:
        assets = path.get("assets") or []
        if assets:
            parts.append(" -> ".join(str(asset) for asset in assets))
    if parts:
        return "Found relationship path: " + "; ".join(parts)
    return "Found relationship path evidence for the requested tables."


def _database_schema_payload(evidence: tuple[Evidence, ...]) -> dict[str, Any]:
    asset_scoped = [
        item.payload
        for item in evidence
        if item.kind == "schema.asset_profile"
        and _schema_scope(item) == "asset"
        and _tables_from_schema_payload(item.payload)
    ]
    if asset_scoped:
        tables = []
        for payload in asset_scoped:
            tables.extend(_tables_from_schema_payload(payload))
        return {"tables": tables}
    scoped = next(
        (
            item.payload
            for item in evidence
            if item.kind == "schema.asset_profile" and _schema_scope(item) == "database"
        ),
        None,
    )
    if scoped is not None:
        return scoped
    return next(
        (
            item.payload
            for item in evidence
            if item.kind == "schema.asset_profile"
            and _tables_from_schema_payload(item.payload)
        ),
        {},
    )


def _schema_scope(evidence: Evidence) -> str | None:
    if evidence.metadata.get("scope"):
        return str(evidence.metadata["scope"])
    payload_metadata = evidence.payload.get("metadata")
    if isinstance(payload_metadata, dict) and payload_metadata.get("scope"):
        return str(payload_metadata["scope"])
    if isinstance(evidence.payload.get("asset"), dict):
        return "asset"
    if isinstance(evidence.payload.get("table"), dict):
        return "asset"
    return None


def _data_answer(evidence: tuple[Evidence, ...]) -> str:
    query_result = next(
        (item.payload for item in evidence if item.kind == "query.result"), None
    )
    if query_result is None:
        return "No query result was produced."
    rows = query_result.get("rows", []) or []
    if len(rows) == 1 and "count" in rows[0]:
        return f"The count is {rows[0]['count']}."
    if not rows:
        return "The query returned no rows."
    return f"Returned {len(rows)} row{'s' if len(rows) != 1 else ''}."


def _tables_from_schema_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    nested_schema = payload.get("schema")
    if isinstance(nested_schema, dict):
        nested_tables = _tables_from_schema_payload(nested_schema)
        if nested_tables:
            return nested_tables
    if payload.get("tables"):
        return [
            table
            for table in payload.get("tables", []) or []
            if isinstance(table, dict)
        ]
    asset = payload.get("asset")
    if isinstance(asset, dict):
        return [
            {
                "name": asset.get("name") or payload.get("table_name"),
                "columns": payload.get("fields") or payload.get("columns") or [],
            }
        ]
    table = payload.get("table")
    if isinstance(table, dict):
        return [
            {
                "name": table.get("name") or payload.get("table_name"),
                "columns": payload.get("fields") or payload.get("columns") or [],
            }
        ]
    return []


def _columns_from_schema_table(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        column
        for column in (table.get("columns") or table.get("fields") or [])
        if isinstance(column, dict)
    ]


@dataclass(frozen=True)
class DbAnswerCitation:
    """Citation to accepted runtime evidence used by final synthesis."""

    id: str
    kind: str
    purpose: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "kind": self.kind, "purpose": self.purpose}


@dataclass(frozen=True)
class DbAnswerSynthesisPayload:
    """Typed payload for `answer.synthesis` evidence."""

    answer: str
    reasoning_summary: str
    cited_evidence_refs: tuple[DbAnswerCitation, ...]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    warnings: tuple[str, ...]
    follow_up_questions: tuple[str, ...]
    sufficiency: str
    confidence: float
    truncation: dict[str, bool]
    grounding: dict[str, bool]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "reasoning_summary": self.reasoning_summary,
            "cited_evidence_refs": [
                citation.to_dict() for citation in self.cited_evidence_refs
            ],
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "follow_up_questions": list(self.follow_up_questions),
            "sufficiency": self.sufficiency,
            "confidence": self.confidence,
            "truncation": dict(self.truncation),
            "grounding": dict(self.grounding),
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class DbSynthesisContext:
    """Bounded accepted-evidence context supplied to final synthesis."""

    payload: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DbAnswerSynthesisExecutor:
    """Executor for runtime-owned `db.answer.synthesize` tasks."""

    runtime: Any
    id: str = "db.answer.synthesize.runtime"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.answer.synthesize"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: dict[str, Any],
    ) -> list[Evidence]:
        runtime = _runtime_from_boundary(self.runtime)
        request = runtime._db_request_from_operation(operation)
        intent = runtime._db_intent_from_operation(operation)
        contract = runtime._db_contract_from_context(operation)
        dependency_evidence = await _accepted_dependency_evidence(
            runtime, task, operation
        )
        verification_evidence = next(
            (
                item
                for item in dependency_evidence
                if item.kind == "verification.result"
            ),
            None,
        )
        if verification_evidence is None:
            raise RuntimeError("verification.result evidence is required")
        verification = verification_from_evidence(verification_evidence)
        synthesis_context = build_synthesis_context(
            request=request,
            intent=intent,
            contract=contract,
            evidence=dependency_evidence,
            row_budget=int(task.input.get("row_budget") or _DEFAULT_CONTEXT_ROW_BUDGET),
            char_budget=int(
                task.input.get("char_budget") or _DEFAULT_CONTEXT_CHAR_BUDGET
            ),
        )
        payload: DbAnswerSynthesisPayload | None = None
        fallback_reason: str | None = None
        if runtime.db_llm_service.available:
            try:
                response = await runtime.db_llm_service.generate_synthesis_json(
                    _synthesis_messages(synthesis_context.payload)
                )
                parsed = parse_synthesis_json(response.content)
                payload = validate_synthesis_payload(
                    parsed,
                    dependency_evidence=dependency_evidence,
                    context_metadata=synthesis_context.metadata,
                    llm_diagnostics=response.diagnostics,
                )
            except Exception as exc:
                fallback_reason = f"{type(exc).__name__}:{exc}"
        else:
            fallback_reason = "db_llm_service_unavailable"

        if payload is None:
            payload = deterministic_synthesis_payload(
                request=request,
                intent=intent,
                contract=contract,
                evidence=dependency_evidence,
                verification=verification,
                synthesizer=runtime.synthesizer,
                context_metadata=synthesis_context.metadata,
                fallback_reason=fallback_reason or "llm_synthesis_invalid",
            )

        return [
            Evidence(
                kind="answer.synthesis",
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                accepted=True,
                payload=payload.to_dict(),
                metadata={
                    "mode": payload.diagnostics.get("mode"),
                    "sufficiency": payload.sufficiency,
                    "cited_evidence_refs": [
                        citation.id for citation in payload.cited_evidence_refs
                    ],
                },
            )
        ]


def verification_from_evidence(evidence: Evidence) -> DbVerificationResult:
    """Rehydrate verifier result from compact `verification.result` evidence."""
    payload = evidence.payload
    evidence_refs = tuple(
        dict(item)
        for item in payload.get("evidence_details", ())
        if isinstance(item, dict)
    )
    if not evidence_refs:
        evidence_refs = tuple(
            {"id": item, "kind": None, "owner": None, "task_id": None}
            for item in payload.get("evidence_refs", ())
        )
    return DbVerificationResult(
        passed=bool(payload.get("passed")),
        missing_evidence=tuple(payload.get("missing_evidence") or ()),
        warnings=tuple(str(item) for item in payload.get("warnings") or ()),
        diagnostics=dict(payload.get("diagnostics") or {}),
        evidence_refs=evidence_refs,
    )


def build_synthesis_context(
    *,
    request: DbRequest,
    intent: DbIntent,
    contract: DbOperationContract,
    evidence: tuple[Evidence, ...],
    row_budget: int = _DEFAULT_CONTEXT_ROW_BUDGET,
    char_budget: int = _DEFAULT_CONTEXT_CHAR_BUDGET,
) -> DbSynthesisContext:
    """Build bounded synthesis context from accepted evidence records only."""
    accepted = tuple(item for item in evidence if item.accepted)
    evidence_summaries = [_evidence_summary(item) for item in accepted]
    query_result = next(
        (item for item in accepted if item.kind == "query.result"), None
    )
    rows, row_metadata = _bounded_rows(query_result, row_budget=row_budget)
    semantics = _catalog_semantics(accepted)
    payload = {
        "prompt": request.prompt,
        "intent_kind": intent.kind.value,
        "operation_type": contract.operation_type,
        "evidence": evidence_summaries,
        "query_result": {
            "evidence_id": query_result.id if query_result else None,
            "columns": row_metadata["columns"],
            "column_types": row_metadata["column_types"],
            "rows": rows,
            "total_rows": row_metadata["total_rows"],
            "sampled_rows": row_metadata["sampled_rows"],
            "truncated": row_metadata["rows_truncated"],
        },
        "planning": _planning_summary(accepted),
        "semantics": semantics,
        "verification": _verification_summary(accepted),
    }
    required_caveats = _required_caveats(payload, row_metadata)
    payload["required_caveats"] = list(required_caveats)
    rendered = json.dumps(payload, sort_keys=True, default=str)
    context_chars_truncated = len(rendered) > char_budget
    if context_chars_truncated:
        payload["rendered_context"] = rendered[:char_budget]
        required_caveats = tuple((*required_caveats, "synthesis_context_truncated"))
        payload["required_caveats"] = list(required_caveats)
    else:
        payload["rendered_context"] = rendered
    truncation = {
        "rows_truncated": bool(row_metadata["rows_truncated"]),
        "fields_truncated": bool(row_metadata["fields_truncated"]),
        "context_chars_truncated": context_chars_truncated,
    }
    metadata = {
        "dependency_evidence_refs": [
            {"id": item.id, "kind": item.kind, "operation_id": item.operation_id}
            for item in accepted
            if item.id
        ],
        "accepted_evidence_ids": [item.id for item in accepted if item.id],
        "required_caveats": list(required_caveats),
        "truncation": truncation,
        "redaction": {
            "values_redacted": bool(row_metadata["redacted_value_count"]),
            "redacted_value_count": row_metadata["redacted_value_count"],
        },
        "context_budget": {
            "row_budget": row_budget,
            "char_budget": char_budget,
            "field_char_budget": _DEFAULT_FIELD_CHAR_BUDGET,
        },
    }
    payload["truncation"] = truncation
    payload["redaction"] = metadata["redaction"]
    return DbSynthesisContext(payload=payload, metadata=metadata)


def deterministic_synthesis_payload(
    *,
    request: DbRequest,
    intent: DbIntent,
    contract: DbOperationContract,
    evidence: tuple[Evidence, ...],
    verification: DbVerificationResult,
    synthesizer: DbSynthesizer | None = None,
    context_metadata: dict[str, Any],
    fallback_reason: str | None,
) -> DbAnswerSynthesisPayload:
    """Build deterministic fallback through the same answer evidence path."""
    result = (synthesizer or DbSynthesizer()).synthesize(
        request, intent, contract, evidence, verification
    )
    citations = tuple(
        DbAnswerCitation(
            id=str(item["id"]),
            kind=str(item.get("kind") or "unknown"),
            purpose="verified runtime evidence",
        )
        for item in _accepted_citation_refs(evidence, verification)
        if item.get("id")
    )
    caveats = tuple(str(item) for item in context_metadata.get("required_caveats", ()))
    truncation = dict(context_metadata.get("truncation") or {})
    sufficiency = "partial" if any(truncation.values()) else "answered"
    return DbAnswerSynthesisPayload(
        answer=result.answer,
        reasoning_summary="Deterministic summary generated from verified evidence.",
        cited_evidence_refs=citations,
        assumptions=(),
        limitations=caveats,
        warnings=tuple((*result.warnings, *caveats)),
        follow_up_questions=(),
        sufficiency=sufficiency,
        confidence=0.8 if verification.passed else 0.0,
        truncation={
            "rows_truncated": bool(truncation.get("rows_truncated")),
            "fields_truncated": bool(truncation.get("fields_truncated")),
            "context_chars_truncated": bool(truncation.get("context_chars_truncated")),
        },
        grounding={"all_claims_from_evidence": True},
        diagnostics={
            "mode": "deterministic_fallback",
            "model": None,
            "provider": None,
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "estimated_cost": None,
            "fallback_reason": fallback_reason,
            "evidence_refs": [citation.id for citation in citations],
            "context": context_metadata,
            "sufficiency": sufficiency,
        },
    )


def parse_synthesis_json(content: str) -> dict[str, Any]:
    raw = _strip_json_fence(content)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("synthesis_json_not_object")
    return parsed


def validate_synthesis_payload(
    parsed: dict[str, Any],
    *,
    dependency_evidence: tuple[Evidence, ...],
    context_metadata: dict[str, Any],
    llm_diagnostics: dict[str, Any],
) -> DbAnswerSynthesisPayload:
    """Validate strict LLM synthesis before accepting it as evidence."""
    answer = str(parsed.get("answer") or "").strip()
    if not answer:
        raise ValueError("synthesis_answer_empty")
    if _requests_db_work(parsed):
        raise ValueError("synthesis_requests_db_work")
    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise ValueError("synthesis_confidence_out_of_bounds")
    sufficiency = str(parsed.get("sufficiency") or "")
    if sufficiency not in _ALLOWED_SUFFICIENCY:
        raise ValueError("synthesis_sufficiency_invalid")
    grounding = dict(parsed.get("grounding") or {})
    if grounding.get("all_claims_from_evidence") is not True:
        raise ValueError("synthesis_grounding_missing")

    accepted_by_id = {
        item.id: item for item in dependency_evidence if item.accepted and item.id
    }
    citations = _parse_citations(parsed.get("cited_evidence_refs"), accepted_by_id)
    if not citations:
        raise ValueError("synthesis_citations_missing")
    _validate_required_citations(citations, dependency_evidence)
    _validate_caveats_preserved(parsed, context_metadata)
    _validate_relationship_claims(answer, citations, accepted_by_id)
    truncation = dict(parsed.get("truncation") or {})
    context_truncation = dict(context_metadata.get("truncation") or {})
    normalized_truncation = {
        "rows_truncated": bool(
            truncation.get("rows_truncated") or context_truncation.get("rows_truncated")
        ),
        "fields_truncated": bool(
            truncation.get("fields_truncated")
            or context_truncation.get("fields_truncated")
        ),
        "context_chars_truncated": bool(
            truncation.get("context_chars_truncated")
            or context_truncation.get("context_chars_truncated")
        ),
    }
    diagnostics = _diagnostics_from_llm(llm_diagnostics)
    diagnostics.update(
        {
            "mode": "llm",
            "fallback_reason": None,
            "evidence_refs": [citation.id for citation in citations],
            "context": context_metadata,
            "sufficiency": sufficiency,
        }
    )
    return DbAnswerSynthesisPayload(
        answer=answer,
        reasoning_summary=str(parsed.get("reasoning_summary") or "").strip(),
        cited_evidence_refs=tuple(citations),
        assumptions=_string_tuple(parsed.get("assumptions")),
        limitations=_string_tuple(parsed.get("limitations")),
        warnings=_string_tuple(parsed.get("warnings")),
        follow_up_questions=_string_tuple(parsed.get("follow_up_questions")),
        sufficiency=sufficiency,
        confidence=float(confidence),
        truncation=normalized_truncation,
        grounding={"all_claims_from_evidence": True},
        diagnostics=diagnostics,
    )


async def _accepted_dependency_evidence(
    runtime: Any,
    task: Task,
    operation: Operation,
) -> tuple[Evidence, ...]:
    evidence: list[Evidence] = []
    for dependency in task.dependencies:
        if dependency.kind.value != "evidence":
            continue
        item = await runtime._accepted_evidence_for_dependency(operation.id, dependency)
        if item is not None and item.accepted and item.operation_id == operation.id:
            evidence.append(item)
    return tuple(evidence)


def _synthesis_messages(context_payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You synthesize final database answers inside a governed runtime. "
                "Use only the provided accepted evidence. Return strict JSON only. "
                "Never request SQL execution, tool calls, connector access, or any "
                "new database work."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "context": context_payload,
                    "schema": {
                        "answer": "string",
                        "reasoning_summary": "string",
                        "cited_evidence_refs": [
                            {
                                "id": "accepted evidence id",
                                "kind": "evidence kind",
                                "purpose": "why cited",
                            }
                        ],
                        "assumptions": ["string"],
                        "limitations": ["string"],
                        "warnings": ["string"],
                        "follow_up_questions": ["string"],
                        "sufficiency": sorted(_ALLOWED_SUFFICIENCY),
                        "confidence": "number from 0 to 1",
                        "truncation": {
                            "rows_truncated": "boolean",
                            "fields_truncated": "boolean",
                            "context_chars_truncated": "boolean",
                        },
                        "grounding": {"all_claims_from_evidence": True},
                    },
                },
                sort_keys=True,
                default=str,
            ),
        },
    ]


def _runtime_from_boundary(boundary: Any) -> Any:
    runtime = getattr(boundary, "runtime", None)
    return runtime if runtime is not None else boundary


def _accepted_citation_refs(
    evidence: tuple[Evidence, ...],
    verification: DbVerificationResult,
) -> tuple[dict[str, str | None], ...]:
    accepted_by_id = {item.id: item for item in evidence if item.accepted and item.id}
    refs = [
        item for item in verification.evidence_refs if item.get("id") in accepted_by_id
    ]
    if refs:
        return tuple(refs)
    return tuple(
        {
            "id": item.id,
            "kind": item.kind,
            "owner": item.owner,
            "task_id": item.task_id,
        }
        for item in evidence
        if item.accepted and item.id
    )


def _evidence_summary(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "operation_id": evidence.operation_id,
        "task_id": evidence.task_id,
        "payload_fingerprint": evidence.metadata.get("payload_fingerprint"),
        "payload_keys": sorted(str(key) for key in evidence.payload),
        "accepted": evidence.accepted,
    }


def _bounded_rows(
    query_result: Evidence | None,
    *,
    row_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if query_result is None:
        return [], {
            "columns": [],
            "column_types": {},
            "total_rows": 0,
            "sampled_rows": 0,
            "rows_truncated": False,
            "fields_truncated": False,
            "redacted_value_count": 0,
        }
    raw_rows = query_result.payload.get("rows") or []
    rows = raw_rows if isinstance(raw_rows, list) else []
    bounded: list[dict[str, Any]] = []
    fields_truncated = False
    redacted_count = 0
    for row in rows[: max(row_budget, 0)]:
        if not isinstance(row, dict):
            row = {"value": row}
        redacted_row: dict[str, Any] = {}
        for key, value in row.items():
            redacted, redacted_value, truncated = _redact_value(str(key), value)
            redacted_row[str(key)] = redacted_value
            redacted_count += int(redacted)
            fields_truncated = fields_truncated or truncated
        bounded.append(redacted_row)
    columns = sorted({column for row in bounded for column in row})
    column_types = {
        column: _first_type_for_column(bounded, column) for column in columns
    }
    total_rows = int(query_result.payload.get("total_rows") or len(rows))
    rows_truncated = bool(query_result.payload.get("truncated")) or len(rows) > len(
        bounded
    )
    return bounded, {
        "columns": columns,
        "column_types": column_types,
        "total_rows": total_rows,
        "sampled_rows": len(bounded),
        "rows_truncated": rows_truncated,
        "fields_truncated": fields_truncated,
        "redacted_value_count": redacted_count,
    }


def _redact_value(key: str, value: Any) -> tuple[bool, Any, bool]:
    if _SENSITIVE_FIELD_RE.search(key):
        return True, "[REDACTED]", False
    if isinstance(value, str):
        redacted = False
        rendered = value
        if _EMAIL_RE.search(rendered) or _PHONE_RE.fullmatch(rendered.strip()):
            rendered = "[REDACTED]"
            redacted = True
        truncated = len(rendered) > _DEFAULT_FIELD_CHAR_BUDGET
        if truncated:
            rendered = rendered[:_DEFAULT_FIELD_CHAR_BUDGET] + "...[TRUNCATED]"
        return redacted, rendered, truncated
    return False, value, False


def _first_type_for_column(rows: list[dict[str, Any]], column: str) -> str:
    for row in rows:
        value = row.get(column)
        if value is not None:
            return type(value).__name__
    return "null"


def _catalog_semantics(evidence: tuple[Evidence, ...]) -> dict[str, Any]:
    planning = next(
        (item for item in evidence if item.kind == "planning.context"), None
    )
    if planning is None:
        planning = next(
            (
                item
                for item in evidence
                if item.kind.startswith("catalog.")
                or item.kind == "schema.asset_profile"
            ),
            None,
        )
    payload = planning.payload if planning is not None else {}
    schema = (
        payload.get("schema") if isinstance(payload.get("schema"), dict) else payload
    )
    tables = []
    for table in _tables_from_schema_payload(schema):
        tables.append(
            {
                "name": table.get("name"),
                "columns": [
                    {
                        "name": column.get("name") or column.get("column_name"),
                        "data_type": column.get("data_type") or column.get("type"),
                    }
                    for column in _columns_from_schema_table(table)
                    if column.get("name") or column.get("column_name")
                ],
                "metadata": dict(table.get("metadata") or {}),
            }
        )
    return {
        "evidence_id": planning.id if planning is not None else None,
        "tables": tables,
        "relationships": list(schema.get("foreign_keys", []) or [])[:50],
    }


def _planning_summary(evidence: tuple[Evidence, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for kind in (
        "planning.context",
        "query.plan.proposal",
        "query.plan.validation",
        "sql.validation",
    ):
        item = next(
            (candidate for candidate in evidence if candidate.kind == kind), None
        )
        if item is None:
            continue
        summary[kind] = {
            "evidence_id": item.id,
            "payload_fingerprint": item.metadata.get("payload_fingerprint"),
            "payload": _compact_payload(item.payload),
        }
    return summary


def _verification_summary(evidence: tuple[Evidence, ...]) -> dict[str, Any]:
    item = next(
        (
            candidate
            for candidate in evidence
            if candidate.kind == "verification.result"
        ),
        None,
    )
    return dict(item.payload) if item is not None else {}


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "valid",
        "warnings",
        "assumptions",
        "strategy",
        "sql_fingerprint",
        "schema_fingerprint",
        "plan_fingerprint",
        "tables",
        "columns",
        "statement_type",
        "is_read",
        "accepted_sql",
    }
    return {key: value for key, value in payload.items() if key in allowed}


def _required_caveats(
    payload: dict[str, Any], row_metadata: dict[str, Any]
) -> tuple[str, ...]:
    caveats: list[str] = []
    if row_metadata["rows_truncated"]:
        caveats.append("query_result_truncated")
    if row_metadata["fields_truncated"]:
        caveats.append("fields_truncated")
    if row_metadata["redacted_value_count"]:
        caveats.append("sensitive_values_redacted")
    verification = payload.get("verification") or {}
    for warning in verification.get("warnings") or ():
        caveats.append(str(warning))
    return tuple(dict.fromkeys(caveats))


def _parse_citations(
    value: Any, accepted_by_id: dict[str | None, Evidence]
) -> list[DbAnswerCitation]:
    if not isinstance(value, list):
        raise ValueError("synthesis_citations_not_list")
    citations: list[DbAnswerCitation] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("synthesis_citation_not_object")
        evidence_id = str(item.get("id") or "")
        if evidence_id not in accepted_by_id:
            raise ValueError(f"synthesis_unknown_citation:{evidence_id}")
        evidence = accepted_by_id[evidence_id]
        kind = str(item.get("kind") or evidence.kind)
        if kind != evidence.kind:
            raise ValueError(f"synthesis_citation_kind_mismatch:{evidence_id}")
        citations.append(
            DbAnswerCitation(
                id=evidence_id,
                kind=kind,
                purpose=str(item.get("purpose") or "supporting evidence"),
            )
        )
    return citations


def _validate_required_citations(
    citations: list[DbAnswerCitation],
    dependency_evidence: tuple[Evidence, ...],
) -> None:
    cited_kinds = {citation.kind for citation in citations}
    available_kinds = {item.kind for item in dependency_evidence if item.accepted}
    if "query.result" in available_kinds and "query.result" not in cited_kinds:
        raise ValueError("synthesis_query_result_citation_missing")
    if (
        "verification.result" in available_kinds
        and "verification.result" not in cited_kinds
    ):
        raise ValueError("synthesis_verification_citation_missing")


def _validate_caveats_preserved(
    parsed: dict[str, Any], context_metadata: dict[str, Any]
) -> None:
    required = tuple(context_metadata.get("required_caveats") or ())
    if not required:
        return
    provided = " ".join(
        str(item)
        for item in (
            *(parsed.get("limitations") or ()),
            *(parsed.get("warnings") or ()),
            *(parsed.get("assumptions") or ()),
        )
    )
    for caveat in required:
        if str(caveat) not in provided:
            raise ValueError(f"synthesis_dropped_caveat:{caveat}")


def _validate_relationship_claims(
    answer: str,
    citations: list[DbAnswerCitation],
    accepted_by_id: dict[str | None, Evidence],
) -> None:
    lowered = answer.lower()
    if not any(
        term in lowered
        for term in ("relationship", "foreign key", "joined", "join path", "linked")
    ):
        return
    cited_kinds = {accepted_by_id[citation.id].kind for citation in citations}
    if not any(
        kind == "planning.context"
        or kind == "schema.asset_profile"
        or kind.startswith("catalog.")
        for kind in cited_kinds
    ):
        raise ValueError("synthesis_ungrounded_relationship_claim")


def _requests_db_work(parsed: dict[str, Any]) -> bool:
    text = json.dumps(parsed, sort_keys=True, default=str).lower()
    forbidden = (
        "execute sql",
        "run sql",
        "run a query",
        "query the database",
        "call a tool",
        "tool call",
        "connector access",
        "db.sql.",
    )
    return any(term in text for term in forbidden)


def _diagnostics_from_llm(diagnostics: dict[str, Any]) -> dict[str, Any]:
    tokens = (
        diagnostics.get("tokens") if isinstance(diagnostics.get("tokens"), dict) else {}
    )
    return {
        "model": diagnostics.get("model"),
        "provider": diagnostics.get("provider"),
        "latency_ms": diagnostics.get("latency_ms"),
        "input_tokens": (
            diagnostics.get("input_tokens")
            or tokens.get("input_tokens")
            or tokens.get("prompt_tokens")
        ),
        "output_tokens": (
            diagnostics.get("output_tokens")
            or tokens.get("output_tokens")
            or tokens.get("completion_tokens")
        ),
        "estimated_cost": (
            diagnostics.get("estimated_cost") or diagnostics.get("estimated_cost_usd")
        ),
    }


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return (str(value),)


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped
