"""
Evidence-driven final answer synthesis for DB runtime operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from daita.runtime import Evidence

from .context import DbContextRenderer
from .models import DbIntent, DbIntentKind, DbOperationContract, DbRequest
from .verification import DbVerificationResult


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
    schema = next(
        (item.payload for item in evidence if item.kind == "schema.asset_profile"),
        {},
    )
    tables = schema.get("tables", []) or []
    parts = []
    for table in tables:
        columns = [
            str(column.get("name"))
            for column in table.get("columns", []) or []
            if column.get("name")
        ]
        parts.append(f"{table.get('name')}: {', '.join(columns)}")
    return f"Found {len(tables)} tables. " + "; ".join(parts)


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
