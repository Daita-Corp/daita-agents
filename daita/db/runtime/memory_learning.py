"""Worker-owned deterministic DB memory learning for ``DbRuntime``."""

from __future__ import annotations

from typing import Any

from daita.runtime import Operation, OperationStatus

from ..analysis import stable_fingerprint
from ..memory import db_memory_options_from_runtime_metadata
from ..models import DbOperationResult

_ELIGIBLE_OPERATION_TYPES = frozenset(
    {
        "data.query",
        "query",
        "read",
        "db.read",
    }
)


class DbRuntimeMemoryLearningMixin:
    """Enqueue cold-path DB memory learning after verified operations."""

    async def _enqueue_memory_learning_after_result(
        self,
        result: DbOperationResult,
        *,
        operation: Operation | None,
    ) -> None:
        if operation is None or not self._memory_learning_result_eligible(result):
            return
        if await self._memory_learning_child_exists(operation.id):
            return

        memory_options = db_memory_options_from_runtime_metadata(self.config.metadata)
        source_identity = str(memory_options.get("source_identity") or "").strip()
        learning_mode = str(memory_options.get("learning") or "off").strip() or "off"
        source_schema_fingerprint = _schema_fingerprint_from_result(result)
        evidence_refs = [
            str(item.id)
            for item in result.evidence
            if item.id and item.accepted and item.operation_id == operation.id
        ]
        if not evidence_refs:
            return

        child = await self.kernel.create_operation(
            operation_type="db.memory.learning",
            request={
                "source_operation_id": operation.id,
                "source_operation_type": operation.operation_type,
                "source_identity": source_identity,
                "source_schema_fingerprint": source_schema_fingerprint,
                "learning_mode": learning_mode,
            },
            required_evidence=frozenset({"db.memory.learning.enqueue"}),
            metadata={
                "owner": "db_runtime",
                "queue": "memory_learning",
                "source_operation_id": operation.id,
                "source_operation_type": operation.operation_type,
                "source_identity": source_identity,
                "source_schema_fingerprint": source_schema_fingerprint,
                "learning_mode": learning_mode,
            },
            evaluate_governance=False,
        )
        await self.execute_task_spec_once(
            child,
            self.memory_learning_enqueue_task_spec(
                source_operation_id=operation.id,
                source_operation_type=operation.operation_type,
                source_identity=source_identity,
                source_schema_fingerprint=source_schema_fingerprint,
                source_evidence_ids=evidence_refs,
                learning_mode=learning_mode,
            ),
        )

    def _memory_learning_result_eligible(self, result: DbOperationResult) -> bool:
        if result.status is not OperationStatus.SUCCEEDED:
            return False
        if result.contract.operation_type not in _ELIGIBLE_OPERATION_TYPES:
            return False
        if result.contract.operation_type in {"memory.update", "db.memory.learning"}:
            return False
        verification = result.diagnostics.get("verification")
        if not isinstance(verification, dict) or verification.get("passed") is not True:
            return False
        memory_options = db_memory_options_from_runtime_metadata(self.config.metadata)
        if not bool(memory_options.get("enabled", False)):
            return False
        if str(memory_options.get("learning") or "off") == "off":
            return False
        if not result.evidence or not any(
            item.accepted and item.id for item in result.evidence
        ):
            return False
        try:
            self.registry.get_capability("db.memory.learning.run", owner="db_runtime")
            self.registry.get_capability("memory.semantic.write", owner="memory")
        except Exception:
            return False
        return True

    async def _memory_learning_child_exists(self, source_operation_id: str) -> bool:
        for operation in await self.store.list_operations():
            if operation.operation_type != "db.memory.learning":
                continue
            if operation.metadata.get("source_operation_id") == source_operation_id:
                return True
        return False


def _schema_fingerprint_from_result(result: DbOperationResult) -> str | None:
    for evidence in reversed(result.evidence):
        payload = evidence.payload if isinstance(evidence.payload, dict) else {}
        if evidence.kind == "planning.context":
            value = payload.get("schema_fingerprint")
            if value:
                return str(value)
        if evidence.kind == "verification.result":
            diagnostics = payload.get("diagnostics")
            if isinstance(diagnostics, dict):
                value = diagnostics.get("schema_fingerprint")
                if value:
                    return str(value)
    for evidence in result.evidence:
        if evidence.kind in {"schema.asset_profile", "catalog.source"}:
            return stable_fingerprint(evidence.payload)
    return None
