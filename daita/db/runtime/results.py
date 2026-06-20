"""Operation result and audit recording helpers for ``DbRuntime``."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from daita.runtime import Evidence, Operation, OperationStatus

from ..models import DbOperationResult


class DbRuntimeResultsMixin:
    @property
    def operation_results(self) -> tuple[DbOperationResult, ...]:
        """Typed operation results retained by this in-memory runtime."""
        return tuple(self._operation_results)

    @property
    def audit_log(self) -> tuple[dict[str, Any], ...]:
        """Redacted operation audit summaries retained by this runtime."""
        return tuple(dict(entry) for entry in self._audit_log)

    async def _record_operation_result(
        self,
        result: DbOperationResult,
        *,
        operation: Operation | None = None,
    ) -> DbOperationResult:
        operation_for_trace = operation
        if operation_for_trace is None:
            try:
                operation_for_trace = await self.store.load_operation(
                    result.operation_id
                )
            except Exception:
                operation_for_trace = None
        if operation_for_trace is not None:
            operation_for_trace = await self._persist_trace_correlation(
                operation_for_trace,
                intent_kind=result.intent.kind.value,
            )
            result = self._result_with_observability(result, operation_for_trace)
        else:
            result = self._result_with_observability(result, None)
        self._operation_results.append(result)
        self._audit_log.append(_audit_entry_from_result(result))
        if operation is not None:
            message = (
                f"Operation {result.operation_id} finished with "
                f"{result.status.value}."
            )
            payload = {"warnings": list(result.warnings)}
            if result.status in {
                OperationStatus.SUCCEEDED,
                OperationStatus.FAILED,
                OperationStatus.CANCELLED,
            }:
                await self.kernel.complete_operation(
                    result.operation_id,
                    status=result.status,
                    message=message,
                    payload=payload,
                )
            elif result.status is OperationStatus.BLOCKED:
                await self.kernel.block_operation(
                    result.operation_id,
                    message=message,
                    payload=payload,
                )
            else:
                await self.kernel.update_operation(
                    result.operation_id,
                    result.status,
                    message=message,
                    payload=payload,
                )
        return result

    async def _persist_trace_correlation(
        self,
        operation: Operation,
        *,
        intent_kind: str | None = None,
    ) -> Operation:
        trace = self._current_trace_metadata()
        self._record_active_span_correlation(operation, intent_kind=intent_kind)
        if not trace:
            return operation
        updated = replace(
            operation,
            metadata={
                **operation.metadata,
                "trace": trace,
            },
        )
        await self.store.save_operation(updated)
        return updated

    def _result_with_observability(
        self,
        result: DbOperationResult,
        operation: Operation | None,
    ) -> DbOperationResult:
        diagnostics = dict(result.diagnostics)
        trace = (
            dict(operation.metadata.get("trace") or {})
            if operation is not None
            else self._current_trace_metadata()
        )
        if trace:
            diagnostics["trace"] = trace
        observed = replace(result, diagnostics=diagnostics)
        diagnostics["telemetry"] = observed.telemetry
        return replace(observed, diagnostics=diagnostics)

    def _record_active_span_correlation(
        self,
        operation: Operation,
        *,
        intent_kind: str | None = None,
    ) -> None:
        try:
            from daita.core.tracing import get_trace_manager

            trace_manager = get_trace_manager()
            span_id = trace_manager.trace_context.current_span_id
            if not span_id:
                return
            trace_manager.record_runtime_correlation(
                span_id,
                operation_id=operation.id,
                execution_id=operation.id,
                runtime_id=self.runtime_id,
                runtime_kind=self.runtime_kind,
                operation_type=operation.operation_type,
                intent_kind=intent_kind or operation.metadata.get("intent_kind"),
                command_kind=operation.metadata.get("command_kind"),
                control_plane=operation.metadata.get("control_plane"),
                monitor_id=operation.metadata.get("monitor_id"),
                monitor_name=operation.metadata.get("monitor_name"),
            )
        except Exception:
            return

    def _current_trace_metadata(self) -> dict[str, str]:
        try:
            from daita.core.tracing import get_trace_manager

            return get_trace_manager().current_trace_metadata(
                span_name="from_db_operation"
            )
        except Exception:
            return {}

    def _current_trace_ids(self) -> tuple[str | None, str | None]:
        try:
            from daita.core.tracing import get_trace_manager

            context = get_trace_manager().trace_context
            return context.current_trace_id, context.current_span_id
        except Exception:
            return None, None


def _audit_entry_from_result(result: DbOperationResult) -> dict[str, Any]:
    """Build a redacted, JSON-safe summary for operation inspection."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation_id": result.operation_id,
        "prompt": result.request.prompt,
        "status": result.status.value,
        "intent_kind": result.intent.kind.value,
        "operation_type": result.contract.operation_type,
        "warnings": list(result.warnings),
        "evidence": [_evidence_audit_summary(item) for item in result.evidence],
        "evidence_refs": (
            result.diagnostics.get("execution", {}).get("evidence_refs", [])
            if isinstance(result.diagnostics.get("execution"), dict)
            else []
        ),
    }


def _evidence_audit_summary(evidence: Evidence) -> dict[str, Any]:
    payload = evidence.payload
    summary: dict[str, Any] = {
        "kind": evidence.kind,
        "owner": evidence.owner,
        "task_id": evidence.task_id,
        "accepted": evidence.accepted,
        "payload_keys": sorted(str(key) for key in payload),
    }
    if "sql" in payload:
        summary["sql"] = payload["sql"]
    if isinstance(payload.get("rows"), list):
        summary["row_count"] = len(payload["rows"])
    if "total_rows" in payload:
        summary["total_rows"] = payload["total_rows"]
    if "truncated" in payload:
        summary["truncated"] = payload["truncated"]
    if "success" in payload:
        summary["success"] = payload["success"]
    if "error" in payload:
        summary["error"] = payload["error"]
    return summary
