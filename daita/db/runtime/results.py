"""Operation result and audit recording helpers for ``DbRuntime``."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import re
from typing import Any, Mapping

from daita.runtime import (
    Evidence,
    Operation,
    OperationStatus,
    RuntimeEventType,
    RuntimeKernel,
    RuntimeStore,
)

from ..context_projection import (
    ProjectionContext,
    ProjectionMode,
    policy_summary_from_source,
    project_operation_result,
    project_runtime_stream_terminal_payload,
)
from ..fingerprints import sensitive_fingerprint
from ..models import DbOperationResult, DbRuntimeConfig

_AUDIT_MACHINE_CODE = re.compile(r"[a-z0-9_.:-]+")
_AUDIT_ERROR_TYPE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]*")
_AUDIT_CODE_MAX_LENGTH = 128
_REDACTED_WARNING_CODE = "db_runtime_warning_redacted"
_REDACTED_ERROR_CODE = "db_runtime_error_redacted"


class DbRuntimeResultsMixin:
    _operation_results: list[DbOperationResult]
    _audit_log: list[dict[str, Any]]
    _audit_fingerprint_key: bytes
    store: RuntimeStore
    kernel: RuntimeKernel
    config: DbRuntimeConfig
    runtime_id: str
    runtime_kind: str

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
        raw_result = result
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
        projection = ProjectionContext(
            mode=ProjectionMode.PUBLIC_RESULT,
            operation_intent=result.intent.kind.value,
            safety_frame=(
                operation_for_trace.metadata.get("safety_frame")
                if operation_for_trace is not None
                else None
            ),
            policy_summary=policy_summary_from_source(_result_projection_source(self)),
            source_identity=_projection_source_identity(self.config.metadata),
            session_id=result.request.session_id,
            user_id=result.request.user_id,
        )
        result = project_operation_result(result, projection)
        self._operation_results.append(result)
        self._audit_log.append(
            _audit_entry_from_result(
                result,
                raw_prompt=raw_result.request.prompt,
                raw_evidence=raw_result.evidence,
                raw_warnings=raw_result.warnings,
                raw_error=raw_result.diagnostics.get("error"),
                audit_key=self._audit_fingerprint_key,
            )
        )
        if operation is not None:
            enqueue_learning = getattr(
                self,
                "_enqueue_memory_learning_after_result",
                None,
            )
            if enqueue_learning is not None:
                try:
                    await enqueue_learning(raw_result, operation=operation)
                except Exception:
                    await self.kernel.append_event(
                        RuntimeEventType.DIAGNOSTIC,
                        operation_id=result.operation_id,
                        message="DB memory learning enqueue was skipped after error.",
                        payload={"reason": "memory_learning_enqueue_failed"},
                    )
            message = (
                f"Operation {result.operation_id} finished with "
                f"{result.status.value}."
            )
            payload = {
                "warnings": list(raw_result.warnings),
                **project_runtime_stream_terminal_payload(result),
            }
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


def _audit_entry_from_result(
    result: DbOperationResult,
    *,
    raw_prompt: str,
    raw_evidence: tuple[Evidence, ...],
    raw_warnings: tuple[str, ...],
    raw_error: Any,
    audit_key: bytes,
) -> dict[str, Any]:
    """Build a schema-v2 operational index without diagnostic excerpts."""
    entry = {
        "schema_version": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation_id": result.operation_id,
        "status": result.status.value,
        "intent_kind": result.intent.kind.value,
        "operation_type": result.contract.operation_type,
        "prompt_fingerprint": sensitive_fingerprint(raw_prompt, audit_key),
        "prompt_length": len(raw_prompt),
        "warning_codes": list(_audit_warning_codes(raw_warnings)),
        "evidence_count": len(raw_evidence),
        "accepted_evidence_count": sum(item.accepted for item in raw_evidence),
        "evidence_refs": [_evidence_ref(item) for item in result.evidence],
        "evidence_summaries": [
            _evidence_audit_summary(item, audit_key=audit_key) for item in raw_evidence
        ],
        "telemetry": dict(result.telemetry),
    }
    entry.update(_audit_error_facts(raw_error))
    return entry


def _evidence_ref(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "task_id": evidence.task_id,
        "accepted": evidence.accepted,
    }


def _evidence_audit_summary(
    evidence: Evidence,
    *,
    audit_key: bytes,
) -> dict[str, Any]:
    payload = evidence.payload
    summary: dict[str, Any] = {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "task_id": evidence.task_id,
        "accepted": evidence.accepted,
        "payload_keys": sorted(str(key) for key in payload),
    }
    if "sql" in payload:
        summary["sql_fingerprint"] = sensitive_fingerprint(
            payload["sql"],
            audit_key,
        )
        summary.update(_sql_audit_facts(payload))
    if isinstance(payload.get("rows"), list):
        summary["row_count"] = len(payload["rows"])
    elif (row_count := _non_negative_int(payload.get("row_count"))) is not None:
        summary["row_count"] = row_count
    if (total_rows := _non_negative_int(payload.get("total_rows"))) is not None:
        summary["total_rows"] = total_rows
    if isinstance(payload.get("truncated"), bool):
        summary["truncated"] = payload["truncated"]
    if isinstance(payload.get("success"), bool):
        summary["success"] = payload["success"]
    if any(
        key in payload
        for key in ("error", "error_message", "exception", "error_type", "error_code")
    ):
        summary.update(_audit_error_facts(payload))
    return summary


def _sql_audit_facts(payload: Mapping[str, Any]) -> dict[str, Any]:
    statement_facts = payload.get("statement_facts")
    facts = statement_facts if isinstance(statement_facts, Mapping) else {}
    result: dict[str, Any] = {}
    statement_type = _audit_machine_code(
        facts.get("statement_type") or payload.get("statement_type")
    )
    if statement_type is not None:
        result["statement_type"] = statement_type
    referenced_tables = _first_collection(
        payload.get("referenced_tables"),
        payload.get("tables"),
        facts.get("target_resources"),
    )
    if referenced_tables is not None:
        result["referenced_table_count"] = len(referenced_tables)
    referenced_columns = _first_collection(
        payload.get("referenced_columns"),
        payload.get("columns"),
    )
    if referenced_columns is not None:
        result["referenced_column_count"] = len(referenced_columns)
    return result


def _audit_warning_codes(value: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for item in value:
        code = _audit_machine_code(item) or _REDACTED_WARNING_CODE
        if code not in result:
            result.append(code)
    return tuple(result)


def _audit_error_facts(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    source = value if isinstance(value, Mapping) else {}
    nested = source.get("error") if isinstance(source, Mapping) else None
    if isinstance(nested, Mapping):
        source = {**source, **nested}
    error_type = _audit_error_type(source.get("type") or source.get("error_type"))
    error_code = _audit_machine_code(source.get("code") or source.get("error_code"))
    return {
        "error_type": error_type or "Error",
        "error_code": error_code or _REDACTED_ERROR_CODE,
    }


def _audit_machine_code(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if (
        len(text) > _AUDIT_CODE_MAX_LENGTH
        or _AUDIT_MACHINE_CODE.fullmatch(text) is None
    ):
        return None
    return text


def _audit_error_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) > _AUDIT_CODE_MAX_LENGTH or _AUDIT_ERROR_TYPE.fullmatch(text) is None:
        return None
    return text


def _first_collection(*values: Any) -> list[Any] | tuple[Any, ...] | None:
    for value in values:
        if isinstance(value, (list, tuple)):
            return value
    return None


def _non_negative_int(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return max(0, value)


def _result_projection_source(runtime: Any) -> Any:
    if runtime.source is not None:
        return runtime.source
    for plugin in getattr(runtime.config, "plugins", ()) or ():
        if any(
            hasattr(plugin, attribute)
            for attribute in ("blocked_tables", "blocked_columns", "allowed_tables")
        ):
            return plugin
    return None


def _projection_source_identity(metadata: Mapping[str, Any]) -> str | None:
    options = metadata.get("from_db_options")
    if not isinstance(options, Mapping):
        return None
    memory = options.get("memory")
    if not isinstance(memory, Mapping):
        return None
    value = memory.get("source_identity")
    return str(value) if value is not None else None
