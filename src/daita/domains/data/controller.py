"""Data tools: projection, validation, execution, and model-visible results."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Protocol, cast

from ..._json import FrozenJsonObject
from ...capabilities import (
    AccessMode,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    Capability,
    CapabilityInputError,
    CapabilityRegistry,
    Executor,
    SideEffectExecutor,
    ToolApplicability,
    ToolExecution,
    ToolOutput,
    ToolOutputValidationError,
)
from ...memory.capabilities import MEMORY_SET_CAPABILITY_ID
from ...observation import (
    AgentEvent,
    AgentEventKind,
    AgentObserver,
    _emit_safely,
)
from ...catalog.capabilities import (
    CATALOG_INSPECT_CAPABILITY_ID,
    CATALOG_SCHEMA_CAPABILITY_ID,
    CATALOG_SEARCH_CAPABILITY_ID,
    CATALOG_TRAVERSE_CAPABILITY_ID,
)
from ...llm.models import ToolCall, ToolDefinition, ToolResultBlock
from ...loop.models import RunInput
from ...skills.capabilities import (
    SKILL_DELETE_CAPABILITY_ID,
    SKILL_SAVE_CAPABILITY_ID,
    SKILL_VIEW_CAPABILITY_ID,
)
from ...skills.store import (
    SkillNotFoundError,
    SkillStoreError,
    SkillValidationError,
    validate_skill_name,
)
from .file_capabilities import LOCAL_FILE_READ_CAPABILITY_ID
from .sql import ResourceSchema, validate_postgresql_read, validate_sqlite_read

SQLITE_QUERY_CAPABILITY_ID = "data.sqlite.query"
SQLITE_QUERY_EVIDENCE_KIND = "data.sqlite.query_result"
POSTGRESQL_QUERY_CAPABILITY_ID = "data.postgresql.query"
POSTGRESQL_QUERY_EVIDENCE_KIND = "data.postgresql.query_result"
SQLITE_UPDATE_IMPACT_CAPABILITY_ID = "data.sqlite.update_impact"
SQLITE_UPDATE_IMPACT_EVIDENCE_KIND = "data.sqlite.update_impact"
SQLITE_UPDATE_IMPACT_TOOL_NAME = "data_preview_sqlite_update"
SQLITE_UPDATE_CAPABILITY_ID = "data.sqlite.update"
SQLITE_UPDATE_EVIDENCE_KIND = "data.sqlite.update_result"
SQLITE_UPDATE_TOOL_NAME = "data_update_sqlite"

_MVP_CAPABILITIES = frozenset(
    {
        CATALOG_SEARCH_CAPABILITY_ID,
        CATALOG_SCHEMA_CAPABILITY_ID,
        CATALOG_INSPECT_CAPABILITY_ID,
        CATALOG_TRAVERSE_CAPABILITY_ID,
        SQLITE_QUERY_CAPABILITY_ID,
        POSTGRESQL_QUERY_CAPABILITY_ID,
        LOCAL_FILE_READ_CAPABILITY_ID,
        SKILL_VIEW_CAPABILITY_ID,
        SKILL_SAVE_CAPABILITY_ID,
        SKILL_DELETE_CAPABILITY_ID,
        MEMORY_SET_CAPABILITY_ID,
    }
)


class CatalogSchemaReader(Protocol):
    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]: ...


class CatalogDataReader(CatalogSchemaReader, Protocol):
    async def source_routing_facts(
        self,
        agent_id: str,
        configuration_flags: tuple[str, ...],
    ) -> tuple[Mapping[str, object], ...]: ...

    async def source_adapter_id(self, agent_id: str, source_id: str) -> str | None: ...

    async def is_current_tabular_file(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
    ) -> bool: ...


class DataToolRuntime:
    """Project, authorize, and execute built-in MVP tools."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        catalog: CatalogDataReader,
        *,
        approval_handler: ApprovalHandler | None = None,
        mutation_lock: asyncio.Lock | None = None,
        observer: AgentObserver | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(registry, CapabilityRegistry):
            raise TypeError("registry must be CapabilityRegistry")
        if not callable(getattr(catalog, "source_routing_facts", None)):
            raise TypeError("catalog must provide source_routing_facts")
        if approval_handler is not None and not callable(approval_handler):
            raise TypeError("approval_handler must be callable or None")
        if mutation_lock is not None and not isinstance(mutation_lock, asyncio.Lock):
            raise TypeError("mutation_lock must be an asyncio.Lock or None")
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable or None")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable or None")
        self._registry = registry
        self._catalog = catalog
        self._approval_handler = approval_handler
        self._mutation_lock = mutation_lock or asyncio.Lock()
        self._observer = observer
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def definitions(self, run: RunInput) -> tuple[ToolDefinition, ...]:
        names = await self._projected_tool_names(run)
        return tuple(self._registry.tool_definition(name) for name in names)

    async def execute_all(
        self,
        run: RunInput,
        calls: tuple[ToolCall, ...],
    ) -> tuple[ToolResultBlock, ...]:
        if not isinstance(run, RunInput):
            raise TypeError("run must be RunInput")
        calls = tuple(calls)
        if any(not isinstance(call, ToolCall) for call in calls):
            raise TypeError("calls must contain ToolCall records")
        projected = frozenset(await self._projected_tool_names(run))
        results: list[ToolResultBlock | None] = [None] * len(calls)
        reads: list[tuple[int, ToolCall]] = []

        async def finish_reads() -> None:
            if not reads:
                return
            indexes = tuple(index for index, _ in reads)
            completed = await asyncio.gather(
                *(self._execute_one(run, call, projected) for _, call in reads)
            )
            for index, result in zip(indexes, completed, strict=True):
                results[index] = result
            reads.clear()

        for index, call in enumerate(calls):
            if self._is_side_effecting(call, projected):
                await finish_reads()
                results[index] = await self._execute_one(run, call, projected)
            else:
                reads.append((index, call))
        await finish_reads()
        if any(result is None for result in results):
            raise RuntimeError("tool result scheduling left an incomplete call")
        return cast(tuple[ToolResultBlock, ...], tuple(results))

    async def _execute_one(
        self,
        run: RunInput,
        call: ToolCall,
        projected: frozenset[str],
    ) -> ToolResultBlock:
        started = (
            asyncio.get_running_loop().time() if self._observer is not None else None
        )
        view = None
        capability = None
        if call.name in projected:
            try:
                view, capability = self._registry.resolve_tool(call.name)
            except KeyError:
                pass
        self._emit_tool_started(run, call, capability)
        cancelled_after_mutation = False
        try:
            if view is None or capability is None:
                result = _error(
                    call,
                    "tool_not_available",
                    "The requested tool is not available for the attached sources.",
                    {"tool_name": call.name},
                )
                self._emit_tool_completed(run, call, result, started)
                return result
            arguments = self._registry.validate_arguments(capability.id, call.arguments)
            validation_error = await self._validate(run, capability, arguments)
            if validation_error is not None:
                code, message, details = validation_error
                result = _error(call, code, message, details)
                self._emit_tool_completed(run, call, result, started)
                return result
            resolved_capability, executor = self._registry.resolve_execution(
                capability.id
            )
            if resolved_capability != capability or view.capability_id != capability.id:
                raise ValueError("tool execution identity changed")
            execution = ToolExecution(
                run_id=run.id,
                capability_id=capability.id,
                arguments=arguments,
            )
            if capability.side_effecting:
                result, cancelled_after_mutation = await self._execute_side_effect(
                    run,
                    call,
                    capability,
                    executor,
                    execution,
                )
            else:
                candidate = await executor.execute(execution)
                output = self._registry.validate_output(capability.id, candidate)
                result = _success(call, output)
        except CapabilityInputError as error:
            if (
                capability is not None
                and capability.id
                in {
                    SKILL_VIEW_CAPABILITY_ID,
                    SKILL_SAVE_CAPABILITY_ID,
                    SKILL_DELETE_CAPABILITY_ID,
                }
                and "name" in call.arguments
                and error.details.get("name") == "name"
            ):
                result = _error(
                    call,
                    "skill_invalid_name",
                    "Skill names must match [a-z][a-z0-9-]{0,63}.",
                )
            else:
                result = _error(call, error.code, str(error), error.details)
        except ToolOutputValidationError as error:
            result = _error(call, "invalid_tool_result", str(error))
        except SkillNotFoundError:
            result = _error(
                call,
                "skill_not_found",
                "The requested skill is not available.",
                {"name": call.arguments.get("name")},
            )
        except SkillStoreError:
            result = _error(
                call,
                "skill_unavailable",
                "The requested skill document is unavailable or invalid.",
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            result = _error(
                call,
                "tool_execution_failed",
                f"{type(error).__name__}: {error}",
            )
        self._emit_tool_completed(run, call, result, started)
        if cancelled_after_mutation:
            raise asyncio.CancelledError
        return result

    async def _execute_side_effect(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        executor: Executor,
        execution: ToolExecution,
    ) -> tuple[ToolResultBlock, bool]:
        if (
            capability.access_mode is not AccessMode.WRITE
            or not capability.side_effecting
        ):
            raise ValueError("side-effect execution requires a write capability")
        preflight = getattr(executor, "preflight", None)
        if not callable(preflight):
            raise ValueError("side-effecting executor must provide preflight")
        side_effect = cast(SideEffectExecutor, executor)
        fingerprint = await side_effect.preflight(execution)
        if not isinstance(fingerprint, FrozenJsonObject):
            raise ValueError("side-effect preflight must return FrozenJsonObject")
        if self._approval_handler is None:
            return (
                _error(
                    call,
                    "approval_required",
                    "This side effect requires an approval handler.",
                    {"capability_id": capability.id},
                ),
                False,
            )

        request = ApprovalRequest(
            run_id=run.id,
            call_id=call.id,
            tool_name=call.name,
            capability_id=capability.id,
            arguments=cast(FrozenJsonObject, execution.arguments),
            reason="Allow this exact side-effecting tool invocation once?",
        )
        self._emit_approval_requested(run, call, capability)
        try:
            decision = await self._approval_handler(request)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._emit_approval_decided(run, call, "failed")
            return (
                _error(
                    call,
                    "approval_failed",
                    "The approval handler failed closed.",
                    {"capability_id": capability.id},
                ),
                False,
            )
        if not isinstance(decision, ApprovalDecision):
            self._emit_approval_decided(run, call, "failed")
            return (
                _error(
                    call,
                    "approval_failed",
                    "The approval handler returned an invalid decision.",
                    {"capability_id": capability.id},
                ),
                False,
            )
        if decision is ApprovalDecision.DENY:
            self._emit_approval_decided(run, call, "denied")
            return (
                _error(
                    call,
                    "approval_denied",
                    "The side effect was denied.",
                    {"capability_id": capability.id},
                ),
                False,
            )

        self._emit_approval_decided(run, call, "approved")
        async with self._mutation_lock:
            try:
                current = await side_effect.preflight(execution)
            except asyncio.CancelledError:
                raise
            except (CapabilityInputError, SkillNotFoundError):
                return (
                    _error(
                        call,
                        "state_changed",
                        "The validated state changed while approval was pending.",
                        {"capability_id": capability.id},
                    ),
                    False,
                )
            if not isinstance(current, FrozenJsonObject):
                raise ValueError("side-effect preflight must return FrozenJsonObject")
            if current != fingerprint:
                return (
                    _error(
                        call,
                        "state_changed",
                        "The validated state changed while approval was pending.",
                        {"capability_id": capability.id},
                    ),
                    False,
                )
            candidate, execution_error, cancelled = await _execute_definitely(
                side_effect,
                execution,
            )
            if execution_error is not None:
                if isinstance(execution_error, asyncio.CancelledError):
                    raise execution_error
                return _exception_result(call, execution_error), cancelled
            output = self._registry.validate_output(capability.id, candidate)
            return _success(call, output), cancelled

    def _is_side_effecting(
        self,
        call: ToolCall,
        projected: frozenset[str],
    ) -> bool:
        if call.name not in projected:
            return False
        try:
            _, capability = self._registry.resolve_tool(call.name)
        except KeyError:
            return False
        return capability.side_effecting

    def _emit_tool_started(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability | None,
    ) -> None:
        data: dict[str, object] = {
            "call_id": call.id,
            "tool_name": call.name,
        }
        if capability is not None:
            data["capability_id"] = capability.id
        self._emit(AgentEventKind.TOOL_STARTED, run, data)

    def _emit_approval_requested(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
    ) -> None:
        self._emit(
            AgentEventKind.APPROVAL_REQUESTED,
            run,
            {
                "call_id": call.id,
                "tool_name": call.name,
                "capability_id": capability.id,
            },
        )

    def _emit_approval_decided(
        self,
        run: RunInput,
        call: ToolCall,
        outcome: str,
    ) -> None:
        self._emit(
            AgentEventKind.APPROVAL_DECIDED,
            run,
            {"call_id": call.id, "outcome": outcome},
        )

    def _emit_tool_completed(
        self,
        run: RunInput,
        call: ToolCall,
        result: ToolResultBlock,
        started: float | None,
    ) -> None:
        if self._observer is None:
            return
        assert started is not None
        self._emit(
            AgentEventKind.TOOL_COMPLETED,
            run,
            {
                "call_id": call.id,
                "tool_name": call.name,
                "duration_ms": _duration_ms(started),
                "success": not result.is_error,
                "error_code": _result_error_code(result),
            },
        )

    def _emit(
        self,
        kind: AgentEventKind,
        run: RunInput,
        data: Mapping[str, object],
    ) -> None:
        if self._observer is None:
            return
        try:
            event = AgentEvent(
                kind=kind,
                occurred_at=self._clock(),
                run_id=run.id,
                conversation_id=run.conversation_id or run.id,
                data=FrozenJsonObject.from_mapping(data),
            )
        except Exception:
            return
        _emit_safely(self._observer, event)

    async def _validate(
        self,
        run: RunInput,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> tuple[str, str, Mapping[str, object]] | None:
        if capability.access_mode is AccessMode.WRITE:
            if (
                capability.id
                in {
                    MEMORY_SET_CAPABILITY_ID,
                    SKILL_SAVE_CAPABILITY_ID,
                    SKILL_DELETE_CAPABILITY_ID,
                }
                and capability.side_effecting
            ):
                return None
            return (
                "write_not_enabled",
                "Write tools are not enabled in the MVP agent loop.",
                {"capability_id": capability.id},
            )
        if capability.id == SKILL_VIEW_CAPABILITY_ID:
            name = arguments.get("name")
            try:
                if not isinstance(name, str):
                    raise TypeError("skill name must be text")
                validate_skill_name(name)
            except (TypeError, SkillValidationError):
                return (
                    "skill_invalid_name",
                    "Skill names must match [a-z][a-z0-9-]{0,63}.",
                    {},
                )
        if capability.id == CATALOG_SEARCH_CAPABILITY_ID:
            query = arguments.get("query")
            limit = arguments.get("limit", 12)
            if (
                not isinstance(query, str)
                or not query.strip()
                or len(query) > 1_024
                or not isinstance(limit, int)
                or isinstance(limit, bool)
                or not 1 <= limit <= 50
            ):
                return (
                    "catalog_invalid_search",
                    "Catalog search requires a non-empty query and limit from 1 to 50.",
                    {},
                )
        if capability.id == CATALOG_INSPECT_CAPABILITY_ID:
            resource_id = arguments.get("resource_id")
            if not isinstance(resource_id, str) or not resource_id.strip():
                return (
                    "catalog_invalid_resource",
                    "Catalog inspection requires a resource_id.",
                    {},
                )
        if capability.id == CATALOG_SCHEMA_CAPABILITY_ID:
            query = arguments.get("query")
            resource_ids = arguments.get("resource_ids", ())
            source_id = arguments.get("source_id")
            limit = arguments.get("limit", 12)
            include_relationships = arguments.get("include_relationships", True)
            valid_query = query is None or (
                isinstance(query, str) and bool(query.strip()) and len(query) <= 1_024
            )
            valid_resources = (
                isinstance(resource_ids, tuple)
                and len(resource_ids) <= 50
                and len(resource_ids) == len(set(resource_ids))
                and all(
                    isinstance(resource_id, str) and bool(resource_id.strip())
                    for resource_id in resource_ids
                )
            )
            if (
                not valid_query
                or not valid_resources
                or (query is None and not resource_ids)
                or (
                    source_id is not None
                    and (not isinstance(source_id, str) or not bool(source_id.strip()))
                )
                or not isinstance(limit, int)
                or isinstance(limit, bool)
                or not 1 <= limit <= 50
                or not isinstance(include_relationships, bool)
            ):
                return (
                    "catalog_invalid_schema",
                    (
                        "Catalog schema requires a non-empty query or explicit "
                        "resource IDs, optional current source scope, and a limit "
                        "from 1 to 50."
                    ),
                    {},
                )
        if capability.id in {
            SQLITE_QUERY_CAPABILITY_ID,
            POSTGRESQL_QUERY_CAPABILITY_ID,
        }:
            return await self._validate_sql(run, capability, arguments)
        if capability.id == LOCAL_FILE_READ_CAPABILITY_ID:
            source_id = arguments.get("source_id")
            resource_id = arguments.get("resource_id")
            if (
                not isinstance(source_id, str)
                or not isinstance(resource_id, str)
                or not await self._catalog.is_current_tabular_file(
                    run.agent_id, source_id, resource_id
                )
            ):
                return (
                    "file_not_current_or_tabular",
                    "The selected file is not a current tabular catalog resource.",
                    {"resource_id": resource_id, "source_id": source_id},
                )
        return None

    async def _validate_sql(
        self,
        run: RunInput,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> tuple[str, str, Mapping[str, object]] | None:
        source_id = arguments.get("source_id")
        sql = arguments.get("sql")
        parameters = arguments.get("parameters", ())
        if (
            not isinstance(source_id, str)
            or not isinstance(sql, str)
            or not sql.strip()
            or not isinstance(parameters, tuple)
        ):
            return (
                "sql_invalid_input",
                "SQL reads require source_id, non-empty sql, and an array of parameters.",
                {},
            )
        expected_adapter = (
            "postgresql"
            if capability.id == POSTGRESQL_QUERY_CAPABILITY_ID
            else "sqlite"
        )
        actual_adapter = await self._catalog.source_adapter_id(run.agent_id, source_id)
        if actual_adapter != expected_adapter:
            return (
                "sql_source_adapter_mismatch",
                "The selected SQL tool does not match the source adapter.",
                {
                    "actual_adapter": actual_adapter,
                    "expected_adapter": expected_adapter,
                    "source_id": source_id,
                },
            )
        resources = await self._catalog.resource_schemas(run.agent_id, source_id)
        validator = (
            validate_postgresql_read
            if expected_adapter == "postgresql"
            else validate_sqlite_read
        )
        result = validator(
            sql,
            source_id=source_id,
            resources=resources,
            parameters=parameters,
        )
        if result.valid:
            return None
        return (
            "sql_validation_failed",
            "The SQL read is invalid. Correct all reported issues before retrying.",
            {
                "issues": [
                    {
                        "code": issue.code,
                        "message": issue.message,
                        "details": issue.details,
                    }
                    for issue in result.issues
                ],
                "source_id": source_id,
            },
        )

    async def _projected_tool_names(self, run: RunInput) -> tuple[str, ...]:
        candidates: list[tuple[str, ToolApplicability]] = []
        required_flags: set[str] = set()
        for name in sorted(self._registry.tool_names):
            view, capability = self._registry.resolve_tool(name)
            if capability.id not in _MVP_CAPABILITIES:
                continue
            candidates.append((name, view.applicability))
            required_flags.update(view.applicability.required_configuration_flags)
        facts = await self._catalog.source_routing_facts(
            run.agent_id, tuple(sorted(required_flags))
        )
        return tuple(
            name
            for name, applicability in candidates
            if _applicable(applicability, facts)
        )


async def _execute_definitely(
    executor: SideEffectExecutor,
    execution: ToolExecution,
) -> tuple[ToolOutput | None, BaseException | None, bool]:
    worker = asyncio.create_task(executor.execute(execution))
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
            continue
    try:
        return worker.result(), None, cancelled
    except BaseException as error:
        return None, error, cancelled


def _exception_result(call: ToolCall, error: BaseException) -> ToolResultBlock:
    if isinstance(error, ToolOutputValidationError):
        return _error(call, "invalid_tool_result", str(error))
    return _error(
        call,
        "tool_execution_failed",
        f"{type(error).__name__}: {error}",
    )


def _result_error_code(result: ToolResultBlock) -> str | None:
    if not result.is_error:
        return None
    error = result.output.get("error")
    if not isinstance(error, Mapping):
        return "unknown_tool_error"
    code = error.get("code")
    return code if isinstance(code, str) else "unknown_tool_error"


def _duration_ms(started: float) -> int:
    elapsed = asyncio.get_running_loop().time() - started
    return max(0, int(elapsed * 1_000))


def _applicable(
    applicability: ToolApplicability,
    facts: tuple[Mapping[str, object], ...],
) -> bool:
    matching = []
    for fact in facts:
        adapter_id = fact.get("adapter_id")
        if applicability.source_adapter_ids and adapter_id not in set(
            applicability.source_adapter_ids
        ):
            continue
        flags = fact.get("configuration_flags", {})
        if not isinstance(flags, Mapping) or any(
            flags.get(flag) is not True
            for flag in applicability.required_configuration_flags
        ):
            continue
        matching.append(fact)
    return len(matching) >= applicability.minimum_active_sources


def _success(call: ToolCall, result: ToolOutput) -> ToolResultBlock:
    output: dict[str, object] = {
        "kind": result.kind,
        "data": result.data,
    }
    if result.artifact is not None:
        output["artifact"] = {
            "media_type": result.artifact.media_type,
            "retention": result.artifact.retention,
            "sensitivity": result.artifact.sensitivity,
        }
    return ToolResultBlock(call_id=call.id, output=output)


def _error(
    call: ToolCall,
    code: str,
    message: str,
    details: Mapping[str, object] | None = None,
) -> ToolResultBlock:
    return ToolResultBlock(
        call_id=call.id,
        is_error=True,
        output={
            "error": {
                "code": code,
                "message": message,
                "details": {} if details is None else details,
            }
        },
    )


# The data loop intentionally has no final-answer readiness evaluator and no
# observation schema. Model text completes the run directly.

__all__ = [
    "CatalogDataReader",
    "CatalogSchemaReader",
    "DataToolRuntime",
    "POSTGRESQL_QUERY_CAPABILITY_ID",
    "POSTGRESQL_QUERY_EVIDENCE_KIND",
    "SQLITE_QUERY_CAPABILITY_ID",
    "SQLITE_QUERY_EVIDENCE_KIND",
]
