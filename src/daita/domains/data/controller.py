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
from ...llm.models import MessageRole, ToolCall, ToolDefinition, ToolResultBlock
from ...loop.models import RunInput, Transcript
from ...semantics import (
    SEMANTIC_DELETE_CAPABILITY_ID,
    SEMANTIC_LIST_CAPABILITY_ID,
    SEMANTIC_SAVE_CAPABILITY_ID,
    SEMANTIC_VIEW_CAPABILITY_ID,
    SemanticAnnotation,
    SemanticAnnotationState,
    SemanticAnnotationView,
    SemanticDigestMismatchError,
    SemanticEvidenceKind,
    SemanticNotFoundError,
    SemanticResourceFact,
    SemanticValidationError,
    inspect_semantic_annotations,
    semantic_annotation_from_mapping,
    semantic_maintenance_intersects,
)
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
from .file_capabilities import (
    LOCAL_FILE_READ_CAPABILITY_ID,
    LOCAL_FILE_READ_EVIDENCE_KIND,
)
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
        SEMANTIC_LIST_CAPABILITY_ID,
        SEMANTIC_VIEW_CAPABILITY_ID,
        SEMANTIC_SAVE_CAPABILITY_ID,
        SEMANTIC_DELETE_CAPABILITY_ID,
    }
)
_SEMANTIC_CAPABILITIES = frozenset(
    {
        SEMANTIC_LIST_CAPABILITY_ID,
        SEMANTIC_VIEW_CAPABILITY_ID,
        SEMANTIC_SAVE_CAPABILITY_ID,
        SEMANTIC_DELETE_CAPABILITY_ID,
    }
)
_SEMANTIC_MANAGEMENT_SIGNALS = (
    "business meaning",
    "correct the definition",
    "define ",
    "definition",
    "explicit teaching request",
    "learn ",
    " means ",
    "remember ",
    "replace ",
    "resource/field-scoped semantic annotation",
    "semantic",
    "should mean",
    "supersede",
    "teach ",
    "teaching material:",
    "we mean",
    "when we say",
)


class CatalogSchemaReader(Protocol):
    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]: ...


class CatalogDataReader(CatalogSchemaReader, Protocol):
    async def catalog_context(
        self,
        agent_id: str,
        query: str,
        *,
        limit: int,
        source_ids: tuple[str, ...] = (),
        resource_ids: tuple[str, ...] = (),
    ) -> FrozenJsonObject: ...

    async def source_routing_facts(
        self,
        agent_id: str,
        configuration_flags: tuple[str, ...],
        source_ids: tuple[str, ...] = (),
    ) -> tuple[Mapping[str, object], ...]: ...

    async def source_adapter_id(self, agent_id: str, source_id: str) -> str | None: ...

    async def resource_identity(
        self,
        agent_id: str,
        resource_id: str,
    ) -> tuple[str, str, str] | None: ...

    async def is_current_tabular_file(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
    ) -> bool: ...

    async def semantic_resource_facts(
        self,
        agent_id: str,
        resource_ids: tuple[str, ...],
    ) -> tuple[SemanticResourceFact, ...]: ...


class TranscriptReader(Protocol):
    async def load(self, run_id: str) -> Transcript: ...

    async def list_semantic_annotations(
        self,
        agent_id: str,
    ) -> tuple[SemanticAnnotation, ...]: ...


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
        transcripts: TranscriptReader | None = None,
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
        if transcripts is not None and not callable(getattr(transcripts, "load", None)):
            raise TypeError("transcripts must provide load")
        self._registry = registry
        self._catalog = catalog
        self._approval_handler = approval_handler
        self._mutation_lock = mutation_lock or asyncio.Lock()
        self._observer = observer
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._transcripts = transcripts

    async def definitions(self, run: RunInput) -> tuple[ToolDefinition, ...]:
        names = await self._projected_tool_names(run)
        return tuple(self._registry.tool_definition(name) for name in names)

    async def validate_semantic_annotation(
        self,
        agent_id: str,
        annotation: SemanticAnnotation,
    ) -> None:
        """Validate direct public semantic content against current authoritative state."""

        if not isinstance(annotation, SemanticAnnotation):
            raise TypeError("annotation must be SemanticAnnotation")
        issue = await self._semantic_annotation_issue(agent_id, annotation)
        if issue is not None:
            _code, message, _details = issue
            raise SemanticValidationError(message)

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
            raw_arguments = (
                _without_runtime_owned_semantic_evidence(call.arguments)
                if capability.id == SEMANTIC_SAVE_CAPABILITY_ID
                else call.arguments
            )
            arguments = self._registry.validate_arguments(
                capability.id,
                raw_arguments,
            )
            arguments = self._apply_source_scope(run, capability, arguments)
            validation_error = await self._validate(run, capability, arguments)
            if validation_error is not None:
                code, message, details = validation_error
                result = _error(call, code, message, details)
                self._emit_tool_completed(run, call, result, started)
                return result
            if capability.id == SEMANTIC_SAVE_CAPABILITY_ID:
                arguments = await self._bind_current_semantic_evidence(
                    run,
                    arguments,
                )
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
                if capability.id == SEMANTIC_VIEW_CAPABILITY_ID:
                    candidate = await self._decorate_semantic_view(
                        run,
                        arguments,
                        candidate,
                    )
                elif capability.id == SEMANTIC_LIST_CAPABILITY_ID:
                    self._registry.validate_output(capability.id, candidate)
                    candidate = await self._filter_semantic_list(
                        run,
                        arguments,
                        candidate,
                    )
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
        except SemanticNotFoundError:
            result = _error(
                call,
                "semantic_not_found",
                "The requested semantic annotation is not available.",
                {"id": call.arguments.get("id")},
            )
        except SemanticDigestMismatchError as error:
            code = (
                "semantic_expected_sha256_required"
                if "requires expected_sha256" in str(error)
                else "semantic_stale_digest"
            )
            result = _error(call, code, str(error), {"id": call.arguments.get("id")})
        except SemanticValidationError as error:
            result = _error(
                call,
                "semantic_invalid_annotation",
                str(error),
                {"id": call.arguments.get("id")},
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
        await self._validate_semantic_preflight(run, capability, fingerprint)
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
                await self._validate_semantic_preflight(run, capability, current)
            except asyncio.CancelledError:
                raise
            except (
                CapabilityInputError,
                SemanticDigestMismatchError,
                SemanticNotFoundError,
                SemanticValidationError,
                SkillNotFoundError,
            ):
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

    async def _validate_semantic_preflight(
        self,
        run: RunInput,
        capability: Capability,
        fingerprint: FrozenJsonObject,
    ) -> None:
        if capability.id != SEMANTIC_SAVE_CAPABILITY_ID:
            return
        raw_annotation = fingerprint.get("annotation")
        if not isinstance(raw_annotation, Mapping):
            raise ValueError("semantic preflight omitted its candidate annotation")
        annotation = semantic_annotation_from_mapping(raw_annotation)
        if annotation.agent_id != run.agent_id:
            raise CapabilityInputError(
                "semantic_foreign_agent",
                "The semantic annotation belongs to another agent.",
            )
        issue = await self._semantic_annotation_issue(run.agent_id, annotation)
        if issue is not None:
            code, message, details = issue
            raise CapabilityInputError(code, message, details)

    async def _bind_current_semantic_evidence(
        self,
        run: RunInput,
        arguments: Mapping[str, object],
    ) -> FrozenJsonObject:
        """Replace model-authored provenance with exact current transcript facts."""

        if self._transcripts is None:
            raise CapabilityInputError(
                "semantic_evidence_unavailable",
                "Semantic evidence validation is unavailable.",
            )
        try:
            transcript = await self._transcripts.load(run.id)
        except KeyError as error:
            raise CapabilityInputError(
                "semantic_invalid_evidence",
                "The current semantic write transcript is unavailable.",
                {"run_id": run.id},
            ) from error
        if transcript.run.id != run.id or transcript.run.agent_id != run.agent_id:
            raise CapabilityInputError(
                "semantic_invalid_evidence",
                "The current semantic write transcript identity does not match.",
                {"run_id": run.id},
            )

        raw_evidence = arguments.get("evidence")
        if not isinstance(raw_evidence, tuple):
            raise CapabilityInputError(
                "semantic_invalid_evidence",
                "Semantic evidence must be a bounded array.",
            )
        user_positions = tuple(
            position
            for position, message in enumerate(transcript.messages)
            if message.role is MessageRole.USER
        )
        bound: list[dict[str, object]] = []
        for item in raw_evidence:
            if not isinstance(item, Mapping):
                raise CapabilityInputError(
                    "semantic_invalid_evidence",
                    "Semantic evidence entries must be objects.",
                )
            kind_value = item.get("kind")
            try:
                kind = SemanticEvidenceKind(kind_value)
            except (TypeError, ValueError) as error:
                raise CapabilityInputError(
                    "semantic_invalid_evidence",
                    "Semantic evidence kind is not supported.",
                ) from error
            note = item.get("note")
            entry: dict[str, object] = {
                "kind": kind.value,
                "run_id": run.id,
            }
            if note is not None:
                entry["note"] = note
            if kind in {
                SemanticEvidenceKind.USER_ASSERTION,
                SemanticEvidenceKind.USER_CONFIRMATION,
            }:
                if len(user_positions) != 1:
                    raise CapabilityInputError(
                        "semantic_invalid_evidence",
                        "Current-run user evidence must resolve to exactly one message.",
                        {"run_id": run.id},
                    )
                entry["message_position"] = user_positions[0]
            else:
                tool_call_id = item.get("tool_call_id")
                if not isinstance(tool_call_id, str):
                    raise CapabilityInputError(
                        "semantic_invalid_evidence",
                        "Tool-result evidence requires a tool_call_id.",
                    )
                result_positions = tuple(
                    position
                    for position, message in enumerate(transcript.messages)
                    if message.role is MessageRole.TOOL
                    and any(
                        isinstance(block, ToolResultBlock)
                        and block.call_id == tool_call_id
                        for block in message.content
                    )
                )
                if len(result_positions) != 1:
                    raise CapabilityInputError(
                        "semantic_invalid_evidence",
                        (
                            "Tool-result evidence must reference exactly one result "
                            "from an earlier completed tool step in the current run."
                        ),
                        {"run_id": run.id, "tool_call_id": tool_call_id},
                    )
                entry["message_position"] = result_positions[0]
                entry["tool_call_id"] = tool_call_id
            bound.append(entry)

        normalized = (
            arguments.to_dict()
            if isinstance(arguments, FrozenJsonObject)
            else dict(arguments)
        )
        normalized["evidence"] = bound
        return FrozenJsonObject.from_mapping(normalized)

    async def _decorate_semantic_view(
        self,
        run: RunInput,
        arguments: Mapping[str, object],
        output: ToolOutput,
    ) -> ToolOutput:
        annotation_id = arguments.get("id")
        if not isinstance(annotation_id, str):
            raise CapabilityInputError(
                "semantic_invalid_id",
                "Semantic view requires an annotation id.",
            )
        selected = next(
            (
                view
                for view in await self._current_semantic_views(run.agent_id)
                if view.annotation.id == annotation_id
            ),
            None,
        )
        if selected is None:
            raise SemanticNotFoundError(annotation_id)
        if output.data.get("current_sha256") != selected.sha256:
            raise CapabilityInputError(
                "semantic_state_changed",
                "The semantic annotation changed during inspection; view it again.",
                {"id": annotation_id},
            )
        data = dict(output.data)
        data["maintenance"] = {
            "state": selected.state.value,
            "usable_as_current_meaning": selected.usable_as_current_meaning,
            "requires_revalidation": selected.requires_revalidation,
            "stale_reasons": selected.stale_reasons,
            "conflicting_ids": selected.conflicting_ids,
            "duplicate_ids": selected.duplicate_ids,
            "duplicate_of_id": selected.duplicate_of_id,
            "superseded_by_id": selected.superseded_by_id,
        }
        return ToolOutput(kind=output.kind, data=data)

    async def _filter_semantic_list(
        self,
        run: RunInput,
        arguments: Mapping[str, object],
        output: ToolOutput,
    ) -> ToolOutput:
        raw = output.data.get("annotations")
        if not isinstance(raw, tuple):
            raise ToolOutputValidationError(
                "semantic list output annotations must be an array"
            )
        source_id = arguments.get("source_id")
        resource_id = arguments.get("resource_id")
        kind = arguments.get("kind")
        limit = arguments.get("limit", 24)
        assert source_id is None or isinstance(source_id, str)
        assert resource_id is None or isinstance(resource_id, str)
        assert kind is None or isinstance(kind, str)
        assert isinstance(limit, int) and not isinstance(limit, bool)
        active = tuple(
            view
            for view in await self._current_semantic_views(run.agent_id)
            if view.state is SemanticAnnotationState.ACTIVE
            and (source_id is None or source_id in view.annotation.subject.source_ids)
            and (
                resource_id is None
                or resource_id in view.annotation.subject.resource_ids
            )
            and (kind is None or view.annotation.kind.value == kind)
        )[:limit]
        annotations = tuple(
            {
                "id": view.annotation.id,
                "kind": view.annotation.kind.value,
                "resource_ids": view.annotation.subject.resource_ids,
                "field_count": len(view.annotation.subject.fields),
                "statement_preview": view.annotation.statement[:240],
                "current_sha256": view.sha256,
            }
            for view in active
        )
        return ToolOutput(
            kind=output.kind,
            data={"annotations": annotations, "count": len(annotations)},
        )

    async def _current_semantic_views(
        self,
        agent_id: str,
    ) -> tuple[SemanticAnnotationView, ...]:
        if self._transcripts is None:
            raise CapabilityInputError(
                "semantic_state_unavailable",
                "Semantic state validation is unavailable.",
            )
        annotations = await self._transcripts.list_semantic_annotations(agent_id)
        resource_ids = tuple(
            sorted(
                {
                    resource_id
                    for annotation in annotations
                    for resource_id in annotation.subject.resource_ids
                }
            )
        )
        facts = await self._catalog.semantic_resource_facts(agent_id, resource_ids)
        return inspect_semantic_annotations(annotations, facts)

    async def _semantic_annotation_issue(
        self,
        agent_id: str,
        annotation: SemanticAnnotation,
    ) -> tuple[str, str, Mapping[str, object]] | None:
        if annotation.agent_id != agent_id:
            return (
                "semantic_foreign_agent",
                "The semantic annotation belongs to another agent.",
                {"annotation_id": annotation.id},
            )
        facts = await self._catalog.semantic_resource_facts(
            agent_id,
            annotation.subject.resource_ids,
        )
        fact_by_id = {item.resource_id: item for item in facts}
        for resource_id in annotation.subject.resource_ids:
            if resource_id not in fact_by_id:
                return (
                    "semantic_unknown_resource",
                    "A semantic subject resource is not current for this agent.",
                    {"resource_id": resource_id},
                )
        actual_sources = tuple(
            sorted(
                {fact_by_id[item].source_id for item in annotation.subject.resource_ids}
            )
        )
        if actual_sources != annotation.subject.source_ids:
            return (
                "semantic_source_mismatch",
                "Semantic source scope does not match the current catalog resources.",
                {
                    "actual_source_ids": actual_sources,
                    "subject_source_ids": annotation.subject.source_ids,
                },
            )
        revisions = {
            item.resource_id: item.revision for item in annotation.catalog_revisions
        }
        for resource_id in annotation.subject.resource_ids:
            current_revision = fact_by_id[resource_id].revision
            if revisions[resource_id] != current_revision:
                return (
                    "semantic_stale_revision",
                    "A semantic revision binding does not match the current catalog.",
                    {
                        "current_revision": current_revision,
                        "resource_id": resource_id,
                        "requested_revision": revisions[resource_id],
                    },
                )
        for field in annotation.subject.fields:
            if field.field_name not in fact_by_id[field.resource_id].field_names:
                return (
                    "semantic_unknown_field",
                    "A semantic subject field is not current for its resource.",
                    {
                        "field_name": field.field_name,
                        "resource_id": field.resource_id,
                    },
                )
        return await self._semantic_evidence_issue(agent_id, annotation)

    async def _semantic_evidence_issue(
        self,
        agent_id: str,
        annotation: SemanticAnnotation,
    ) -> tuple[str, str, Mapping[str, object]] | None:
        if self._transcripts is None:
            return (
                "semantic_evidence_unavailable",
                "Semantic evidence validation is unavailable.",
                {"annotation_id": annotation.id},
            )
        for evidence in annotation.evidence:
            position = evidence.message_position
            assert position is not None
            try:
                transcript = await self._transcripts.load(evidence.run_id)
            except KeyError:
                return (
                    "semantic_invalid_evidence",
                    "Semantic evidence references an unknown run.",
                    {"run_id": evidence.run_id},
                )
            if transcript.run.agent_id != agent_id:
                return (
                    "semantic_invalid_evidence",
                    "Semantic evidence references a run owned by another agent.",
                    {"run_id": evidence.run_id},
                )
            if position >= len(transcript.messages):
                return (
                    "semantic_invalid_evidence",
                    "Semantic evidence references a missing transcript message.",
                    {
                        "message_position": evidence.message_position,
                        "run_id": evidence.run_id,
                    },
                )
            message = transcript.messages[position]
            if evidence.kind in {
                SemanticEvidenceKind.USER_ASSERTION,
                SemanticEvidenceKind.USER_CONFIRMATION,
            }:
                if message.role is not MessageRole.USER:
                    return (
                        "semantic_invalid_evidence",
                        "User semantic evidence must reference an exact user message.",
                        {
                            "message_position": evidence.message_position,
                            "run_id": evidence.run_id,
                        },
                    )
                continue
            if message.role is not MessageRole.TOOL:
                return (
                    "semantic_invalid_evidence",
                    "Tool-result evidence must reference an exact tool-result message.",
                    {
                        "message_position": evidence.message_position,
                        "run_id": evidence.run_id,
                    },
                )
            result = next(
                (
                    block
                    for block in message.content
                    if isinstance(block, ToolResultBlock)
                    and block.call_id == evidence.tool_call_id
                ),
                None,
            )
            if (
                result is None
                or result.is_error
                or result.output.get("kind")
                not in {
                    SQLITE_QUERY_EVIDENCE_KIND,
                    POSTGRESQL_QUERY_EVIDENCE_KIND,
                    LOCAL_FILE_READ_EVIDENCE_KIND,
                }
            ):
                return (
                    "semantic_invalid_evidence",
                    "Tool-result evidence must reference a successful validated data read.",
                    {
                        "run_id": evidence.run_id,
                        "tool_call_id": evidence.tool_call_id,
                    },
                )
            call_exists = any(
                call.id == evidence.tool_call_id
                for prior in transcript.messages[: evidence.message_position]
                if prior.role is MessageRole.ASSISTANT
                for call in prior.tool_calls
            )
            if not call_exists:
                return (
                    "semantic_invalid_evidence",
                    "Tool-result evidence has no matching prior tool call.",
                    {
                        "run_id": evidence.run_id,
                        "tool_call_id": evidence.tool_call_id,
                    },
                )
        return None

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
        source_scope_error = await self._validate_source_scope(
            run,
            capability,
            arguments,
        )
        if source_scope_error is not None:
            return source_scope_error
        if capability.access_mode is AccessMode.WRITE:
            if (
                capability.id
                in {
                    MEMORY_SET_CAPABILITY_ID,
                    SEMANTIC_SAVE_CAPABILITY_ID,
                    SEMANTIC_DELETE_CAPABILITY_ID,
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

    def _apply_source_scope(
        self,
        run: RunInput,
        capability: Capability,
        arguments: FrozenJsonObject,
    ) -> FrozenJsonObject:
        if (
            run.source_id is None
            or capability.id
            not in {
                CATALOG_SEARCH_CAPABILITY_ID,
                CATALOG_SCHEMA_CAPABILITY_ID,
            }
            or arguments.get("source_id") is not None
        ):
            return arguments
        scoped = arguments.to_dict()
        scoped["source_id"] = run.source_id
        return self._registry.validate_arguments(capability.id, scoped)

    async def _validate_source_scope(
        self,
        run: RunInput,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> tuple[str, str, Mapping[str, object]] | None:
        selected_source_id = run.source_id
        if selected_source_id is None:
            return None
        supplied_source_id = arguments.get("source_id")
        if supplied_source_id is not None and supplied_source_id != selected_source_id:
            return (
                "source_scope_violation",
                "This run can only access the source selected by the user.",
                {
                    "selected_source_id": selected_source_id,
                    "requested_source_id": supplied_source_id,
                },
            )
        resource_ids: tuple[object, ...] = ()
        if capability.id == CATALOG_INSPECT_CAPABILITY_ID:
            resource_ids = (arguments.get("resource_id"),)
        elif capability.id == CATALOG_SCHEMA_CAPABILITY_ID:
            value = arguments.get("resource_ids", ())
            resource_ids = value if isinstance(value, tuple) else ()
        elif capability.id == CATALOG_TRAVERSE_CAPABILITY_ID:
            from_ids = arguments.get("from_resource_ids", ())
            to_ids = arguments.get("to_resource_ids", ())
            resource_ids = (
                *(from_ids if isinstance(from_ids, tuple) else ()),
                *(to_ids if isinstance(to_ids, tuple) else ()),
            )
        for resource_id in resource_ids:
            if not isinstance(resource_id, str):
                continue
            identity = await self._catalog.resource_identity(
                run.agent_id,
                resource_id,
            )
            if identity is None or identity[0] != selected_source_id:
                return (
                    "source_scope_violation",
                    "This run can only access resources from the selected source.",
                    {
                        "resource_id": resource_id,
                        "selected_source_id": selected_source_id,
                    },
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
        semantic_requested = _semantic_management_requested(run.message)
        if not semantic_requested:
            semantic_requested = await self._semantic_maintenance_requested(run)
        for name in sorted(self._registry.tool_names):
            view, capability = self._registry.resolve_tool(name)
            if capability.id not in _MVP_CAPABILITIES:
                continue
            if capability.id in _SEMANTIC_CAPABILITIES and not semantic_requested:
                continue
            candidates.append((name, view.applicability))
            required_flags.update(view.applicability.required_configuration_flags)
        required = tuple(sorted(required_flags))
        facts = (
            await self._catalog.source_routing_facts(
                run.agent_id,
                required,
                (run.source_id,),
            )
            if run.source_id is not None
            else await self._catalog.source_routing_facts(
                run.agent_id,
                required,
            )
        )
        return tuple(
            name
            for name, applicability in candidates
            if _applicable(applicability, facts)
        )

    async def _semantic_maintenance_requested(self, run: RunInput) -> bool:
        if self._transcripts is None:
            return False
        views = await self._current_semantic_views(run.agent_id)
        if not any(
            view.requires_revalidation
            or view.state is SemanticAnnotationState.DUPLICATE
            or bool(view.duplicate_ids)
            for view in views
        ):
            return False
        catalog = await self._catalog.catalog_context(
            run.agent_id,
            run.message[:4_000],
            limit=12,
            source_ids=(() if run.source_id is None else (run.source_id,)),
        )
        resources = catalog.get("resources")
        if not isinstance(resources, tuple):
            return False
        selected_resource_ids = tuple(
            resource_id
            for resource in resources
            if isinstance(resource, FrozenJsonObject)
            and isinstance((resource_id := resource.get("resource_id")), str)
        )
        return semantic_maintenance_intersects(
            views,
            selected_resource_ids=selected_resource_ids,
            query=run.message,
        )


def _semantic_management_requested(message: str) -> bool:
    normalized = " ".join(message.casefold().split())
    return any(signal in normalized for signal in _SEMANTIC_MANAGEMENT_SIGNALS)


def _without_runtime_owned_semantic_evidence(
    arguments: Mapping[str, object],
) -> Mapping[str, object]:
    """Remove provenance fields that only the runtime is allowed to establish."""

    raw_evidence = arguments.get("evidence")
    if not isinstance(raw_evidence, tuple):
        return arguments
    normalized = dict(arguments)
    normalized["evidence"] = [
        (
            {
                key: value
                for key, value in item.items()
                if key not in {"run_id", "message_position"}
            }
            if isinstance(item, Mapping)
            else item
        )
        for item in raw_evidence
    ]
    return normalized


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
