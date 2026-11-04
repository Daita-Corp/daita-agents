"""Project data tools and revalidate their catalog and permission bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from difflib import get_close_matches
from hashlib import sha256
from typing import Protocol, cast

from ..._json import FrozenJsonObject, canonical_json
from ...adapters.local_file_query import LocalFileQueryError
from ...adapters.local_workspace import LocalWorkspaceError
from ...adapters.registry import SourceOperation, SourceTypeRegistry
from ...artifacts.models import (
    ArtifactAuthorship,
    ArtifactDraft,
    ArtifactError,
)
from ...artifacts.renderers import ARROW_MEDIA_TYPE, arrow_schema
from ...capabilities import (
    AccessMode,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    OperationalEffect,
    RecoveryKind,
    ToolExecution,
    ToolOutput,
    ToolOutputValidationError,
    ToolRecovery,
)
from ...capability_runtime import CapabilityFailure, SideEffectPlan
from ...catalog.capabilities import (
    CATALOG_DESCRIBE_CAPABILITY_ID,
    CATALOG_SEARCH_CAPABILITY_ID,
)
from ...catalog.protocols import CatalogStoreError
from ...llm.models import ModelSensitivity, ToolCall
from ...loop.models import RunInput, RunOrigin
from ...loop.session import RunSession
from ...scope import (
    EffectiveSourceScope,
    narrowed_read_access,
    resolve_effective_source_scope,
    run_read_narrowings,
)
from ...storage.sqlite_records import RelationalWriteScope, SourcePermissionStateError
from .sql import (
    CheckedRead,
    DuckDBReadValidationError,
    ResourceSchema,
    SqlValidationIssue,
    WriteFacts,
    checked_reads,
    validate_duckdb_read,
    validate_read,
    write_facts,
    write_permission_issue,
)
from .sql.contracts import _SqlDialect

DATA_DOMAIN_OWNER_ID = "data"
DATA_QUERY_CAPABILITY_ID = "data.query"
DATA_QUERY_EVIDENCE_KIND = "data.query_result"
DATA_QUERY_COMBINED_CAPABILITY_ID = "data.query_combined"
DATA_QUERY_COMBINED_EVIDENCE_KIND = "data.query_combined_result"
DATA_EXPORT_TABULAR_CAPABILITY_ID = "data.export_tabular"
RELATIONAL_WRITE_PREVIEW_CAPABILITY_ID = "data.preview_write"
RELATIONAL_WRITE_PREVIEW_EVIDENCE_KIND = "data.preview_write"
RELATIONAL_WRITE_CAPABILITY_ID = "data.write"
RELATIONAL_WRITE_EVIDENCE_KIND = "data.write_result"
FILE_SEARCH_CAPABILITY_ID = "data.search_files"
FILE_SEARCH_OUTPUT_KIND = "data.search_files_result"
TEXT_READ_CAPABILITY_ID = "data.read_text"
TEXT_READ_OUTPUT_KIND = "data.read_text_result"
PATH_QUERY_CAPABILITY_ID = "data.query_files"
PATH_QUERY_OUTPUT_KIND = "data.query_files_result"
# Folder-source reads whose results carry their own binding provenance.
SOURCE_FILE_CAPABILITY_IDS = frozenset(
    {FILE_SEARCH_CAPABILITY_ID, TEXT_READ_CAPABILITY_ID, PATH_QUERY_CAPABILITY_ID}
)

_CATALOG_CAPABILITIES = frozenset(
    {
        CATALOG_SEARCH_CAPABILITY_ID,
        CATALOG_DESCRIBE_CAPABILITY_ID,
    }
)
_RELATIONAL_READ_CAPABILITIES = frozenset(
    {DATA_QUERY_CAPABILITY_ID, DATA_EXPORT_TABULAR_CAPABILITY_ID}
)
_WRITE_CAPABILITIES = frozenset(
    {RELATIONAL_WRITE_PREVIEW_CAPABILITY_ID, RELATIONAL_WRITE_CAPABILITY_ID}
)
_PERMISSION_ISSUE_CODES = frozenset(
    {
        "resource_write_not_allowed",
        "write_operation_not_allowed",
        "insert_column_not_allowed",
        "update_column_not_allowed",
    }
)
_RESOURCE_LIST_ARGUMENT_CAPABILITIES = _RELATIONAL_READ_CAPABILITIES
_RESOURCE_READ_DENIED_MESSAGE = (
    "The requested table is not available for reading in this run. Use the exact "
    "table names the catalog shows; do not guess or substitute another table."
)
# The source operation a source's registered type must declare for each capability.
_CAPABILITY_OPERATIONS = {
    DATA_QUERY_CAPABILITY_ID: SourceOperation.SQL_READ,
    DATA_EXPORT_TABULAR_CAPABILITY_ID: SourceOperation.TABULAR_EXPORT,
    FILE_SEARCH_CAPABILITY_ID: SourceOperation.FILE_SEARCH,
    TEXT_READ_CAPABILITY_ID: SourceOperation.TEXT_READ,
    PATH_QUERY_CAPABILITY_ID: SourceOperation.PATH_QUERY,
    **{item: SourceOperation.RELATIONAL_WRITE for item in _WRITE_CAPABILITIES},
}
# Combined-query inputs each need SQL_READ or TABULAR_READ, checked per input.
_SOURCE_CAPABILITIES = frozenset().union(
    _CATALOG_CAPABILITIES,
    _CAPABILITY_OPERATIONS,
    {DATA_QUERY_COMBINED_CAPABILITY_ID},
)


def _sql_statements(
    arguments: Mapping[str, object],
) -> tuple[tuple[str, tuple[object, ...]], ...] | None:
    """Return validated (sql, parameters) pairs, or None for a malformed shape."""

    raw_statements = arguments.get("statements")
    if raw_statements is not None:
        if "sql" in arguments or "parameters" in arguments:
            return None
        if not isinstance(raw_statements, tuple) or not raw_statements:
            return None
        items: tuple[object, ...] = raw_statements
    else:
        items = (
            {
                "sql": arguments.get("sql"),
                "parameters": arguments.get("parameters", ()),
            },
        )
    statements: list[tuple[str, tuple[object, ...]]] = []
    for item in items:
        if not isinstance(item, Mapping):
            return None
        sql = item.get("sql")
        parameters = item.get("parameters", ())
        if not isinstance(sql, str) or not sql.strip():
            return None
        if not isinstance(parameters, tuple):
            return None
        statements.append((sql, parameters))
    return tuple(statements)


_SCOPE_ISSUE_CODES = frozenset({"resource_out_of_scope", "schema_not_configured"})
_COPY_CATALOG_IDS = ToolRecovery(
    RecoveryKind.CORRECT_ARGUMENTS,
    "Use exact table names from this run's catalog; a table outside it "
    "cannot be read in this run.",
)
_CORRECT_SQL = ToolRecovery(
    RecoveryKind.CORRECT_ARGUMENTS,
    "Fix every reported issue in the SQL or its source and table names; "
    "the same call will be refused again.",
)
_WRITE_PERMISSION_REQUIRED = ToolRecovery(
    RecoveryKind.REQUIRES_HUMAN_PERMISSION,
    "The person must grant write permission for this table before any write.",
)


class CatalogSchemaReader(Protocol):
    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]: ...


class ReadScopedCatalogReader(CatalogSchemaReader, Protocol):
    async def readable_resource_ids(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> frozenset[str]: ...


class RelationalWriteScopeReader(Protocol):
    async def relational_write_scopes(
        self, agent_id: str, source_id: str
    ) -> Mapping[str, RelationalWriteScope]: ...


class DataDomainCatalog(ReadScopedCatalogReader, RelationalWriteScopeReader, Protocol):
    async def source_routing_facts(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> tuple[Mapping[str, object], ...]: ...

    async def relational_write_applicable_source_ids(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> frozenset[str]: ...

    async def source_adapter_id(self, agent_id: str, source_id: str) -> str | None: ...

    async def source_schemas(
        self, agent_id: str, source_id: str
    ) -> tuple[str, ...]: ...

    async def source_read_access(
        self, agent_id: str, source_ids: tuple[str, ...]
    ) -> Mapping[str, frozenset[str] | None]: ...

    async def resource_identity(
        self,
        agent_id: str,
        resource_id: str,
    ) -> tuple[str, str, str] | None: ...

    async def admitted_model_sensitivity(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> ModelSensitivity | None: ...

    @property
    def session_source_id(self) -> str | None: ...


class WriteDescriber(Protocol):
    """A source's parse of one write, before any source I/O."""

    def describe(
        self,
        sql: str,
        *,
        source_id: str,
        parameters: Sequence[object],
        allowed_resource_ids: frozenset[str] | None,
        schemas: tuple[str, ...],
    ) -> WriteFacts | tuple[SqlValidationIssue, ...]: ...


class DataCapabilityDomain:
    """Own native catalog, SQL, update, and folder-file call semantics."""

    domain_owner_id = DATA_DOMAIN_OWNER_ID

    def __init__(
        self,
        declarations: CapabilityDeclarations,
        catalog: DataDomainCatalog,
        source_types: SourceTypeRegistry,
        *,
        relational_export_available: bool = True,
        combined_query_available: bool = False,
        source_files_available: bool = False,
        path_query_available: bool = False,
        write_backends: Mapping[str, WriteDescriber] | None = None,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("data declarations have the wrong domain owner")
        declared_ids = {item.id for item in declarations.capabilities}
        expected_ids = _SOURCE_CAPABILITIES
        if not relational_export_available:
            expected_ids -= {DATA_EXPORT_TABULAR_CAPABILITY_ID}
        if not combined_query_available:
            expected_ids -= {DATA_QUERY_COMBINED_CAPABILITY_ID}
        if not source_files_available:
            expected_ids -= {FILE_SEARCH_CAPABILITY_ID, TEXT_READ_CAPABILITY_ID}
        if not path_query_available:
            expected_ids -= {PATH_QUERY_CAPABILITY_ID}
        if declared_ids != expected_ids:
            raise ValueError(
                "data declarations must exactly match supported native capabilities"
            )
        if not isinstance(source_types, SourceTypeRegistry):
            raise TypeError("source_types must be a SourceTypeRegistry")
        for method_name in (
            "source_routing_facts",
            "resource_schemas",
            "readable_resource_ids",
            "source_adapter_id",
        ):
            if not callable(getattr(catalog, method_name, None)):
                raise TypeError(f"data catalog must provide {method_name}")
        self._declarations = declarations
        self._catalog = catalog
        self._source_types = source_types
        self._write_backends = dict(write_backends or {})
        self._views = {item.name: item for item in declarations.tool_views}
        self._capabilities = {item.id: item for item in declarations.capabilities}

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(
        self,
        run: RunInput,
        session: RunSession | None = None,
        *,
        tool_names: frozenset[str] | None = None,
    ) -> tuple[str, ...]:
        return (await self._projection(run, session, tool_names))[0]

    project_session = project

    async def project_routes(
        self, run: RunInput, session: RunSession | None = None
    ) -> tuple[tuple[str, ...], frozenset[str]]:
        """Project the tool names and the routed capabilities no source in this
        run's scope can serve, from one scope resolution.

        ``data_query`` is offered for any SQL source, but its export route needs
        a ``TABULAR_EXPORT`` source and its ``path_pattern`` route the live
        session folder; offering them otherwise only invites refused calls.
        """

        return await self._projection(run, session, None)

    async def _projection(
        self,
        run: RunInput,
        session: RunSession | None,
        tool_names: frozenset[str] | None,
    ) -> tuple[tuple[str, ...], frozenset[str]]:
        views = {
            name: view
            for name, view in self._views.items()
            if tool_names is None or name in tool_names
        }
        source_capability_ids = {view.capability_id for view in views.values()}
        scope = (
            await resolve_effective_source_scope(run, self._catalog)
            if source_capability_ids
            else EffectiveSourceScope(frozenset(), frozenset())
        )
        run = replace(run, resolved_source_scope=scope)
        facts: tuple[Mapping[str, object], ...]
        write_source_ids: frozenset[str]
        if not scope.source_ids:
            facts = ()
            write_source_ids = frozenset()
        else:
            facts = await self._catalog.source_routing_facts(
                run.agent_id,
                tuple(sorted(scope.source_ids)),
            )
            write_source_ids = (
                await self._catalog.relational_write_applicable_source_ids(
                    run.agent_id,
                    tuple(sorted(scope.source_ids)),
                )
                if source_capability_ids & _WRITE_CAPABILITIES
                else frozenset()
            )
        active_adapter_ids: set[str] = set()
        for fact in facts:
            adapter_id = fact.get("adapter_id")
            if isinstance(adapter_id, str):
                active_adapter_ids.add(adapter_id)
        names: list[str] = []
        for name in sorted(views):
            view = views[name]
            capability = self._capabilities[view.capability_id]
            if capability.id in _CATALOG_CAPABILITIES:
                # The live session folder source has no catalog to discover.
                if any(
                    fact.get("source_id") != self._catalog.session_source_id
                    for fact in facts
                ):
                    names.append(name)
                continue
            operation = _CAPABILITY_OPERATIONS.get(capability.id)
            if operation is None or not any(
                self._source_types.supports(adapter_id, operation)
                for adapter_id in active_adapter_ids
            ):
                continue
            if capability.id in _WRITE_CAPABILITIES and not write_source_ids:
                continue
            names.append(name)
        withheld = {
            capability_id
            for capability_id, operation in _CAPABILITY_OPERATIONS.items()
            if capability_id in self._capabilities
            and not any(
                self._source_types.supports(adapter_id, operation)
                for adapter_id in active_adapter_ids
            )
        }
        return tuple(names), frozenset(withheld)

    async def project_call(
        self, run: RunInput, call: ToolCall, session: RunSession | None
    ) -> tuple[str, ...]:
        """Revalidate one existing view without projecting unrelated contracts."""
        return await self.project(run, session, tool_names=frozenset({call.name}))

    def normalize_arguments(
        self,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        return arguments

    async def prepare_call(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        *,
        request_sensitivity: ModelSensitivity,
        session: RunSession | None = None,
    ) -> FrozenJsonObject:
        scope = await resolve_effective_source_scope(run, self._catalog)
        run = replace(run, resolved_source_scope=scope)
        await self._validate_source_scope(run, capability, arguments)
        self._validate_execution_resource_scope(run, capability, arguments)
        await self._validate_resource_read_scope(run, capability, arguments)
        if capability.access_mode is AccessMode.WRITE and (
            capability.id != RELATIONAL_WRITE_CAPABILITY_ID
            or capability.operational_effect is not OperationalEffect.MUTATE_DATA
        ):
            raise CapabilityInputError(
                "write_not_enabled",
                "This data write is not enabled in the current runtime.",
                {"capability_id": capability.id},
            )
        if capability.id == CATALOG_SEARCH_CAPABILITY_ID:
            query = cast(str, arguments["query"])
            if not query.strip():
                raise CapabilityInputError(
                    "catalog_invalid_search",
                    "Catalog search requires a non-empty query.",
                )
        if capability.id == DATA_QUERY_COMBINED_CAPABILITY_ID:
            arguments = await self._prepare_combined_query(run, arguments)
        if capability.id in SOURCE_FILE_CAPABILITY_IDS:
            arguments = await self._prepare_source_file_call(run, capability, arguments)
        if capability.id in _RELATIONAL_READ_CAPABILITIES:
            arguments = await self._validate_sql(
                run,
                arguments,
                operation=_CAPABILITY_OPERATIONS[capability.id],
            )
        if capability.id in _WRITE_CAPABILITIES:
            arguments = await self._prepare_write(run, arguments, request_sensitivity)
        return arguments

    prepare_session_call = prepare_call

    async def side_effect_plan(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        execution: ToolExecution,
        fingerprint: FrozenJsonObject,
    ) -> SideEffectPlan:
        if (
            capability.id != RELATIONAL_WRITE_CAPABILITY_ID
            or capability.operational_effect is not OperationalEffect.MUTATE_DATA
        ):
            raise ValueError("data domain received an unsupported side effect")
        write = write_facts(execution.arguments)
        source_id = cast(str, execution.arguments["source_id"])
        source = next(
            (
                item
                for item in await self._catalog.source_routing_facts(
                    run.agent_id, (source_id,)
                )
                if item["source_id"] == source_id
            ),
            None,
        )
        if source is None:
            raise CapabilityInputError(
                "write_state_changed",
                "The source is no longer available.",
            )
        approval_arguments = FrozenJsonObject.from_mapping(
            {
                "arguments": {
                    "source_id": source_id,
                    "sql": execution.arguments["sql"],
                    "parameters": execution.arguments.get("parameters", ()),
                },
                "target": {
                    "source_id": source_id,
                    "resource_id": write.resource_id,
                    "name": write.table,
                    "source_name": source["display_name"],
                },
                "preview": fingerprint["review"],
            }
        )
        return SideEffectPlan(
            approval_reason=(
                "Review the exact SQL and rows. Execution locks existing target rows "
                "and refuses if they or the table changed. Trigger and cascade "
                "effects are not previewed."
            ),
            approval_arguments=approval_arguments,
            recheck_after_approval=True,
            effect_intent=FrozenJsonObject.from_mapping(
                {
                    "source_id": source_id,
                    "resource_id": write.resource_id,
                    "intent_sha256": fingerprint["intent_sha256"],
                }
            ),
        )

    async def finalize_output(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        output: ToolOutput,
        *,
        request_sensitivity: ModelSensitivity,
        session: RunSession | None = None,
    ) -> ToolOutput:
        del request_sensitivity
        if capability.id == DATA_EXPORT_TABULAR_CAPABILITY_ID:
            self._validate_export_summary(output)
            assert output.artifact is not None
            output = replace(
                output,
                artifact=await self._bind_exact_export(
                    run,
                    arguments,
                    output.artifact,
                ),
            )
        return await self._classify(run, capability, arguments, output)

    finalize_session_output = finalize_output

    def _validate_export_summary(self, output: ToolOutput) -> None:
        draft = output.artifact
        if (
            draft is None
            or draft.provenance.authorship is not ArtifactAuthorship.EXACT_SOURCE_DATA
        ):
            raise ToolOutputValidationError(
                "relational export did not return one exact-source artifact"
            )
        valid = (
            bool(draft.provenance.resource_bindings)
            and isinstance(adapter_id := output.data.get("adapter_id"), str)
            and self._source_types.supports(adapter_id, SourceOperation.TABULAR_EXPORT)
            and output.data.get("format") in {"csv", "xlsx", "arrow"}
            and output.data.get("filename") == draft.suggested_filename
            and output.data.get("source_id")
            == draft.provenance.resource_bindings[0].source_id
            and output.data.get("row_count") == draft.provenance.row_count
            and output.data.get("column_count") == len(draft.provenance.columns)
        )
        if output.data.get("format") == "arrow":
            columns, types = arrow_schema(draft.content)
            valid = valid and (
                draft.media_type == ARROW_MEDIA_TYPE
                and columns == draft.provenance.columns
                and output.data.get("columns") == columns
                and output.data.get("column_types") == types
                and output.data.get("byte_size") == len(draft.content)
                and output.data.get("complete") is True
                and output.data.get("coverage") == "complete_query_result"
                and output.data.get("sql_fingerprint")
                == draft.provenance.sql_fingerprint
                and output.data.get("parameters_sha256")
                == draft.provenance.parameters_sha256
            )
        if not valid:
            raise ToolOutputValidationError(
                "relational export summary differs from its execution provenance"
            )

    async def _bind_exact_export(
        self,
        run: RunInput,
        arguments: Mapping[str, object],
        draft: ArtifactDraft,
    ) -> ArtifactDraft:
        source_id = arguments.get("source_id")
        parameters = arguments.get("parameters", ())
        adapter_id = arguments.get("_adapter_id")
        if (
            not isinstance(source_id, str)
            or not isinstance(parameters, tuple)
            or not isinstance(adapter_id, str)
            or not self._source_types.supports(
                adapter_id, SourceOperation.TABULAR_EXPORT
            )
        ):
            raise ToolOutputValidationError(
                "relational export execution arguments are unavailable"
            )
        (read,) = checked_reads(arguments)
        if await self._catalog.source_adapter_id(run.agent_id, source_id) != adapter_id:
            raise _incomplete_export(draft, "source_changed")
        try:
            access = await self._read_access_in_run(run, source_id)
        except CapabilityInputError as error:
            raise _incomplete_export(draft, "permission_state_invalid") from error
        if access is not None and not frozenset(read.resource_ids) <= access:
            raise _incomplete_export(draft, "permission_revoked")
        parameters_sha256 = (
            "sha256:" + sha256(canonical_json(parameters).encode("utf-8")).hexdigest()
        )
        provenance = draft.provenance
        if (
            {
                (item.source_id, item.resource_id)
                for item in provenance.resource_bindings
            }
            != {(source_id, resource_id) for resource_id in read.resource_ids}
            or provenance.sql_fingerprint != read.sql_fingerprint
            or provenance.parameters_sha256 != parameters_sha256
        ):
            raise ToolOutputValidationError(
                "exact artifact provenance differs from its execution"
            )
        return draft

    def normalize_error(
        self,
        call: ToolCall,
        error: BaseException,
    ) -> CapabilityFailure | None:
        del call
        from ...adapters.postgresql_write import (
            RelationalWriteCancelled,
            RelationalWriteError,
        )

        if isinstance(error, RelationalWriteError):
            return CapabilityFailure(
                error.error_code,
                str(error),
                error.details,
                effect_observation=error.effect_observation,
            )
        if isinstance(error, RelationalWriteCancelled):
            return CapabilityFailure(
                "write_interrupted",
                "The native write was interrupted; see its transaction evidence.",
                effect_observation=error.effect_observation,
            )
        if isinstance(error, LocalWorkspaceError):
            return CapabilityFailure(error.code, error.message, error.details)
        if isinstance(error, LocalFileQueryError):
            return CapabilityFailure(error.code, error.message, error.details)
        return None

    async def _prepare_source_file_call(
        self,
        run: RunInput,
        capability: Capability,
        arguments: FrozenJsonObject,
    ) -> FrozenJsonObject:
        """Bind one folder-file read to its source type before any file I/O.

        The live session source answers only its own foreground user; an
        attached folder answers only for the files this run may read.
        """

        source_id = cast(str, arguments["source_id"])
        adapter_id = await self._catalog.source_adapter_id(run.agent_id, source_id)
        live = source_id == self._catalog.session_source_id
        if adapter_id is None or not self._source_types.supports(
            adapter_id, _CAPABILITY_OPERATIONS[capability.id]
        ):
            raise CapabilityInputError(
                "source_operation_unsupported",
                "The selected source does not support this file operation.",
                {"source_id": source_id},
            )
        if live and (
            run.origin is not RunOrigin.USER or run.execution_scope is not None
        ):
            raise CapabilityInputError(
                "workspace_unavailable",
                "Local file access is unavailable for this run.",
            )
        prepared = arguments.to_dict()
        prepared["_adapter_id"] = adapter_id
        if capability.id == TEXT_READ_CAPABILITY_ID:
            has_path = "path" in arguments
            has_cursor = "cursor" in arguments
            if has_path == has_cursor:
                raise CapabilityInputError(
                    "path_invalid",
                    "File read requires exactly one path or cursor.",
                )
            if has_cursor and "position" in arguments:
                raise CapabilityInputError(
                    "cursor_invalid",
                    "A file cursor cannot be combined with a position.",
                )
        if capability.id == PATH_QUERY_CAPABILITY_ID:
            try:
                validated = validate_duckdb_read(cast(str, arguments["sql"]))
            except DuckDBReadValidationError as error:
                raise CapabilityInputError(
                    "folder_query_invalid",
                    error.message,
                    {"reason": error.reason},
                ) from error
            prepared["sql"] = validated.canonical_sql
            prepared["sql_fingerprint"] = validated.sql_fingerprint
        if not live:
            access = await self._read_access_in_run(run, source_id)
            if access is not None:
                prepared["_resource_ids"] = tuple(sorted(access))
        return FrozenJsonObject.from_mapping(prepared)

    async def _read_access_in_run(
        self, run: RunInput, source_id: str
    ) -> frozenset[str] | None:
        """What this run may read of one source; None means all of it."""

        try:
            current = await self._catalog.source_read_access(run.agent_id, (source_id,))
        except SourcePermissionStateError as error:
            raise CapabilityInputError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from error
        return narrowed_read_access(
            source_id, current.get(source_id, frozenset()), run_read_narrowings(run)
        )

    async def _validate_source_scope(
        self,
        run: RunInput,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> None:
        scope = await resolve_effective_source_scope(run, self._catalog)
        selected_source_ids = scope.source_ids
        supplied_source_id = arguments.get("source_id")
        if (
            supplied_source_id is not None
            and supplied_source_id not in selected_source_ids
        ):
            raise CapabilityInputError(
                "source_scope_violation",
                "The requested source is outside this run's effective scope.",
                {
                    "allowed_source_ids": tuple(sorted(selected_source_ids)),
                    "requested_source_id": supplied_source_id,
                },
                recovery=ToolRecovery(
                    RecoveryKind.REQUIRES_HUMAN_PERMISSION,
                    "Use one of allowed_sources; another source needs the "
                    "person to add it to this conversation's scope.",
                ),
            )
        resource_ids: tuple[object, ...] = ()
        if capability.id in _RESOURCE_LIST_ARGUMENT_CAPABILITIES:
            raw = arguments.get("resource_ids", ())
            resource_ids = raw if isinstance(raw, tuple) else ()
        for resource_id in resource_ids:
            if not isinstance(resource_id, str):
                continue
            identity = await self._catalog.resource_identity(
                run.agent_id,
                resource_id,
            )
            if identity is None or identity[0] not in selected_source_ids:
                raise CapabilityInputError(
                    "resource_read_not_allowed",
                    _RESOURCE_READ_DENIED_MESSAGE,
                    recovery=_COPY_CATALOG_IDS,
                )

    async def _validate_resource_read_scope(
        self,
        run: RunInput,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> None:
        resource_ids: tuple[object, ...] = ()
        if capability.id in _RESOURCE_LIST_ARGUMENT_CAPABILITIES:
            raw = arguments.get("resource_ids", ())
            resource_ids = raw if isinstance(raw, tuple) else ()
        requested = tuple(item for item in resource_ids if isinstance(item, str))
        if not requested:
            return
        source_id = arguments.get("source_id")
        source_ids = (source_id,) if isinstance(source_id, str) else ()
        try:
            readable = await self._catalog.readable_resource_ids(
                run.agent_id,
                source_ids,
            )
        except SourcePermissionStateError as error:
            raise CapabilityInputError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from error
        # These tools name one source; a source read in full lists no tables.
        for narrowing in run_read_narrowings(run):
            if source_id not in narrowing[0]:
                readable = readable & narrowing[1]
        if any(resource_id not in readable for resource_id in requested):
            raise CapabilityInputError(
                "resource_read_not_allowed",
                _RESOURCE_READ_DENIED_MESSAGE,
                recovery=_COPY_CATALOG_IDS,
            )

    async def _validate_sql(
        self,
        run: RunInput,
        arguments: Mapping[str, object],
        *,
        operation: SourceOperation,
    ) -> FrozenJsonObject:
        """Parse each statement once and hand the executor what passed policy."""

        source_id = arguments.get("source_id")
        statements = _sql_statements(arguments)
        if not isinstance(source_id, str) or statements is None:
            raise CapabilityInputError(
                "sql_invalid_input",
                "SQL reads require a source and either non-empty sql with an "
                "array of parameters or a statements array of {sql, parameters}, "
                "not both.",
                recovery=_CORRECT_SQL,
            )
        adapter_id = await self._catalog.source_adapter_id(
            run.agent_id,
            source_id,
        )
        if adapter_id is None:
            raise CapabilityInputError(
                "sql_source_not_available",
                "The selected source is not active for this agent.",
                {"source_id": source_id},
            )
        source_type = self._source_types.get(adapter_id)
        dialect = None if source_type is None else source_type.sql_dialect
        if dialect is None or not self._source_types.supports(adapter_id, operation):
            raise CapabilityInputError(
                "sql_source_adapter_unsupported",
                "The selected source does not support relational SQL reads.",
                {"source_id": source_id},
            )
        if dialect == "duckdb":
            raise CapabilityInputError(
                "path_pattern_required",
                "A folder source is queried with path_pattern and sql over the "
                "relation data.",
                {"source_id": source_id},
                recovery=_CORRECT_SQL,
            )
        access = await self._read_access_in_run(run, source_id)
        schemas = await self._catalog.source_schemas(run.agent_id, source_id)
        reads: list[CheckedRead] = []
        for index, (sql, parameters) in enumerate(statements):
            result = validate_read(
                sql,
                dialect=cast(_SqlDialect, dialect),
                source_id=source_id,
                parameters=parameters,
                allowed_resource_ids=access,
                schemas=schemas,
            )
            if not result.valid or result.analysis is None:
                # A scope denial only when scope is all that is wrong: SELECT INTO's
                # target is out of scope too, but its fault is the write.
                if result.issues and all(
                    item.code in _SCOPE_ISSUE_CODES for item in result.issues
                ):
                    raise await self._read_denied(run, source_id, result.issues)
                details: dict[str, object] = {
                    "issues": [
                        {
                            "code": item.code,
                            "message": item.message,
                            "details": item.details,
                        }
                        for item in result.issues
                    ],
                    "source_id": source_id,
                }
                if "statements" in arguments:
                    details["statement_index"] = index
                raise CapabilityInputError(
                    "sql_validation_failed",
                    "The SQL read is invalid. Correct all reported issues before "
                    "retrying.",
                    details,
                    recovery=_CORRECT_SQL,
                )
            if operation is SourceOperation.TABULAR_EXPORT and not result.resource_ids:
                raise CapabilityInputError(
                    "export_requires_table",
                    "An export must read at least one table.",
                    recovery=_CORRECT_SQL,
                )
            reads.append(
                CheckedRead(
                    canonical_sql=result.analysis.canonical_sql,
                    sql_fingerprint=result.analysis.sql_fingerprint,
                    parameters=parameters,
                    function_lookups=result.function_lookups,
                    attribute_lookups=result.attribute_lookups,
                    written_calls=result.written_calls,
                    unqualified_relations=result.unqualified_relations,
                    resource_ids=result.resource_ids,
                    resource_names=result.resource_names,
                    allowed_resource_ids=(
                        None if access is None else tuple(sorted(access))
                    ),
                )
            )
        prepared = dict(arguments)
        prepared["_adapter_id"] = adapter_id
        prepared["_reads"] = tuple(read.to_mapping() for read in reads)
        # The admission policy reads the touched tables here.
        prepared["resource_ids"] = tuple(
            sorted({item for read in reads for item in read.resource_ids})
        )
        return FrozenJsonObject.from_mapping(prepared)

    async def _prepare_write(
        self,
        run: RunInput,
        arguments: FrozenJsonObject,
        request_sensitivity: ModelSensitivity,
    ) -> FrozenJsonObject:
        """Parse the one statement once and hand the executor what passed policy."""

        source_id = cast(str, arguments["source_id"])
        adapter_id = await self._catalog.source_adapter_id(run.agent_id, source_id)
        if adapter_id is None or adapter_id not in self._write_backends:
            raise CapabilityInputError(
                "write_source_not_available",
                "data_write works on PostgreSQL sources; this source cannot be "
                "written.",
                {"source_id": source_id},
            )
        target = await self._catalog.admitted_model_sensitivity(
            run.agent_id, (source_id,)
        )
        if target is None or request_sensitivity.routing_rank > target.routing_rank:
            raise CapabilityInputError(
                "write_sensitivity_denied",
                "The request exceeds the source's admitted classification.",
            )
        access = await self._read_access_in_run(run, source_id)
        try:
            scopes = await self._catalog.relational_write_scopes(
                run.agent_id, source_id
            )
        except SourcePermissionStateError as error:
            raise CapabilityInputError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from error
        checked = self._write_backends[adapter_id].describe(
            cast(str, arguments["sql"]),
            source_id=source_id,
            parameters=cast(tuple[object, ...], arguments.get("parameters", ())),
            allowed_resource_ids=access,
            schemas=await self._catalog.source_schemas(run.agent_id, source_id),
        )
        if not isinstance(checked, tuple):
            issue = write_permission_issue(scopes.get(checked.resource_id), checked)
            checked = checked if issue is None else (issue,)
        if isinstance(checked, tuple):
            if all(item.code in _SCOPE_ISSUE_CODES for item in checked):
                raise await self._read_denied(run, source_id, checked)
            first = checked[0]
            raise CapabilityInputError(
                first.code,
                first.message,
                {
                    "issues": [
                        {
                            "code": item.code,
                            "message": item.message,
                            "details": item.details,
                        }
                        for item in checked
                    ],
                    "source_id": source_id,
                },
                recovery=(
                    _WRITE_PERMISSION_REQUIRED
                    if first.code in _PERMISSION_ISSUE_CODES
                    else _CORRECT_SQL
                ),
            )
        prepared: dict[str, object] = dict(arguments)
        prepared["_adapter_id"] = adapter_id
        prepared["_write"] = checked.to_mapping()
        # The admission policy reads the touched tables here.
        prepared["resource_ids"] = checked.resource_ids
        return FrozenJsonObject.from_mapping(prepared)

    async def _read_denied(
        self,
        run: RunInput,
        source_id: str,
        issues: tuple[SqlValidationIssue, ...],
    ) -> CapabilityInputError:
        """Name what is outside this run's scope; the catalog may offer close names."""

        relations = [cast(str, item.details["relation"]) for item in issues]
        try:
            resources = await self._catalog.resource_schemas(run.agent_id, source_id)
            readable = await self._catalog.readable_resource_ids(
                run.agent_id, (source_id,)
            )
        except (CatalogStoreError, SourcePermissionStateError):
            resources, readable = (), frozenset()
        names = [
            next((alias for alias in item.aliases if "." in alias), item.name)
            for item in resources
            if item.resource_id in readable
        ]
        return CapabilityInputError(
            "resource_read_not_allowed",
            " ".join(item.message for item in issues),
            {
                "tables": tuple(relations),
                "did_you_mean": tuple(
                    dict.fromkeys(
                        match
                        for relation in relations
                        for match in get_close_matches(
                            relation, names, n=4, cutoff=0.45
                        )
                    )
                ),
            },
            recovery=_COPY_CATALOG_IDS,
        )

    async def _prepare_combined_query(
        self, run: RunInput, arguments: Mapping[str, object]
    ) -> FrozenJsonObject:
        """Validate every input as one single-source read, then the combining SQL.

        Each input takes the data_query scope and policy path; declared catalog
        links grant no read authority here.
        """

        single = self._capabilities[DATA_QUERY_CAPABILITY_ID]
        inputs = cast(tuple[Mapping[str, object], ...], arguments["inputs"])
        aliases = tuple(cast(str, item["alias"]) for item in inputs)
        if len(set(aliases)) != len(aliases):
            raise CapabilityInputError(
                "combined_query_invalid",
                "Every input needs a distinct alias.",
                {"aliases": aliases},
                recovery=_CORRECT_SQL,
            )
        prepared: list[dict[str, object]] = []
        for alias, item in zip(aliases, inputs, strict=True):
            tabular = "sql" not in item
            # Both forms take the same scope checks; a tabular input names
            # its one resource as resource_id.
            read = FrozenJsonObject.from_mapping(
                {
                    "source_id": item["source_id"],
                    **(
                        {"resource_ids": (item["resource_id"],)}
                        if tabular
                        else {"sql": item["sql"]}
                    ),
                }
            )
            try:
                await self._validate_source_scope(run, single, read)
                self._validate_execution_resource_scope(run, single, read)
                await self._validate_resource_read_scope(run, single, read)
                if tabular:
                    validated = await self._validate_tabular_read(run, item)
                elif "path_pattern" in item:
                    validated = await self._validate_folder_input(run, item)
                else:
                    validated = await self._validate_sql(
                        run, read, operation=SourceOperation.SQL_READ
                    )
            except CapabilityInputError as error:
                raise CapabilityInputError(
                    error.code,
                    str(error),
                    {**error.details.to_dict(), "input_alias": alias},
                ) from error
            prepared.append({**validated.to_dict(), "alias": alias})
        try:
            combining = validate_duckdb_read(cast(str, arguments["sql"]), aliases)
        except DuckDBReadValidationError as error:
            raise CapabilityInputError(
                "combined_query_invalid",
                error.message,
                {"reason": error.reason, "aliases": aliases},
                recovery=_CORRECT_SQL,
            ) from error
        return FrozenJsonObject.from_mapping(
            {
                "inputs": tuple(prepared),
                "sql": combining.canonical_sql,
                "_sql_fingerprint": combining.sql_fingerprint,
            }
        )

    async def _validate_folder_input(
        self, run: RunInput, item: Mapping[str, object]
    ) -> FrozenJsonObject:
        """Check one attached folder's input; its files bind when the read runs."""

        source_id = cast(str, item["source_id"])
        adapter_id = await self._catalog.source_adapter_id(run.agent_id, source_id)
        if (
            adapter_id is None
            or source_id == self._catalog.session_source_id
            or not self._source_types.supports(adapter_id, SourceOperation.PATH_QUERY)
        ):
            raise CapabilityInputError(
                "combined_query_invalid",
                "path_pattern applies to an attached folder source.",
                {"source_id": source_id},
                recovery=_CORRECT_SQL,
            )
        try:
            validated = validate_duckdb_read(cast(str, item["sql"]))
        except DuckDBReadValidationError as error:
            raise CapabilityInputError(
                "folder_query_invalid",
                error.message,
                {"reason": error.reason},
                recovery=_CORRECT_SQL,
            ) from error
        access = await self._read_access_in_run(run, source_id)
        read = CheckedRead(
            canonical_sql=validated.canonical_sql,
            sql_fingerprint=validated.sql_fingerprint,
            path_pattern=cast(str, item["path_pattern"]),
            allowed_resource_ids=None if access is None else tuple(sorted(access)),
        )
        return FrozenJsonObject.from_mapping(
            {
                "source_id": source_id,
                "sql": item["sql"],
                "_adapter_id": adapter_id,
                "_reads": (read.to_mapping(),),
            }
        )

    async def _validate_tabular_read(
        self, run: RunInput, item: Mapping[str, object]
    ) -> FrozenJsonObject:
        """Check one tabular input against the current catalog before any I/O."""

        # capabilities imports this module, so its records load at call time.
        from .capabilities import TabularFilter

        source_id = cast(str, item["source_id"])
        resource_id = cast(str, item["resource_id"])
        adapter_id = await self._catalog.source_adapter_id(run.agent_id, source_id)
        if adapter_id is None:
            raise CapabilityInputError(
                "sql_source_not_available",
                "The selected source is not active for this agent.",
                {"source_id": source_id},
            )
        if not self._source_types.supports(adapter_id, SourceOperation.TABULAR_READ):
            raise CapabilityInputError(
                "tabular_source_adapter_unsupported",
                "The selected source does not support tabular reads; send this "
                "input as sql instead.",
                {"source_id": source_id},
                recovery=_CORRECT_SQL,
            )
        resource = next(
            (
                schema
                for schema in await self._catalog.resource_schemas(
                    run.agent_id, source_id
                )
                if schema.resource_id == resource_id
            ),
            None,
        )
        if resource is None:
            raise CapabilityInputError(
                "sql_resource_source_mismatch",
                "The resource must belong to the selected source.",
                {"resource_id": resource_id, "source_id": source_id},
                recovery=_CORRECT_SQL,
            )
        if resource.revision is None or resource.source_revision is None:
            raise CapabilityInputError(
                "catalog_provenance_missing",
                "The resource has no current catalog revision; refresh the source.",
                {"resource_id": resource_id},
            )
        try:
            filters = tuple(
                TabularFilter.from_mapping(cast(Mapping[str, object], value))
                for value in cast(tuple[object, ...], item.get("filters", ()))
            )
        except ValueError as error:
            raise CapabilityInputError(
                "tabular_filter_invalid",
                str(error),
                {"resource_id": resource_id},
                recovery=_CORRECT_SQL,
            ) from error
        columns = cast(tuple[str, ...], item["columns"])
        unknown = sorted(
            {*columns, *(value.column for value in filters)} - set(resource.columns)
        )
        if unknown:
            raise CapabilityInputError(
                "tabular_column_unknown",
                "Every column and filter column must exist in the resource's "
                "current catalog schema.",
                {
                    "resource_id": resource_id,
                    "unknown_columns": tuple(unknown),
                    "available_columns": resource.columns,
                },
                recovery=_CORRECT_SQL,
            )
        return FrozenJsonObject.from_mapping(
            {
                **item,
                "_adapter_id": adapter_id,
                "_resource_name": resource.name,
            }
        )

    def _validate_execution_resource_scope(
        self,
        run: RunInput,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> None:
        scope = run.execution_scope
        if scope is None:
            return
        unrestricted = frozenset(scope.unrestricted_source_ids)
        source_id = arguments.get("source_id")
        if capability.id == CATALOG_SEARCH_CAPABILITY_ID:
            # Discovery is fine where the run may read everything anyway.
            touched = (
                {source_id}
                if isinstance(source_id, str)
                else set(scope.allowed_source_ids)
            )
            if not touched <= unrestricted:
                raise CapabilityInputError(
                    "execution_scope_resource_violation",
                    "This autonomous run can search only sources it reads in full.",
                )
            return
        resource_ids: tuple[object, ...] = ()
        if capability.id in _RESOURCE_LIST_ARGUMENT_CAPABILITIES:
            raw = arguments.get("resource_ids", ())
            resource_ids = raw if isinstance(raw, tuple) else ()
        if source_id in unrestricted:
            return
        allowed = frozenset(scope.allowed_resource_ids)
        if any(
            not isinstance(resource_id, str) or resource_id not in allowed
            for resource_id in resource_ids
        ):
            raise CapabilityInputError(
                "execution_scope_resource_violation",
                "The requested resource is outside this run's immutable scope.",
            )

    async def _classify(
        self,
        run: RunInput,
        capability: Capability,
        arguments: FrozenJsonObject,
        output: ToolOutput,
    ) -> ToolOutput:
        if capability.id in SOURCE_FILE_CAPABILITY_IDS:
            if output.sensitivity is None or not output.sensitivity_provenance:
                raise CapabilityInputError(
                    "result_classification_unavailable",
                    "The local file result cannot be classified safely.",
                )
            return output
        if output.sensitivity is not None:
            return output
        scope = await resolve_effective_source_scope(run, self._catalog)
        run = replace(run, resolved_source_scope=scope)
        # Resolved arguments carry full IDs; the call keeps what the model sent.
        source_id = arguments.get("source_id")
        inputs = arguments.get("inputs")
        source_ids = (
            tuple(sorted({source_id, *_endpoint_source_ids(output.data)}))
            if isinstance(source_id, str)
            else (
                tuple(
                    sorted(
                        {
                            cast(str, cast(Mapping[str, object], item)["source_id"])
                            for item in inputs
                        }
                    )
                )
                if isinstance(inputs, tuple)
                else tuple(sorted(scope.source_ids))
            )
        )
        sensitivity = await self._catalog.admitted_model_sensitivity(
            run.agent_id,
            source_ids,
        )
        if sensitivity is None:
            raise CapabilityInputError(
                "result_classification_unavailable",
                "The current admitted result scope cannot be classified safely.",
                {"capability_id": capability.id},
            )
        # A read reports what it touched. Catalog tools list what they show by name,
        # so their classification rests on the sources alone.
        reported = output.data.get("resource_ids")
        readable = (
            frozenset(cast(tuple[str, ...], reported))
            if capability.id
            in _RELATIONAL_READ_CAPABILITIES | {DATA_QUERY_COMBINED_CAPABILITY_ID}
            and isinstance(reported, tuple)
            else frozenset()
        )
        return replace(
            output,
            sensitivity=sensitivity,
            sensitivity_provenance={
                "authority": (
                    "artifact_domain_current_scope"
                    if capability.id == DATA_EXPORT_TABULAR_CAPABILITY_ID
                    else "current_admitted_resource_scope"
                ),
                "capability_id": capability.id,
                "source_ids": source_ids,
                "resource_ids": tuple(sorted(readable)),
            },
        )


def _endpoint_source_ids(data: Mapping[str, object]) -> set[str]:
    """Other sources a description names through its declared links."""

    relationships = data.get("relationships")
    return {
        cast(str, relationship[key])
        for relationship in cast(
            tuple[Mapping[str, object], ...],
            relationships if isinstance(relationships, (tuple, list)) else (),
        )
        for key in ("from_source_id", "to_source_id")
        if key in relationship
    }


def _incomplete_export(draft: ArtifactDraft, reason: str) -> ArtifactError:
    return ArtifactError(
        "artifact_incomplete_export",
        "The source or this run's permission changed during the export.",
        {
            "reason": reason,
            "completed_rows": draft.provenance.row_count or 0,
            "completed_columns": len(draft.provenance.columns),
            "completed_bytes": len(draft.content),
        },
    )


__all__ = [
    "CatalogSchemaReader",
    "DATA_DOMAIN_OWNER_ID",
    "DataCapabilityDomain",
    "DataDomainCatalog",
    "DATA_QUERY_CAPABILITY_ID",
    "DATA_QUERY_COMBINED_CAPABILITY_ID",
    "DATA_QUERY_COMBINED_EVIDENCE_KIND",
    "DATA_QUERY_EVIDENCE_KIND",
    "DATA_EXPORT_TABULAR_CAPABILITY_ID",
    "RELATIONAL_WRITE_CAPABILITY_ID",
    "RELATIONAL_WRITE_EVIDENCE_KIND",
    "RELATIONAL_WRITE_PREVIEW_CAPABILITY_ID",
    "RELATIONAL_WRITE_PREVIEW_EVIDENCE_KIND",
    "WriteDescriber",
    "RelationalWriteScopeReader",
    "ReadScopedCatalogReader",
]
