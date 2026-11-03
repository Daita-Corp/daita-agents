"""Project data tools and revalidate their catalog and permission bindings."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
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
    CallProvenance,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    CapabilityRegistry,
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
from ...catalog.models import ResourceKind, catalog_resource_id
from ...catalog.protocols import CatalogStoreError
from ...llm.models import ModelSensitivity, ToolCall, ToolResultBlock
from ...loop.models import RunInput, RunOrigin, Transcript
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
    RelationalUpdateIntent,
    ResourceSchema,
    SqlValidationIssue,
    checked_reads,
    validate_duckdb_read,
    validate_read,
    validate_relational_update_intent,
)
from .sql.contracts import _SqlDialect
from .sql.relational_upsert import (
    RelationalUpsertIntent,
    validate_relational_upsert_intent,
)

DATA_DOMAIN_OWNER_ID = "data"
DATA_QUERY_CAPABILITY_ID = "data.query"
DATA_QUERY_EVIDENCE_KIND = "data.query_result"
DATA_QUERY_COMBINED_CAPABILITY_ID = "data.query_combined"
DATA_QUERY_COMBINED_EVIDENCE_KIND = "data.query_combined_result"
DATA_EXPORT_TABULAR_CAPABILITY_ID = "data.export_tabular"
RELATIONAL_UPDATE_PREVIEW_CAPABILITY_ID = "data.preview_update_rows"
RELATIONAL_UPDATE_PREVIEW_EVIDENCE_KIND = "data.preview_update_rows"
RELATIONAL_UPDATE_CAPABILITY_ID = "data.update_rows"
RELATIONAL_UPDATE_EVIDENCE_KIND = "data.update_rows_result"
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
_UPDATE_CAPABILITIES = frozenset(
    {RELATIONAL_UPDATE_PREVIEW_CAPABILITY_ID, RELATIONAL_UPDATE_CAPABILITY_ID}
)
_UPSERT_CAPABILITIES = frozenset({"data.preview_upsert_rows", "data.upsert_rows"})
_NATIVE_WRITE_CAPABILITIES = frozenset(
    {RELATIONAL_UPDATE_CAPABILITY_ID, "data.upsert_rows"}
)


_RESOURCE_ARGUMENT_CAPABILITIES = frozenset(
    {
        *_UPSERT_CAPABILITIES,
        RELATIONAL_UPDATE_PREVIEW_CAPABILITY_ID,
        RELATIONAL_UPDATE_CAPABILITY_ID,
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
    **{item: SourceOperation.RELATIONAL_UPDATE for item in _UPDATE_CAPABILITIES},
    **{item: SourceOperation.RELATIONAL_UPSERT for item in _UPSERT_CAPABILITIES},
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


class RelationalWriteCatalogReader(CatalogSchemaReader, Protocol):
    async def relational_write_scope_issue(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
        assignment_columns: tuple[str, ...],
        *,
        operation: str = "update",
        insert_columns: tuple[str, ...] = (),
        key_columns: tuple[str, ...] = (),
        row_count: int | None = None,
    ) -> tuple[str, str] | None: ...

    async def load_relational_write_scope(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
    ) -> RelationalWriteScope | None: ...


class DataDomainCatalog(
    ReadScopedCatalogReader, RelationalWriteCatalogReader, Protocol
):
    async def source_routing_facts(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> tuple[Mapping[str, object], ...]: ...

    async def relational_write_applicable_source_ids(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
        *,
        operation: str = "update",
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


def _reads_resource(
    data: Mapping[str, object], source_id: object, resource_id: object
) -> bool:
    """Whether one read result covers the exact source resource.

    Query results name their source and the tables they read; a description names
    the one table it read, and older ones list resources with their own source.
    """

    touched = data.get("resource_ids", ())
    if data.get("source_id") == source_id and (
        data.get("resource_id") == resource_id
        or (isinstance(touched, tuple) and resource_id in touched)
    ):
        return True
    name, kind = data.get("name"), data.get("kind")
    if (
        isinstance(source_id, str)
        and data.get("source_id") == source_id
        and isinstance(name, str)
        and kind in {"table", "view"}
        and catalog_resource_id(source_id, ResourceKind(str(kind)), name) == resource_id
    ):
        return True
    resources = data.get("resources", ())
    return isinstance(resources, tuple) and any(
        isinstance(item, Mapping)
        and item.get("source_id") == source_id
        and item.get("resource_id") == resource_id
        for item in resources
    )


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
        transcript_loader: Callable[[str], Awaitable[Transcript]] | None = None,
        source_files_available: bool = False,
        path_query_available: bool = False,
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
        self._transcript_loader = transcript_loader
        self._registry: CapabilityRegistry | None = None
        self._declarations = declarations
        self._catalog = catalog
        self._source_types = source_types
        self._views = {item.name: item for item in declarations.tool_views}
        self._capabilities = {item.id: item for item in declarations.capabilities}

    def bind_capability_registry(self, registry: CapabilityRegistry) -> None:
        registry.validate_declarations(self._declarations)
        self._registry = registry

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
        update_source_ids: frozenset[str]
        if not scope.source_ids:
            facts = ()
            update_source_ids = frozenset()
        else:
            facts = await self._catalog.source_routing_facts(
                run.agent_id,
                tuple(sorted(scope.source_ids)),
            )
            update_source_ids = (
                await self._catalog.relational_write_applicable_source_ids(
                    run.agent_id,
                    tuple(sorted(scope.source_ids)),
                )
                if source_capability_ids & _UPDATE_CAPABILITIES
                else frozenset()
            )
        upsert_source_ids = (
            await self._catalog.relational_write_applicable_source_ids(
                run.agent_id, tuple(sorted(scope.source_ids)), operation="upsert"
            )
            if scope.source_ids and source_capability_ids & _UPSERT_CAPABILITIES
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
            if capability.id in _UPDATE_CAPABILITIES and not update_source_ids:
                continue
            if capability.id in _UPSERT_CAPABILITIES and not upsert_source_ids:
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
            capability.id not in _NATIVE_WRITE_CAPABILITIES
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
        if capability.id in _UPDATE_CAPABILITIES | _UPSERT_CAPABILITIES:
            await self._validate_target_sensitivity(
                run.agent_id, arguments, request_sensitivity
            )
        if capability.id in _UPSERT_CAPABILITIES:
            try:
                intent = RelationalUpsertIntent.from_mapping(arguments)
            except ValueError:
                raise CapabilityInputError(
                    "upsert_invalid_batch",
                    "Upsert requires uniform scalar rows with explicit columns, "
                    "at most 1,000 rows, and a batch within 64 KiB and the exact "
                    "approval display bound. Reduce the batch or correct its shape; "
                    "no write was dispatched.",
                ) from None
            permission = await self._catalog.load_relational_write_scope(
                run.agent_id, intent.source_id, intent.resource_id
            )
            if permission is None:
                raise CapabilityInputError(
                    "resource_write_not_allowed",
                    "The exact write permission is unavailable or stale.",
                    recovery=_WRITE_PERMISSION_REQUIRED,
                )
            issue = await self._catalog.relational_write_scope_issue(
                run.agent_id,
                intent.source_id,
                intent.resource_id,
                intent.update_columns,
                operation="upsert",
                insert_columns=intent.insert_columns,
                key_columns=intent.key_columns,
                row_count=len(intent.rows),
            )
            if issue is not None:
                raise CapabilityInputError(*issue)
            resource = next(
                item
                for item in await self._catalog.resource_schemas(
                    run.agent_id, intent.source_id
                )
                if item.resource_id == intent.resource_id
            )
            validate_relational_upsert_intent(
                intent,
                resource=resource,
                generated_identity_columns=permission.generated_identity_columns,
                max_rows=permission.max_rows,
            )
            await self._authenticate_upsert_evidence(
                run, call, capability, intent.evidence_call_ids
            )
        if capability.id in _UPDATE_CAPABILITIES:
            await self._validate_relational_update_call(run, arguments)
            if capability.id == RELATIONAL_UPDATE_CAPABILITY_ID:
                permission = await self._catalog.load_relational_write_scope(
                    run.agent_id,
                    cast(str, arguments["source_id"]),
                    cast(str, arguments["resource_id"]),
                )
                if permission is None:
                    raise CapabilityInputError(
                        "resource_write_not_allowed",
                        "The current write permission is unavailable.",
                        recovery=_WRITE_PERMISSION_REQUIRED,
                    )
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
            capability.id not in _NATIVE_WRITE_CAPABILITIES
            or capability.operational_effect is not OperationalEffect.MUTATE_DATA
        ):
            raise ValueError("data domain received an unsupported side effect")
        source_id = cast(str, execution.arguments["source_id"])
        resource_id = cast(str, execution.arguments["resource_id"])
        resource = next(
            (
                item
                for item in await self._catalog.resource_schemas(
                    run.agent_id, source_id
                )
                if item.resource_id == resource_id
            ),
            None,
        )
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
        if (
            resource is None
            or source is None
            or resource.revision != fingerprint["resource_revision"]
        ):
            raise CapabilityInputError(
                "write_state_changed",
                "The exact review target is no longer current.",
            )
        approval_arguments = FrozenJsonObject.from_mapping(
            {
                "arguments": execution.arguments,
                "target": {
                    "source_id": source_id,
                    "resource_id": resource_id,
                    "name": resource.name,
                    "aliases": resource.aliases,
                    "source_name": source["display_name"],
                    "revision": resource.revision,
                },
                "preview": fingerprint["review"],
            }
        )
        approval_reason = (
            "Review the exact batch. Execution briefly holds an EXCLUSIVE table lock; ordinary reads may continue."
            if capability.id == "data.upsert_rows"
            else "Review the exact selected rows and assignments. Execution locks matching rows and rejects a changed preview."
        )
        if (
            capability.id == "data.upsert_rows"
            and cast(Mapping[str, object], fingerprint["review"])[
                "identity_sequence_gaps_possible"
            ]
        ):
            approval_reason += " Identity sequence gaps can remain after rollback."
        return SideEffectPlan(
            approval_reason=approval_reason,
            approval_arguments=approval_arguments,
            recheck_after_approval=True,
            effect_intent=FrozenJsonObject.from_mapping(
                {
                    "source_id": execution.arguments["source_id"],
                    "resource_id": execution.arguments["resource_id"],
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
            RelationalUpdateExecutionCancelled,
            RelationalUpdateExecutionError,
        )

        if isinstance(error, RelationalUpdateExecutionError):
            return CapabilityFailure(
                error.error_code,
                str(error),
                error.details,
                effect_observation=error.effect_observation,
            )
        if isinstance(error, RelationalUpdateExecutionCancelled):
            return CapabilityFailure(
                "write_interrupted",
                "The native update was interrupted; see its transaction evidence.",
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
        if capability.id in _RESOURCE_ARGUMENT_CAPABILITIES:
            resource_ids = (arguments.get("resource_id"),)
        elif capability.id in _RESOURCE_LIST_ARGUMENT_CAPABILITIES:
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
        if capability.id in _RESOURCE_ARGUMENT_CAPABILITIES:
            resource_ids = (arguments.get("resource_id"),)
        elif capability.id in _RESOURCE_LIST_ARGUMENT_CAPABILITIES:
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
        if capability.id in _RESOURCE_ARGUMENT_CAPABILITIES:
            resource_ids = (arguments.get("resource_id"),)
        elif capability.id in _RESOURCE_LIST_ARGUMENT_CAPABILITIES:
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

    async def _validate_relational_update_call(
        self,
        run: RunInput,
        arguments: Mapping[str, object],
    ) -> None:
        source_id = arguments.get("source_id")
        if not isinstance(source_id, str):
            raise CapabilityInputError(
                "write_source_not_available",
                "PostgreSQL update requires an exact current source.",
            )
        adapter_id = await self._catalog.source_adapter_id(run.agent_id, source_id)
        if adapter_id is None or not self._source_types.supports(
            adapter_id, SourceOperation.RELATIONAL_UPDATE
        ):
            raise CapabilityInputError(
                "write_source_not_available",
                "The selected source is not an active PostgreSQL source owned by "
                "this agent.",
                {"source_id": source_id},
            )
        try:
            intent = RelationalUpdateIntent.from_mapping(arguments)
        except (TypeError, ValueError) as error:
            raise CapabilityInputError(
                "write_assignment_invalid",
                "The PostgreSQL update intent is malformed.",
            ) from error
        try:
            issue = await self._catalog.relational_write_scope_issue(
                run.agent_id,
                source_id,
                intent.resource_id,
                tuple(item.column for item in intent.assignments),
            )
        except SourcePermissionStateError as error:
            raise CapabilityInputError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from error
        if issue is not None:
            raise CapabilityInputError(issue[0], issue[1])
        validation = validate_relational_update_intent(
            intent,
            resources=await self._catalog.resource_schemas(
                run.agent_id,
                source_id,
            ),
        )
        if not validation.valid:
            first = validation.issues[0]
            raise CapabilityInputError(
                first.code,
                first.message,
                {"source_id": source_id, "resource_id": intent.resource_id},
            )

    async def _validate_target_sensitivity(
        self,
        agent_id: str,
        arguments: Mapping[str, object],
        sensitivity: ModelSensitivity,
    ) -> None:
        resource = next(
            (
                item
                for item in await self._catalog.resource_schemas(
                    agent_id, cast(str, arguments["source_id"])
                )
                if item.resource_id == arguments["resource_id"]
            ),
            None,
        )
        try:
            target = ModelSensitivity(
                "unknown" if resource is None else resource.sensitivity_class
            )
        except ValueError:
            raise CapabilityInputError(
                "write_sensitivity_denied", "Target classification is unknown."
            ) from None
        if sensitivity.routing_rank > target.routing_rank:
            raise CapabilityInputError(
                "write_sensitivity_denied",
                "The request exceeds the target's current admitted classification.",
            )

    async def _authenticate_upsert_evidence(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        evidence_ids: tuple[str, ...],
    ) -> None:
        if not evidence_ids:
            return
        _, results = await self._authenticated_prior_results(run, call, capability)
        if (
            tuple(block.call_id for block in results if block.call_id in evidence_ids)
            != evidence_ids
        ):
            known = {block.call_id for block in results}
            unknown = [item for item in evidence_ids if item not in known]
            raise CapabilityInputError(
                "write_evidence_invalid",
                "Evidence must identify ordered successful authenticated current-run results"
                + (
                    f"; not a successful call id in this run: {', '.join(unknown[:4])}."
                    if unknown
                    else "; the call ids are out of order."
                ),
            )

    async def derive_call_provenance(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        *,
        session: RunSession | None = None,
    ) -> CallProvenance:
        """Derive upsert evidence: earlier authenticated reads of the same table."""

        del session
        registry = self._registry
        if capability.id not in _UPSERT_CAPABILITIES or registry is None:
            return CallProvenance()
        try:
            _, results = await self._authenticated_prior_results(run, call, capability)
        except CapabilityInputError:
            return CallProvenance()
        source_id, resource_id = arguments["source_id"], arguments["resource_id"]
        evidence: list[str] = []
        for block in results:
            data = block.output.get("data")
            if (
                block.capability_id is None
                or block.capability_id in _UPSERT_CAPABILITIES | _UPDATE_CAPABILITIES
                or registry.capability(block.capability_id).access_mode
                is not AccessMode.READ
                or not isinstance(data, Mapping)
            ):
                continue
            if _reads_resource(data, source_id, resource_id):
                evidence.append(block.call_id)
        return CallProvenance(tuple(evidence[-32:]))

    async def _authenticated_prior_results(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
    ) -> tuple[dict[str, ToolCall], list[ToolResultBlock]]:
        """Return this call's earlier successful results with verified lineage."""

        if self._transcript_loader is None or self._registry is None:
            raise CapabilityInputError(
                "write_preview_not_authenticated",
                "Current-run native lineage is unavailable.",
            )
        transcript = await self._transcript_loader(run.id)
        if transcript.run.agent_id != run.agent_id or transcript.run.id != run.id:
            raise CapabilityInputError(
                "write_preview_not_authenticated",
                "The transcript belongs to another run or agent.",
            )
        registry = self._registry

        def target(candidate: ToolCall) -> ToolCall:
            # A mode-routed tool persists its model call with the mode selector;
            # its capability received the same call without it.
            return replace(
                candidate,
                arguments=registry.target_arguments(
                    candidate.name, candidate.arguments
                ),
            )

        # Calls are compared by the arguments they executed with, not the
        # run-scoped refs and handles the model sent.
        try:
            executed = {
                candidate.id: target(candidate)
                for candidate, _ in registry.executed_tool_pairs(
                    transcript, source_scope=run.resolved_source_scope
                )
            }
        except ValueError as error:
            raise CapabilityInputError(
                "write_preview_not_authenticated",
                "Current-run native lineage is malformed.",
            ) from error
        positions = [
            index
            for index, message in enumerate(transcript.messages)
            for candidate in message.tool_calls
            if candidate.id == call.id
        ]
        # The runtime's routed call, resolved by the same resolver.
        executing = (
            None
            if len(positions) != 1
            else registry.executed_call(
                call,
                messages=transcript.messages[: positions[0]],
                source_scope=run.resolved_source_scope,
                capability=capability,
            )
        )
        if executing is None or executed.get(call.id) != executing:
            raise CapabilityInputError(
                "write_preview_not_authenticated",
                "The native call does not match its persisted current-run identity.",
            )
        current_index = positions[0]
        calls = {
            candidate.id: executed[candidate.id]
            for message in transcript.messages[:current_index]
            for candidate in message.tool_calls
        }
        results: list[ToolResultBlock] = []
        for result_index, message in enumerate(transcript.messages[:current_index]):
            for block in message.content:
                if (
                    not isinstance(block, ToolResultBlock)
                    or block.is_error
                    or block.call_id not in calls
                    or block.capability_id is None
                ):
                    continue
                producers = [
                    index
                    for index, item in enumerate(transcript.messages)
                    for candidate in item.tool_calls
                    if candidate.id == block.call_id
                ]
                result_count = sum(
                    1
                    for item in transcript.messages
                    for other in item.content
                    if isinstance(other, ToolResultBlock)
                    and other.call_id == block.call_id
                )
                if (
                    len(producers) != 1
                    or result_count != 1
                    or not producers[0] < result_index < current_index
                ):
                    continue
                if (
                    block.output_sha256
                    != "sha256:"
                    + sha256(canonical_json(block.output).encode("utf-8")).hexdigest()
                    or block.sensitivity is None
                ):
                    continue
                try:
                    _, result_capability = self._registry.resolve_tool(
                        calls[block.call_id].name,
                        capability_id=block.capability_id,
                    )
                    if (
                        result_capability.id != block.capability_id
                        or result_capability.executor_id != block.executor_id
                        or result_capability.operational_effect
                        is not OperationalEffect.NONE
                    ):
                        continue
                    await self._registry.validate_arguments_async(
                        result_capability.id, calls[block.call_id].arguments
                    )
                    result_data = block.output.get("data")
                    if not isinstance(result_data, Mapping):
                        continue
                    self._registry.validate_output(
                        result_capability.id,
                        ToolOutput(
                            kind=cast(str, block.output["kind"]), data=result_data
                        ),
                    )
                    source_id = result_data.get("source_id")
                    revisions = result_data.get("resource_revisions", ())
                    if (
                        isinstance(source_id, str)
                        and isinstance(revisions, tuple)
                        and revisions
                    ):
                        schemas = {
                            item.resource_id: item.revision
                            for item in await self._catalog.resource_schemas(
                                run.agent_id, source_id
                            )
                        }
                        if any(
                            not isinstance(item, Mapping)
                            or schemas.get(cast(str, item.get("resource_id")))
                            != item.get("revision")
                            for item in revisions
                        ):
                            continue
                except (KeyError, TypeError, ValueError):
                    continue
                results.append(block)
        return calls, results

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
    "RELATIONAL_UPDATE_CAPABILITY_ID",
    "RELATIONAL_UPDATE_EVIDENCE_KIND",
    "RELATIONAL_UPDATE_PREVIEW_CAPABILITY_ID",
    "RELATIONAL_UPDATE_PREVIEW_EVIDENCE_KIND",
    "RelationalWriteCatalogReader",
    "ReadScopedCatalogReader",
]
