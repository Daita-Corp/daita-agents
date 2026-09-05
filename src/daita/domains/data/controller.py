"""Project data tools and revalidate their catalog and permission bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from typing import Protocol, cast

from ..._json import FrozenJsonObject, canonical_json
from ...adapters.local_file_query import LocalFileQueryError
from ...adapters.local_workspace import LocalWorkspaceError
from ...artifacts.models import (
    ArtifactAuthorship,
    ArtifactDraft,
    ArtifactError,
    ArtifactResourceBinding,
)
from ...capabilities import (
    AccessMode,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    OperationalEffect,
    ToolExecution,
    ToolOutput,
    ToolOutputValidationError,
)
from ...capability_runtime import CapabilityFailure, SideEffectPlan
from ...catalog.capabilities import (
    CATALOG_INSPECT_CAPABILITY_ID,
    CATALOG_SCHEMA_CAPABILITY_ID,
    CATALOG_SEARCH_CAPABILITY_ID,
    CATALOG_TRAVERSE_CAPABILITY_ID,
)
from ...catalog.models import Sensitivity
from ...llm.models import ModelSensitivity, ToolCall
from ...loop.models import RunInput, RunOrigin
from ...storage.sqlite_records import SourcePermissionStateError
from ..learning import LearningCandidateGuard
from .file_capabilities import (
    LOCAL_FILE_CAPABILITY_IDS,
    LOCAL_FILE_QUERY_CAPABILITY_ID,
    LOCAL_FILE_READ_CAPABILITY_ID,
)
from .routine_precheck import RESOURCE_REVISION_OBSERVATION_CAPABILITY_ID
from .sql import (
    DuckDBReadValidationError,
    PostgreSQLUpdateCommand,
    PostgreSQLUpdateIntent,
    ResourceSchema,
    validate_duckdb_read,
    validate_postgresql_read,
    validate_postgresql_update_intent,
    validate_sqlite_read,
)

DATA_DOMAIN_OWNER_ID = "data"
DATA_QUERY_CAPABILITY_ID = "data.query"
DATA_QUERY_EVIDENCE_KIND = "data.query_result"
DATA_EXPORT_TABULAR_CAPABILITY_ID = "data.export_tabular"
POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID = "data.postgresql.update_impact"
POSTGRESQL_UPDATE_PREVIEW_EVIDENCE_KIND = "data.postgresql.update_impact"
POSTGRESQL_UPDATE_CAPABILITY_ID = "data.postgresql.update"
POSTGRESQL_UPDATE_EVIDENCE_KIND = "data.postgresql.update_result"

_CATALOG_CAPABILITIES = frozenset(
    {
        CATALOG_SEARCH_CAPABILITY_ID,
        CATALOG_SCHEMA_CAPABILITY_ID,
        CATALOG_INSPECT_CAPABILITY_ID,
        CATALOG_TRAVERSE_CAPABILITY_ID,
        RESOURCE_REVISION_OBSERVATION_CAPABILITY_ID,
    }
)
_RELATIONAL_READ_CAPABILITIES = frozenset(
    {DATA_QUERY_CAPABILITY_ID, DATA_EXPORT_TABULAR_CAPABILITY_ID}
)
_UPDATE_CAPABILITIES = frozenset(
    {POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID, POSTGRESQL_UPDATE_CAPABILITY_ID}
)
_RESOURCE_ARGUMENT_CAPABILITIES = frozenset(
    {
        CATALOG_INSPECT_CAPABILITY_ID,
        RESOURCE_REVISION_OBSERVATION_CAPABILITY_ID,
        POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID,
        POSTGRESQL_UPDATE_CAPABILITY_ID,
    }
)
_RESOURCE_LIST_ARGUMENT_CAPABILITIES = _RELATIONAL_READ_CAPABILITIES
_RELATIONAL_ADAPTER_IDS = frozenset({"sqlite", "postgresql"})
_ADAPTER_CAPABILITIES = {
    "postgresql": frozenset(
        {
            POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID,
            POSTGRESQL_UPDATE_CAPABILITY_ID,
        }
    ),
}
_SOURCE_CAPABILITIES = frozenset().union(
    _CATALOG_CAPABILITIES,
    _RELATIONAL_READ_CAPABILITIES,
    *_ADAPTER_CAPABILITIES.values(),
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


class PostgreSQLUpdateCatalogReader(CatalogSchemaReader, Protocol):
    async def postgresql_update_scope_issue(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
        assignment_columns: tuple[str, ...],
    ) -> tuple[str, str] | None: ...


class DataDomainCatalog(
    ReadScopedCatalogReader, PostgreSQLUpdateCatalogReader, Protocol
):
    async def source_routing_facts(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> tuple[Mapping[str, object], ...]: ...

    async def postgresql_update_applicable_source_ids(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> frozenset[str]: ...

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

    async def admitted_model_sensitivity(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> ModelSensitivity | None: ...


class DataCapabilityDomain:
    """Own native catalog, SQL, update, and local-file call semantics."""

    domain_owner_id = DATA_DOMAIN_OWNER_ID

    def __init__(
        self,
        declarations: CapabilityDeclarations,
        catalog: DataDomainCatalog,
        learning: LearningCandidateGuard,
        *,
        relational_export_available: bool = True,
        workspace_id: str | None = None,
        workspace_sensitivity: ModelSensitivity | None = None,
        files_only_run_ids: set[str] | None = None,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("data declarations have the wrong domain owner")
        declared_ids = {item.id for item in declarations.capabilities}
        expected_ids = _SOURCE_CAPABILITIES
        if not relational_export_available:
            expected_ids -= {DATA_EXPORT_TABULAR_CAPABILITY_ID}
        if workspace_id is not None:
            expected_ids |= LOCAL_FILE_CAPABILITY_IDS
        if declared_ids != expected_ids:
            raise ValueError(
                "data declarations must exactly match supported native capabilities"
            )
        if (workspace_id is None) != (workspace_sensitivity is None):
            raise ValueError(
                "workspace identity and sensitivity must be present together"
            )
        if workspace_id is not None and (
            not isinstance(workspace_id, str)
            or not workspace_id.startswith("workspace:sha256:")
        ):
            raise ValueError("workspace_id must be one admitted workspace identity")
        if workspace_sensitivity is not None and not isinstance(
            workspace_sensitivity, ModelSensitivity
        ):
            raise TypeError("workspace_sensitivity must be ModelSensitivity")
        for adapter_id, capability_ids in _ADAPTER_CAPABILITIES.items():
            if not capability_ids <= declared_ids:
                raise ValueError(
                    f"data adapter capability admission is incomplete: {adapter_id}"
                )
        for method_name in (
            "source_routing_facts",
            "resource_schemas",
            "readable_resource_ids",
            "source_adapter_id",
        ):
            if not callable(getattr(catalog, method_name, None)):
                raise TypeError(f"data catalog must provide {method_name}")
        if not isinstance(learning, LearningCandidateGuard):
            raise TypeError("learning must be LearningCandidateGuard")
        self._declarations = declarations
        self._catalog = catalog
        self._learning = learning
        self._workspace_id = workspace_id
        self._workspace_sensitivity = workspace_sensitivity
        self._files_only_run_ids = (
            files_only_run_ids if files_only_run_ids is not None else set()
        )
        self._views = {item.name: item for item in declarations.tool_views}
        self._capabilities = {item.id: item for item in declarations.capabilities}

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(self, run: RunInput) -> tuple[str, ...]:
        files_only = run.id in self._files_only_run_ids
        facts: tuple[Mapping[str, object], ...]
        update_source_ids: frozenset[str]
        if files_only:
            facts = ()
            update_source_ids = frozenset()
        else:
            facts = await self._catalog.source_routing_facts(
                run.agent_id,
                (() if run.source_id is None else (run.source_id,)),
            )
            update_source_ids = (
                await self._catalog.postgresql_update_applicable_source_ids(
                    run.agent_id,
                    (() if run.source_id is None else (run.source_id,)),
                )
            )
        active_adapter_ids: set[str] = set()
        for fact in facts:
            adapter_id = fact.get("adapter_id")
            if isinstance(adapter_id, str):
                active_adapter_ids.add(adapter_id)
        names: list[str] = []
        for name in sorted(self._views):
            view = self._views[name]
            capability = self._capabilities[view.capability_id]
            if not self._learning.allows(
                run.id,
                name,
                effectful=capability.operational_effect is not OperationalEffect.NONE,
            ):
                continue
            if capability.id in _CATALOG_CAPABILITIES:
                if facts:
                    names.append(name)
                continue
            if capability.id in LOCAL_FILE_CAPABILITY_IDS:
                if self._workspace_id is not None and run.origin is RunOrigin.USER:
                    names.append(name)
                continue
            if capability.id in _RELATIONAL_READ_CAPABILITIES:
                if active_adapter_ids & _RELATIONAL_ADAPTER_IDS:
                    names.append(name)
                continue
            required_adapter = next(
                (
                    adapter_id
                    for adapter_id, capability_ids in _ADAPTER_CAPABILITIES.items()
                    if capability.id in capability_ids
                ),
                None,
            )
            if required_adapter not in active_adapter_ids:
                continue
            if capability.id in _UPDATE_CAPABILITIES and not update_source_ids:
                continue
            names.append(name)
        return tuple(names)

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
    ) -> FrozenJsonObject:
        del request_sensitivity
        if capability.id in LOCAL_FILE_CAPABILITY_IDS:
            if (
                self._workspace_id is None
                or self._workspace_sensitivity is None
                or run.origin is not RunOrigin.USER
                or run.execution_scope is not None
            ):
                raise CapabilityInputError(
                    "workspace_unavailable",
                    "Workspace file access is unavailable for this run.",
                )
            if "source_id" in arguments or "resource_id" in arguments:
                raise CapabilityInputError(
                    "source_scope_violation",
                    "Workspace file tools do not accept source authority.",
                )
            if capability.id == LOCAL_FILE_READ_CAPABILITY_ID:
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
            if capability.id == LOCAL_FILE_QUERY_CAPABILITY_ID:
                try:
                    validated = validate_duckdb_read(cast(str, arguments["sql"]))
                except DuckDBReadValidationError as error:
                    raise CapabilityInputError(
                        "file_query_invalid",
                        error.message,
                        {"reason": error.reason},
                    ) from error
                prepared = arguments.to_dict()
                prepared["sql"] = validated.canonical_sql
                prepared["sql_fingerprint"] = validated.sql_fingerprint
                return FrozenJsonObject.from_mapping(prepared)
            return arguments
        if capability.operational_effect is not OperationalEffect.NONE:
            self._learning.validate_effect(run.id, call)
        arguments = self._apply_source_scope(run, capability, arguments)
        await self._validate_source_scope(run, capability, arguments)
        self._validate_execution_resource_scope(run, capability, arguments)
        await self._validate_resource_read_scope(run, capability, arguments)
        if capability.access_mode is AccessMode.WRITE and (
            capability.id != POSTGRESQL_UPDATE_CAPABILITY_ID
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
        if capability.id == CATALOG_INSPECT_CAPABILITY_ID:
            resource_id = cast(str, arguments["resource_id"])
            if not resource_id.strip():
                raise CapabilityInputError(
                    "catalog_invalid_resource",
                    "Catalog inspection requires a resource_id.",
                )
        if capability.id == CATALOG_SCHEMA_CAPABILITY_ID:
            schema_query = cast(str | None, arguments.get("query"))
            resource_ids = cast(tuple[str, ...], arguments.get("resource_ids", ()))
            schema_source_id = cast(str | None, arguments.get("source_id"))
            if (
                (schema_query is not None and not schema_query.strip())
                or any(not item.strip() for item in resource_ids)
                or (schema_query is None and not resource_ids)
                or (schema_source_id is not None and not schema_source_id.strip())
            ):
                raise CapabilityInputError(
                    "catalog_invalid_schema",
                    "Catalog schema requires a non-empty query or explicit resource "
                    "IDs and an optional non-empty current source scope.",
                )
        if capability.id in _RELATIONAL_READ_CAPABILITIES:
            arguments = await self._validate_sql(run, arguments)
        if capability.id in _UPDATE_CAPABILITIES:
            await self._validate_postgresql_update_call(
                run,
                arguments,
                execution=capability.id == POSTGRESQL_UPDATE_CAPABILITY_ID,
            )
        return arguments

    async def side_effect_plan(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        execution: ToolExecution,
        fingerprint: FrozenJsonObject,
    ) -> SideEffectPlan:
        if (
            capability.id not in _UPDATE_CAPABILITIES
            or capability.operational_effect is not OperationalEffect.MUTATE_DATA
        ):
            raise ValueError("data domain received an unsupported side effect")
        return SideEffectPlan(recheck_after_approval=False)

    async def finalize_output(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        output: ToolOutput,
        *,
        request_sensitivity: ModelSensitivity,
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
        if capability.operational_effect is not OperationalEffect.NONE:
            self._learning.mark_effect_succeeded(run.id)
        return await self._classify(run, call, capability, output)

    @staticmethod
    def _validate_export_summary(output: ToolOutput) -> None:
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
            and output.data.get("adapter_id") in _RELATIONAL_ADAPTER_IDS
            and output.data.get("format") in {"csv", "xlsx"}
            and output.data.get("filename") == draft.suggested_filename
            and output.data.get("source_id")
            == draft.provenance.resource_bindings[0].source_id
            and output.data.get("row_count") == draft.provenance.row_count
            and output.data.get("column_count") == len(draft.provenance.columns)
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
        resource_ids = arguments.get("resource_ids")
        sql = arguments.get("sql")
        parameters = arguments.get("parameters", ())
        adapter_id = arguments.get("_adapter_id")
        if (
            not isinstance(source_id, str)
            or not isinstance(resource_ids, tuple)
            or not resource_ids
            or any(not isinstance(item, str) for item in resource_ids)
            or not isinstance(sql, str)
            or not isinstance(parameters, tuple)
            or adapter_id not in _RELATIONAL_ADAPTER_IDS
        ):
            raise ToolOutputValidationError(
                "relational export execution arguments are unavailable"
            )
        if await self._catalog.source_adapter_id(run.agent_id, source_id) != adapter_id:
            raise _incomplete_export(draft, "catalog_changed")
        resources = await self._catalog.resource_schemas(run.agent_id, source_id)
        try:
            readable = await self._catalog.readable_resource_ids(
                run.agent_id,
                (source_id,),
            )
        except SourcePermissionStateError as error:
            raise _incomplete_export(draft, "permission_state_invalid") from error
        if run.execution_scope is not None:
            readable = readable & frozenset(run.execution_scope.allowed_resource_ids)
        validator = (
            validate_postgresql_read
            if adapter_id == "postgresql"
            else validate_sqlite_read
        )
        validation = validator(
            sql,
            source_id=source_id,
            resources=resources,
            parameters=parameters,
            allowed_resource_ids=readable,
        )
        if (
            not validation.valid
            or validation.analysis is None
            or validation.source_revision is None
            or not validation.resource_ids
            or frozenset(validation.resource_ids) != frozenset(resource_ids)
            or len(validation.resource_revisions) != len(validation.resource_ids)
        ):
            raise _incomplete_export(draft, "catalog_changed")
        bindings = tuple(
            ArtifactResourceBinding(
                source_id=source_id,
                source_revision=validation.source_revision,
                resource_id=resource_id,
                resource_revision=revision,
            )
            for resource_id, revision in sorted(validation.resource_revisions)
        )
        parameters_sha256 = (
            "sha256:" + sha256(canonical_json(parameters).encode("utf-8")).hexdigest()
        )
        provenance = draft.provenance
        if (
            provenance.resource_bindings != bindings
            or provenance.sql_fingerprint != validation.analysis.sql_fingerprint
            or provenance.parameters_sha256 != parameters_sha256
        ):
            raise ToolOutputValidationError(
                "exact artifact provenance differs from current runtime facts"
            )
        schemas = {item.resource_id: item for item in resources}
        current: list[Sensitivity] = []
        for resource_id in validation.resource_ids:
            schema = schemas.get(resource_id)
            if schema is None:
                raise ToolOutputValidationError(
                    "exact artifact resource is absent from current catalog facts"
                )
            try:
                current.append(Sensitivity(schema.sensitivity_class))
            except ValueError:
                current.append(Sensitivity.RESTRICTED)
        if draft.sensitivity is not _resolved_sensitivity(tuple(current)):
            raise _incomplete_export(draft, "catalog_changed")
        return replace(
            draft,
            provenance=replace(provenance, resource_bindings=bindings),
        )

    def normalize_error(
        self,
        call: ToolCall,
        error: BaseException,
    ) -> CapabilityFailure | None:
        del call
        if isinstance(error, LocalWorkspaceError):
            return CapabilityFailure(error.code, error.message, error.details)
        if isinstance(error, LocalFileQueryError):
            return CapabilityFailure(error.code, error.message, error.details)
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
            not in {CATALOG_SEARCH_CAPABILITY_ID, CATALOG_SCHEMA_CAPABILITY_ID}
            or arguments.get("source_id") is not None
        ):
            return arguments
        scoped = arguments.to_dict()
        scoped["source_id"] = run.source_id
        return FrozenJsonObject.from_mapping(scoped)

    async def _validate_source_scope(
        self,
        run: RunInput,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> None:
        selected_source_id = run.source_id
        if selected_source_id is None:
            return
        supplied_source_id = arguments.get("source_id")
        if supplied_source_id is not None and supplied_source_id != selected_source_id:
            raise CapabilityInputError(
                "source_scope_violation",
                "This run can only access the source selected by the user.",
                {
                    "selected_source_id": selected_source_id,
                    "requested_source_id": supplied_source_id,
                },
            )
        resource_ids: tuple[object, ...] = ()
        if capability.id in _RESOURCE_ARGUMENT_CAPABILITIES:
            resource_ids = (arguments.get("resource_id"),)
        elif capability.id in _RESOURCE_LIST_ARGUMENT_CAPABILITIES:
            raw = arguments.get("resource_ids", ())
            resource_ids = raw if isinstance(raw, tuple) else ()
        elif capability.id == CATALOG_SCHEMA_CAPABILITY_ID:
            raw = arguments.get("resource_ids", ())
            resource_ids = raw if isinstance(raw, tuple) else ()
        elif capability.id == CATALOG_TRAVERSE_CAPABILITY_ID:
            raw_from = arguments.get("from_resource_ids", ())
            raw_to = arguments.get("to_resource_ids", ())
            resource_ids = (
                *(raw_from if isinstance(raw_from, tuple) else ()),
                *(raw_to if isinstance(raw_to, tuple) else ()),
            )
        for resource_id in resource_ids:
            if not isinstance(resource_id, str):
                continue
            identity = await self._catalog.resource_identity(
                run.agent_id,
                resource_id,
            )
            if identity is None or identity[0] != selected_source_id:
                raise CapabilityInputError(
                    "source_scope_violation",
                    "This run can only access resources from the selected source.",
                    {
                        "resource_id": resource_id,
                        "selected_source_id": selected_source_id,
                    },
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
        elif capability.id == CATALOG_SCHEMA_CAPABILITY_ID:
            raw = arguments.get("resource_ids", ())
            resource_ids = raw if isinstance(raw, tuple) else ()
        elif capability.id == CATALOG_TRAVERSE_CAPABILITY_ID:
            raw_from = arguments.get("from_resource_ids", ())
            raw_to = arguments.get("to_resource_ids", ())
            resource_ids = (
                *(raw_from if isinstance(raw_from, tuple) else ()),
                *(raw_to if isinstance(raw_to, tuple) else ()),
            )
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
        if run.execution_scope is not None:
            readable = readable & frozenset(run.execution_scope.allowed_resource_ids)
        if any(resource_id not in readable for resource_id in requested):
            raise CapabilityInputError(
                "resource_read_not_allowed",
                "The requested resource is not available for reading.",
            )

    async def _validate_sql(
        self,
        run: RunInput,
        arguments: Mapping[str, object],
    ) -> FrozenJsonObject:
        source_id = arguments.get("source_id")
        requested_resource_ids = arguments.get("resource_ids")
        sql = arguments.get("sql")
        parameters = arguments.get("parameters", ())
        if (
            not isinstance(source_id, str)
            or not isinstance(requested_resource_ids, tuple)
            or not requested_resource_ids
            or any(not isinstance(item, str) for item in requested_resource_ids)
            or not isinstance(sql, str)
            or not sql.strip()
            or not isinstance(parameters, tuple)
        ):
            raise CapabilityInputError(
                "sql_invalid_input",
                "SQL reads require an exact source_id, one or more resource_ids, "
                "non-empty sql, and an array of parameters.",
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
        if adapter_id not in _RELATIONAL_ADAPTER_IDS:
            raise CapabilityInputError(
                "sql_source_adapter_unsupported",
                "The selected source does not support relational SQL reads.",
                {"source_id": source_id},
            )
        for resource_id in requested_resource_ids:
            identity = await self._catalog.resource_identity(run.agent_id, resource_id)
            if identity is None:
                raise CapabilityInputError(
                    "resource_read_not_allowed",
                    "The requested resource is not available for reading.",
                )
            if identity[0] != source_id:
                raise CapabilityInputError(
                    "sql_resource_source_mismatch",
                    "Every requested resource must belong to the selected source.",
                    {"resource_id": resource_id, "source_id": source_id},
                )
        resources = await self._catalog.resource_schemas(run.agent_id, source_id)
        try:
            readable = await self._catalog.readable_resource_ids(
                run.agent_id,
                (source_id,),
            )
        except SourcePermissionStateError as error:
            raise CapabilityInputError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from error
        if run.execution_scope is not None:
            readable = readable & frozenset(run.execution_scope.allowed_resource_ids)
        validator = (
            validate_postgresql_read
            if adapter_id == "postgresql"
            else validate_sqlite_read
        )
        result = validator(
            sql,
            source_id=source_id,
            resources=resources,
            parameters=parameters,
            allowed_resource_ids=readable,
        )
        if result.valid:
            if frozenset(result.resource_ids) != frozenset(requested_resource_ids):
                raise CapabilityInputError(
                    "sql_resource_target_mismatch",
                    "The SQL query must reference exactly the selected catalog resources.",
                    {
                        "source_id": source_id,
                        "requested_resource_ids": requested_resource_ids,
                    },
                )
            prepared = dict(arguments)
            prepared["_adapter_id"] = adapter_id
            return FrozenJsonObject.from_mapping(prepared)
        if {item.code for item in result.issues} & {
            "resource_out_of_scope",
            "unknown_resource",
        }:
            raise CapabilityInputError(
                "resource_read_not_allowed",
                "One or more requested resources are not available for reading.",
            )
        raise CapabilityInputError(
            "sql_validation_failed",
            "The SQL read is invalid. Correct all reported issues before retrying.",
            {
                "issues": [
                    {
                        "code": item.code,
                        "message": item.message,
                        "details": item.details,
                    }
                    for item in result.issues
                ],
                "source_id": source_id,
            },
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
        if capability.id in {
            CATALOG_SEARCH_CAPABILITY_ID,
            CATALOG_TRAVERSE_CAPABILITY_ID,
        }:
            raise CapabilityInputError(
                "execution_scope_resource_violation",
                "This autonomous run cannot perform open-ended catalog discovery.",
            )
        resource_ids: tuple[object, ...] = ()
        if capability.id in _RESOURCE_ARGUMENT_CAPABILITIES:
            resource_ids = (arguments.get("resource_id"),)
        elif capability.id in _RESOURCE_LIST_ARGUMENT_CAPABILITIES:
            raw = arguments.get("resource_ids", ())
            resource_ids = raw if isinstance(raw, tuple) else ()
        elif capability.id == CATALOG_SCHEMA_CAPABILITY_ID:
            raw = arguments.get("resource_ids", ())
            resource_ids = raw if isinstance(raw, tuple) else ()
            if not resource_ids:
                raise CapabilityInputError(
                    "execution_scope_resource_violation",
                    "This autonomous run requires exact catalog resource IDs.",
                )
        allowed = frozenset(scope.allowed_resource_ids)
        if any(
            not isinstance(resource_id, str) or resource_id not in allowed
            for resource_id in resource_ids
        ):
            raise CapabilityInputError(
                "execution_scope_resource_violation",
                "The requested resource is outside this run's immutable scope.",
            )

    async def _validate_postgresql_update_call(
        self,
        run: RunInput,
        arguments: Mapping[str, object],
        *,
        execution: bool,
    ) -> None:
        source_id = arguments.get("source_id")
        if not isinstance(source_id, str):
            raise CapabilityInputError(
                "write_source_not_available",
                "PostgreSQL update requires an exact current source.",
            )
        if (
            await self._catalog.source_adapter_id(run.agent_id, source_id)
            != "postgresql"
        ):
            raise CapabilityInputError(
                "write_source_not_available",
                "The selected source is not an active PostgreSQL source owned by "
                "this agent.",
                {"source_id": source_id},
            )
        try:
            intent = (
                PostgreSQLUpdateCommand.from_mapping(arguments).intent
                if execution
                else PostgreSQLUpdateIntent.from_mapping(arguments)
            )
        except (TypeError, ValueError) as error:
            raise CapabilityInputError(
                "write_assignment_invalid",
                "The PostgreSQL update intent is malformed.",
            ) from error
        try:
            issue = await self._catalog.postgresql_update_scope_issue(
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
        validation = validate_postgresql_update_intent(
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

    async def _classify(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        output: ToolOutput,
    ) -> ToolOutput:
        if output.sensitivity is not None:
            return output
        if capability.id in LOCAL_FILE_CAPABILITY_IDS:
            if self._workspace_id is None or self._workspace_sensitivity is None:
                raise CapabilityInputError(
                    "result_classification_unavailable",
                    "The workspace result cannot be classified safely.",
                )
            relative_paths: list[str] = []
            physical_revisions: list[str] = []
            path = output.data.get("path")
            revision = output.data.get("physical_revision")
            if isinstance(path, str) and isinstance(revision, str):
                relative_paths.append(path)
                physical_revisions.append(revision)
            matches = output.data.get("matches")
            if isinstance(matches, tuple):
                for match in matches:
                    if not isinstance(match, Mapping):
                        continue
                    match_path = match.get("path")
                    match_revision = match.get("physical_revision")
                    if isinstance(match_path, str) and isinstance(match_revision, str):
                        relative_paths.append(match_path)
                        physical_revisions.append(match_revision)
            return replace(
                output,
                sensitivity=self._workspace_sensitivity,
                sensitivity_provenance={
                    "authority": "local_workspace_binding",
                    "workspace_id": self._workspace_id,
                    "capability_id": capability.id,
                    "relative_paths": tuple(relative_paths),
                    "physical_revisions": tuple(physical_revisions),
                },
            )
        source_id = call.arguments.get("source_id")
        source_ids = (
            (source_id,)
            if isinstance(source_id, str)
            else (() if run.source_id is None else (run.source_id,))
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
        readable = await self._catalog.readable_resource_ids(
            run.agent_id,
            source_ids,
        )
        if run.execution_scope is not None:
            readable = readable & frozenset(run.execution_scope.allowed_resource_ids)
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


def _incomplete_export(draft: ArtifactDraft, reason: str) -> ArtifactError:
    return ArtifactError(
        "artifact_incomplete_export",
        "Current catalog facts no longer prove the exact artifact.",
        {
            "reason": reason,
            "completed_rows": draft.provenance.row_count or 0,
            "completed_columns": len(draft.provenance.columns),
            "completed_bytes": len(draft.content),
        },
    )


def _resolved_sensitivity(values: tuple[Sensitivity, ...]) -> Sensitivity:
    ordered = {
        Sensitivity.PUBLIC: 0,
        Sensitivity.INTERNAL: 1,
        Sensitivity.CONFIDENTIAL: 2,
        Sensitivity.RESTRICTED: 3,
        Sensitivity.UNKNOWN: 3,
    }
    selected = max(values, key=lambda item: ordered[item])
    return Sensitivity.RESTRICTED if selected is Sensitivity.UNKNOWN else selected


__all__ = [
    "CatalogSchemaReader",
    "DATA_DOMAIN_OWNER_ID",
    "DataCapabilityDomain",
    "DataDomainCatalog",
    "DATA_QUERY_CAPABILITY_ID",
    "DATA_QUERY_EVIDENCE_KIND",
    "DATA_EXPORT_TABULAR_CAPABILITY_ID",
    "POSTGRESQL_UPDATE_CAPABILITY_ID",
    "POSTGRESQL_UPDATE_EVIDENCE_KIND",
    "POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID",
    "POSTGRESQL_UPDATE_PREVIEW_EVIDENCE_KIND",
    "PostgreSQLUpdateCatalogReader",
    "ReadScopedCatalogReader",
]
