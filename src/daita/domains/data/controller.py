"""Project data tools and revalidate their catalog and permission bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Protocol, cast

from ..._json import FrozenJsonObject
from ...capabilities import (
    AccessMode,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    OperationalEffect,
    ToolExecution,
    ToolOutput,
)
from ...capability_runtime import CapabilityFailure, SideEffectPlan
from ...catalog.capabilities import (
    CATALOG_INSPECT_CAPABILITY_ID,
    CATALOG_SCHEMA_CAPABILITY_ID,
    CATALOG_SEARCH_CAPABILITY_ID,
    CATALOG_TRAVERSE_CAPABILITY_ID,
)
from ...llm.models import ModelSensitivity, ToolCall
from ...loop.models import RunInput
from ...storage.sqlite_records import SourcePermissionStateError
from ..learning import LearningCandidateGuard
from .file_capabilities import LOCAL_FILE_READ_CAPABILITY_ID
from .sql import (
    PostgreSQLUpdateCommand,
    PostgreSQLUpdateIntent,
    ResourceSchema,
    validate_postgresql_read,
    validate_postgresql_update_intent,
    validate_sqlite_read,
)

DATA_DOMAIN_OWNER_ID = "data"
SQLITE_QUERY_CAPABILITY_ID = "data.sqlite.query"
SQLITE_QUERY_EVIDENCE_KIND = "data.sqlite.query_result"
POSTGRESQL_QUERY_CAPABILITY_ID = "data.postgresql.query"
POSTGRESQL_QUERY_EVIDENCE_KIND = "data.postgresql.query_result"
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
    }
)
_QUERY_CAPABILITIES = frozenset(
    {SQLITE_QUERY_CAPABILITY_ID, POSTGRESQL_QUERY_CAPABILITY_ID}
)
_UPDATE_CAPABILITIES = frozenset(
    {POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID, POSTGRESQL_UPDATE_CAPABILITY_ID}
)
_RESOURCE_ARGUMENT_CAPABILITIES = frozenset(
    {
        CATALOG_INSPECT_CAPABILITY_ID,
        LOCAL_FILE_READ_CAPABILITY_ID,
        POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID,
        POSTGRESQL_UPDATE_CAPABILITY_ID,
    }
)
_ADAPTER_CAPABILITIES = {
    "sqlite": frozenset({SQLITE_QUERY_CAPABILITY_ID}),
    "postgresql": frozenset(
        {
            POSTGRESQL_QUERY_CAPABILITY_ID,
            POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID,
            POSTGRESQL_UPDATE_CAPABILITY_ID,
        }
    ),
    "local-directory": frozenset({LOCAL_FILE_READ_CAPABILITY_ID}),
}
_OWNED_CAPABILITIES = frozenset().union(
    _CATALOG_CAPABILITIES,
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
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("data declarations have the wrong domain owner")
        declared_ids = {item.id for item in declarations.capabilities}
        if declared_ids != _OWNED_CAPABILITIES:
            raise ValueError(
                "data declarations must exactly match supported native capabilities"
            )
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
        self._views = {item.name: item for item in declarations.tool_views}
        self._capabilities = {item.id: item for item in declarations.capabilities}

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(self, run: RunInput) -> tuple[str, ...]:
        facts = await self._catalog.source_routing_facts(
            run.agent_id,
            (() if run.source_id is None else (run.source_id,)),
        )
        active_adapter_ids = {
            adapter_id
            for fact in facts
            if isinstance((adapter_id := fact.get("adapter_id")), str)
        }
        update_source_ids = await self._catalog.postgresql_update_applicable_source_ids(
            run.agent_id,
            (() if run.source_id is None else (run.source_id,)),
        )
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
        if capability.operational_effect is not OperationalEffect.NONE:
            self._learning.validate_effect(run.id, call)
        arguments = self._apply_source_scope(run, capability, arguments)
        await self._validate_source_scope(run, capability, arguments)
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
        if capability.id in _QUERY_CAPABILITIES:
            await self._validate_sql(run, capability, arguments)
        if capability.id in _UPDATE_CAPABILITIES:
            await self._validate_postgresql_update_call(
                run,
                arguments,
                execution=capability.id == POSTGRESQL_UPDATE_CAPABILITY_ID,
            )
        if capability.id == LOCAL_FILE_READ_CAPABILITY_ID:
            file_source_id = arguments.get("source_id")
            file_resource_id = arguments.get("resource_id")
            if (
                not isinstance(file_source_id, str)
                or not isinstance(file_resource_id, str)
                or not await self._catalog.is_current_tabular_file(
                    run.agent_id,
                    file_source_id,
                    file_resource_id,
                )
            ):
                raise CapabilityInputError(
                    "file_not_current_or_tabular",
                    "The selected file is not a current tabular catalog resource.",
                    {
                        "resource_id": file_resource_id,
                        "source_id": file_source_id,
                    },
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
        if capability.operational_effect is not OperationalEffect.NONE:
            self._learning.mark_effect_succeeded(run.id)
        return await self._classify(run, call, capability, output)

    def normalize_error(
        self,
        call: ToolCall,
        error: BaseException,
    ) -> CapabilityFailure | None:
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
        if any(resource_id not in readable for resource_id in requested):
            raise CapabilityInputError(
                "resource_read_not_allowed",
                "The requested resource is not available for reading.",
            )

    async def _validate_sql(
        self,
        run: RunInput,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> None:
        source_id = arguments.get("source_id")
        sql = arguments.get("sql")
        parameters = arguments.get("parameters", ())
        if (
            not isinstance(source_id, str)
            or not isinstance(sql, str)
            or not sql.strip()
            or not isinstance(parameters, tuple)
        ):
            raise CapabilityInputError(
                "sql_invalid_input",
                "SQL reads require source_id, non-empty sql, and an array of "
                "parameters.",
            )
        expected_adapter = (
            "postgresql"
            if capability.id == POSTGRESQL_QUERY_CAPABILITY_ID
            else "sqlite"
        )
        actual_adapter = await self._catalog.source_adapter_id(
            run.agent_id,
            source_id,
        )
        if actual_adapter != expected_adapter:
            raise CapabilityInputError(
                "sql_source_adapter_mismatch",
                "The selected SQL tool does not match the source adapter.",
                {
                    "actual_adapter": actual_adapter,
                    "expected_adapter": expected_adapter,
                    "source_id": source_id,
                },
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
            allowed_resource_ids=readable,
        )
        if result.valid:
            return
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
        return replace(
            output,
            sensitivity=sensitivity,
            sensitivity_provenance={
                "authority": "current_admitted_resource_scope",
                "capability_id": capability.id,
                "source_ids": source_ids,
                "resource_ids": tuple(sorted(readable)),
            },
        )


__all__ = [
    "CatalogSchemaReader",
    "DATA_DOMAIN_OWNER_ID",
    "DataCapabilityDomain",
    "DataDomainCatalog",
    "POSTGRESQL_QUERY_CAPABILITY_ID",
    "POSTGRESQL_QUERY_EVIDENCE_KIND",
    "POSTGRESQL_UPDATE_CAPABILITY_ID",
    "POSTGRESQL_UPDATE_EVIDENCE_KIND",
    "POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID",
    "POSTGRESQL_UPDATE_PREVIEW_EVIDENCE_KIND",
    "PostgreSQLUpdateCatalogReader",
    "ReadScopedCatalogReader",
    "SQLITE_QUERY_CAPABILITY_ID",
    "SQLITE_QUERY_EVIDENCE_KIND",
]
