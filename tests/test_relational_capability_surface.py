"""Focused contracts for the semantic relational model-tool surface."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

import pytest

from daita.capabilities import (
    CapabilityDeclarations,
    CapabilityInputError,
    ToolExecution,
)
from daita.catalog.capabilities import CatalogProjection, catalog_declarations
from daita.catalog.models import Sensitivity
from daita.domains.data import (
    DATA_QUERY_CAPABILITY_ID,
    DATA_QUERY_EVIDENCE_KIND,
    DATA_QUERY_TOOL_NAME,
    DATA_EXPORT_TABULAR_CAPABILITY_ID,
    DATA_EXPORT_TABULAR_TOOL_NAME,
    DataCapabilityDomain,
    DataDomainCatalog,
    DataExportTabularExecutor,
    DataQueryExecutor,
    ExactTabularExportResult,
    PostgreSQLReadResult,
    ResourceSchema,
    SQLiteReadResult,
    postgresql_update_capability_declarations,
    postgresql_update_preview_capability_declarations,
    data_query_capability_declarations,
    data_export_tabular_capability_declarations,
    project_result_rows,
    resource_revision_observation_declarations,
)
from daita.domains.learning import LearningCandidateGuard
from daita.domains.data.routine_precheck import ResourceRevisionCatalog
from daita.loop.models import RunInput


def test_data_query_is_one_backend_neutral_model_tool() -> None:
    declarations = data_query_capability_declarations()

    assert tuple(item.id for item in declarations.capabilities) == (
        DATA_QUERY_CAPABILITY_ID,
    )
    assert tuple(item.name for item in declarations.tool_views) == (
        DATA_QUERY_TOOL_NAME,
    )
    capability = declarations.capabilities[0]
    properties = capability.input_schema["properties"]
    assert isinstance(properties, Mapping)
    assert set(properties) == {"source_id", "resource_ids", "sql", "parameters"}
    assert set(cast(list[str], capability.input_schema["required"])) == {
        "source_id",
        "resource_ids",
        "sql",
    }
    assert set(properties).isdisjoint(
        {"adapter", "adapter_id", "backend", "connection", "executor_id"}
    )


def test_data_export_is_one_backend_neutral_model_tool() -> None:
    declarations = data_export_tabular_capability_declarations()

    assert tuple(item.id for item in declarations.capabilities) == (
        DATA_EXPORT_TABULAR_CAPABILITY_ID,
    )
    assert tuple(item.name for item in declarations.tool_views) == (
        DATA_EXPORT_TABULAR_TOOL_NAME,
    )
    capability = declarations.capabilities[0]
    properties = capability.input_schema["properties"]
    assert isinstance(properties, Mapping)
    assert set(properties) == {
        "source_id",
        "resource_ids",
        "sql",
        "parameters",
        "format",
        "filename",
    }
    assert set(cast(list[str], capability.input_schema["required"])) == {
        "source_id",
        "resource_ids",
        "sql",
        "format",
    }
    assert set(properties).isdisjoint(
        {"adapter", "adapter_id", "backend", "connection", "executor_id"}
    )


class _Catalog:
    def __init__(self) -> None:
        self.schemas = {
            "source-sqlite": ResourceSchema(
                resource_id="resource-sqlite",
                source_id="source-sqlite",
                name="items",
                columns=("id",),
                revision="sha256:" + "1" * 64,
                source_revision="sha256:" + "2" * 64,
                resource_kind="table",
            ),
            "source-postgresql": ResourceSchema(
                resource_id="resource-postgresql",
                source_id="source-postgresql",
                name="items",
                columns=("id",),
                aliases=("public.items",),
                revision="sha256:" + "3" * 64,
                source_revision="sha256:" + "4" * 64,
                resource_kind="table",
            ),
        }
        self.sqlite_other = ResourceSchema(
            resource_id="resource-sqlite-other",
            source_id="source-sqlite",
            name="other_items",
            columns=("id",),
            revision="sha256:" + "8" * 64,
            source_revision="sha256:" + "2" * 64,
            resource_kind="table",
        )

    async def source_adapter_id(self, agent_id: str, source_id: str) -> str | None:
        del agent_id
        return {
            "source-sqlite": "sqlite",
            "source-postgresql": "postgresql",
            "source-unsupported": "other",
        }.get(source_id)

    async def source_routing_facts(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> tuple[Mapping[str, object], ...]:
        del agent_id
        facts = (
            {"source_id": "source-sqlite", "adapter_id": "sqlite"},
            {"source_id": "source-postgresql", "adapter_id": "postgresql"},
        )
        return tuple(
            fact for fact in facts if not source_ids or fact["source_id"] in source_ids
        )

    async def postgresql_update_applicable_source_ids(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> frozenset[str]:
        del agent_id, source_ids
        return frozenset()

    async def resource_identity(
        self, agent_id: str, resource_id: str
    ) -> tuple[str, str, str] | None:
        del agent_id
        schema = next(
            (
                item
                for item in (*self.schemas.values(), self.sqlite_other)
                if item.resource_id == resource_id
            ),
            None,
        )
        if schema is None or schema.revision is None:
            return None
        return schema.source_id, "table", schema.revision

    async def resource_schemas(
        self, agent_id: str, source_id: str
    ) -> tuple[ResourceSchema, ...]:
        del agent_id
        schema = self.schemas.get(source_id)
        if schema is None:
            return ()
        return (
            (schema, self.sqlite_other) if source_id == "source-sqlite" else (schema,)
        )

    async def readable_resource_ids(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> frozenset[str]:
        del agent_id
        return frozenset(
            item.resource_id
            for item in (*self.schemas.values(), self.sqlite_other)
            if not source_ids or item.source_id in source_ids
        )


def _run() -> RunInput:
    return RunInput(
        id="run-relational",
        agent_id="agent-relational",
        message="query exact resources",
        created_at=datetime(2026, 9, 3, tzinfo=UTC),
    )


async def test_mixed_relational_catalog_projects_each_semantic_tool_once() -> None:
    catalog = _Catalog()
    catalog_bundle = catalog_declarations(
        "agent-relational", cast(CatalogProjection, catalog)
    )
    query_bundle = data_query_capability_declarations()
    export_bundle = data_export_tabular_capability_declarations()
    preview_bundle = postgresql_update_preview_capability_declarations()
    update_bundle = postgresql_update_capability_declarations()
    revision_bundle = resource_revision_observation_declarations(
        agent_id="agent-relational",
        catalog=cast(ResourceRevisionCatalog, catalog),
        clock=lambda: datetime(2026, 9, 3, tzinfo=UTC),
    )
    capabilities = (
        *catalog_bundle.capabilities,
        *query_bundle.capabilities,
        *export_bundle.capabilities,
        *preview_bundle.capabilities,
        *update_bundle.capabilities,
        *revision_bundle.capabilities,
    )
    domain = DataCapabilityDomain(
        CapabilityDeclarations(
            domain_owner_id="data",
            capabilities=capabilities,
            executor_ids=tuple(item.executor_id for item in capabilities),
            tool_views=(
                *catalog_bundle.tool_views,
                *query_bundle.tool_views,
                *export_bundle.tool_views,
                *preview_bundle.tool_views,
                *update_bundle.tool_views,
            ),
        ),
        cast(DataDomainCatalog, catalog),
        LearningCandidateGuard(),
    )

    projected = await domain.project(_run())

    assert projected.count(DATA_QUERY_TOOL_NAME) == 1
    assert projected.count(DATA_EXPORT_TABULAR_TOOL_NAME) == 1


async def test_data_domain_derives_private_adapter_and_fails_closed() -> None:
    domain = object.__new__(DataCapabilityDomain)
    domain._catalog = cast(DataDomainCatalog, _Catalog())

    sqlite = await domain._validate_sql(
        _run(),
        {
            "source_id": "source-sqlite",
            "resource_ids": ("resource-sqlite",),
            "sql": "SELECT id FROM items",
        },
    )
    postgresql = await domain._validate_sql(
        _run(),
        {
            "source_id": "source-postgresql",
            "resource_ids": ("resource-postgresql",),
            "sql": "SELECT id FROM public.items",
        },
    )

    assert sqlite["_adapter_id"] == "sqlite"
    assert postgresql["_adapter_id"] == "postgresql"
    with pytest.raises(CapabilityInputError, match="belong to the selected source"):
        await domain._validate_sql(
            _run(),
            {
                "source_id": "source-sqlite",
                "resource_ids": ("resource-postgresql",),
                "sql": "SELECT id FROM items",
            },
        )
    with pytest.raises(CapabilityInputError, match="does not support relational"):
        await domain._validate_sql(
            _run(),
            {
                "source_id": "source-unsupported",
                "resource_ids": ("resource-sqlite",),
                "sql": "SELECT id FROM items",
            },
        )
    with pytest.raises(CapabilityInputError) as unknown_source:
        await domain._validate_sql(
            _run(),
            {
                "source_id": "source-missing",
                "resource_ids": ("resource-sqlite",),
                "sql": "SELECT id FROM items",
            },
        )
    assert unknown_source.value.code == "sql_source_not_available"
    with pytest.raises(CapabilityInputError) as unknown_resource:
        await domain._validate_sql(
            _run(),
            {
                "source_id": "source-sqlite",
                "resource_ids": ("resource-missing",),
                "sql": "SELECT id FROM items",
            },
        )
    assert unknown_resource.value.code == "resource_read_not_allowed"
    with pytest.raises(CapabilityInputError) as target_mismatch:
        await domain._validate_sql(
            _run(),
            {
                "source_id": "source-sqlite",
                "resource_ids": ("resource-sqlite",),
                "sql": "SELECT id FROM other_items",
            },
        )
    assert target_mismatch.value.code == "sql_resource_target_mismatch"


class _ReadBackend:
    def __init__(self, adapter_id: str) -> None:
        self.adapter_id = adapter_id
        self.calls = 0

    async def execute_read(self, **arguments: object):
        self.calls += 1
        source_id = str(arguments["source_id"])
        resource_id = f"resource-{self.adapter_id}"
        projection = project_result_rows(({"id": 1},), max_rows=100, max_bytes=65_536)
        result_type = (
            PostgreSQLReadResult
            if self.adapter_id == "postgresql"
            else SQLiteReadResult
        )
        return result_type(
            source_id=source_id,
            canonical_sql="SELECT id FROM items",
            sql_fingerprint="sha256:" + "5" * 64,
            resource_ids=(resource_id,),
            resource_revisions=((resource_id, "sha256:" + "6" * 64),),
            source_revision="sha256:" + "7" * 64,
            columns=("id",),
            projection=projection,
        )


async def test_data_query_routes_each_catalog_binding_to_exactly_one_backend() -> None:
    sqlite = _ReadBackend("sqlite")
    postgresql = _ReadBackend("postgresql")
    executor = DataQueryExecutor("agent-relational", sqlite, postgresql)

    for adapter_id in ("sqlite", "postgresql"):
        output = await executor.execute(
            ToolExecution(
                run_id="run-relational",
                call_id=f"call-{adapter_id}",
                capability_id=DATA_QUERY_CAPABILITY_ID,
                arguments={
                    "source_id": f"source-{adapter_id}",
                    "resource_ids": (f"resource-{adapter_id}",),
                    "sql": "SELECT id FROM items",
                    "_adapter_id": adapter_id,
                },
            )
        )
        assert output.kind == DATA_QUERY_EVIDENCE_KIND
        assert output.data["adapter_id"] == adapter_id
        assert output.data["resource_ids"] == (f"resource-{adapter_id}",)
        assert output.data["row_limit"] == 100
        assert output.data["byte_limit"] == 65_536

    assert sqlite.calls == 1
    assert postgresql.calls == 1


class _ExportBackend:
    def __init__(self, adapter_id: str) -> None:
        self.adapter_id = adapter_id
        self.calls = 0

    async def execute_exact_tabular(
        self, **arguments: object
    ) -> ExactTabularExportResult:
        self.calls += 1
        resource_id = f"resource-{self.adapter_id}"
        return ExactTabularExportResult(
            format=str(arguments["format_name"]),
            adapter_id=self.adapter_id,
            source_id=str(arguments["source_id"]),
            source_revision="sha256:" + "7" * 64,
            sql_fingerprint="sha256:" + "5" * 64,
            resource_revisions=((resource_id, "sha256:" + "6" * 64),),
            columns=("id",),
            row_count=1,
            content=b"id\r\n1\r\n",
            sensitivity=Sensitivity.PUBLIC,
        )


async def test_data_export_routes_each_catalog_binding_to_exactly_one_backend() -> None:
    sqlite = _ExportBackend("sqlite")
    postgresql = _ExportBackend("postgresql")
    executor = DataExportTabularExecutor(
        "agent-relational",
        sqlite,
        postgresql,
        clock=lambda: datetime(2026, 9, 3, tzinfo=UTC),
    )

    for adapter_id in ("sqlite", "postgresql"):
        output = await executor.execute(
            ToolExecution(
                run_id="run-relational",
                call_id=f"export-{adapter_id}",
                capability_id=DATA_EXPORT_TABULAR_CAPABILITY_ID,
                arguments={
                    "source_id": f"source-{adapter_id}",
                    "resource_ids": (f"resource-{adapter_id}",),
                    "sql": "SELECT id FROM items",
                    "format": "csv",
                    "_adapter_id": adapter_id,
                },
            )
        )
        assert output.data["adapter_id"] == adapter_id
        assert output.data["source_id"] == f"source-{adapter_id}"
        revisions = output.data["resource_revisions"]
        assert isinstance(revisions, tuple) and len(revisions) == 1
        assert revisions[0]["resource_id"] == f"resource-{adapter_id}"
        assert revisions[0]["revision"] == "sha256:" + "6" * 64
        assert output.artifact is not None
        assert output.artifact.provenance.resource_bindings[0].resource_id == (
            f"resource-{adapter_id}"
        )

    assert sqlite.calls == 1
    assert postgresql.calls == 1
