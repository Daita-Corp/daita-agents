"""Focused contracts for the semantic relational model-tool surface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import pytest

from daita._json import FrozenJsonObject, canonical_json, thaw_json
from daita.adapters.registry import SourceOperation, SourceTypeRegistry
from daita.capabilities import (
    CallTargetLane,
    CapabilityDeclarations,
    CapabilityInputError,
    CapabilityRegistry,
    ToolExecution,
    ToolOutput,
)
from daita.catalog.capabilities import CatalogProjection, catalog_declarations
from daita.catalog.models import Sensitivity
from daita.domains.data import (
    DATA_EXPORT_TABULAR_CAPABILITY_ID,
    DATA_QUERY_CAPABILITY_ID,
    DATA_QUERY_EVIDENCE_KIND,
    DATA_QUERY_TOOL_NAME,
    PATH_QUERY_CAPABILITY_ID,
    DataCapabilityDomain,
    DataDomainCatalog,
    DataExportTabularExecutor,
    DataQueryExecutor,
    ExactTabularExportResult,
    PostgreSQLReadResult,
    ResourceSchema,
    SQLiteReadResult,
    data_query_capability_declarations,
    exact_export_capability_declarations,
    exact_export_outputs,
    project_result_rows,
    relational_update_capability_declarations,
    relational_update_preview_capability_declarations,
    relational_upsert_capability_declarations,
    source_file_capability_declarations,
)
from daita.llm.models import ModelSensitivity, ToolCall
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
    assert set(properties) == {
        "source_id",
        "resource_ids",
        "sql",
        "parameters",
        "statements",
    }
    # Exactly one of sql or statements is enforced by the data domain.
    assert set(cast(list[str], capability.input_schema["required"])) == {
        "source_id",
        "resource_ids",
    }
    assert set(properties).isdisjoint(
        {"adapter", "adapter_id", "backend", "connection", "executor_id"}
    )


def test_data_export_is_one_backend_neutral_model_tool() -> None:
    declarations = exact_export_capability_declarations(
        adapter_ids=("sqlite", "postgresql")
    )

    assert tuple(item.id for item in declarations.capabilities) == (
        DATA_EXPORT_TABULAR_CAPABILITY_ID,
    )
    assert declarations.tool_views == ()
    capability = declarations.capabilities[0]
    properties = capability.input_schema["properties"]
    assert isinstance(properties, Mapping)
    assert set(properties) == {
        "source_id",
        "resource_ids",
        "sql",
        "parameters",
        "output",
        "filename",
    }
    assert set(cast(list[str], capability.input_schema["required"])) == {
        "source_id",
        "resource_ids",
        "sql",
        "output",
    }
    assert set(properties).isdisjoint(
        {"adapter", "adapter_id", "backend", "connection", "executor_id"}
    )


def test_data_query_file_outputs_route_to_exact_export_with_one_limits_clause() -> None:
    exports = exact_export_capability_declarations(adapter_ids=("sqlite", "postgresql"))
    query = data_query_capability_declarations(
        file_outputs=exact_export_outputs(allow_arrow=False),
        file_capability_id=DATA_EXPORT_TABULAR_CAPABILITY_ID,
    )

    class Executor:
        def __init__(self, executor_id: str) -> None:
            self.executor_id = executor_id

        async def execute(self, request: ToolExecution) -> ToolOutput:
            raise AssertionError("Registration cannot execute a query")

    registry = CapabilityRegistry(
        declarations=(
            CapabilityDeclarations(
                domain_owner_id="data",
                capabilities=(*query.capabilities, *exports.capabilities),
                executor_ids=(*query.executor_ids, *exports.executor_ids),
                tool_views=query.tool_views,
            ),
        ),
        executors=tuple(
            Executor(item) for item in (*query.executor_ids, *exports.executor_ids)
        ),
    )
    view = registry.resolve_tool(DATA_QUERY_TOOL_NAME)[0]
    assert view.mode_argument == "output"
    assert view.default_mode == "rows"
    assert view.mode_capability_ids == (
        ("rows", DATA_QUERY_CAPABILITY_ID),
        ("csv", DATA_EXPORT_TABULAR_CAPABILITY_ID),
        ("xlsx", DATA_EXPORT_TABULAR_CAPABILITY_ID),
    )
    definition = registry.tool_definition(
        DATA_QUERY_TOOL_NAME, lane=CallTargetLane.FOREGROUND
    )
    assert definition.description.count("Limits:") == 1
    assert len(definition.description.encode()) <= 600
    properties = definition.input_schema["properties"]
    assert isinstance(properties, Mapping)
    assert thaw_json(properties["output"]) == {
        "type": "string",
        "enum": ["rows", "csv", "xlsx"],
    }
    assert "output" not in cast(list[str], definition.input_schema["required"])


class _Catalog:
    # No live session folder source unless a test adds one.
    @property
    def session_source_id(self) -> str | None:
        return None

    def __init__(self) -> None:
        self.write_applicability_calls: list[str] = []
        self.routing_enabled = True
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
        if not self.routing_enabled:
            return ()
        facts = (
            {"source_id": "source-sqlite", "adapter_id": "sqlite"},
            {"source_id": "source-postgresql", "adapter_id": "postgresql"},
        )
        return tuple(
            fact for fact in facts if not source_ids or fact["source_id"] in source_ids
        )

    async def relational_write_applicable_source_ids(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
        *,
        operation: str = "update",
    ) -> frozenset[str]:
        del agent_id, source_ids
        self.write_applicability_calls.append(operation)
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
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
        *,
        operation: str = "update",
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


def _domain(catalog: _Catalog, *, source_files: bool = False) -> DataCapabilityDomain:
    catalog_bundle = catalog_declarations(
        "agent-relational", cast(CatalogProjection, catalog)
    )
    query_bundle = data_query_capability_declarations(
        file_outputs=exact_export_outputs(allow_arrow=False),
        file_capability_id=DATA_EXPORT_TABULAR_CAPABILITY_ID,
        path_query=source_files,
    )
    export_bundle = exact_export_capability_declarations(
        adapter_ids=("sqlite", "postgresql")
    )
    preview_bundle = relational_update_preview_capability_declarations()
    update_bundle = relational_update_capability_declarations()
    upsert_bundle = relational_upsert_capability_declarations()
    file_bundle = source_file_capability_declarations()
    capabilities = (
        *catalog_bundle.capabilities,
        *query_bundle.capabilities,
        *export_bundle.capabilities,
        *preview_bundle.capabilities,
        *update_bundle.capabilities,
        *upsert_bundle.capabilities,
        *(file_bundle.capabilities if source_files else ()),
    )
    return DataCapabilityDomain(
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
                *upsert_bundle.tool_views,
                *(file_bundle.tool_views if source_files else ()),
            ),
        ),
        cast(DataDomainCatalog, catalog),
        SourceTypeRegistry(),
        source_files_available=source_files,
        path_query_available=source_files,
    )


@pytest.mark.parametrize("tool_name", (None, DATA_QUERY_TOOL_NAME))
async def test_mixed_relational_catalog_projects_each_semantic_tool_once(
    tool_name,
) -> None:
    catalog = _Catalog()
    domain = _domain(catalog)
    run = _run()
    projected = (
        await domain.project(run)
        if tool_name is None
        else await domain.project_call(run, ToolCall(id="read", name=tool_name), None)
    )
    assert projected.count(DATA_QUERY_TOOL_NAME) == (
        tool_name in {None, DATA_QUERY_TOOL_NAME}
    )
    assert catalog.write_applicability_calls == (
        ["update", "upsert"] if tool_name is None else []
    )


_SESSION_SOURCE = "source:workspace:sha256:" + "5" * 64


class _FolderCatalog(_Catalog):
    """The relational catalog plus a live session folder source when present."""

    def __init__(self, *, session: bool) -> None:
        super().__init__()
        self._session = session

    @property
    def session_source_id(self) -> str | None:
        return _SESSION_SOURCE if self._session else None

    async def source_adapter_id(self, agent_id: str, source_id: str) -> str | None:
        if self._session and source_id == _SESSION_SOURCE:
            return "folder"
        return await super().source_adapter_id(agent_id, source_id)

    async def source_routing_facts(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> tuple[Mapping[str, object], ...]:
        facts = await super().source_routing_facts(agent_id, source_ids)
        if self._session and (not source_ids or _SESSION_SOURCE in source_ids):
            facts += ({"source_id": _SESSION_SOURCE, "adapter_id": "folder"},)
        return facts

    async def readable_resource_ids(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
        *,
        operation: str = "update",
    ) -> frozenset[str]:
        return await super().readable_resource_ids(
            agent_id, tuple(item for item in source_ids if item != _SESSION_SOURCE)
        )


@pytest.mark.parametrize("tool_name", ("file_search", "file_read", "unknown"))
@pytest.mark.parametrize("session", (True, False))
async def test_folder_file_tools_project_only_with_a_folder_source_in_scope(
    tool_name,
    session,
):
    domain = _domain(_FolderCatalog(session=session), source_files=True)
    projected = await domain.project_call(
        _run(), ToolCall(id="local", name=tool_name), None
    )
    assert projected == (() if tool_name == "unknown" or not session else (tool_name,))


@pytest.mark.parametrize(
    ("tool_name", "arguments", "error"),
    (
        ("file_read", {"source_id": _SESSION_SOURCE, "path": "x.txt"}, None),
        ("file_search", {"source_id": _SESSION_SOURCE, "query": "x"}, None),
        (
            "data_query",
            {
                "source_id": _SESSION_SOURCE,
                "path_pattern": "x.csv",
                "sql": "SELECT * FROM data",
            },
            None,
        ),
        (
            "file_read",
            {"source_id": "source:sha256:" + "0" * 64, "path": "x"},
            "source_scope_violation",
        ),
        (
            "file_read",
            {"source_id": _SESSION_SOURCE, "path": "x", "cursor": "cursor"},
            "path_invalid",
        ),
        (
            "data_query",
            {
                "source_id": "source-sqlite",
                "path_pattern": "x.csv",
                "sql": "SELECT * FROM data",
            },
            "path_pattern_unsupported",
        ),
        (
            "data_query",
            {
                "source_id": _SESSION_SOURCE,
                "path_pattern": "x.csv",
                "sql": "DELETE FROM data",
            },
            "folder_query_invalid",
        ),
    ),
)
async def test_folder_file_calls_bind_their_source_before_file_io(
    tool_name, arguments, error
):
    domain = _domain(_FolderCatalog(session=True), source_files=True)
    capability_id = (
        PATH_QUERY_CAPABILITY_ID
        if "path_pattern" in arguments
        else next(
            view.capability_id
            for view in domain.declarations.tool_views
            if view.name == tool_name
        )
    )
    capability = next(
        item for item in domain.declarations.capabilities if item.id == capability_id
    )
    call = ToolCall(id="local", name=tool_name, arguments=arguments)
    if error is None:
        bound = await domain.prepare_call(
            _run(),
            call,
            capability,
            FrozenJsonObject.from_mapping(arguments),
            request_sensitivity=ModelSensitivity.RESTRICTED,
        )
        assert bound["_adapter_id"] == "folder"
        assert "_resource_ids" not in bound
        if "path_pattern" in arguments:
            assert bound["sql"] == "SELECT * FROM data"
            assert isinstance(bound["sql_fingerprint"], str)
    else:
        with pytest.raises(CapabilityInputError) as raised:
            await domain.prepare_call(
                _run(),
                call,
                capability,
                FrozenJsonObject.from_mapping(arguments),
                request_sensitivity=ModelSensitivity.RESTRICTED,
            )
        assert raised.value.code == error


async def test_exact_relational_projection_rechecks_current_source_admission():
    catalog = _Catalog()
    domain = _domain(catalog)
    run = replace(_run(), source_scope_ids=("source-sqlite",))
    call = ToolCall(id="read", name=DATA_QUERY_TOOL_NAME)
    assert await domain.project_call(run, call, None) == (DATA_QUERY_TOOL_NAME,)
    catalog.routing_enabled = False
    assert await domain.project_call(run, call, None) == ()


async def test_data_domain_derives_private_adapter_and_fails_closed() -> None:
    domain = object.__new__(DataCapabilityDomain)
    domain._catalog = cast(DataDomainCatalog, _Catalog())
    domain._source_types = SourceTypeRegistry()

    sqlite = await domain._validate_sql(
        _run(),
        operation=SourceOperation.SQL_READ,
        arguments={
            "source_id": "source-sqlite",
            "resource_ids": ("resource-sqlite",),
            "sql": "SELECT id FROM items",
        },
    )
    postgresql = await domain._validate_sql(
        _run(),
        operation=SourceOperation.SQL_READ,
        arguments={
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
            operation=SourceOperation.SQL_READ,
            arguments={
                "source_id": "source-sqlite",
                "resource_ids": ("resource-postgresql",),
                "sql": "SELECT id FROM items",
            },
        )
    with pytest.raises(CapabilityInputError, match="does not support relational"):
        await domain._validate_sql(
            _run(),
            operation=SourceOperation.SQL_READ,
            arguments={
                "source_id": "source-unsupported",
                "resource_ids": ("resource-sqlite",),
                "sql": "SELECT id FROM items",
            },
        )
    with pytest.raises(CapabilityInputError) as unknown_source:
        await domain._validate_sql(
            _run(),
            operation=SourceOperation.SQL_READ,
            arguments={
                "source_id": "source-missing",
                "resource_ids": ("resource-sqlite",),
                "sql": "SELECT id FROM items",
            },
        )
    assert unknown_source.value.code == "sql_source_not_available"
    with pytest.raises(CapabilityInputError) as unknown_resource:
        await domain._validate_sql(
            _run(),
            operation=SourceOperation.SQL_READ,
            arguments={
                "source_id": "source-sqlite",
                "resource_ids": ("resource-missing",),
                "sql": "SELECT id FROM items",
            },
        )
    assert unknown_resource.value.code == "resource_read_not_allowed"
    with pytest.raises(CapabilityInputError) as target_mismatch:
        await domain._validate_sql(
            _run(),
            operation=SourceOperation.SQL_READ,
            arguments={
                "source_id": "source-sqlite",
                "resource_ids": ("resource-sqlite",),
                "sql": "SELECT id FROM other_items",
            },
        )
    assert target_mismatch.value.code == "sql_resource_target_mismatch"


class _ReadBackend:
    def __init__(
        self,
        adapter_id: str,
        rows: tuple[dict[str, object], ...] = ({"id": 1},),
    ) -> None:
        self.adapter_id = adapter_id
        self.rows = rows
        self.calls = 0

    async def execute_read(self, **arguments: object):
        self.calls += 1
        source_id = str(arguments["source_id"])
        resource_id = f"resource-{self.adapter_id}"
        max_rows = arguments["max_rows"]
        max_bytes = arguments["max_bytes"]
        assert isinstance(max_rows, int)
        assert isinstance(max_bytes, int)
        projection = project_result_rows(
            self.rows,
            max_rows=max_rows,
            max_bytes=max_bytes,
        )
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
    executor = DataQueryExecutor(
        "agent-relational", {"sqlite": sqlite, "postgresql": postgresql}
    )

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
        assert output.data["byte_limit"] == 16_384

    assert sqlite.calls == 1
    assert postgresql.calls == 1


async def test_data_query_inline_byte_limit_reports_truncation() -> None:
    rows: tuple[dict[str, object], ...] = tuple(
        {"id": index, "detail": "x" * 1_000} for index in range(100)
    )
    sqlite = _ReadBackend("sqlite", rows)
    executor = DataQueryExecutor(
        "agent-relational", {"sqlite": sqlite, "postgresql": _ReadBackend("postgresql")}
    )

    output = await executor.execute(
        ToolExecution(
            run_id="run-relational",
            call_id="call-sqlite",
            capability_id=DATA_QUERY_CAPABILITY_ID,
            arguments={
                "source_id": "source-sqlite",
                "resource_ids": ("resource-sqlite",),
                "sql": "SELECT id, detail FROM items",
                "_adapter_id": "sqlite",
            },
        )
    )

    assert output.data["byte_limit"] == 16_384
    assert output.data["truncated"] is True
    assert output.data["truncation_reasons"] == ("byte_limit",)
    returned_rows = output.data["returned_rows"]
    utf8_bytes = output.data["utf8_bytes"]
    byte_limit = output.data["byte_limit"]
    assert isinstance(returned_rows, int)
    assert isinstance(utf8_bytes, int)
    assert isinstance(byte_limit, int)
    assert 0 < returned_rows < len(rows)
    assert utf8_bytes <= byte_limit


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
        {"sqlite": sqlite, "postgresql": postgresql},
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
                    "output": "csv",
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


@pytest.mark.parametrize("slack", [0, -1])
def test_result_projection_keeps_only_whole_rows_within_the_byte_limit(slack: int):
    rows = tuple({"id": index, "label": "x" * (index * 7)} for index in range(6))
    # Three rows encode as '[r0,r1,r2]': brackets, rows and two separators.
    three_rows = 2 + sum(len(canonical_json(row)) for row in rows[:3]) + 2

    projection = project_result_rows(rows, max_rows=5, max_bytes=three_rows + slack)

    kept = 3 if slack == 0 else 2
    assert projection.returned_rows == kept
    assert projection.utf8_bytes == len(canonical_json(list(rows[:kept])))
    assert projection.truncation_reasons == ("row_limit", "byte_limit")
