"""Phase 5 deterministic tool-selection and catalog-scale evaluations."""

from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from _capability_runtime_support import StaticTestDomain
from _toolbox_model_support import ToolboxAwareMockModelProvider

from daita import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    LocalWorkspace,
    SQLiteSource,
)
from daita._json import canonical_json
from daita.adapters.models import (
    DiscoveryRequest,
    DiscoveryResult,
    ResourceRef,
    ResourceSnapshot,
    SourceHealth,
    SourceRegistration,
)
from daita.adapters.protocols import ResourceAdapter
from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityRegistry,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
)
from daita.capability_runtime import CapabilityRuntime
from daita.catalog.models import CatalogSync, CatalogSyncStatus, SourceCatalogSnapshot
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopLimits, RunInput, Transcript

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)

# This is the Phase 5 requirement-to-test matrix. Every value is asserted as the
# exact ordered tool sequence in a deterministic public-agent test below.
PHASE5_ROUTES = {
    "find_forecast": (
        "Find the Q4 forecast file",
        ("file_search",),
    ),
    "read_todo": (
        "Read notes/todo.md",
        ("file_read",),
    ),
    "file_revenue": (
        "Total revenue by region in sales.csv",
        ("toolbox_search", "toolbox_load", "file_query"),
    ),
    "customer_columns": (
        "What columns exist in the connected customer table?",
        ("catalog_schema",),
    ),
    "postgres_query": (
        "Run SELECT customer_id FROM customers against Postgres",
        ("data_query_postgresql",),
    ),
    "mixed_compare": (
        "Compare local sales.csv to the connected customer table",
        (
            "toolbox_search",
            "toolbox_load",
            "file_query",
            "catalog_schema",
            "data_query_sqlite",
        ),
    ),
    "files_only": (
        "Use only files for this question: find the Q4 forecast file",
        ("file_search",),
    ),
    "latest_log": (
        "Find the latest log and show its end",
        ("file_search", "file_read"),
    ),
    "markdown_report": (
        "Create a Markdown report from these results and save it locally",
        (
            "toolbox_search",
            "toolbox_load",
            "artifact_create_document",
            "toolbox_search",
            "toolbox_load",
            "artifact_save_local",
        ),
    ),
    "text_edit": (
        "Change the timeout in config.yaml from 30 to 60",
        (
            "file_read",
            "toolbox_search",
            "toolbox_load",
            "artifact_edit_text",
            "toolbox_search",
            "toolbox_load",
            "artifact_save_local",
        ),
    ),
    "unsupported": (
        "Use a shell command to rewrite every image in this workspace",
        (),
    ),
}


def _profile(provider: object) -> ModelProfile:
    provider_id = getattr(provider, "provider_id")
    assert isinstance(provider_id, str)
    return ModelProfile(
        id=provider_id,
        context_window_tokens=128_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=False,
    )


def _tool(call_id: str, name: str, arguments: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _workspace(tmp_path: Path) -> LocalWorkspace:
    path = tmp_path / "workspace"
    path.mkdir()
    return LocalWorkspace(path)


def _calls(transcript: Transcript) -> tuple[ToolCall, ...]:
    return tuple(call for message in transcript.messages for call in message.tool_calls)


def _results(transcript: Transcript) -> dict[str, ToolResultBlock]:
    return {
        block.call_id: block
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }


def _request_results(request: ModelRequest) -> dict[str, ToolResultBlock]:
    return {
        block.call_id: block
        for message in request.messages
        if message.role is MessageRole.TOOL
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }


def _data(block: ToolResultBlock) -> Mapping[str, object]:
    value = block.output.get("data")
    assert isinstance(value, Mapping)
    return value


def _assert_route(transcript: Transcript, route: str) -> None:
    expected = PHASE5_ROUTES[route][1]
    assert tuple(call.name for call in _calls(transcript)) == expected


def _assert_no_workspace_path(
    workspace: LocalWorkspace,
    provider: object,
    transcript: Transcript,
) -> None:
    requests = getattr(provider, "requests")
    assert str(workspace.root) not in repr(requests)
    assert str(workspace.root) not in repr(transcript)


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                segment TEXT NOT NULL
            );
            INSERT INTO customers VALUES (1, 'Ada', 'enterprise');
            INSERT INTO customers VALUES (2, 'Lin', 'self-serve');
            INSERT INTO customers VALUES (3, 'Jo', 'enterprise');
            """)


def _ids() -> Callable[[str], str]:
    counts: defaultdict[str, int] = defaultdict(int)

    def create(prefix: str) -> str:
        counts[prefix] += 1
        if prefix in {"run", "conversation", "artifact", "destination"}:
            return f"{prefix}-{counts[prefix]:032x}"
        return f"{prefix}-{counts[prefix]}"

    return create


async def test_direct_file_first_tools_are_stable_and_grounded(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    forecast = workspace.root / "planning" / "Q4-forecast.md"
    todo = workspace.root / "notes" / "todo.md"
    forecast.parent.mkdir()
    todo.parent.mkdir()
    forecast.write_text("forecast_token=Q4_7F21\n", encoding="utf-8")
    todo.write_text("todo_token=TODO_8C32\n", encoding="utf-8")

    cases = (
        (
            "find_forecast",
            (
                _tool(
                    "find-forecast",
                    "file_search",
                    {"query": "Q4", "mode": "paths"},
                ),
                _stop("The Q4 forecast is planning/Q4-forecast.md."),
            ),
            "find-forecast",
            "planning/Q4-forecast.md",
        ),
        (
            "read_todo",
            (
                _tool("read-todo", "file_read", {"path": "notes/todo.md"}),
                _stop("The todo token is TODO_8C32."),
            ),
            "read-todo",
            "TODO_8C32",
        ),
    )
    for index, (route, script, result_id, grounding) in enumerate(cases):
        provider = ToolboxAwareMockModelProvider(
            script,
            provider_id=f"mock:phase5-direct-{index}",
        )
        agent = await Agent.create(
            f"phase5-direct-{index}",
            root=tmp_path / f"state-{index}",
            workspace=workspace,
            model=provider,
            model_profile=_profile(provider),
        )
        try:
            result = await agent.run(PHASE5_ROUTES[route][0])
            transcript = await agent.transcript(result.run_id)
            _assert_route(transcript, route)
            block = _results(transcript)[result_id]
            assert not block.is_error
            assert grounding in canonical_json(block.output)
            assert result.final_text is not None and grounding in result.final_text
            assert block.sensitivity_provenance["authority"] == (
                "local_workspace_binding"
            )
            assert result.artifacts == ()
            assert result.artifact_deliveries == ()
            _assert_no_workspace_path(workspace, provider, transcript)
        finally:
            await agent.close()


async def test_file_query_search_load_route_is_exact_and_grounded(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    (workspace.root / "sales.csv").write_text(
        "region,revenue\nnorth,10\nsouth,20\nnorth,5\n",
        encoding="utf-8",
    )
    provider = MockModelProvider(
        (
            _tool(
                "search-query",
                "toolbox_search",
                {
                    "query": "total revenue by region sales csv",
                    "toolboxes": ["files"],
                    "limit": 5,
                },
            ),
            _tool("load-query", "toolbox_load", {"tool_names": ["file_query"]}),
            _tool(
                "query-sales",
                "file_query",
                {
                    "path_pattern": "sales.csv",
                    "sql": (
                        "SELECT region, SUM(revenue) AS total_revenue FROM data "
                        "GROUP BY region ORDER BY region"
                    ),
                },
            ),
            _stop("North revenue is 15 and south revenue is 20."),
        ),
        provider_id="mock:phase5-file-query",
    )
    agent = await Agent.create(
        "phase5-file-query",
        root=tmp_path / "state",
        workspace=workspace,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        result = await agent.run(PHASE5_ROUTES["file_revenue"][0])
        transcript = await agent.transcript(result.run_id)
        _assert_route(transcript, "file_revenue")
        results = _results(transcript)
        matches = _data(results["search-query"])["matches"]
        assert isinstance(matches, tuple) and matches
        assert matches[0]["tool_name"] == "file_query"
        assert _data(results["load-query"])["loaded_names"] == ("file_query",)
        query_data = _data(results["query-sales"])
        rows = query_data["rows"]
        assert isinstance(rows, tuple)
        assert tuple(dict(row) for row in rows) == (
            {"region": "north", "total_revenue": 15},
            {"region": "south", "total_revenue": 20},
        )
        assert results["query-sales"].sensitivity_provenance["authority"] == (
            "local_workspace_binding"
        )
        assert result.final_text is not None
        assert "15" in result.final_text and "20" in result.final_text
        assert "file_query" not in {tool.name for tool in provider.requests[0].tools}
        assert "file_query" in {tool.name for tool in provider.requests[2].tools}
        _assert_no_workspace_path(workspace, provider, transcript)
    finally:
        await agent.close()


async def test_connected_customer_schema_is_the_first_and_only_tool(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    database = tmp_path / "customers.sqlite"
    _database(database)
    provider = ToolboxAwareMockModelProvider(
        (
            _tool(
                "customer-schema",
                "catalog_schema",
                {
                    "query": "customer table columns",
                    "limit": 5,
                    "include_relationships": False,
                },
            ),
            _stop("The customer columns are customer_id, name, and segment."),
        ),
        provider_id="mock:phase5-customer-schema",
    )
    agent = await Agent.create(
        "phase5-customer-schema",
        root=tmp_path / "state",
        workspace=workspace,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="Customers"))
        result = await agent.run(
            PHASE5_ROUTES["customer_columns"][0],
            source_id=source.id,
        )
        transcript = await agent.transcript(result.run_id)
        _assert_route(transcript, "customer_columns")
        block = _results(transcript)["customer-schema"]
        assert not block.is_error
        encoded = canonical_json(block.output)
        assert all(name in encoded for name in ("customer_id", "name", "segment"))
        assert result.final_text is not None
        assert all(
            name in result.final_text for name in ("customer_id", "name", "segment")
        )
        assert result.artifacts == ()
        _assert_no_workspace_path(workspace, provider, transcript)
    finally:
        await agent.close()


class _OfflinePostgresSource:
    def __init__(self, agent_id: str) -> None:
        self.registration = SourceRegistration.build(
            agent_id=agent_id,
            adapter_id="postgresql",
            native_identity="postgresql://offline-phase5/customers",
            display_name="Offline PostgreSQL",
            configuration={},
            attached_at=NOW,
        )

    async def open(
        self,
        *,
        agent_id: str,
        attached_at: datetime,
        clock: Callable[[], datetime],
    ) -> ResourceAdapter:
        del clock
        assert agent_id == self.registration.agent_id
        assert attached_at == NOW
        registration = self.registration

        class _Adapter:
            @property
            def registration(self) -> SourceRegistration:
                return registration

            async def discover(self, request: DiscoveryRequest) -> DiscoveryResult:
                sync = CatalogSync(
                    id=request.sync_id,
                    agent_id=request.agent_id,
                    source_id=request.source_id,
                    adapter_id="postgresql",
                    status=CatalogSyncStatus.SUCCEEDED,
                    started_at=request.requested_at,
                    completed_at=NOW,
                    source_revision="catalog:phase5-offline",
                )
                return DiscoveryResult(
                    request=request,
                    snapshot=SourceCatalogSnapshot(
                        sync=sync,
                        resources=(),
                        revisions=(),
                    ),
                    completed_at=NOW,
                )

            async def inspect(self, resource: ResourceRef) -> ResourceSnapshot:
                raise AssertionError(resource)

            async def health(self) -> SourceHealth:
                raise AssertionError("attachment does not need a second health call")

            async def close(self) -> None:
                return None

        return _Adapter()


async def test_postgresql_query_is_direct_and_structured_failure_has_no_fallback(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    provider = ToolboxAwareMockModelProvider(
        (), provider_id="mock:phase5-postgres-query"
    )
    agent = await Agent.create(
        "phase5-postgres-query",
        root=tmp_path / "state",
        workspace=workspace,
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: NOW,
    )
    try:
        source = await agent.attach(_OfflinePostgresSource(agent.id))
        provider.replace_script(
            (
                _tool(
                    "postgres-query",
                    "data_query_postgresql",
                    {
                        "source_id": source.id,
                        "sql": "SELECT customer_id FROM customers",
                        "parameters": (),
                    },
                ),
                _stop(
                    "The PostgreSQL query could not run because no current customer "
                    "table schema is admitted."
                ),
            )
        )
        result = await agent.run(
            PHASE5_ROUTES["postgres_query"][0], source_id=source.id
        )
        transcript = await agent.transcript(result.run_id)
        _assert_route(transcript, "postgres_query")
        assert "data_query_postgresql" in {
            tool.name for tool in provider.requests[0].tools
        }
        failure = _results(transcript)["postgres-query"]
        assert failure.is_error
        error = failure.output.get("error")
        assert isinstance(error, Mapping)
        assert isinstance(error.get("code"), str)
        assert result.final_text is not None and "could not run" in result.final_text
        assert not any(
            name in agent._embedded._capabilities.tool_names
            for name in ("shell", "shell_run", "terminal", "terminal_run")
        )
        _assert_no_workspace_path(workspace, provider, transcript)
    finally:
        await agent.close()


async def test_mixed_file_and_source_queries_remain_separate_and_grounded(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    (workspace.root / "sales.csv").write_text(
        "region,revenue\nnorth,10\nsouth,20\nnorth,5\n",
        encoding="utf-8",
    )
    database = tmp_path / "customers.sqlite"
    _database(database)
    provider = ToolboxAwareMockModelProvider((), provider_id="mock:phase5-mixed")
    agent = await Agent.create(
        "phase5-mixed",
        root=tmp_path / "state",
        workspace=workspace,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="Customers"))
        provider.replace_script(
            (
                _tool(
                    "mixed-search",
                    "toolbox_search",
                    {
                        "query": "aggregate local sales csv",
                        "toolboxes": ["files"],
                        "limit": 5,
                    },
                ),
                _tool(
                    "mixed-load",
                    "toolbox_load",
                    {"tool_names": ["file_query"]},
                ),
                _tool(
                    "mixed-file-query",
                    "file_query",
                    {
                        "path_pattern": "sales.csv",
                        "sql": (
                            "SELECT region, SUM(revenue) AS total_revenue FROM data "
                            "GROUP BY region ORDER BY region"
                        ),
                    },
                ),
                _tool(
                    "mixed-schema",
                    "catalog_schema",
                    {
                        "query": "customer segment",
                        "limit": 5,
                        "include_relationships": False,
                    },
                ),
                _tool(
                    "mixed-source-query",
                    "data_query_sqlite",
                    {
                        "source_id": source.id,
                        "sql": (
                            "SELECT segment, COUNT(*) AS customer_count FROM customers "
                            "GROUP BY segment ORDER BY segment"
                        ),
                        "parameters": (),
                    },
                ),
                _stop(
                    "Local revenue is north 15 and south 20; connected customers "
                    "are enterprise 2 and self-serve 1."
                ),
            )
        )
        result = await agent.run(PHASE5_ROUTES["mixed_compare"][0], source_id=source.id)
        transcript = await agent.transcript(result.run_id)
        _assert_route(transcript, "mixed_compare")
        calls = {call.id: call for call in _calls(transcript)}
        assert calls["mixed-file-query"].arguments["path_pattern"] == "sales.csv"
        assert set(calls["mixed-file-query"].arguments) == {"path_pattern", "sql"}
        assert set(calls["mixed-source-query"].arguments) == {
            "source_id",
            "sql",
            "parameters",
        }
        assert "data" in str(calls["mixed-file-query"].arguments["sql"])
        assert "customers" not in str(calls["mixed-file-query"].arguments["sql"])
        assert "sales.csv" not in str(calls["mixed-source-query"].arguments["sql"])
        results = _results(transcript)
        assert results["mixed-file-query"].sensitivity_provenance["authority"] == (
            "local_workspace_binding"
        )
        assert results["mixed-source-query"].sensitivity_provenance["authority"] == (
            "current_admitted_resource_scope"
        )
        assert result.final_text is not None
        assert all(token in result.final_text for token in ("15", "20", "2", "1"))
        _assert_no_workspace_path(workspace, provider, transcript)
    finally:
        await agent.close()


async def test_files_only_with_connections_projects_and_uses_only_file_tools(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    (workspace.root / "Q4-forecast.md").write_text("Q4_ONLY_4B91\n", encoding="utf-8")
    database = tmp_path / "customers.sqlite"
    _database(database)
    provider = ToolboxAwareMockModelProvider(
        (
            _tool(
                "files-only-search",
                "file_search",
                {"query": "Q4", "mode": "paths"},
            ),
            _stop("The files-only forecast is Q4-forecast.md."),
        ),
        provider_id="mock:phase5-files-only",
    )
    agent = await Agent.create(
        "phase5-files-only",
        root=tmp_path / "state",
        workspace=workspace,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        await agent.attach(SQLiteSource(database, name="Customers"))
        result = await agent.run(PHASE5_ROUTES["files_only"][0], files_only=True)
        transcript = await agent.transcript(result.run_id)
        _assert_route(transcript, "files_only")
        initial = {tool.name for tool in provider.requests[0].tools}
        assert {"file_search", "file_read"} <= initial
        assert not any(
            name.startswith(("catalog_", "data_query_", "mcp_")) for name in initial
        )
        assert "Q4-forecast.md" in canonical_json(
            _results(transcript)["files-only-search"].output
        )
        _assert_no_workspace_path(workspace, provider, transcript)
    finally:
        await agent.close()


async def test_latest_log_search_then_tail_read_is_exact(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    logs = workspace.root / "logs"
    logs.mkdir()
    older = logs / "older.log"
    latest = logs / "latest.log"
    older.write_text("OLD_LOG\n", encoding="utf-8")
    latest.write_text("header\nLATEST_TAIL_9D11\n", encoding="utf-8")
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(latest, (1_800_000_000, 1_800_000_000))
    provider = ToolboxAwareMockModelProvider(
        (
            _tool(
                "latest-search",
                "file_search",
                {
                    "query": "*.log",
                    "mode": "paths",
                    "order_by": "modified_desc",
                },
            ),
            _tool(
                "latest-read",
                "file_read",
                {"path": "logs/latest.log", "position": "end"},
            ),
            _stop("The latest log ends with LATEST_TAIL_9D11."),
        ),
        provider_id="mock:phase5-latest-log",
    )
    agent = await Agent.create(
        "phase5-latest-log",
        root=tmp_path / "state",
        workspace=workspace,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        result = await agent.run(PHASE5_ROUTES["latest_log"][0])
        transcript = await agent.transcript(result.run_id)
        _assert_route(transcript, "latest_log")
        calls = {call.id: call for call in _calls(transcript)}
        assert calls["latest-search"].arguments["order_by"] == "modified_desc"
        assert dict(calls["latest-read"].arguments) == {
            "path": "logs/latest.log",
            "position": "end",
        }
        read = _results(transcript)["latest-read"]
        assert "LATEST_TAIL_9D11" in canonical_json(read.output)
        assert result.final_text is not None and "LATEST_TAIL_9D11" in result.final_text
        _assert_no_workspace_path(workspace, provider, transcript)
    finally:
        await agent.close()


async def test_markdown_report_searches_loads_creates_and_delivers_only_on_request(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    artifact_id = "artifact-00000000000000000000000000000001"
    provider = ToolboxAwareMockModelProvider(
        (
            _tool(
                "report-search-create",
                "toolbox_search",
                {
                    "query": "create markdown report document",
                    "toolboxes": ["artifacts"],
                    "limit": 5,
                },
            ),
            _tool(
                "report-load-create",
                "toolbox_load",
                {"tool_names": ["artifact_create_document"]},
            ),
            _tool(
                "report-create",
                "artifact_create_document",
                {
                    "format": "markdown",
                    "filename": "report.md",
                    "content": "# Report\n\nNorth: 15\nSouth: 20\n",
                },
            ),
            _tool(
                "report-search-save",
                "toolbox_search",
                {
                    "query": "save deliver local report",
                    "toolboxes": ["artifacts"],
                    "limit": 5,
                },
            ),
            _tool(
                "report-load-save",
                "toolbox_load",
                {"tool_names": ["artifact_save_local"]},
            ),
            _tool(
                "report-save",
                "artifact_save_local",
                {
                    "artifact_id": artifact_id,
                    "mode": "create_new",
                    "destination_id": "default",
                },
            ),
            _stop("The Markdown report was saved locally."),
        ),
        provider_id="mock:phase5-report",
    )
    agent = await Agent.create(
        "phase5-report",
        root=tmp_path / "state",
        workspace=workspace,
        downloads_directory=downloads,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
    )
    try:
        result = await agent.run(PHASE5_ROUTES["markdown_report"][0])
        transcript = await agent.transcript(result.run_id)
        _assert_route(transcript, "markdown_report")
        results = _results(transcript)
        create_matches = _data(results["report-search-create"])["matches"]
        save_matches = _data(results["report-search-save"])["matches"]
        assert create_matches[0]["tool_name"] == "artifact_create_document"  # type: ignore[index]
        assert save_matches[0]["tool_name"] == "artifact_save_local"  # type: ignore[index]
        save_request = provider.requests[5]
        save_surface = {tool.name for tool in save_request.tools}
        assert "artifact_save_local" in save_surface
        assert "artifact_create_document" not in save_surface
        assert tuple(item.artifact_id for item in result.artifacts) == (artifact_id,)
        assert tuple(item.artifact_id for item in result.artifact_deliveries) == (
            artifact_id,
        )
        assert (downloads / "report.md").read_text(encoding="utf-8") == (
            "# Report\n\nNorth: 15\nSouth: 20\n"
        )
        _assert_no_workspace_path(workspace, provider, transcript)
    finally:
        await agent.close()


class _EditRoutingProvider:
    provider_id = "mock:phase5-edit"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return False

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        results = _request_results(request)
        if "edit-read" not in results:
            return _tool("edit-read", "file_read", {"path": "config.yaml"})
        if "edit-search" not in results:
            return _tool(
                "edit-search",
                "toolbox_search",
                {
                    "query": "edit timeout config yaml text replace workspace",
                    "toolboxes": ["artifacts"],
                    "limit": 5,
                },
            )
        if "edit-load" not in results:
            return _tool(
                "edit-load",
                "toolbox_load",
                {"tool_names": ["artifact_edit_text"]},
            )
        if "edit-create" not in results:
            read_data = _data(results["edit-read"])
            binding = read_data.get("binding")
            assert isinstance(binding, str)
            return _tool(
                "edit-create",
                "artifact_edit_text",
                {
                    "binding": binding,
                    "replacements": [
                        {
                            "old_text": "timeout: 30",
                            "new_text": "timeout: 60",
                            "expected_occurrences": 1,
                        }
                    ],
                },
            )
        if "save-search" not in results:
            return _tool(
                "save-search",
                "toolbox_search",
                {
                    "query": "save replace bound workspace file",
                    "toolboxes": ["artifacts"],
                    "limit": 5,
                },
            )
        if "save-load" not in results:
            return _tool(
                "save-load",
                "toolbox_load",
                {"tool_names": ["artifact_save_local"]},
            )
        if "edit-save" not in results:
            artifact = results["edit-create"].output.get("artifact")
            assert isinstance(artifact, Mapping)
            artifact_id = artifact.get("artifact_id")
            assert isinstance(artifact_id, str)
            return _tool(
                "edit-save",
                "artifact_save_local",
                {"artifact_id": artifact_id, "mode": "replace_bound_file"},
            )
        return _stop("config.yaml now has timeout 60.")


async def test_text_edit_search_load_replacement_and_one_approval_are_cohesive(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    target = workspace.root / "config.yaml"
    target.write_text("service: demo\ntimeout: 30\n", encoding="utf-8")
    provider = _EditRoutingProvider()
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "phase5-edit",
        root=tmp_path / "state",
        workspace=workspace,
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approve,
        id_factory=_ids(),
    )
    try:
        result = await agent.run(PHASE5_ROUTES["text_edit"][0])
        transcript = await agent.transcript(result.run_id)
        _assert_route(transcript, "text_edit")
        assert target.read_text(encoding="utf-8") == "service: demo\ntimeout: 60\n"
        results = _results(transcript)
        edit_matches = _data(results["edit-search"])["matches"]
        save_matches = _data(results["save-search"])["matches"]
        assert edit_matches[0]["tool_name"] == "artifact_edit_text"  # type: ignore[index]
        assert save_matches[0]["tool_name"] == "artifact_save_local"  # type: ignore[index]
        assert len(approvals) == 1
        approval = approvals[0]
        assert approval.tool_name == "artifact_save_local"
        assert approval.arguments["mode"] == "replace_bound_file"
        assert approval.arguments["relative_path"] == "config.yaml"
        assert str(workspace.root) not in (approval.render_arguments_for_review() or "")

        edit_request = provider.requests[3]
        save_request = provider.requests[6]
        assert "artifact_edit_text" in {tool.name for tool in edit_request.tools}
        assert "artifact_save_local" not in {tool.name for tool in edit_request.tools}
        assert "artifact_save_local" in {tool.name for tool in save_request.tools}
        assert "artifact_edit_text" not in {tool.name for tool in save_request.tools}
        receipt = result.artifact_deliveries[0]
        assert receipt.saved_path == "config.yaml"
        assert receipt.relative_path == "config.yaml"
        assert result.final_text is not None and "60" in result.final_text
        _assert_no_workspace_path(workspace, provider, transcript)
    finally:
        await agent.close()


async def test_unsupported_workflow_has_no_invented_tool_or_terminal_fallback(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    provider = ToolboxAwareMockModelProvider(
        (
            _stop(
                "That image rewrite workflow is unsupported in this release; no "
                "shell or terminal command was run."
            ),
        ),
        provider_id="mock:phase5-unsupported",
    )
    agent = await Agent.create(
        "phase5-unsupported",
        root=tmp_path / "state",
        workspace=workspace,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        result = await agent.run(PHASE5_ROUTES["unsupported"][0])
        transcript = await agent.transcript(result.run_id)
        _assert_route(transcript, "unsupported")
        names = set(agent._embedded._capabilities.tool_names)
        assert names.isdisjoint(
            {"shell", "shell_run", "terminal", "terminal_run", "file_write"}
        )
        assert result.final_text is not None
        assert "unsupported" in result.final_text and "no shell" in result.final_text
        assert result.artifacts == ()
        assert result.artifact_deliveries == ()
    finally:
        await agent.close()


async def test_forged_terminal_call_is_rejected_without_execution_or_fallback(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    sentinel = workspace.root / "unchanged.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    provider = MockModelProvider(
        (
            _tool(
                "forged-terminal",
                "terminal_run",
                {"command": "rewrite every image"},
            ),
            _stop(
                "The requested workflow is unsupported; the unavailable terminal "
                "tool did not execute."
            ),
        ),
        provider_id="mock:phase5-forged-terminal",
    )
    agent = await Agent.create(
        "phase5-forged-terminal",
        root=tmp_path / "state",
        workspace=workspace,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        result = await agent.run(PHASE5_ROUTES["unsupported"][0])
        transcript = await agent.transcript(result.run_id)
        assert tuple(call.name for call in _calls(transcript)) == ("terminal_run",)
        rejected = _results(transcript)["forged-terminal"]
        assert rejected.is_error
        error = rejected.output.get("error")
        assert isinstance(error, Mapping)
        assert error["code"] == "tool_not_available"
        assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
        assert result.final_text is not None and "did not execute" in result.final_text
        assert result.artifacts == ()
        assert result.artifact_deliveries == ()
        _assert_no_workspace_path(workspace, provider, transcript)
    finally:
        await agent.close()


@dataclass(slots=True)
class _GeneratedExecutor:
    executor_id: str

    async def execute(self, request: ToolExecution) -> ToolOutput:
        return ToolOutput(
            kind="test.phase5.generated.output",
            data={"value": request.arguments.get("value", "ok")},
        )


def _definition_bytes(definitions: tuple[ToolDefinition, ...]) -> int:
    return len(
        canonical_json(
            [
                {
                    "name": item.name,
                    "description": item.description,
                    "input_schema": item.input_schema,
                }
                for item in definitions
            ]
        ).encode("utf-8")
    )


def _generated_runtime() -> tuple[CapabilityRuntime, LoopLimits]:
    limits = LoopLimits()
    capabilities: list[Capability] = []
    views: list[ToolView] = []
    executors: list[_GeneratedExecutor] = []
    toolboxes = tuple(ToolboxId)
    for index in range(limits.max_run_tool_catalog_entries):
        name = "file_query" if index == 511 else f"generated_tool_{index:03d}"
        executor = _GeneratedExecutor(f"test.phase5.generated.executor.{index:03d}")
        capability = Capability(
            id=f"test.phase5.generated.capability.{index:03d}",
            description=f"Execute generated bounded operation {index:03d}.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
            output_kind="test.phase5.generated.output",
            output_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            executor_id=executor.executor_id,
            access_mode=AccessMode.READ,
        )
        relevant = name == "file_query"
        toolbox_id = ToolboxId.FILES if relevant else toolboxes[index % len(toolboxes)]
        views.append(
            ToolView(
                name=name,
                capability_id=capability.id,
                description=capability.description,
                presentation=ToolPresentation(
                    toolbox_id=toolbox_id,
                    load_mode=(
                        ToolLoadMode.PINNED
                        if index < limits.max_pinned_tools
                        else ToolLoadMode.ON_DEMAND
                    ),
                    text_trust=ToolTextTrust.CODE,
                    summary=(
                        "Analyze one structured workspace CSV dataset by region."
                        if relevant
                        else f"Generated bounded operation {index:03d}."
                    ),
                    when_to_use=(
                        "Use for total revenue aggregation in a local sales CSV."
                        if relevant
                        else f"Use for generated fixture operation {index:03d}."
                    ),
                    keywords=(
                        ("file", "query", "csv", "revenue", "region", "aggregate")
                        if relevant
                        else ("generated", f"fixture{index:03d}")
                    ),
                ),
            )
        )
        capabilities.append(capability)
        executors.append(executor)

    domain = StaticTestDomain(
        tuple(capabilities),
        tuple(views),
        domain_owner_id="phase5_generated",
    )
    registry = CapabilityRegistry(
        declarations=(domain.declarations,),
        executors=tuple(executors),
    )
    return CapabilityRuntime(registry, (domain,), limits=limits), limits


async def _execute_control(
    runtime: CapabilityRuntime,
    run: RunInput,
    projection: object,
    call: ToolCall,
    *,
    messages: tuple[CanonicalMessage, ...] = (),
) -> ToolResultBlock:
    outcome = await runtime.execute_all(
        run,
        (call,),
        projection=projection,
        messages=messages,
        sensitivity=ModelSensitivity.INTERNAL,
    )
    return outcome.ordered_results[0]


def _append_exchange(
    messages: tuple[CanonicalMessage, ...],
    call: ToolCall,
    result: ToolResultBlock,
) -> tuple[CanonicalMessage, ...]:
    return (
        *messages,
        CanonicalMessage(MessageRole.ASSISTANT, tool_calls=(call,)),
        CanonicalMessage(MessageRole.TOOL, content=(result,)),
    )


async def test_generated_maximum_catalog_is_bounded_searchable_and_replacing(
    tmp_path: Path,
) -> None:
    runtime, limits = _generated_runtime()
    context_provider = ToolboxAwareMockModelProvider(
        (), provider_id="mock:phase5-maximum-catalog"
    )
    workspace = _workspace(tmp_path)
    agent = await Agent.create(
        "phase5-maximum-catalog",
        root=tmp_path / "state",
        workspace=workspace,
        model=context_provider,
        model_profile=_profile(context_provider),
    )
    try:
        run = RunInput(
            id="run-phase5-maximum-catalog",
            agent_id=agent.id,
            message="Total revenue by region in sales.csv",
            created_at=NOW,
            conversation_id="conversation-phase5-maximum-catalog",
        )
        start = run.start_message()
        catalog = await runtime.prepare_run(run)
        initial = runtime.project(catalog, (start,))

        builder = agent._embedded._data_context_builder
        assert builder is not None
        snapshot = await builder.prepare(run, (start,), catalog)
        request = builder.project(
            snapshot,
            (start,),
            step=1,
            tool_context=initial,
        )
        system_text = "\n".join(
            block.text
            for message in request.messages
            if message.role is MessageRole.SYSTEM
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert request.tools == initial.provider_definitions
        assert "generated_tool_500" not in system_text
        assert "file_query" not in system_text
        assert '"on_demand_count":' in system_text

        assert len(catalog.entries) == limits.max_run_tool_catalog_entries == 512
        assert catalog.aggregate_bytes <= limits.max_run_tool_catalog_bytes
        assert len(catalog.toolbox_manifest) == limits.max_toolbox_manifest_entries == 5
        assert catalog.manifest_bytes <= limits.max_toolbox_manifest_bytes
        assert (catalog.manifest_bytes + 3) // 4 <= limits.max_toolbox_manifest_tokens
        assert len(catalog.pinned_provider_definitions) == limits.max_pinned_tools
        assert _definition_bytes(catalog.pinned_provider_definitions) <= (
            limits.max_pinned_tool_definition_bytes
        )
        assert len(initial.provider_definitions) == limits.max_pinned_tools + 2
        assert len(initial.provider_definitions) <= limits.max_step_tools
        assert _definition_bytes(initial.provider_definitions) <= (
            limits.max_step_tool_definition_bytes
        )

        manifest_json = canonical_json(
            [
                {
                    "toolbox_id": item.toolbox_id.value,
                    "label": item.label,
                    "summary": item.summary,
                    "pinned_count": item.pinned_count,
                    "on_demand_count": item.on_demand_count,
                }
                for item in catalog.toolbox_manifest
            ]
        )
        assert "generated_tool_" not in manifest_json
        assert "file_query" not in manifest_json
        assert len(manifest_json.encode("utf-8")) == catalog.manifest_bytes

        search = ToolCall(
            id="maximum-search",
            name="toolbox_search",
            arguments={
                "query": "total revenue by region sales csv",
                "toolboxes": ["files"],
                "limit": 5,
            },
        )
        first = await _execute_control(
            runtime,
            run,
            initial,
            search,
            messages=(start,),
        )
        second = await _execute_control(
            runtime,
            run,
            initial,
            search,
            messages=(start,),
        )
        assert first.output == second.output
        assert len(canonical_json(first.output).encode("utf-8")) <= (
            limits.max_toolbox_search_result_bytes
        )
        raw_matches = _data(first)["matches"]
        assert isinstance(raw_matches, tuple)
        matches: list[Mapping[str, object]] = []
        for match in raw_matches:
            assert isinstance(match, Mapping)
            matches.append(match)
        assert 0 < len(matches) <= 5
        assert matches[0]["tool_name"] == "file_query"
        assert all("domain_owner_id" not in match for match in matches)

        load_query = ToolCall(
            id="maximum-load-query",
            name="toolbox_load",
            arguments={"tool_names": ["file_query"]},
        )
        load_query_result = await _execute_control(
            runtime,
            run,
            initial,
            load_query,
            messages=(start,),
        )
        assert not load_query_result.is_error
        messages = _append_exchange((start,), load_query, load_query_result)
        query_projection = runtime.project(catalog, messages)
        assert tuple(entry.view.name for entry in query_projection.loaded_entries) == (
            "file_query",
        )
        assert query_projection.loaded_definition_bytes <= (
            limits.max_loaded_tool_definition_bytes
        )
        assert _definition_bytes(query_projection.provider_definitions) <= (
            limits.max_step_tool_definition_bytes
        )

        replacement_name = "generated_tool_032"
        load_replacement = ToolCall(
            id="maximum-load-replacement",
            name="toolbox_load",
            arguments={"tool_names": [replacement_name]},
        )
        replacement_result = await _execute_control(
            runtime,
            run,
            query_projection,
            load_replacement,
            messages=messages,
        )
        assert not replacement_result.is_error
        replaced_messages = _append_exchange(
            messages,
            load_replacement,
            replacement_result,
        )
        replaced = runtime.project(catalog, replaced_messages)
        assert tuple(entry.view.name for entry in replaced.loaded_entries) == (
            replacement_name,
        )
        names = {definition.name for definition in replaced.provider_definitions}
        assert replacement_name in names
        assert "file_query" not in names
        assert len(names) == limits.max_pinned_tools + 2 + 1
        assert _definition_bytes(replaced.provider_definitions) <= (
            limits.max_step_tool_definition_bytes
        )
    finally:
        await agent.close()
