"""Phase 4 contracts for bounded structured local-file queries."""

from __future__ import annotations

import asyncio
import builtins
import json
import multiprocessing
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from daita import Agent, LocalWorkspace
from daita.adapters import local_file_query as query_module
from daita.adapters.local_file_query import (
    LocalFileQueryError,
    LocalFileQueryLimits,
)
from daita.adapters.local_workspace import (
    LocalWorkspaceBackend,
    LocalWorkspaceError,
    LocalWorkspaceLimits,
)
from daita.domains.data import (
    LOCAL_FILE_QUERY_CAPABILITY_ID,
    LOCAL_FILE_QUERY_EXECUTOR_ID,
    LOCAL_FILE_QUERY_TOOL_NAME,
)
from daita.domains.data.sql import (
    DuckDBReadValidationError,
    validate_duckdb_read,
)
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider


def _call(call_id: str, name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def _stop(text: str = "done") -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=64_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


async def _backend(
    tmp_path: Path,
    *,
    workspace_limits: LocalWorkspaceLimits | None = None,
    query_limits: LocalFileQueryLimits | None = None,
) -> tuple[LocalWorkspaceBackend, Path, Path]:
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    home = state / "agent"
    workspace.mkdir(parents=True)
    home.mkdir(parents=True)
    backend = await LocalWorkspaceBackend.open(
        LocalWorkspace(workspace),
        agent_root=state,
        agent_home=home,
        limits=workspace_limits,
        query_limits=query_limits,
    )
    return backend, workspace, home


async def _query(
    backend: LocalWorkspaceBackend,
    pattern: str,
    sql: str = (
        "SELECT region, SUM(amount) AS total FROM data "
        "GROUP BY region ORDER BY region"
    ),
):
    validated = validate_duckdb_read(sql)
    return await backend.query(
        run_id="phase4-query",
        path_pattern=pattern,
        canonical_sql=validated.canonical_sql,
        sql_fingerprint=validated.sql_fingerprint,
    )


@pytest.mark.parametrize(
    ("sql", "accepted"),
    (
        ("SELECT region, SUM(amount) FROM data GROUP BY region", True),
        ("SELECT * FROM data WHERE amount > 10 ORDER BY amount LIMIT 5", True),
        (
            "WITH filtered AS (SELECT * FROM data WHERE amount > 0) SELECT COUNT(*) FROM filtered",
            True,
        ),
        ("SELECT AVG(amount) FROM (SELECT amount FROM data) AS selected", True),
        ("ATTACH '/tmp/evil.db' AS evil", False),
        ("COPY data TO '/tmp/evil.csv'", False),
        ("EXPORT DATABASE '/tmp/evil'", False),
        ("IMPORT DATABASE '/tmp/evil'", False),
        ("PRAGMA version", False),
        ("INSTALL httpfs", False),
        ("LOAD httpfs", False),
        ("CREATE TABLE evil AS SELECT * FROM data", False),
        ("CREATE PERSISTENT SECRET x (TYPE s3, KEY_ID 'x')", False),
        ("INSERT INTO data VALUES (1)", False),
        ("UPDATE data SET amount = 0", False),
        ("DELETE FROM data", False),
        ("SET enable_external_access = true", False),
        ("SELECT * FROM data; SELECT 1", False),
        ("SELECT * FROM other", False),
        ("SELECT * FROM main.data", False),
        ("SELECT * FROM data a JOIN data b USING (id)", False),
        ("SELECT * FROM read_csv('/etc/hosts')", False),
        ("SELECT * FROM read_json('https://example.test/a.json')", False),
        ("SELECT * FROM parquet_scan('/tmp/a.parquet')", False),
        ("SELECT * FROM glob('/tmp/*')", False),
        ("SELECT * FROM range(10)", False),
        ("SELECT * FROM query_table('data')", False),
        ("SELECT * FROM data JOIN (VALUES (1)) v(x) ON true", False),
        ("SELECT * FROM data JOIN (SELECT 1) v ON true", False),
        ("SELECT * FROM data UNION ALL SELECT * FROM data", False),
        ("SELECT * FROM data WHERE region = 'https://example.test'", False),
        ("SELECT * FROM data WHERE region = '/etc/passwd'", False),
        ("SELECT 1", False),
    ),
)
def test_duckdb_sql_validator_accepts_only_one_canonical_data_read(
    sql: str, accepted: bool
) -> None:
    if accepted:
        result = validate_duckdb_read(sql)
        assert result.canonical_sql
        assert result.sql_fingerprint.startswith("sha256:")
        assert validate_duckdb_read(result.canonical_sql) == result
    else:
        with pytest.raises(DuckDBReadValidationError):
            validate_duckdb_read(sql)


def test_production_query_limits_lock_the_selected_buffer_and_product_bounds() -> None:
    limits = LocalFileQueryLimits()
    assert limits.duckdb_memory_bytes == 256 * 1_024 * 1_024
    assert limits.max_query_seconds == 30.0
    assert limits.max_result_rows == 100
    assert limits.max_result_bytes == 48 * 1_024
    assert limits.max_spill_bytes == 2 * 1_024 * 1_024 * 1_024
    assert limits.max_worker_rss_bytes == 1_536 * 1_024 * 1_024
    assert limits.duckdb_threads == 2


async def test_exact_manifest_is_descriptor_bound_complete_and_bounded(
    tmp_path: Path,
) -> None:
    backend, workspace, _home = await _backend(tmp_path)
    parts = workspace / "parts"
    parts.mkdir()
    for index in range(300):
        (parts / f"orders-{index:03d}.csv").write_text(
            f"id,region,amount\n{index},north,{index}.5\n", encoding="utf-8"
        )
    try:
        manifest = await backend.bind_query_manifest(
            run_id="manifest", path_pattern="parts/orders-*.csv"
        )
        try:
            assert len(manifest.bindings) == 300
            assert manifest.input_bytes == sum(
                item.size_bytes for item in manifest.bindings
            )
            assert manifest.encoded_bytes <= 256 * 1_024
            assert len({item.relative_path for item in manifest.bindings}) == 300
            assert all(item.descriptor >= 0 for item in manifest.bindings)
            assert all(
                item.physical_revision.startswith("sha256:")
                for item in manifest.bindings
            )
            assert str(workspace) not in canonical_manifest(manifest)
            manifest.revalidate()
            validated = validate_duckdb_read("SELECT COUNT(*) AS rows FROM data")
            request = query_module._request_mapping(
                manifest,
                validated.canonical_sql,
                validated.sql_fingerprint,
                LocalFileQueryLimits(),
            )
            assert (
                query_module._validate_request(
                    json.dumps(request, separators=(",", ":")).encode("utf-8")
                )["manifest_sha256"]
                == manifest.manifest_sha256
            )
            request["manifest_sha256"] = "sha256:" + "0" * 64
            with pytest.raises(RuntimeError):
                query_module._validate_request(
                    json.dumps(request, separators=(",", ":")).encode("utf-8")
                )
        finally:
            manifest.close()
    finally:
        await backend.close()


def canonical_manifest(manifest) -> str:
    return json.dumps(manifest.provenance_mapping(), default=list, sort_keys=True)


async def test_manifest_rejects_empty_mixed_unsupported_symlink_and_bounds(
    tmp_path: Path,
) -> None:
    limits = LocalWorkspaceLimits(
        max_query_files=2,
        max_query_input_bytes=128,
        max_query_manifest_bytes=1_024,
    )
    backend, workspace, _home = await _backend(tmp_path, workspace_limits=limits)
    mixed = workspace / "mixed"
    many = workspace / "many"
    links = workspace / "links"
    mixed.mkdir()
    many.mkdir()
    links.mkdir()
    (mixed / "one.csv").write_text("id\n1\n", encoding="utf-8")
    (mixed / "two.json").write_text('[{"id":2}]', encoding="utf-8")
    for index in range(3):
        (many / f"part-{index}.csv").write_text(f"id\n{index}\n", encoding="utf-8")
    (workspace / "notes.txt").write_text("not structured", encoding="utf-8")
    big = workspace / "big.csv"
    big.write_bytes(b"id\n" + b"1" * 129)
    long = workspace / "long"
    long.mkdir()
    for index in range(2):
        (long / (f"part-{index}-" + "x" * 180 + ".csv")).write_text(
            "id\n1\n", encoding="utf-8"
        )
    (links / "target.csv").write_text("id\n1\n", encoding="utf-8")
    (links / "link.csv").symlink_to(links / "target.csv")
    try:
        cases = (
            ("missing-*.csv", "file_pattern_empty"),
            ("mixed/*", "format_unsupported"),
            ("notes.txt", "format_unsupported"),
            ("links/link.csv", "symlink_not_allowed"),
            ("many/*.csv", "file_pattern_too_broad"),
            ("big.csv", "file_pattern_too_broad"),
            ("long/*.csv", "file_pattern_too_broad"),
        )
        for pattern, code in cases:
            with pytest.raises(LocalWorkspaceError) as failure:
                await backend.bind_query_manifest(
                    run_id="manifest-failure", path_pattern=pattern
                )
            assert failure.value.code == code
        with pytest.raises(LocalWorkspaceError) as absolute:
            await backend.bind_query_manifest(
                run_id="manifest-failure", path_pattern="/etc/*.csv"
            )
        assert absolute.value.code == "path_invalid"
    finally:
        await backend.close()


async def test_manifest_revision_drift_fails_before_query_execution(
    tmp_path: Path,
) -> None:
    backend, workspace, _home = await _backend(tmp_path)
    target = workspace / "orders.csv"
    target.write_text("id\n1\n", encoding="utf-8")
    manifest = await backend.bind_query_manifest(
        run_id="revision-drift", path_pattern="orders.csv"
    )
    try:
        target.write_text("id\n1\n2\n", encoding="utf-8")
        with pytest.raises(LocalWorkspaceError) as failure:
            manifest.revalidate()
        assert failure.value.code == "file_changed"
    finally:
        manifest.close()
        await backend.close()


async def test_csv_tsv_json_ndjson_and_parquet_aggregate_in_private_workers(
    tmp_path: Path,
) -> None:
    import duckdb

    backend, workspace, _home = await _backend(tmp_path)
    fixtures = {
        "csv": "region,amount\nnorth,10\nsouth,20\n",
        "tsv": "region\tamount\nnorth\t10\nsouth\t20\n",
        "json": '[{"region":"north","amount":10},{"region":"south","amount":20}]',
        "ndjson": (
            '{"region":"north","amount":10}\n' '{"region":"south","amount":20}\n'
        ),
    }
    for suffix, content in fixtures.items():
        (workspace / f"orders.{suffix}").write_text(content, encoding="utf-8")
    parquet = workspace / "orders.parquet"
    generator = duckdb.connect(":memory:")
    try:
        generator.execute(
            "COPY (SELECT * FROM VALUES ('north', 10), ('south', 20) "
            "orders(region, amount)) TO ? (FORMAT PARQUET)",
            [str(parquet)],
        )
    finally:
        generator.close()
    try:
        for suffix in ("csv", "tsv", "json", "ndjson", "parquet"):
            result = await _query(backend, f"orders.{suffix}")
            rows = result.data["rows"]
            assert isinstance(rows, tuple)
            assert [row["region"] for row in rows] == ["north", "south"]
            assert [row["total"] for row in rows] == [10, 20]
            assert result.data["input_file_count"] == 1
            assert result.data["truncated"] is False
            provenance = result.sensitivity_provenance
            assert provenance["authority"] == "local_workspace_binding"
            assert len(provenance["bindings"]) == 1
    finally:
        await backend.close()


async def test_incompatible_partition_schema_is_rejected_and_scratch_is_cleaned(
    tmp_path: Path,
) -> None:
    backend, workspace, home = await _backend(tmp_path)
    (workspace / "one.csv").write_text("id,amount\n1,2\n", encoding="utf-8")
    (workspace / "two.csv").write_text("id,amount,extra\n2,3,x\n", encoding="utf-8")
    try:
        with pytest.raises(LocalFileQueryError) as failure:
            await _query(backend, "*.csv", "SELECT COUNT(*) AS rows FROM data")
        assert failure.value.code == "file_query_invalid"
        scratch = home / "file-query-scratch"
        assert not scratch.exists() or not tuple(scratch.iterdir())
    finally:
        await backend.close()


async def test_malformed_csv_json_and_parquet_fail_as_bounded_results(
    tmp_path: Path,
) -> None:
    backend, workspace, home = await _backend(tmp_path)
    (workspace / "broken.csv").write_bytes(b"value\n\xff\xfe\n")
    (workspace / "broken.json").write_text('[{"value": 1}', encoding="utf-8")
    (workspace / "broken.parquet").write_bytes(b"not-a-parquet-file")
    try:
        for name in ("broken.csv", "broken.json", "broken.parquet"):
            with pytest.raises(LocalFileQueryError) as failure:
                await _query(backend, name, "SELECT COUNT(*) AS rows FROM data")
            assert failure.value.code == "file_query_invalid"
            assert str(workspace) not in failure.value.message
        scratch = home / "file-query-scratch"
        assert not scratch.exists() or not tuple(scratch.iterdir())
    finally:
        await backend.close()


async def test_result_row_byte_and_column_bounds_are_enforced(tmp_path: Path) -> None:
    backend, workspace, _home = await _backend(tmp_path)
    rows = "id,payload\n" + "".join(f"{index},{'x' * 800}\n" for index in range(150))
    (workspace / "rows.csv").write_text(rows, encoding="utf-8")
    wide_columns = [f"column_{index}" for index in range(129)]
    (workspace / "wide.csv").write_text(
        ",".join(wide_columns) + "\n" + ",".join("1" for _ in wide_columns) + "\n",
        encoding="utf-8",
    )
    try:
        limited = await _query(
            backend,
            "rows.csv",
            "SELECT id, payload FROM data ORDER BY id",
        )
        assert limited.data["truncated"] is True
        assert "byte_limit" in limited.data["truncation_reasons"]
        assert limited.data["returned_rows"] < 100
        assert limited.data["utf8_bytes"] <= 48 * 1_024
        with pytest.raises(LocalFileQueryError) as columns:
            await _query(backend, "wide.csv", "SELECT * FROM data")
        assert columns.value.code == "file_query_limited"
    finally:
        await backend.close()


def test_security_profile_is_ordered_locked_local_only_and_httpfs_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb

    allowed = tmp_path / "allowed.csv"
    blocked = tmp_path / "blocked.csv"
    scratch = tmp_path / "scratch"
    allowed.write_text("value\n1\n", encoding="utf-8")
    blocked.write_text("value\n2\n", encoding="utf-8")
    scratch.mkdir()
    descriptor = os.open(allowed, os.O_RDONLY)
    path = f"/dev/fd/{descriptor}"
    request = {
        "threads": 2,
        "memory_bytes": 256 * 1_024 * 1_024,
        "spill_bytes": 2 * 1_024 * 1_024 * 1_024,
    }
    observed_order: list[str] = []
    original = query_module._set

    def record(connection, name: str, value: str) -> None:
        observed_order.append(name)
        original(connection, name, value)

    monkeypatch.setattr(query_module, "_set", record)
    connection = duckdb.connect(":memory:")
    try:
        query_module._apply_security_profile(connection, (path,), scratch, request)
        assert observed_order.index("allowed_paths") < observed_order.index("threads")
        assert observed_order.index("allowed_directories") < observed_order.index(
            "memory_limit"
        )
        assert observed_order.index("allow_persistent_secrets") < observed_order.index(
            "enable_external_access"
        )
        assert observed_order[-2:] == ["allowed_configs", "lock_configuration"]
        observed_paths = connection.execute(
            "SELECT current_setting('allowed_paths')"
        ).fetchone()
        assert observed_paths is not None
        assert isinstance(observed_paths[0], list)
        assert len(observed_paths[0]) == 1
        assert observed_paths[0][0].startswith("/dev/fd/")
        assert connection.execute("SELECT * FROM read_csv(?)", [path]).fetchall() == [
            (1,)
        ]
        prohibited: tuple[tuple[str, list[str]], ...] = (
            ("SELECT * FROM read_csv(?)", [str(blocked)]),
            ("SELECT * FROM read_text('/etc/hosts')", []),
            ("SELECT * FROM read_csv('https://example.test/a.csv')", []),
            ("INSTALL httpfs", []),
            ("LOAD httpfs", []),
            (
                "CREATE PERSISTENT SECRET phase4 (TYPE HTTP, BEARER_TOKEN 'x')",
                [],
            ),
            ("SET enable_external_access = true", []),
            ("SET lock_configuration = false", []),
        )
        for sql, parameters in prohibited:
            with pytest.raises(duckdb.Error):
                connection.execute(sql, parameters).fetchall()
    finally:
        connection.close()
        os.close(descriptor)


def test_security_profile_rejects_any_extra_retained_allowed_path(
    tmp_path: Path,
) -> None:
    import duckdb

    allowed = tmp_path / "allowed.csv"
    extra = tmp_path / "extra.csv"
    scratch = tmp_path / "scratch"
    allowed.write_text("value\n1\n", encoding="utf-8")
    extra.write_text("value\n2\n", encoding="utf-8")
    scratch.mkdir()
    descriptor = os.open(allowed, os.O_RDONLY)
    path = f"/dev/fd/{descriptor}"
    connection = duckdb.connect(":memory:")

    class ExtraAllowedPathCursor:
        def __init__(self, row: tuple[object, ...]) -> None:
            self._row = row

        def fetchone(self) -> tuple[object, ...]:
            return self._row

    class ExtraAllowedPathConnection:
        def __init__(self) -> None:
            self._allowed_path_reads = 0

        def execute(self, sql: str, parameters: list[object] | None = None) -> object:
            cursor = (
                connection.execute(sql)
                if parameters is None
                else connection.execute(sql, parameters)
            )
            if sql == "SELECT current_setting('allowed_paths')":
                self._allowed_path_reads += 1
                if self._allowed_path_reads == 1:
                    return cursor
                observed = cursor.fetchone()
                assert observed is not None
                assert isinstance(observed[0], list)
                return ExtraAllowedPathCursor(([*observed[0], str(extra)],))
            return cursor

    request = {
        "threads": 2,
        "memory_bytes": 256 * 1_024 * 1_024,
        "spill_bytes": 2 * 1_024 * 1_024 * 1_024,
    }
    try:
        with pytest.raises(query_module._WorkerFailure) as failure:
            query_module._apply_security_profile(
                ExtraAllowedPathConnection(), (path,), scratch, request
            )
        assert failure.value.code == "dependency_unavailable"
    finally:
        connection.close()
        os.close(descriptor)


@pytest.mark.parametrize(
    ("observer_name", "limit_name", "detail_name"),
    (
        ("_process_rss_bytes", "max_worker_rss_bytes", "rss_limit_bytes"),
        ("_directory_bytes", "max_spill_bytes", "spill_limit_bytes"),
    ),
)
async def test_rss_and_spill_overruns_reap_worker_clean_scratch_and_allow_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observer_name: str,
    limit_name: str,
    detail_name: str,
) -> None:
    limits = LocalFileQueryLimits()
    backend, workspace, home = await _backend(tmp_path, query_limits=limits)
    (workspace / "orders.csv").write_text("region,amount\nnorth,1\n", encoding="utf-8")
    before = {child.pid for child in multiprocessing.active_children()}
    limit = getattr(limits, limit_name)
    try:
        with monkeypatch.context() as overrun:
            overrun.setattr(query_module, observer_name, lambda _value: limit + 1)
            with pytest.raises(LocalFileQueryError) as failure:
                await _query(backend, "orders.csv")
        assert failure.value.code == "file_query_limited"
        assert failure.value.details[detail_name] == limit
        scratch = home / "file-query-scratch"
        assert not scratch.exists() or not tuple(scratch.iterdir())
        assert {child.pid for child in multiprocessing.active_children()} <= before

        recovered = await _query(backend, "orders.csv")
        recovered_rows = recovered.data["rows"]
        assert isinstance(recovered_rows, tuple)
        assert len(recovered_rows) == 1
        assert recovered_rows[0]["region"] == "north"
        assert recovered_rows[0]["total"] == 1
    finally:
        await backend.close()
    assert {child.pid for child in multiprocessing.active_children()} <= before


async def test_timeout_and_cancellation_terminate_workers_and_remove_scratch(
    tmp_path: Path,
) -> None:
    timeout_limits = LocalFileQueryLimits(max_query_seconds=0.01)
    backend, workspace, home = await _backend(
        tmp_path / "timeout", query_limits=timeout_limits
    )
    (workspace / "orders.csv").write_text("region,amount\nnorth,1\n", encoding="utf-8")
    before = {child.pid for child in multiprocessing.active_children()}
    try:
        with pytest.raises(LocalFileQueryError) as timeout:
            await _query(backend, "orders.csv")
        assert timeout.value.code == "file_query_timeout"
        scratch = home / "file-query-scratch"
        assert not scratch.exists() or not tuple(scratch.iterdir())
    finally:
        await backend.close()

    cancelled_backend, cancelled_workspace, cancelled_home = await _backend(
        tmp_path / "cancelled"
    )
    (cancelled_workspace / "orders.csv").write_text(
        "region,amount\nnorth,1\n", encoding="utf-8"
    )
    task = asyncio.create_task(_query(cancelled_backend, "orders.csv"))
    await asyncio.sleep(0.01)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        scratch = cancelled_home / "file-query-scratch"
        assert not scratch.exists() or not tuple(scratch.iterdir())
    finally:
        await cancelled_backend.close()
    assert {child.pid for child in multiprocessing.active_children()} <= before


def test_duckdb_is_lazy_and_missing_dependency_has_repair_guidance(
    tmp_path: Path,
) -> None:
    script = """
import asyncio, sys
from pathlib import Path
from daita.hosting.embedded import EmbeddedAgent
assert 'duckdb' not in sys.modules
async def check():
    root = Path(sys.argv[1])
    agent = await EmbeddedAgent.create('hosted-lazy', hosted=True, root=root)
    try:
        assert 'duckdb' not in sys.modules
        assert 'file_query' not in agent._capabilities.tool_names
    finally:
        await agent.close()
asyncio.run(check())
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "hosted")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    original_import = builtins.__import__

    def unavailable(name: str, *args, **kwargs):
        if name == "duckdb":
            raise ImportError("injected missing DuckDB")
        return original_import(name, *args, **kwargs)

    try:
        builtins.__import__ = unavailable
        with pytest.raises(ImportError, match="pipx reinstall daita-agents"):
            query_module._load_duckdb()
    finally:
        builtins.__import__ = original_import


async def test_public_agent_loads_and_executes_file_query_through_one_runtime(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "orders.csv").write_text(
        "region,amount\nnorth,10\nsouth,20\nnorth,5\n", encoding="utf-8"
    )
    provider = MockModelProvider(
        (
            _call(
                "load-query",
                "toolbox_load",
                {"tool_names": [LOCAL_FILE_QUERY_TOOL_NAME]},
            ),
            _call(
                "query",
                LOCAL_FILE_QUERY_TOOL_NAME,
                {
                    "path_pattern": "orders.csv",
                    "sql": (
                        "SELECT region, SUM(amount) AS total FROM data "
                        "GROUP BY region ORDER BY region"
                    ),
                },
            ),
            _stop("north totals 15"),
        ),
        provider_id="mock:phase4-public",
    )
    agent = await Agent.create(
        "phase4-public",
        workspace=LocalWorkspace(workspace),
        root=tmp_path / "state",
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        result = await agent.run("Total sales by region in orders.csv")
        assert result.final_text == "north totals 15"
        assert LOCAL_FILE_QUERY_TOOL_NAME not in {
            tool.name for tool in provider.requests[0].tools
        }
        assert LOCAL_FILE_QUERY_TOOL_NAME in {
            tool.name for tool in provider.requests[1].tools
        }
        view, capability, owner = agent._embedded._capabilities.resolve_tool_owner(
            LOCAL_FILE_QUERY_TOOL_NAME
        )
        assert capability.id == LOCAL_FILE_QUERY_CAPABILITY_ID
        assert capability.executor_id == LOCAL_FILE_QUERY_EXECUTOR_ID
        assert owner == "data"
        assert view.presentation.load_mode.value == "on_demand"
        transcript = await agent.transcript(result.run_id)
        query_result = next(
            block
            for message in transcript.messages
            if message.role is MessageRole.TOOL
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "query"
        )
        assert not query_result.is_error
        data = query_result.output["data"]
        assert isinstance(data, Mapping)
        assert data["input_file_count"] == 1
        assert data["returned_rows"] == 2
        assert query_result.sensitivity_provenance["authority"] == (
            "local_workspace_binding"
        )
        bindings = query_result.sensitivity_provenance["bindings"]
        assert isinstance(bindings, tuple)
        assert len(bindings) == 1
    finally:
        await agent.close()


async def test_invalid_file_query_sql_fails_before_workspace_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "orders.csv").write_text("id\n1\n", encoding="utf-8")
    provider = MockModelProvider(
        (
            _call(
                "load-query",
                "toolbox_load",
                {"tool_names": [LOCAL_FILE_QUERY_TOOL_NAME]},
            ),
            _call(
                "invalid-query",
                LOCAL_FILE_QUERY_TOOL_NAME,
                {
                    "path_pattern": "orders.csv",
                    "sql": "SELECT * FROM read_csv('/etc/hosts')",
                },
            ),
            _stop(),
        ),
        provider_id="mock:phase4-invalid-sql",
    )
    agent = await Agent.create(
        "phase4-invalid-sql",
        workspace=LocalWorkspace(workspace),
        root=tmp_path / "state",
        model=provider,
        model_profile=_profile(provider),
    )

    async def unexpected_binding(**_arguments):
        raise AssertionError("invalid SQL reached workspace manifest binding")

    assert agent._embedded._workspace_backend is not None
    monkeypatch.setattr(
        agent._embedded._workspace_backend,
        "bind_query_manifest",
        unexpected_binding,
    )
    try:
        result = await agent.run("Read an arbitrary host file")
        transcript = await agent.transcript(result.run_id)
        rejected = next(
            block
            for message in transcript.messages
            if message.role is MessageRole.TOOL
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "invalid-query"
        )
        assert rejected.is_error
        error = rejected.output["error"]
        assert isinstance(error, Mapping)
        assert error["code"] == "file_query_invalid"
    finally:
        await agent.close()


async def test_hosted_registry_contains_no_file_query_authority(tmp_path: Path) -> None:
    from daita.hosting.embedded import EmbeddedAgent

    hosted = await EmbeddedAgent.create("phase4-hosted", hosted=True, root=tmp_path)
    try:
        assert LOCAL_FILE_QUERY_TOOL_NAME not in hosted._capabilities.tool_names
        assert LOCAL_FILE_QUERY_CAPABILITY_ID not in hosted._capabilities._capabilities
        assert LOCAL_FILE_QUERY_EXECUTOR_ID not in hosted._capabilities._executors
    finally:
        await hosted.close()
