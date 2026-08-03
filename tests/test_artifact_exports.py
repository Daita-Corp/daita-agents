from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import threading
import time as time_module
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest

from daita import Agent, SQLiteSource
from daita._json import canonical_json
from daita.adapters import postgresql_query as postgresql_query_module
from daita.adapters import sqlite_query as sqlite_query_module
from daita.adapters.models import SourceRegistration
from daita.artifacts.models import (
    ArtifactAuthorship,
    ArtifactError,
    artifact_delivery_receipt_to_mapping,
    artifact_ref_to_mapping,
)
from daita.artifacts.renderers import (
    MAX_CSV_BYTES,
    MAX_CSV_COLUMNS,
    MAX_CSV_ROWS,
    MAX_CSV_SECONDS,
    ExactCsvRenderer,
    render_exact_csv,
)
from daita.catalog.models import Sensitivity
from daita.domains.data.sql import ResourceSchema
from daita.domains.data.export_capabilities import (
    POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID,
    POSTGRESQL_TABULAR_EXPORT_TOOL_NAME,
    SQLITE_TABULAR_EXPORT_CAPABILITY_ID,
    SQLITE_TABULAR_EXPORT_TOOL_NAME,
    artifact_extension_declarations,
)
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopExitKind
from daita.security import EmptySecretProvider


def _ids():
    counts: defaultdict[str, int] = defaultdict(int)

    def create(prefix: str) -> str:
        counts[prefix] += 1
        if prefix in {"run", "conversation", "artifact", "destination"}:
            return f"{prefix}-{counts[prefix]:032x}"
        return f"{prefix}-{counts[prefix]}"

    return create


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _tools(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls)


def _stop(text: str = "done") -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _error_code(block: ToolResultBlock) -> str:
    error = block.output.get("error")
    assert isinstance(error, Mapping)
    code = error.get("code")
    assert isinstance(code, str)
    return code


def test_exact_csv_frozen_scalar_escaping_and_dialect_contract() -> None:
    columns = (
        "text",
        "null",
        "boolean",
        "integer",
        "float",
        "decimal",
        "date",
        "time",
        "timestamp",
        "binary",
        "uuid",
    )
    content = render_exact_csv(
        columns,
        (
            (
                'comma,"quote"\r\nline',
                None,
                True,
                -123456789012345678901234567890,
                -0.0,
                Decimal("12.3400"),
                date(2026, 8, 1),
                time(1, 2, 3, 4, tzinfo=timezone(timedelta(hours=-5))),
                datetime(2026, 8, 1, 1, 2, 3, 4, tzinfo=timezone.utc),
                b"\x00\xff",
                UUID("ABCDEFAB-CDEF-ABCD-EFAB-CDEFABCDEFAB"),
            ),
        ),
    )
    assert content == (
        b'"text","null","boolean","integer","float","decimal","date",'
        b'"time","timestamp","binary","uuid"\r\n'
        b'"comma,""quote""\r\nline",\\N,TRUE,'
        b"-123456789012345678901234567890,-0.0,12.3400,2026-08-01,"
        b'"01:02:03.000004-05:00","2026-08-01T01:02:03.000004+00:00",'
        b'\\BAP8=,"abcdefab-cdef-abcd-efab-cdefabcdefab"\r\n'
    )
    assert not content.startswith(b"\xef\xbb\xbf")
    assert content.endswith(b"\r\n")
    assert render_exact_csv(("only",), ()) == b'"only"\r\n'


@pytest.mark.parametrize(
    ("value", "field"),
    (
        ("=1", "'=1"),
        (" +1", "' +1"),
        ("\t-1", "'\t-1"),
        ("\r\n@x", "'\r\n@x"),
        ("'=1", "''=1"),
        ("''  +x", "'''  +x"),
    ),
)
def test_exact_csv_formula_injection_protection_is_exact_and_reversible(
    value: str,
    field: str,
) -> None:
    expected = ('"value"\r\n"' + field.replace('"', '""') + '"\r\n').encode("utf-8")
    assert render_exact_csv(("value",), ((value,),)) == expected


def test_exact_csv_reserved_tokens_headers_and_source_unicode_are_not_ambiguous() -> (
    None
):
    decomposed = "e\u0301"
    content = render_exact_csv(
        ("=header", "literal"),
        ((r"\N", "NULL"), (r"\Bvalue", decomposed), ("", "1+1")),
    )
    assert content == (
        b'"\'=header","literal"\r\n'
        b'"\\\\N","NULL"\r\n'
        + '"\\\\Bvalue","e\u0301"\r\n'.encode("utf-8")
        + b'"","1+1"\r\n'
    )
    assert decomposed.encode("utf-8") in content


def test_exact_csv_remaining_scalar_variants_follow_the_frozen_encoding() -> None:
    content = render_exact_csv(
        (
            "false",
            "float",
            "naive_time",
            "naive_datetime",
            "bytearray",
            "memoryview",
        ),
        (
            (
                False,
                1.25,
                time(1, 2, 3),
                datetime(2026, 8, 1, 1, 2, 3),
                bytearray(b"a"),
                memoryview(b"b"),
            ),
        ),
    )
    assert content == (
        b'"false","float","naive_time","naive_datetime","bytearray",'
        b'"memoryview"\r\n'
        b'FALSE,1.25,"01:02:03.000000","2026-08-01T01:02:03.000000",'
        b"\\BYQ==,\\BYg==\r\n"
    )


@pytest.mark.parametrize(
    "value",
    (
        float("nan"),
        float("inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
        [],
        {},
        (1, 2),
        timedelta(days=1),
        "\ud800",
        object(),
    ),
)
def test_exact_csv_unsupported_values_fail_without_exposing_the_value(
    value: object,
) -> None:
    with pytest.raises(ArtifactError) as failure:
        render_exact_csv(("value",), ((value,),))
    assert failure.value.code == "artifact_unsupported_value"
    details = failure.value.details.to_dict()
    assert details["row_index"] == 0
    assert details["column_index"] == 0
    assert details["column_name"] == "value"
    assert "value" not in details


@pytest.mark.parametrize(
    ("columns", "reason"),
    (
        ((), "missing_columns"),
        ((1,), "not_text"),
        (("",), "empty"),
        ((" ",), "empty"),
        (("same", "same"), "duplicate"),
        (("x" * 257,), "too_long"),
        (("\ud800",), "invalid_unicode"),
    ),
)
def test_exact_csv_invalid_columns_are_rejected_without_renaming(
    columns: tuple[object, ...],
    reason: str,
) -> None:
    with pytest.raises(ArtifactError) as failure:
        render_exact_csv(cast(tuple[str, ...], columns), ())
    assert failure.value.code == "artifact_unsupported_value"
    assert failure.value.details["reason"] == reason


def test_exact_csv_row_column_byte_and_time_boundaries_fail_closed() -> None:
    assert (MAX_CSV_ROWS, MAX_CSV_COLUMNS, MAX_CSV_BYTES, MAX_CSV_SECONDS) == (
        100_000,
        256,
        64 * 1024 * 1024,
        60.0,
    )
    assert render_exact_csv(("x",), ((1,),), max_rows=1) == b'"x"\r\n1\r\n'
    assert render_exact_csv(("a", "b"), (), max_columns=2) == b'"a","b"\r\n'
    assert render_exact_csv(("x",), (), max_bytes=5) == b'"x"\r\n'
    with pytest.raises(ArtifactError) as row_failure:
        render_exact_csv(("x",), ((1,), (2,)), max_rows=1)
    assert row_failure.value.code == "artifact_incomplete_export"
    assert row_failure.value.details["reason"] == "row_limit"

    with pytest.raises(ArtifactError) as column_failure:
        render_exact_csv(("a", "b", "c"), (), max_columns=2)
    assert column_failure.value.code == "artifact_quota_exceeded"
    assert column_failure.value.details.to_dict() == {
        "scope": "artifact",
        "limit_kind": "columns",
        "limit": 2,
        "attempted": 3,
    }

    with pytest.raises(ArtifactError) as byte_failure:
        render_exact_csv(("x",), (("12345",),), max_bytes=10)
    assert byte_failure.value.code == "artifact_incomplete_export"
    assert byte_failure.value.details["reason"] == "byte_limit"

    ticks = iter((0.0, 0.0, 2.0))
    with pytest.raises(ArtifactError) as time_failure:
        render_exact_csv(
            ("x",),
            ((1,),),
            max_seconds=1.0,
            clock=lambda: next(ticks),
        )
    assert time_failure.value.code == "artifact_incomplete_export"
    assert time_failure.value.details["reason"] == "time_limit"


def test_exact_csv_renderer_never_returns_a_prefix_after_source_failure() -> None:
    def rows():
        yield (1,)
        raise RuntimeError("source failed")

    with pytest.raises(RuntimeError, match="source failed"):
        render_exact_csv(("x",), rows())


class _Attribute:
    def __init__(self, name: str) -> None:
        self.name = name


class _Cursor:
    def __init__(self, rows: tuple[tuple[object, ...], ...]) -> None:
        self._rows = rows
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._rows):
            raise StopAsyncIteration
        value = self._rows[self._index]
        self._index += 1
        return value


class _Statement:
    def __init__(
        self,
        *,
        names: tuple[str, ...] = (),
        rows: tuple[tuple[object, ...], ...] = (),
    ) -> None:
        self._names = names
        self._rows = rows
        self.cursor_arguments: tuple[object, ...] | None = None

    def get_attributes(self):
        return tuple(_Attribute(name) for name in self._names)

    def cursor(self, *parameters: object, **kwargs: object):
        del kwargs
        self.cursor_arguments = parameters
        return _Cursor(self._rows)


class _Connection:
    def __init__(
        self,
        names: tuple[str, ...],
        rows: tuple[tuple[object, ...], ...],
    ) -> None:
        self.shape = _Statement(names=names)
        self.result = _Statement(rows=rows)
        self.prepared_sql: list[str] = []

    async def prepare(self, sql: str, *, timeout: float):
        del timeout
        self.prepared_sql.append(sql)
        return self.shape if len(self.prepared_sql) == 1 else self.result


async def test_postgresql_exact_adapter_streams_typed_values_without_json_projection() -> (
    None
):
    connection = _Connection(
        ("amount", "when", "payload"),
        (
            (
                Decimal("1.2300"),
                datetime(2026, 8, 1, tzinfo=timezone.utc),
                b"abc",
                True,
            ),
        ),
    )
    content, columns, row_count = (
        await postgresql_query_module._execute_exact_tabular_query(
            connection,
            'SELECT amount, "when", payload FROM public.orders WHERE id = $1',
            (7,),
            format_name="csv",
            xlsx_provenance=None,
            max_rows=100_000,
            max_columns=256,
            max_bytes=64 * 1024 * 1024,
            timeout_seconds=60,
        )
    )
    assert columns == ("amount", "when", "payload")
    assert row_count == 1
    assert content == (
        b'"amount","when","payload"\r\n'
        b'1.2300,"2026-08-01T00:00:00.000000+00:00",\\BYWJj\r\n'
    )
    assert connection.result.cursor_arguments == (7,)
    executed = connection.prepared_sql[1]
    assert "pg_catalog.pg_column_size" in executed
    assert "LIMIT 100001" in executed
    assert "to_json" not in executed.casefold()
    assert "jsonb" not in executed.casefold()


async def test_postgresql_exact_adapter_rejects_invalid_columns_and_server_byte_sentinel() -> (
    None
):
    duplicate = _Connection(("same", "same"), ())
    with pytest.raises(ArtifactError) as duplicate_failure:
        await postgresql_query_module._execute_exact_tabular_query(
            duplicate,
            "SELECT 1, 2",
            (),
            format_name="csv",
            xlsx_provenance=None,
            max_rows=100_000,
            max_columns=256,
            max_bytes=64 * 1024 * 1024,
            timeout_seconds=60,
        )
    assert duplicate_failure.value.code == "artifact_unsupported_value"
    assert len(duplicate.prepared_sql) == 1

    byte_limited = _Connection(("value",), ((None, False),))
    with pytest.raises(ArtifactError) as byte_failure:
        await postgresql_query_module._execute_exact_tabular_query(
            byte_limited,
            "SELECT value FROM public.orders",
            (),
            format_name="csv",
            xlsx_provenance=None,
            max_rows=100_000,
            max_columns=256,
            max_bytes=64 * 1024 * 1024,
            timeout_seconds=60,
        )
    assert byte_failure.value.code == "artifact_incomplete_export"
    assert byte_failure.value.details["reason"] == "byte_limit"


class _SourceStore:
    def __init__(self, registration: SourceRegistration | None) -> None:
        self.registration = registration

    async def register_source(self, registration: SourceRegistration):
        self.registration = registration
        return registration

    async def load_source(self, agent_id: str, source_id: str):
        if (
            self.registration is not None
            and self.registration.agent_id == agent_id
            and self.registration.id == source_id
        ):
            return self.registration
        return None

    async def list_sources(self, agent_id: str):
        return (
            (self.registration,)
            if self.registration is not None and self.registration.agent_id == agent_id
            else ()
        )

    async def detach_source(
        self,
        agent_id: str,
        source_id: str,
        detached_at: datetime,
    ):
        del agent_id, source_id, detached_at
        assert self.registration is not None
        return self.registration


class _CatalogSchemas:
    def __init__(self, resources: tuple[ResourceSchema, ...]) -> None:
        self.resources = resources

    async def resource_schemas(self, agent_id: str, source_id: str):
        del agent_id
        return tuple(item for item in self.resources if item.source_id == source_id)


class _Transaction:
    def __init__(self) -> None:
        self.started = False
        self.committed = False

    async def start(self) -> None:
        self.started = True

    async def commit(self) -> None:
        self.committed = True


class _BackendConnection:
    def __init__(self) -> None:
        self.transaction_record = _Transaction()
        self.settings: list[tuple[str, tuple[object, ...]]] = []

    def transaction(self, **kwargs: object):
        assert kwargs == {"isolation": "repeatable_read", "readonly": True}
        return self.transaction_record

    async def execute(self, sql: str, *parameters: object):
        self.settings.append((sql, parameters))


@pytest.mark.parametrize("format_name", ("csv", "xlsx"))
async def test_postgresql_backend_contract_revalidates_and_executes_once_without_live_db(
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
) -> None:
    agent_id = "agent-postgresql"
    registration = SourceRegistration.build(
        agent_id=agent_id,
        adapter_id="postgresql",
        native_identity="offline-contract",
        display_name="Offline PostgreSQL",
        configuration={},
        attached_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    source_revision = "sha256:" + "3" * 64
    resource = ResourceSchema(
        resource_id="resource-orders",
        source_id=registration.id,
        name="orders",
        aliases=("public.orders",),
        columns=("amount",),
        revision="sha256:" + "2" * 64,
        source_revision=source_revision,
        resource_kind="table",
        sensitivity_class="confidential",
    )
    connection = _BackendConnection()
    exact_calls = 0
    closed = False

    async def connect(*args: object, **kwargs: object):
        del args, kwargs
        return connection

    async def load_structure(*args: object, **kwargs: object):
        del args, kwargs
        return SimpleNamespace(source_revision=source_revision)

    async def execute_exact(*args: object, **kwargs: object):
        nonlocal exact_calls
        exact_calls += 1
        assert args[2] == (Decimal("1.20"),)
        assert kwargs["max_rows"] == 100_000
        assert kwargs["format_name"] == format_name
        return b'"amount"\r\n1.20\r\n', ("amount",), 1

    async def close(*args: object, **kwargs: object):
        nonlocal closed
        del args, kwargs
        closed = True

    monkeypatch.setattr(postgresql_query_module, "_connect", connect)
    monkeypatch.setattr(postgresql_query_module, "_load_structure", load_structure)
    monkeypatch.setattr(
        postgresql_query_module, "_execute_exact_tabular_query", execute_exact
    )
    monkeypatch.setattr(postgresql_query_module, "_close_postgresql_connection", close)
    backend = postgresql_query_module.PostgreSQLQueryBackend(
        _SourceStore(registration),
        _CatalogSchemas((resource,)),
        EmptySecretProvider(),
    )
    result = await backend.execute_exact_tabular(
        agent_id=agent_id,
        source_id=registration.id,
        sql="SELECT amount FROM public.orders WHERE amount = $1",
        parameters=(Decimal("1.20"),),
        format_name=format_name,
        parameters_sha256="sha256:" + "0" * 64,
        created_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        max_rows=100_000,
        max_columns=256,
        max_bytes=64 * 1024 * 1024,
        timeout_seconds=60,
    )
    assert exact_calls == 1
    assert connection.transaction_record.started
    assert connection.transaction_record.committed
    assert closed
    assert result.content == b'"amount"\r\n1.20\r\n'
    assert result.format == format_name
    assert result.resource_revisions == ((resource.resource_id, resource.revision),)
    assert result.source_revision == source_revision


def test_source_specific_csv_tool_schemas_cannot_accept_rows_bytes_or_provenance() -> (
    None
):
    extension = artifact_extension_declarations()
    views = {
        item.name: item
        for item in extension.tool_views
        if item.name
        in {SQLITE_TABULAR_EXPORT_TOOL_NAME, POSTGRESQL_TABULAR_EXPORT_TOOL_NAME}
    }
    assert set(views) == {
        SQLITE_TABULAR_EXPORT_TOOL_NAME,
        POSTGRESQL_TABULAR_EXPORT_TOOL_NAME,
    }
    for name, expected_capability in (
        (SQLITE_TABULAR_EXPORT_TOOL_NAME, SQLITE_TABULAR_EXPORT_CAPABILITY_ID),
        (
            POSTGRESQL_TABULAR_EXPORT_TOOL_NAME,
            POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID,
        ),
    ):
        view = views[name]
        assert view.capability_id == expected_capability
        capability = next(
            item for item in extension.capabilities if item.id == view.capability_id
        )
        properties = capability.input_schema["properties"]
        assert isinstance(properties, Mapping)
        assert set(properties) == {
            "source_id",
            "sql",
            "parameters",
            "format",
            "filename",
        }
        assert set(properties).isdisjoint(
            {"rows", "content", "bytes", "artifact", "provenance", "sensitivity"}
        )
        assert capability.artifact_policy is not None
        assert capability.artifact_policy.max_bytes_per_artifact == 64 * 1024 * 1024


async def _sqlite_export_agent(
    tmp_path: Path,
    *,
    name: str,
    rows: tuple[tuple[str, int], ...],
    downloads: Path,
    script: tuple[ModelResponse, ...] = (),
    observer=None,
) -> tuple[Agent, MockModelProvider, str, Path]:
    database = tmp_path / f"{name}.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE records(label TEXT, number INTEGER)")
    connection.executemany("INSERT INTO records VALUES (?, ?)", rows)
    connection.commit()
    connection.close()
    provider = MockModelProvider(script, provider_id=f"mock:{name}")
    agent = await Agent.create(
        name,
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        downloads_directory=downloads,
        observer=observer,
    )
    source = await agent.attach(SQLiteSource(database))
    return agent, provider, source.id, database


async def test_sqlite_public_exact_csv_creation_delivery_restart_and_redelivery(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    secret_row = "ROW_SECRET_7d0bda20"
    events: list[object] = []
    agent, provider, source_id, _database = await _sqlite_export_agent(
        tmp_path,
        name="csv-public",
        rows=((secret_row, 1), ("=formula", 2)),
        downloads=downloads,
        observer=events.append,
    )
    provider._script = (
        _tools(
            ToolCall(
                id="export",
                name=SQLITE_TABULAR_EXPORT_TOOL_NAME,
                arguments={
                    "source_id": source_id,
                    "sql": (
                        "SELECT label, number FROM records "
                        "WHERE number >= ? ORDER BY number"
                    ),
                    "parameters": [1],
                    "format": "csv",
                    "filename": "records.csv",
                },
            )
        ),
        _tools(
            ToolCall(
                id="save",
                name="artifact_save_local",
                arguments={
                    "artifact_id": "artifact-00000000000000000000000000000001",
                    "destination_id": "default",
                },
            )
        ),
        _stop("saved"),
    )
    expected = (
        b'"label","number"\r\n'
        + f'"{secret_row}",1\r\n'.encode()
        + b'"\'=formula",2\r\n'
    )
    try:
        result = await agent.run("Export every record as an exact CSV file.")
        transcript = await agent.transcript(result.run_id)
        assert result.kind is LoopExitKind.COMPLETED
        assert len(result.artifacts) == len(result.artifact_deliveries) == 1
        ref = result.artifacts[0]
        assert ref.call_id == "export"
        assert ref.capability_id == SQLITE_TABULAR_EXPORT_CAPABILITY_ID
        assert ref.media_type == "text/csv"
        assert ref.sensitivity is Sensitivity.INTERNAL
        assert ref.sensitivity is not Sensitivity.PUBLIC
        assert ref.provenance.authorship is ArtifactAuthorship.EXACT_SOURCE_DATA
        assert ref.provenance.columns == ("label", "number")
        assert ref.provenance.row_count == 2
        assert ref.provenance.parameters_sha256 == (
            "sha256:" + sha256(canonical_json((1,)).encode()).hexdigest()
        )
        assert await agent.read_artifact(ref.artifact_id) == (
            await agent.read_artifact(ref.artifact_id)
        )
        assert (await agent.read_artifact(ref.artifact_id)).content == expected
        receipt = result.artifact_deliveries[0]
        assert Path(receipt.saved_path).read_bytes() == expected

        export_result = transcript.messages[2].content[0]
        assert isinstance(export_result, ToolResultBlock)
        assert export_result.output["delivery_status"] == "not_delivered"
        export_data = export_result.output["data"]
        assert isinstance(export_data, Mapping)
        assert set(export_data) == {
            "format",
            "filename",
            "row_count",
            "column_count",
        }
        serialized_transcript = canonical_json(
            [
                block.output
                for message in transcript.messages
                for block in message.content
                if isinstance(block, ToolResultBlock)
            ]
        )
        assert secret_row not in serialized_transcript
        assert secret_row not in repr(result)
        assert secret_row.encode() not in (agent.home / "state.db").read_bytes()
        assert secret_row not in canonical_json(
            {
                "artifacts": tuple(
                    artifact_ref_to_mapping(item) for item in result.artifacts
                ),
                "artifact_deliveries": tuple(
                    artifact_delivery_receipt_to_mapping(item)
                    for item in result.artifact_deliveries
                ),
            }
        )
        assert secret_row not in canonical_json(
            [getattr(event, "data", {}) for event in events]
        )
        assert secret_row not in caplog.text
        first_request_text = "\n".join(
            block.text
            for message in provider.requests[0].messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert "data_export_sqlite" in first_request_text
        assert "artifact_save_local" in first_request_text
        assert (
            "Never put source rows or artifact bytes in arguments" in first_request_text
        )
    finally:
        await agent.close()

    reopened = await Agent.open(
        "csv-public", root=tmp_path, downloads_directory=downloads
    )
    try:
        ref = result.artifacts[0]
        assert (await reopened.read_artifact(ref.artifact_id)).content == expected
        redelivery = await reopened.save_artifact(ref.artifact_id)
        assert Path(redelivery.saved_path).read_bytes() == expected
        assert redelivery.sha256 == ref.sha256
    finally:
        await reopened.close()


@pytest.mark.parametrize(
    ("sql", "parameters", "expected"),
    (
        ("DELETE FROM records", (), "sql_validation_failed"),
        (
            "SELECT label FROM records; SELECT number FROM records",
            (),
            "sql_validation_failed",
        ),
        (
            "SELECT label FROM records WHERE number = ?",
            (),
            "sql_validation_failed",
        ),
        ("SELECT * FROM missing", (), "sql_validation_failed"),
    ),
)
async def test_csv_export_reuses_current_sql_and_catalog_validation(
    tmp_path: Path,
    sql: str,
    parameters: tuple[object, ...],
    expected: str,
) -> None:
    downloads = tmp_path / f"downloads-{abs(hash(sql))}"
    downloads.mkdir()
    agent, provider, source_id, _database = await _sqlite_export_agent(
        tmp_path,
        name=f"invalid-{abs(hash(sql))}",
        rows=(("row", 1),),
        downloads=downloads,
    )
    provider._script = (
        _tools(
            ToolCall(
                id="invalid",
                name=SQLITE_TABULAR_EXPORT_TOOL_NAME,
                arguments={
                    "source_id": source_id,
                    "sql": sql,
                    "parameters": list(parameters),
                    "format": "csv",
                },
            )
        ),
        _stop(),
    )
    try:
        result = await agent.run("Export this as CSV.")
        transcript = await agent.transcript(result.run_id)
        block = transcript.messages[2].content[0]
        assert isinstance(block, ToolResultBlock)
        assert _error_code(block) == expected
        assert result.artifacts == ()
    finally:
        await agent.close()


async def test_csv_export_rejects_detached_mismatched_and_stale_sources(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads-source-validation"
    downloads.mkdir()
    agent, provider, source_id, database = await _sqlite_export_agent(
        tmp_path,
        name="source-validation",
        rows=(("row", 1),),
        downloads=downloads,
    )
    try:
        provider._script = (
            _tools(
                ToolCall(
                    id="mismatch",
                    name=POSTGRESQL_TABULAR_EXPORT_TOOL_NAME,
                    arguments={
                        "source_id": source_id,
                        "sql": "SELECT label FROM records",
                        "format": "csv",
                    },
                )
            ),
            _stop(),
        )
        mismatch = await agent.run("Export this as CSV.")
        mismatch_block = (
            (await agent.transcript(mismatch.run_id)).messages[2].content[0]
        )
        assert isinstance(mismatch_block, ToolResultBlock)
        assert _error_code(mismatch_block) == "tool_not_available"

        provider._script = (
            _tools(
                ToolCall(
                    id="stale",
                    name=SQLITE_TABULAR_EXPORT_TOOL_NAME,
                    arguments={
                        "source_id": source_id,
                        "sql": "SELECT label FROM records",
                        "format": "csv",
                    },
                )
            ),
            _stop(),
        )
        provider._cursor = 0
        with sqlite3.connect(database) as connection:
            connection.execute("ALTER TABLE records ADD COLUMN changed TEXT")
        stale = await agent.run("Export this as CSV.")
        stale_block = (await agent.transcript(stale.run_id)).messages[2].content[0]
        assert isinstance(stale_block, ToolResultBlock)
        assert _error_code(stale_block) == "tool_execution_failed"
        assert stale.artifacts == ()

        await agent.detach(source_id)
        provider._script = (
            _tools(
                ToolCall(
                    id="detached",
                    name=SQLITE_TABULAR_EXPORT_TOOL_NAME,
                    arguments={
                        "source_id": source_id,
                        "sql": "SELECT label FROM records",
                        "format": "csv",
                    },
                )
            ),
            _stop(),
        )
        provider._cursor = 0
        detached = await agent.run("Export this as CSV.")
        detached_block = (
            (await agent.transcript(detached.run_id)).messages[2].content[0]
        )
        assert isinstance(detached_block, ToolResultBlock)
        assert _error_code(detached_block) == "tool_not_available"
    finally:
        await agent.close()


async def test_concurrent_csv_exports_keep_call_order_and_failed_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads = tmp_path / "downloads-order"
    downloads.mkdir()
    agent, provider, source_id, _database = await _sqlite_export_agent(
        tmp_path,
        name="csv-order",
        rows=(("one", 1), ("two", 2)),
        downloads=downloads,
    )
    original = sqlite_query_module._run_exact_tabular_sync

    def delayed(*args: Any, **kwargs: Any):
        if "slow_export" in args[1]:
            time_module.sleep(0.1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sqlite_query_module, "_run_exact_tabular_sync", delayed)
    provider._script = (
        _tools(
            ToolCall(
                id="first",
                name=SQLITE_TABULAR_EXPORT_TOOL_NAME,
                arguments={
                    "source_id": source_id,
                    "sql": "SELECT label FROM records /* slow_export */ ORDER BY number",
                    "format": "csv",
                    "filename": "first.csv",
                },
            ),
            ToolCall(
                id="failed",
                name=SQLITE_TABULAR_EXPORT_TOOL_NAME,
                arguments={
                    "source_id": source_id,
                    "sql": "SELECT label AS duplicate, number AS duplicate FROM records",
                    "format": "csv",
                    "filename": "failed.csv",
                },
            ),
            ToolCall(
                id="third",
                name=SQLITE_TABULAR_EXPORT_TOOL_NAME,
                arguments={
                    "source_id": source_id,
                    "sql": "SELECT number FROM records ORDER BY number",
                    "format": "csv",
                    "filename": "third.csv",
                },
            ),
        ),
        _stop(),
    )
    try:
        result = await agent.run("Export these CSV files.")
        transcript = await agent.transcript(result.run_id)
        blocks = tuple(
            message.content[0]
            for message in transcript.messages
            if message.role is MessageRole.TOOL
            and isinstance(message.content[0], ToolResultBlock)
        )
        assert tuple(block.call_id for block in blocks) == ("first", "failed", "third")
        assert isinstance(blocks[1], ToolResultBlock)
        assert _error_code(blocks[1]) == "artifact_unsupported_value"
        assert tuple(ref.call_id for ref in result.artifacts) == ("first", "third")
        assert result.artifacts[0].artifact_id > result.artifacts[1].artifact_id
    finally:
        await agent.close()


async def test_exact_csv_cancellation_emits_no_artifact_or_partial_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads = tmp_path / "downloads-cancel"
    downloads.mkdir()
    agent, provider, source_id, _database = await _sqlite_export_agent(
        tmp_path,
        name="csv-cancel",
        rows=(("row", 1),),
        downloads=downloads,
    )
    started = threading.Event()
    release = threading.Event()

    def blocked(*args: Any, **kwargs: Any):
        del kwargs
        started.set()
        assert release.wait(2)
        expected_schema_version = args[5]
        return (
            b'"label"\r\n',
            ("label",),
            0,
            (f"schema_version:{expected_schema_version}"),
        )

    monkeypatch.setattr(sqlite_query_module, "_run_exact_tabular_sync", blocked)
    provider._script = (
        _tools(
            ToolCall(
                id="cancelled-export",
                name=SQLITE_TABULAR_EXPORT_TOOL_NAME,
                arguments={
                    "source_id": source_id,
                    "sql": "SELECT label FROM records",
                    "format": "csv",
                },
            )
        ),
    )
    try:
        running = asyncio.create_task(agent.run("Export this CSV file."))
        assert await asyncio.to_thread(started.wait, 2)
        running.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert not tuple(downloads.iterdir())
    finally:
        release.set()
        await agent.close()


async def test_delivery_failure_after_csv_commit_retains_valid_internal_artifact(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads-failure"
    downloads.mkdir()
    agent, provider, source_id, _database = await _sqlite_export_agent(
        tmp_path,
        name="csv-delivery-failure",
        rows=(("retained", 1),),
        downloads=downloads,
    )
    provider._script = (
        _tools(
            ToolCall(
                id="export",
                name=SQLITE_TABULAR_EXPORT_TOOL_NAME,
                arguments={
                    "source_id": source_id,
                    "sql": "SELECT label, number FROM records",
                    "format": "csv",
                    "filename": "retained.csv",
                },
            )
        ),
        _tools(
            ToolCall(
                id="delivery",
                name="artifact_save_local",
                arguments={
                    "artifact_id": "artifact-00000000000000000000000000000001",
                    "destination_id": "default",
                },
            )
        ),
        _stop("delivery failed"),
    )
    downloads.rmdir()
    try:
        result = await agent.run("Export this CSV file.")
        transcript = await agent.transcript(result.run_id)
        assert len(result.artifacts) == 1
        assert result.artifact_deliveries == ()
        delivery = transcript.messages[4].content[0]
        assert isinstance(delivery, ToolResultBlock)
        assert delivery.is_error
        assert _error_code(delivery) == "artifact_downloads_unavailable"
        error = delivery.output["error"]
        assert isinstance(error, Mapping)
        details = error["details"]
        assert isinstance(details, Mapping)
        assert details["artifact_retained"] is True
        assert details["artifact_id"] == result.artifacts[0].artifact_id
        payload = await agent.read_artifact(result.artifacts[0].artifact_id)
        assert payload.content == b'"label","number"\r\n"retained",1\r\n'
    finally:
        await agent.close()
