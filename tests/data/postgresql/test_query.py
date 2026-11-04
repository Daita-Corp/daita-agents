from __future__ import annotations

import asyncio
import traceback
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from daita.adapters import postgresql_query as postgresql_query_module
from daita.adapters.postgresql import PostgreSQLSourceError
from daita.capabilities import Capability, ToolExecution, ToolOutput, ToolView
from daita.capability_runtime import CapabilityRuntime
from daita.llm.models import ToolCall
from daita.loop.models import RunInput
from tests.data.postgresql._query_support import _backend
from tests.support.capability_runtime import (
    StaticTestDomain,
    execute_projected,
    presentation_metadata,
    static_registry,
)
from tests.support.checked_read import checked_read


@pytest.mark.parametrize("operation", ("query", "export"))
async def test_postgresql_backend_preserves_safe_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend, registration = _backend()
    diagnostic = "host=db.internal password=database-secret"

    async def failed_connect(*args: object, **kwargs: object) -> object:
        del args, kwargs
        error = PostgreSQLSourceError(
            "postgresql_connect_failed",
            "PostgreSQL source could not be opened.",
            source_id=registration.id,
        )
        error.add_note(diagnostic)
        raise error

    monkeypatch.setattr(postgresql_query_module, "_connect", failed_connect)

    with pytest.raises(postgresql_query_module.PostgreSQLQueryError) as raised:
        if operation == "query":
            await backend.execute_read(
                agent_id=registration.agent_id,
                source_id=registration.id,
                read=checked_read("SELECT amount FROM public.orders"),
                max_rows=100,
                max_bytes=65_536,
            )
        else:
            await backend.execute_exact_tabular(
                agent_id=registration.agent_id,
                source_id=registration.id,
                read=checked_read("SELECT amount FROM public.orders"),
                format_name="csv",
                parameters_sha256="sha256:" + "0" * 64,
                created_at=datetime(2026, 8, 5, tzinfo=UTC),
                max_rows=100_000,
                max_columns=256,
                max_bytes=64 * 1024 * 1024,
                timeout_seconds=60,
            )

    assert raised.value.code == "postgresql_connect_failed"
    assert str(raised.value) == "PostgreSQL source could not be opened."
    assert raised.value.__context__ is None
    assert diagnostic not in "".join(traceback.format_exception(raised.value))


async def test_postgresql_backend_preserves_credential_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, registration = _backend()

    async def failed_connect(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise PostgreSQLSourceError(
            "postgresql_credential_unavailable",
            "PostgreSQL credential resolution failed.",
            source_id=registration.id,
        )

    monkeypatch.setattr(postgresql_query_module, "_connect", failed_connect)

    with pytest.raises(postgresql_query_module.PostgreSQLQueryError) as raised:
        await backend.execute_read(
            agent_id=registration.agent_id,
            source_id=registration.id,
            read=checked_read("SELECT amount FROM public.orders"),
            max_rows=100,
            max_bytes=65_536,
        )

    assert raised.value.code == "postgresql_credential_unavailable"
    assert str(raised.value) == "PostgreSQL credential resolution failed."


class _FailingExecutor:
    executor_id = "test.postgresql.failure.executor"

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def execute(self, request: ToolExecution) -> ToolOutput:
        del request
        raise self.error


async def _failure_result(error: BaseException):
    capability = Capability(
        id="test.postgresql.failure",
        description="Exercise normalized executor failures.",
        input_schema={"type": "object", "properties": {}},
        output_kind="test.postgresql.failure.output",
        output_schema={"type": "object", "properties": {}},
        executor_id=_FailingExecutor.executor_id,
    )
    view = ToolView(
        name="test_postgresql_failure",
        capability_id=capability.id,
        description=capability.description,
        presentation=presentation_metadata(),
    )
    domain = StaticTestDomain((capability,), (view,))
    runtime = CapabilityRuntime(
        static_registry(domain, (_FailingExecutor(error),)),
        (domain,),
    )
    outcome = await execute_projected(
        runtime,
        RunInput(
            id="run-postgresql-failure",
            agent_id="agent-postgresql-failure",
            message="exercise failure normalization",
            created_at=datetime(2026, 8, 19, tzinfo=UTC),
        ),
        (ToolCall(id="call-postgresql", name=view.name),),
    )
    return outcome.ordered_results[0]


async def test_postgresql_query_error_remains_structured_at_tool_result_boundary() -> (
    None
):
    result = await _failure_result(
        postgresql_query_module.PostgreSQLQueryError(
            "postgresql_connect_failed",
            "PostgreSQL source could not be opened.",
        ),
    )

    assert result.is_error is True
    error = result.output["error"]
    assert isinstance(error, Mapping)
    assert error["code"] == "postgresql_connect_failed"
    assert error["message"] == "PostgreSQL source could not be opened."
    details = error["details"]
    assert isinstance(details, Mapping)
    assert dict(details) == {}


async def test_unexpected_executor_error_is_redacted_at_tool_result_boundary() -> None:
    result = await _failure_result(
        RuntimeError("secret=/tmp/private-connection-fragment"),
    )

    assert result.is_error is True
    serialized = repr(result.output)
    assert "secret=" not in serialized
    assert "/tmp/private-connection-fragment" not in serialized
    error = result.output["error"]
    assert isinstance(error, Mapping)
    assert error["code"] == "tool_execution_failed"
    assert error["message"] == (
        "The tool could not complete because of an unexpected internal error."
    )


class _ReadTransaction:
    async def start(self) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class _ReadConnection:
    """Answer the function lookup; the user query is replaced by the test."""

    def __init__(self, functions: tuple[tuple[object, ...], ...] = ()) -> None:
        self.functions = tuple((*item, False)[:4] for item in functions)
        self.settings: dict[str, str] = {}
        self.lookups: list[tuple[str, ...]] = []

    def transaction(self, **kwargs: object) -> _ReadTransaction:
        assert kwargs == {"isolation": "repeatable_read", "readonly": True}
        return _ReadTransaction()

    async def execute(self, sql: str, value: str) -> None:
        self.settings[sql.split("'")[1]] = value

    async def fetch(self, sql: str, names: list[str]) -> list[tuple[object, ...]]:
        assert sql.startswith("/* daita:postgresql.function_policy */")
        self.lookups.append(tuple(names))
        return [item for item in self.functions if item[0] in names]

    def is_closed(self) -> bool:
        return False

    async def close(self, **kwargs: object) -> None:
        return None

    def terminate(self) -> None:
        return None


def _route_to(
    monkeypatch: pytest.MonkeyPatch,
    connection: _ReadConnection,
    query: object = None,
) -> list[str]:
    queries: list[str] = []

    async def connect(*args: object, **kwargs: object) -> _ReadConnection:
        return connection

    async def answer(connection: object, sql: str, parameters: object, **kwargs):
        queries.append(sql)
        return ("n",), ({"n": 1},), False

    monkeypatch.setattr(postgresql_query_module, "_connect", connect)
    monkeypatch.setattr(postgresql_query_module, "_execute_user_query", query or answer)
    return queries


async def _run(backend, registration, read, **kwargs):
    return await backend.execute_read(
        agent_id=registration.agent_id,
        source_id=registration.id,
        read=read,
        max_rows=100,
        max_bytes=65_536,
        **kwargs,
    )


async def test_a_read_without_functions_makes_no_policy_lookup_and_no_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, registration = _backend()
    connection = _ReadConnection()
    queries = _route_to(monkeypatch, connection)

    result = await _run(backend, registration, checked_read("SELECT 1 FROM public.t"))

    assert queries == ["SELECT 1 FROM public.t"]
    assert connection.lookups == []
    assert result.resource_ids == ("resource-orders",)


async def test_reads_pin_the_configured_schemas_with_quoting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, registration = _backend()
    registration = replace(
        registration, configuration={"schemas": ("sales", "Legacy_2")}
    )
    backend._sources.registration = registration  # type: ignore[attr-defined]
    connection = _ReadConnection()
    _route_to(monkeypatch, connection)

    await _run(backend, registration, checked_read("SELECT 1 FROM sales.t"))

    # Registration keeps schemas sorted; several schemas refuse unqualified names.
    assert connection.settings["search_path"] == '"Legacy_2", "sales"'
    assert connection.settings["standard_conforming_strings"] == "on"


@pytest.mark.parametrize(
    ("called", "server", "refused"),
    (
        (("LOWER",), (("lower", "pg_catalog", b"i"),), ()),
        (("NOW",), (("now", "pg_catalog", b"s"),), ()),
        (
            ("RANDOM", "TIMEZONE"),
            (("random", "pg_catalog", b"v"), ("timezone", "pg_catalog", b"v")),
            (),
        ),
        (("PG_SLEEP",), (("pg_sleep", "pg_catalog", b"v"),), ("pg_sleep",)),
        # Any volatile overload of the name refuses every overload.
        (
            ("MIXED",),
            (("mixed", "pg_catalog", b"i"), ("mixed", "pg_catalog", b"v")),
            ("mixed",),
        ),
        (("NOT_THERE",), (), ("not_there",)),
        (("EVIL",), (("evil", "sales", b"i"),), ("evil",)),
        # PostgreSQL could pick the overload outside pg_catalog.
        (
            ("UPPER",),
            (("upper", "pg_catalog", b"i"), ("upper", "sales", b"i")),
            ("upper",),
        ),
        # An explicit pg_catalog call cannot resolve to the namesake.
        (
            ("PG_CATALOG.UPPER",),
            (("upper", "pg_catalog", b"i"), ("upper", "sales", b"i")),
            (),
        ),
    ),
)
async def test_the_server_decides_which_called_functions_a_read_may_use(
    monkeypatch: pytest.MonkeyPatch,
    called: tuple[str, ...],
    server: tuple[tuple[str, str, bytes], ...],
    refused: tuple[str, ...],
) -> None:
    backend, registration = _backend()
    connection = _ReadConnection(server)
    queries = _route_to(monkeypatch, connection)
    read = checked_read("SELECT f() FROM public.t", functions=called)

    if refused:
        with pytest.raises(postgresql_query_module.PostgreSQLQueryError) as raised:
            await _run(backend, registration, read)
        assert raised.value.code == "function_not_allowed"
        assert raised.value.details["functions"] == refused
        assert queries == []
    else:
        await _run(backend, registration, read)
        assert len(queries) == 1
    assert len(connection.lookups) == 1


@pytest.mark.parametrize(
    ("server", "refused"),
    (
        # o.peek is peek(o) when peek takes a row, in any schema on the path.
        ((("peek", "sales", b"s", True),), True),
        # A row-taking built-in is safe unless it is volatile.
        ((("peek", "pg_catalog", b"i", True),), False),
        ((("peek", "pg_catalog", b"v", True),), True),
        # A function that cannot take the row leaves o.peek an ordinary column.
        ((("peek", "sales", b"i", False),), False),
        ((), False),
    ),
)
async def test_a_qualified_name_the_server_would_call_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    server: tuple[tuple[object, ...], ...],
    refused: bool,
) -> None:
    backend, registration = _backend()
    connection = _ReadConnection(server)
    queries = _route_to(monkeypatch, connection)
    read = replace(
        checked_read("SELECT o.peek FROM public.t o"), attribute_lookups=("peek",)
    )

    if refused:
        with pytest.raises(postgresql_query_module.PostgreSQLQueryError) as raised:
            await _run(backend, registration, read)
        assert raised.value.code == "function_not_allowed"
        assert raised.value.details["functions"] == ("peek",)
        assert queries == []
    else:
        await _run(backend, registration, read)
        assert len(queries) == 1
    assert connection.lookups == [("peek",)]


@pytest.mark.parametrize(
    ("server", "refused"),
    (
        # What the parser renames is still what the server runs.
        ((("ceiling", "sales", b"i"), ("ceiling", "pg_catalog", b"i")), True),
        ((("ceiling", "pg_catalog", b"v"),), True),
        ((("ceiling", "pg_catalog", b"i"),), False),
        # A word before "(" that names no function is the server's to explain.
        ((), False),
    ),
)
async def test_a_call_as_written_passes_the_function_rule(
    monkeypatch: pytest.MonkeyPatch,
    server: tuple[tuple[object, ...], ...],
    refused: bool,
) -> None:
    backend, registration = _backend()
    connection = _ReadConnection(server)
    _route_to(monkeypatch, connection)
    read = replace(
        checked_read("SELECT ceiling(1) FROM public.t"), written_calls=("ceiling",)
    )

    if refused:
        with pytest.raises(postgresql_query_module.PostgreSQLQueryError) as raised:
            await _run(backend, registration, read)
        assert raised.value.code == "function_not_allowed"
        assert raised.value.details["functions"] == ("ceiling",)
    else:
        await _run(backend, registration, read)


@pytest.mark.parametrize(
    "sql", ("SELECT 1 AS v -- trailing comment", "SELECT 1 AS v /* note */")
)
def test_the_bounding_wrapper_sits_on_its_own_lines(sql: str) -> None:
    for wrapped in (
        postgresql_query_module._bounded_result_sql(
            sql, column_count=1, max_rows=1, max_bytes=1
        ),
        postgresql_query_module._exact_tabular_bounded_result_sql(
            sql, column_names=("v",), max_rows=1, max_bytes=1
        ),
    ):
        assert f"(\n{sql}\n)" in wrapped


async def test_called_functions_and_qualified_names_share_one_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, registration = _backend()
    connection = _ReadConnection((("lower", "pg_catalog", b"i"),))
    _route_to(monkeypatch, connection)
    read = replace(
        checked_read("SELECT lower(o.a) FROM public.t o", functions=("LOWER",)),
        attribute_lookups=("a",),
    )

    await _run(backend, registration, read)

    assert connection.lookups == [("a", "lower")]


class _ServerError(Exception):
    def __init__(self, sqlstate: str, message: str, **fields: object) -> None:
        super().__init__(f"{message} password=private-diagnostic")
        self.sqlstate = sqlstate
        self.message = message
        for name, value in fields.items():
            setattr(self, name, value)


@pytest.mark.parametrize(
    ("sqlstate", "code", "shown", "undefined"),
    (
        ("42703", "query_invalid", 'column "nope" does not exist', True),
        ("42P01", "query_invalid", 'column "nope" does not exist', True),
        ("42704", "query_invalid", 'column "nope" does not exist', True),
        ("3F000", "postgresql_query_failed", None, True),
        ("42601", "query_invalid", 'column "nope" does not exist', False),
        ("42501", "not_permitted", 'column "nope" does not exist', False),
        ("22012", "query_data_error", None, False),
        ("22P02", "query_data_error", None, False),
        ("25006", "read_only_violation", None, False),
        ("57014", "postgresql_query_timeout", None, False),
        ("08006", "postgresql_connection_failed", None, False),
        ("53300", "postgresql_query_failed", None, False),
    ),
)
async def test_server_failures_cross_by_category_with_only_safe_text(
    monkeypatch: pytest.MonkeyPatch,
    sqlstate: str,
    code: str,
    shown: str | None,
    undefined: bool,
) -> None:
    backend, registration = _backend()

    async def failing(*args: object, **kwargs: object) -> object:
        raise _ServerError(
            sqlstate,
            'column "nope" does not exist',
            hint="Perhaps you meant region.",
            position="8",
            detail="value 'private-value'",
        )

    _route_to(monkeypatch, _ReadConnection(), failing)

    with pytest.raises(postgresql_query_module.PostgreSQLQueryError) as raised:
        await _run(backend, registration, checked_read("SELECT nope FROM public.t"))

    assert raised.value.code == code
    assert raised.value.undefined_object is undefined
    assert "undefined_object" not in raised.value.details
    assert raised.value.details["sqlstate"] == sqlstate
    assert raised.value.__context__ is None
    text = "".join(traceback.format_exception(raised.value))
    assert "private-diagnostic" not in text and "private-value" not in text
    if shown is None:
        assert "does not exist" not in text
    else:
        assert shown in str(raised.value)
    if code == "query_invalid":
        assert raised.value.details["hint"] == "Perhaps you meant region."
        assert raised.value.details["position"] == "8"
        assert raised.value.details["canonical_sql"] == "SELECT nope FROM public.t"


async def test_one_wall_clock_limit_bounds_an_unpooled_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, registration = _backend()
    backend._read_limits = replace(backend._read_limits, acquire_timeout_seconds=0.01)

    async def hanging(*args: object, **kwargs: object) -> None:
        await asyncio.Event().wait()

    _route_to(monkeypatch, _ReadConnection(), hanging)

    async with asyncio.timeout(5):
        with pytest.raises(postgresql_query_module.PostgreSQLQueryError) as raised:
            await _run(
                backend,
                registration,
                checked_read("SELECT 1 FROM public.t"),
                timeout_seconds=0.05,
            )
    assert raised.value.code == "postgresql_query_timeout"


def test_range_values_are_projected_with_their_bounds() -> None:
    from asyncpg.types import Range  # type: ignore[import-untyped]

    assert postgresql_query_module._json_value(Range(1, 5)) == {
        "lower": 1,
        "upper": 5,
        "lower_inclusive": True,
        "upper_inclusive": False,
        "empty": False,
    }
