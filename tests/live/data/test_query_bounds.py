"""Live PostgreSQL coverage for multi-table, multi-statement bounded reads."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from daita import Agent
from daita.adapters.postgresql_query import (
    PostgreSQLQueryBackend,
    PostgreSQLQueryError,
)
from daita.capabilities import CapabilityRegistry
from daita.capability_runtime import CapabilityRuntime
from daita.domains.data import DATA_QUERY_CAPABILITY_ID
from daita.llm.models import ModelProfile, ToolCall
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import RunInput
from daita.security import SecretReference
from tests.catalog._schema_support import _OBSERVED_AT
from tests.data.postgresql._large_fixture_support import ATTACHED_SCHEMAS, _Secrets
from tests.support.capability_runtime import execute_projected
from tests.support.workspace import workspace_for

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get("DAITA_RUN_POSTGRES_LARGE_FIXTURE") != "1",
        reason=(
            "set DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 after starting "
            "tests/fixtures/postgres-large/compose.yaml"
        ),
    ),
]

_PORT = int(os.environ.get("DAITA_LARGE_POSTGRES_PORT", "55433"))


async def _admin():
    import asyncpg  # type: ignore[import-untyped]

    return await asyncpg.connect(
        host="127.0.0.1",
        port=_PORT,
        database="daita_large_fixture",
        user="postgres",
        password=os.environ.get(
            "DAITA_LARGE_POSTGRES_ADMIN_PASSWORD", "fixture_admin_password"
        ),
        ssl=False,
        timeout=5,
        command_timeout=10,
    )


async def _agent(
    tmp_path: Path,
    *,
    username: str,
    password: str,
    schemas: tuple[str, ...],
) -> tuple[Agent, str, dict[str, str]]:
    provider = MockModelProvider(())
    agent = await Agent.create(
        f"query-bounds-{username}",
        root=tmp_path / username,
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=80_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
        secret_provider=_Secrets(password),
        workspace=workspace_for(tmp_path / username),
    )
    source = await agent.attach_postgresql(
        host="127.0.0.1",
        port=_PORT,
        database="daita_large_fixture",
        username=username,
        credential=SecretReference.keychain(f"fixture:postgres-large:{username}"),
        schemas=schemas,
        ssl_mode="disable",
    )
    resources = await agent.list_catalog_resources(source_id=source.id)
    return agent, source.id, {item.native_identity: item.id for item in resources}


async def _query(
    agent: Agent, source_id: str, arguments: Mapping[str, object]
) -> Mapping[str, object]:
    loop = agent._embedded._loop
    assert loop is not None
    runtime = cast(CapabilityRuntime, loop._tools)
    (result,) = await execute_projected(
        runtime,
        RunInput(
            id="query-bounds-run",
            agent_id=agent.id,
            message="query",
            created_at=_OBSERVED_AT,
            conversation_id="query-bounds-conversation",
            source_scope_ids=(source_id,),
        ),
        (ToolCall(id="bounded", name="data_query", arguments=arguments),),
    )
    return cast(Mapping[str, object], result.output)


def _quote(name: str) -> str:
    schema, table = name.split(".", 1)
    return ".".join('"' + part.replace('"', '""') + '"' for part in (schema, table))


async def test_one_read_counts_every_attached_table_including_quoted(
    tmp_path: Path,
):
    agent, source_id, by_name = await _agent(
        tmp_path,
        username="daita_large_reader",
        password=os.environ.get(
            "DAITA_LARGE_POSTGRES_PASSWORD", "daita_large_fixture_password"
        ),
        schemas=ATTACHED_SCHEMAS,
    )
    admin = await _admin()
    try:
        names = sorted(by_name)
        assert len(names) > 4
        assert "core.CustomerSegments" in names
        expected = {
            name: await admin.fetchval(f"SELECT count(*) FROM {_quote(name)}")
            for name in names
        }
        output = await _query(
            agent,
            source_id,
            {
                "source_id": source_id,
                "resource_ids": tuple(by_name[name] for name in names),
                "sql": " UNION ALL ".join(
                    f"SELECT {index} AS position, count(*) AS row_count "
                    f"FROM {_quote(name)}"
                    for index, name in enumerate(names)
                ),
            },
        )
        assert "error" not in output, output
        data = cast(Mapping[str, object], output["data"])
        rows = cast(tuple[Mapping[str, object], ...], data["rows"])
        assert {
            names[cast(int, row["position"])]: row["row_count"] for row in rows
        } == (expected)
        assert len(cast(tuple[object, ...], data["tables_touched"])) == len(names)
    finally:
        await admin.close()
        await agent.close()


async def test_statements_read_empty_tables_and_time_out_with_cleanup(
    tmp_path: Path,
):
    agent, source_id, by_name = await _agent(
        tmp_path,
        username="daita_large_write_tester",
        password=os.environ.get(
            "DAITA_LARGE_POSTGRES_WRITE_TESTER_PASSWORD",
            "daita_large_write_tester_fixture_password",
        ),
        schemas=("write_acceptance",),
    )
    admin = await _admin()
    try:
        cells = await admin.fetchval("SELECT count(*) FROM write_acceptance.cells")
        companies = await admin.fetchval(
            "SELECT count(*) FROM write_acceptance.companies"
        )
        output = await _query(
            agent,
            source_id,
            {
                "source_id": source_id,
                "resource_ids": (
                    by_name["write_acceptance.cells"],
                    by_name["write_acceptance.companies"],
                ),
                "statements": (
                    {"sql": "SELECT count(*) AS n FROM write_acceptance.cells"},
                    {
                        "sql": "SELECT count(*) AS n FROM write_acceptance.companies "
                        "WHERE id > $1",
                        "parameters": (-1,),
                    },
                ),
            },
        )
        assert "error" not in output, output
        statements = cast(
            tuple[Mapping[str, object], ...],
            cast(Mapping[str, object], output["data"])["statements"],
        )
        assert [
            cast(tuple[Mapping[str, object], ...], item["rows"])[0]["n"]
            for item in statements
        ] == [cells, companies]

    finally:
        await admin.close()
        await agent.close()


async def test_statement_deadline_cancels_the_server_query_and_cleans_up(
    tmp_path: Path,
):
    agent, source_id, _by_name = await _agent(
        tmp_path,
        username="daita_large_reader",
        password=os.environ.get(
            "DAITA_LARGE_POSTGRES_PASSWORD", "daita_large_fixture_password"
        ),
        schemas=ATTACHED_SCHEMAS,
    )
    admin = await _admin()
    try:
        registry: CapabilityRegistry = agent._embedded._capabilities
        _capability, executor = registry.resolve_execution(DATA_QUERY_CAPABILITY_ID)
        backend = executor._backends["postgresql"]  # type: ignore[attr-defined]
        assert isinstance(backend, PostgreSQLQueryBackend)
        with pytest.raises(PostgreSQLQueryError) as timed_out:
            await backend.execute_read(
                agent_id=agent.id,
                source_id=source_id,
                sql=(
                    "SELECT count(*) FROM billing.invoice_lines AS a "
                    "CROSS JOIN billing.invoices AS b"
                ),
                parameters=(),
                max_rows=10,
                max_bytes=1_024,
                timeout_seconds=0.3,
            )
        assert timed_out.value.code == "postgresql_query_timeout"
        active = await admin.fetchval(
            "SELECT count(*) FROM pg_stat_activity "
            "WHERE usename = 'daita_large_reader' AND state <> 'idle'"
        )
        assert active == 0

        recovered = await _query(
            agent,
            source_id,
            {
                "source_id": source_id,
                "resource_ids": (_by_name["billing.invoices"],),
                "sql": "SELECT count(*) AS n FROM billing.invoices",
            },
        )
        assert "error" not in recovered, recovered
    finally:
        await admin.close()
        await agent.close()
