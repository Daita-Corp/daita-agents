"""A PostgreSQL change probe moves exactly when a catalog read would record a change.

The probe is one statement per check. It must move for every structural thing a
revision stores and for every grant, revoke and role change that alters what the
connecting role can read, and it must stay still for what a revision does not
store (rows, statistics, table comments, enum labels).

Needs the disposable postgres-large fixture server
(tests/fixtures/postgres-large/compose.yaml). Every case but the last runs
inside one transaction on its ``postgres`` database that always rolls back, so
it creates its schemas, tables and roles there and leaves nothing behind. The
last case only reads: it attaches the fixture database through a real agent:

    DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 DAITA_LARGE_POSTGRES_ADMIN_PASSWORD=... \
    pytest tests/live/data/test_change_probe.py -s
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from daita import Agent
from daita.adapters.models import DiscoveryRequest, SourceProbe, SourceRegistration
from daita.adapters.postgresql import _batches, _load_structure, probe_tokens
from daita.catalog.models import (
    CatalogSync,
    CatalogSyncStatus,
    ResourceKind,
    catalog_resource_id,
)
from daita.catalog.search_text import tabular_parts
from daita.llm.models import ModelProfile
from daita.llm.providers.mock import MockModelProvider
from daita.security import SecretReference
from daita.semantics import (
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticResourceFact,
    SemanticSubject,
    inspect_semantic_annotations,
)
from daita.storage.sqlite import SQLiteStateStore
from tests.data.postgresql._large_fixture_support import ATTACHED_SCHEMAS, _Secrets
from tests.support.source_contract import (
    ProbeMutation,
    assert_change_detection_contract,
)
from tests.support.workspace import workspace_for

pytestmark = [
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get("DAITA_RUN_POSTGRES_LARGE_FIXTURE") != "1"
        or not os.environ.get("DAITA_LARGE_POSTGRES_ADMIN_PASSWORD"),
        reason=(
            "Start tests/fixtures/postgres-large/compose.yaml, then set "
            "DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 and "
            "DAITA_LARGE_POSTGRES_ADMIN_PASSWORD."
        ),
    ),
]

_MAIN = "daita_probe_main"
_AUX = "daita_probe_aux"
_WIDE = "daita_probe_wide"
_ROLE = "daita_probe_reader"
_GRANTOR = "daita_probe_grantor"
_NOW = datetime(2026, 11, 7, tzinfo=UTC)


@pytest.fixture
async def server() -> AsyncIterator[Any]:
    import asyncpg  # type: ignore[import-untyped]

    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=int(os.environ.get("DAITA_LARGE_POSTGRES_PORT", "55433")),
        database="postgres",
        user="postgres",
        password=os.environ["DAITA_LARGE_POSTGRES_ADMIN_PASSWORD"],
        ssl=False,
        timeout=10,
        command_timeout=60,
    )
    # The same isolation lets discovery's own transaction nest as a savepoint.
    transaction = connection.transaction(isolation="repeatable_read")
    await transaction.start()
    try:
        assert await connection.fetchval("SELECT current_database()") == "postgres"
        await connection.execute(
            f"CREATE SCHEMA {_MAIN}; CREATE SCHEMA {_AUX}; CREATE SCHEMA {_WIDE};"
            f"CREATE ROLE {_ROLE}; CREATE ROLE {_GRANTOR};"
            f"GRANT USAGE ON SCHEMA {_MAIN}, {_AUX}, {_WIDE} TO {_ROLE}, {_GRANTOR}"
        )
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()


def _registration(*schemas: str) -> SourceRegistration:
    return SourceRegistration.build(
        agent_id="agent",
        adapter_id="postgresql",
        native_identity="postgres-large/postgres",
        display_name="Change probe",
        configuration={"schemas": schemas},
        attached_at=_NOW,
    )


@dataclass
class _Counting:
    """Passes everything to the connection and records each tagged statement."""

    inner: Any
    tags: list[str] = field(default_factory=list)

    async def fetch(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        if match := re.search(r"/\* (daita:postgresql\.\w+) \*/", sql):
            self.tags.append(match[1])
        return await self.inner.fetch(sql, *args, **kwargs)


class _Watched:
    """One role's view of the schemas, and an administrator who changes them."""

    def __init__(self, server: Any, role: str, *schemas: str) -> None:
        self.server, self.role, self.schemas = server, role, schemas
        self.registration = _registration(*schemas)

    async def probe(self) -> SourceProbe:
        await self.server.execute(f"SET LOCAL ROLE {self.role}")
        try:
            tokens = await probe_tokens(self.server, self.schemas)
        finally:
            await self.server.execute("RESET ROLE")
        return SourceProbe(checked_at=_NOW, tokens=tokens)

    async def revisions(self) -> dict[str, dict[str, str]]:
        """What a catalog read as this role records now: schema -> table -> revision."""

        await self.server.execute(f"SET LOCAL ROLE {self.role}")
        try:
            structure = await _load_structure(
                self.server,
                self.registration,
                max_resources=1_000,
                max_columns=512,
                max_indexes=256,
                max_relationships=2_000,
            )
        finally:
            await self.server.execute("RESET ROLE")
        request = DiscoveryRequest(
            agent_id="agent",
            source_id=self.registration.id,
            sync_id="sync",
            requested_at=_NOW,
        )
        _, detail = _batches(request, structure)
        result: dict[str, dict[str, str]] = {}
        for entry in detail.entries:
            schema, _, _ = entry.resource.native_identity.partition(".")
            result.setdefault(schema, {})[
                entry.resource.native_identity
            ] = entry.revision.revision
        return result

    def change(self, *statements: str) -> Callable[[], Any]:
        async def apply() -> None:
            for statement in statements:
                await self.server.execute(statement)

        return apply


async def _build_structure(server: Any) -> None:
    await server.execute(f"""
        CREATE TYPE {_MAIN}.mood AS ENUM ('sad', 'ok');
        CREATE TABLE {_MAIN}.people (id int, name text, mood {_MAIN}.mood);
        CREATE TABLE {_MAIN}.orders (id int PRIMARY KEY, person_id int);
        CREATE TABLE {_AUX}.items (id int PRIMARY KEY, order_id int);
        CREATE TABLE {_MAIN}.events (id int, day date) PARTITION BY RANGE (day);
        CREATE TABLE {_MAIN}.events_2026 PARTITION OF {_MAIN}.events
            FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
        CREATE TABLE {_MAIN}.events_2027 (id int, day date);
        CREATE VIEW {_MAIN}.people_view AS SELECT id FROM {_MAIN}.people;
        GRANT SELECT ON ALL TABLES IN SCHEMA {_MAIN}, {_AUX} TO {_ROLE}
        """)


async def test_probe_moves_exactly_with_what_a_catalog_read_records(
    server: Any,
) -> None:
    await _build_structure(server)
    watched = _Watched(server, _ROLE, _MAIN, _AUX)
    people, orders = f"{_MAIN}.people", f"{_MAIN}.orders"
    main, both = frozenset({_MAIN}), frozenset({_MAIN, _AUX})

    def move(
        name: str, *statements: str, moves: frozenset[str] = main
    ) -> ProbeMutation:
        return ProbeMutation(name, watched.change(*statements), moves)

    def still(name: str, *statements: str) -> ProbeMutation:
        return ProbeMutation(name, watched.change(*statements))

    await assert_change_detection_contract(
        watched.probe,
        [
            still("rows inserted", f"INSERT INTO {people} VALUES (1, 'a', 'ok')"),
            still("statistics refreshed", f"ANALYZE {people}"),
            still("table comment", f"COMMENT ON TABLE {people} IS 'who we know'"),
            still("enum label added", f"ALTER TYPE {_MAIN}.mood ADD VALUE 'happy'"),
            still("table nobody may read", f"CREATE TABLE {_MAIN}.hidden (id int)"),
            move(
                "table added",
                f"CREATE TABLE {_MAIN}.extra (id int)",
                f"GRANT SELECT ON {_MAIN}.extra TO {_ROLE}",
            ),
            move("table renamed", f"ALTER TABLE {_MAIN}.extra RENAME TO more"),
            move("table dropped", f"DROP TABLE {_MAIN}.more"),
            move("column added", f"ALTER TABLE {people} ADD COLUMN age int"),
            move("column renamed", f"ALTER TABLE {people} RENAME COLUMN age TO years"),
            move(
                "column type changed",
                f"ALTER TABLE {people} ALTER name TYPE varchar(20)",
            ),
            move(
                "column made required", f"ALTER TABLE {people} ALTER name SET NOT NULL"
            ),
            move(
                "column default set", f"ALTER TABLE {people} ALTER years SET DEFAULT 7"
            ),
            move("column comment", f"COMMENT ON COLUMN {people}.name IS 'full name'"),
            move("column dropped", f"ALTER TABLE {people} DROP COLUMN years"),
            move("index added", f"CREATE INDEX people_name ON {people} (name)"),
            move("index dropped", "DROP INDEX " + f"{_MAIN}.people_name"),
            move("primary key added", f"ALTER TABLE {people} ADD PRIMARY KEY (id)"),
            move(
                "foreign key added",
                f"ALTER TABLE {orders} ADD CONSTRAINT o_p "
                f"FOREIGN KEY (person_id) REFERENCES {people} (id)",
            ),
            move("foreign key dropped", f"ALTER TABLE {orders} DROP CONSTRAINT o_p"),
            move(
                "primary key dropped",
                f"ALTER TABLE {people} DROP CONSTRAINT people_pkey",
            ),
            move(
                "foreign key across schemas added",
                f"ALTER TABLE {_AUX}.items ADD CONSTRAINT i_o "
                f"FOREIGN KEY (order_id) REFERENCES {orders} (id)",
                moves=both,
            ),
            move(
                "foreign key across schemas dropped",
                f"ALTER TABLE {_AUX}.items DROP CONSTRAINT i_o",
                moves=both,
            ),
            move(
                "view created",
                f"CREATE VIEW {_MAIN}.names AS SELECT id FROM {people}",
                f"GRANT SELECT ON {_MAIN}.names TO {_ROLE}",
            ),
            move(
                "view replaced with another column",
                f"CREATE OR REPLACE VIEW {_MAIN}.names AS SELECT id, name FROM {people}",
            ),
            still(
                "view replaced with the same columns",
                f"CREATE OR REPLACE VIEW {_MAIN}.names AS "
                f"SELECT id, name FROM {people} WHERE id > 0",
            ),
            move(
                "partition attached",
                f"ALTER TABLE {_MAIN}.events ATTACH PARTITION {_MAIN}.events_2027 "
                "FOR VALUES FROM ('2027-01-01') TO ('2028-01-01')",
            ),
            move(
                "partition detached",
                f"ALTER TABLE {_MAIN}.events DETACH PARTITION {_MAIN}.events_2027",
            ),
        ],
        revisions=watched.revisions,
    )


async def test_probe_moves_with_grants_revokes_and_role_membership(
    server: Any,
) -> None:
    """Measured as the probing role: what it may read is what the probe hashes."""

    await server.execute(f"""
        CREATE TABLE {_MAIN}.a (id int PRIMARY KEY, secret text, note text);
        CREATE TABLE {_MAIN}.b (id int);
        GRANT SELECT ON {_MAIN}.a TO {_ROLE}
        """)
    watched = _Watched(server, _ROLE, _MAIN)
    main = frozenset({_MAIN})

    def move(name: str, *statements: str) -> ProbeMutation:
        return ProbeMutation(name, watched.change(*statements), main)

    await assert_change_detection_contract(
        watched.probe,
        [
            move("grant select", f"GRANT SELECT ON {_MAIN}.b TO {_ROLE}"),
            move("revoke select", f"REVOKE SELECT ON {_MAIN}.b FROM {_ROLE}"),
            move(
                "revoke one column of a table grant",
                f"REVOKE SELECT ON {_MAIN}.a FROM {_ROLE}",
                f"GRANT SELECT (id, note) ON {_MAIN}.a TO {_ROLE}",
            ),
            move("grant a column", f"GRANT SELECT (secret) ON {_MAIN}.a TO {_ROLE}"),
            move(
                "revoke a column", f"REVOKE SELECT (secret) ON {_MAIN}.a FROM {_ROLE}"
            ),
            move(
                "role membership granted",
                f"GRANT SELECT ON {_MAIN}.b TO {_GRANTOR}",
                f"GRANT {_GRANTOR} TO {_ROLE}",
            ),
            move("role membership revoked", f"REVOKE {_GRANTOR} FROM {_ROLE}"),
            move(
                "schema usage revoked", f"REVOKE USAGE ON SCHEMA {_MAIN} FROM {_ROLE}"
            ),
            move("schema usage granted", f"GRANT USAGE ON SCHEMA {_MAIN} TO {_ROLE}"),
        ],
        revisions=watched.revisions,
    )


async def test_notes_follow_schema_changes_and_only_real_damage_makes_one_stale(
    server: Any, tmp_path: Path
) -> None:
    """A note goes stale for a dropped or retyped field or table, never for noise."""

    watched = _Watched(server, _ROLE, _MAIN, _AUX)
    registration = watched.registration
    await server.execute(f"""
        CREATE TABLE {_MAIN}.dim (id int PRIMARY KEY);
        CREATE TABLE {_MAIN}.nt0 (id int, fa int, fb text, pad int);
        CREATE TABLE {_MAIN}.nt1 (id int, fa int, fb text, pad int);
        CREATE TABLE {_MAIN}.nt2 (id int, fa int, fb text, pad int);
        CREATE TABLE {_MAIN}.nt3 (id int, fa int, fb text, pad int);
        GRANT SELECT ON ALL TABLES IN SCHEMA {_MAIN} TO {_ROLE}
        """)
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
        await store.register_source(registration)
        counter = {"sync": 0, "changes": 0, "false_stale": 0, "missed": 0}

        async def commit() -> None:
            counter["sync"] += 1
            sync_id = f"sync-{counter['sync']}"
            await server.execute(f"SET LOCAL ROLE {_ROLE}")
            try:
                structure = await _load_structure(
                    server,
                    registration,
                    max_resources=1_000,
                    max_columns=512,
                    max_indexes=256,
                    max_relationships=2_000,
                )
            finally:
                await server.execute("RESET ROLE")
            request = DiscoveryRequest(
                agent_id="agent",
                source_id=registration.id,
                sync_id=sync_id,
                requested_at=_NOW,
            )
            inventory, detail = _batches(request, structure)
            moment = _NOW + timedelta(seconds=counter["sync"])
            await store.commit_catalog(
                CatalogSync(
                    id=sync_id,
                    agent_id="agent",
                    source_id=registration.id,
                    adapter_id="postgresql",
                    status=CatalogSyncStatus.SUCCEEDED,
                    started_at=moment,
                    completed_at=moment,
                    resource_count=len(inventory.entries),
                    relationship_count=len(detail.relationships),
                ),
                (inventory, detail),
            )

        async def stale_notes() -> set[str]:
            facts = []
            for entry in await store.list_entries("agent", registration.id):
                columns, _ = tabular_parts(entry)
                facts.append(
                    SemanticResourceFact(
                        entry.resource.id,
                        registration.id,
                        entry.resource.current_revision,
                        tuple(name for name, _ in columns),
                        tuple(columns),
                    )
                )
            return {
                note.id
                for note in await store.list_semantic_annotations("agent")
                if inspect_semantic_annotations((note,), tuple(facts))[0].stale_reasons
            }

        expected: set[str] = set()

        async def step(*statements: str) -> None:
            for statement in statements:
                await server.execute(statement)
            await commit()
            counter["changes"] += 1
            stale = await stale_notes()
            counter["false_stale"] += len(stale - expected)
            counter["missed"] += len(expected - stale)

        names = {
            i: {"schema": _MAIN, "table": f"nt{i}", "fa": "fa", "fb": "fb"}
            for i in range(4)
        }

        def rid(i: int) -> str:
            where = f"{names[i]['schema']}.{names[i]['table']}"
            return catalog_resource_id(registration.id, ResourceKind.TABLE, where)

        await commit()
        statements = {}
        for entry in await store.list_entries("agent", registration.id):
            where = entry.resource.native_identity
            for i in range(4):
                if where != f"{_MAIN}.nt{i}":
                    continue
                for suffix, field in (("none", None), ("a", "fa"), ("b", "fb")):
                    note_id, resource_id = f"t{i}-{suffix}", entry.resource.id
                    statements[note_id] = f"Note {note_id}."
                    await store.save_semantic_annotation(
                        "agent",
                        SemanticAnnotation(
                            id=note_id,
                            agent_id="agent",
                            subject=SemanticSubject(
                                (registration.id,),
                                (resource_id,),
                                (
                                    ()
                                    if field is None
                                    else (SemanticFieldReference(resource_id, field),)
                                ),
                            ),
                            kind=SemanticKind.METRIC_DEFINITION,
                            statement=statements[note_id],
                            evidence=(
                                SemanticEvidence(
                                    SemanticEvidenceKind.USER_ASSERTION,
                                    "run-1",
                                    message_position=0,
                                ),
                            ),
                            catalog_revisions=(
                                ResourceRevisionBinding(
                                    resource_id,
                                    entry.resource.current_revision,
                                    tuple(
                                        t
                                        for t in tabular_parts(entry)[0]
                                        if t[0] == field
                                    ),
                                ),
                            ),
                            created_at=_NOW,
                            confirmed_at=_NOW,
                        ),
                    )
        assert len(statements) == 12 and not await stale_notes()

        async def notes_follow_their_fields() -> None:
            for note in await store.list_semantic_annotations("agent"):
                i = int(note.id[1])
                assert note.statement == statements[note.id]
                assert note.subject.resource_ids == (rid(i),)
                field = {"a": "fa", "b": "fb"}.get(note.id.split("-")[1])
                assert note.subject.fields == (
                    ()
                    if field is None
                    else (SemanticFieldReference(rid(i), names[i][field]),)
                )

        for k in range(5):
            i = k % 4
            t = names[i]

            def q(t: dict[str, str] = t) -> str:
                return f"{t['schema']}.{t['table']}"

            dim = f"{_MAIN}.dim"
            await step(f"ALTER TABLE {q()} ADD COLUMN x1 int")
            await step(f"ALTER TABLE {q()} DROP COLUMN x1")
            await step(f"ALTER TABLE {q()} ADD COLUMN x2 int")
            await step(f"ALTER TABLE {q()} ALTER x2 TYPE bigint")
            await step(f"ALTER TABLE {q()} ALTER x2 TYPE text")
            await step(f"ALTER TABLE {q()} DROP COLUMN x2")
            await step(f"COMMENT ON COLUMN {q()}.pad IS 'padding'")
            await step(f"COMMENT ON COLUMN {q()}.pad IS NULL")
            await step(f"CREATE INDEX idx{k} ON {q()} (pad)")
            await step(f"DROP INDEX {t['schema']}.idx{k}")
            await step(f"ALTER TABLE {q()} ADD CONSTRAINT pk{k} PRIMARY KEY (pad)")
            await step(f"ALTER TABLE {q()} DROP CONSTRAINT pk{k}")
            await step(
                f"ALTER TABLE {q()} ADD CONSTRAINT fk{k} "
                f"FOREIGN KEY (pad) REFERENCES {dim} (id)"
            )
            await step(f"ALTER TABLE {q()} DROP CONSTRAINT fk{k}")
            await step(
                f"CREATE TABLE {_MAIN}.tmp{k} (id int, v text)",
                f"GRANT SELECT ON {_MAIN}.tmp{k} TO {_ROLE}",
            )
            await step(f"DROP TABLE {_MAIN}.tmp{k}")
            await step(f"GRANT INSERT ON {dim} TO {_ROLE}")
            await step(f"REVOKE INSERT ON {dim} FROM {_ROLE}")
            await step(f"REVOKE SELECT ON {dim} FROM {_ROLE}")
            await step(f"GRANT SELECT ON {dim} TO {_ROLE}")
            renamed = f"{t['table']}_r{k}"
            await step(f"ALTER TABLE {q()} RENAME TO {renamed}")
            t["table"] = renamed
            await notes_follow_their_fields()
            field = "fa" if k % 2 == 0 else "fb"
            await step(f"ALTER TABLE {q()} RENAME COLUMN {t[field]} TO {t[field]}_r{k}")
            t[field] = f"{t[field]}_r{k}"
            await notes_follow_their_fields()
            await step(f"ALTER TABLE {q()} SET SCHEMA {_AUX}")
            t["schema"] = _AUX
            await notes_follow_their_fields()
            await step(f"ALTER TABLE {q()} SET SCHEMA {_MAIN}")
            t["schema"] = _MAIN
            await notes_follow_their_fields()

        quiet = counter["changes"]
        assert quiet >= 100 and not await stale_notes()

        # Real damage, one table at a time: only the notes that depend on it go stale.
        t0, t1, t2 = (names[i] for i in range(3))
        expected.add("t0-a")
        await step(f"ALTER TABLE {t0['schema']}.{t0['table']} DROP COLUMN {t0['fa']}")
        expected.add("t1-b")
        await step(
            f"ALTER TABLE {t1['schema']}.{t1['table']} ALTER {t1['fb']} TYPE varchar(5)"
        )
        expected.add("t2-b")
        await step(
            f"DROP TABLE {t2['schema']}.{t2['table']}",
            f"CREATE TABLE {t2['schema']}.{t2['table']} (id int, {t2['fa']} int, pad int)",
            f"GRANT SELECT ON {t2['schema']}.{t2['table']} TO {_ROLE}",
        )
        assert await stale_notes() == {"t0-a", "t1-b", "t2-b"}

        print(
            f"\nschema changes: {counter['changes']} "
            f"(first {quiet} unrelated), false_stale: {counter['false_stale']}, "
            f"missed: {counter['missed']}"
        )
        assert counter["false_stale"] == 0 and counter["missed"] == 0
    finally:
        await store.close()


async def _build_wide_tables(server: Any, count: int) -> None:
    """``count`` tables of 30 columns, a key, a second index and some comments."""

    types = ("int", "bigint", "bool", "date", "timestamptz", f"{_WIDE}.state")
    columns = ", ".join(f"c{i:02} {types[i % len(types)]}" for i in range(1, 30))
    await server.execute(
        f"CREATE TYPE {_WIDE}.state AS ENUM ('new', 'open', 'done');"
        f"GRANT USAGE ON TYPE {_WIDE}.state TO {_ROLE}"
    )
    await server.execute(
        f"""
        DO $$
        BEGIN
            FOR i IN 1..{count} LOOP
                EXECUTE format(
                    'CREATE TABLE {_WIDE}.%I (id bigint PRIMARY KEY, {columns})',
                    'wide_' || lpad(i::text, 4, '0'));
                EXECUTE format('CREATE INDEX ON {_WIDE}.%I (c02)',
                    'wide_' || lpad(i::text, 4, '0'));
                IF i % 3 = 0 THEN
                    EXECUTE format('COMMENT ON COLUMN {_WIDE}.%I.c01 IS %L',
                        'wide_' || lpad(i::text, 4, '0'), 'a comment');
                END IF;
            END LOOP;
        END $$
        """,
        timeout=600,
    )
    await server.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {_WIDE} TO {_ROLE}")


async def test_probing_fifteen_hundred_tables_is_one_quick_statement(
    server: Any,
) -> None:
    await _build_wide_tables(server, 1_500)
    await server.execute(f"SET LOCAL ROLE {_ROLE}")
    counting = _Counting(server)

    seconds = []
    for _ in range(3):
        started = time.perf_counter()
        tokens = await probe_tokens(counting, (_WIDE,))
        seconds.append(time.perf_counter() - started)

    print(
        "\nprobe of 1,500 tables:",
        ", ".join(f"{value * 1_000:.0f} ms" for value in seconds),
    )
    assert list(tokens) == [_WIDE]
    assert counting.tags == ["daita:postgresql.change_probe"] * 3


async def test_fill_describes_what_lacks_detail_and_keeps_the_rest(
    server: Any,
) -> None:
    await _build_wide_tables(server, 6)
    await server.execute(f"SET LOCAL ROLE {_ROLE}")
    registration = _registration(_WIDE)
    request = DiscoveryRequest(
        agent_id="agent", source_id=registration.id, sync_id="s", requested_at=_NOW
    )

    async def read(**options: Any) -> Any:
        structure = await _load_structure(
            server,
            registration,
            max_columns=512,
            max_indexes=256,
            max_relationships=2_000,
            **options,
        )
        return structure, _batches(request, structure)[1].entries

    first, kept = await read(max_resources=2)
    assert len(first.tables) == 2 and first.truncated
    middle, kept = await read(max_resources=2, previous=kept, fill=True)
    assert len(middle.tables) == 4 and middle.truncated
    last, kept = await read(max_resources=2, previous=kept, fill=True)
    assert len(last.tables) == 6 and not last.truncated


async def test_attached_agent_stores_the_probe_tokens_the_role_reads(
    tmp_path: Path,
) -> None:
    import asyncpg

    password = os.environ.get(
        "DAITA_LARGE_POSTGRES_PASSWORD", "daita_large_fixture_password"
    )
    port = int(os.environ.get("DAITA_LARGE_POSTGRES_PORT", "55433"))
    provider = MockModelProvider(())
    agent = await Agent.create(
        "change-probe-attached",
        root=tmp_path,
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=80_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
        secret_provider=_Secrets(password),
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach_postgresql(
            host="127.0.0.1",
            port=port,
            database="daita_large_fixture",
            username="daita_large_reader",
            credential=SecretReference.keychain(
                "fixture:postgres-large:daita_large_reader"
            ),
            schemas=ATTACHED_SCHEMAS,
            ssl_mode="disable",
        )
        state = await agent._embedded._store.load_source_state(agent.id, source.id)
        assert state is not None and state.probe_tokens
        connection = await asyncpg.connect(
            host="127.0.0.1",
            port=port,
            database="daita_large_fixture",
            user="daita_large_reader",
            password=password,
            ssl=False,
            timeout=5,
        )
        try:
            direct = await probe_tokens(connection, ATTACHED_SCHEMAS)
        finally:
            await connection.close()
        assert dict(state.probe_tokens) == direct
        assert (await agent.catalog_status(source.id)).error_code is None
    finally:
        await agent.close()
