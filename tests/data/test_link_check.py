"""A link check reads two live columns and reports how well they join."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from daita import Agent, FolderSource, SQLiteSource
from daita.adapters.registry import SourceOperation
from daita.catalog.protocols import CatalogStoreError
from daita.config import AgentConfig
from daita.domains.data.key_read import KeyEndpoint, KeyReadError, checked_key_read
from daita.hosting import embedded as embedded_module
from daita.llm.providers.mock import MockModelProvider
from tests.support.memory_source import MEMORY_NOW, MEMORY_TYPE, MemorySource
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.integration


def _table(
    path: Path, name: str, columns: str, rows: Iterable[tuple[Any, ...]]
) -> Path:
    width = len(columns.split(","))
    with sqlite3.connect(path) as connection:
        connection.execute(f"CREATE TABLE {name} ({columns})")
        connection.executemany(
            f"INSERT INTO {name} VALUES ({', '.join('?' * width)})", list(rows)
        )
    return path


async def _agent(tmp_path: Path, **options: Any) -> Agent:
    provider = MockModelProvider(())
    return await Agent.create(
        "link-check",
        root=tmp_path / "state",
        workspace=workspace_for(tmp_path / "state"),
        model=provider,
        model_profile=provider.model_profile,
        **options,
    )


async def test_match_rate_across_two_sources(tmp_path: Path) -> None:
    orders = _table(
        tmp_path / "orders.sqlite",
        "orders",
        "customer_id INTEGER",
        ((i,) for i in range(1, 101)),
    )
    crm = _table(
        tmp_path / "crm.sqlite",
        "customers",
        "id INTEGER",
        [(i,) for i in range(3, 101)] + [(500,), (501,)],
    )
    agent = await _agent(tmp_path)
    try:
        left = await agent.attach(SQLiteSource(orders, name="Orders"))
        right = await agent.attach(SQLiteSource(crm, name="CRM"))
        check = await agent.check_link(
            left.id, "orders", ("customer_id",), right.id, "customers", ("id",)
        )
    finally:
        await agent.close()

    assert check.error_code is None
    assert (check.from_keys, check.to_keys) == (100, 100)
    assert (check.matched_from, check.matched_to) == (98, 98)
    assert check.match_rate == 0.98
    assert not check.partial
    assert check.from_key_types == check.to_key_types == ("int",)


async def test_one_source_composite_keys_and_text_never_match_numbers(
    tmp_path: Path,
) -> None:
    database = tmp_path / "shop.sqlite"
    _table(
        database, "a", "x INTEGER, y TEXT", [(1, "p"), (2, "q"), (3, "r"), (None, "s")]
    )
    _table(database, "b", "x INTEGER, y TEXT", [(1, "p"), (2, "z"), (3, None)])
    _table(database, "tags", "code TEXT", [("1",), ("2",), (None,)])
    _table(database, "ids", "code INTEGER", [(1,), (2,)])
    agent = await _agent(tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database, name="Shop"))
        # Both sides on one capacity-one source must still finish.
        composite, typed = await asyncio.wait_for(
            asyncio.gather(
                agent.check_link(
                    source.id, "a", ("x", "y"), source.id, "b", ("x", "y")
                ),
                agent.check_link(
                    source.id, "tags", ("code",), source.id, "ids", ("code",)
                ),
            ),
            timeout=10,
        )
    finally:
        await agent.close()

    # Rows with a NULL part are not keys; only (1, "p") is on both sides.
    assert (composite.from_keys, composite.to_keys) == (3, 2)
    assert composite.matched_from == 1
    assert composite.match_rate == pytest.approx(1 / 3)
    assert typed.from_keys == 2
    assert typed.match_rate == 0.0
    assert typed.from_key_types == ("str",)
    assert typed.to_key_types == ("int",)


async def test_a_read_over_the_bound_is_partial_and_says_which_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "wide.sqlite"
    _table(database, "big", "k INTEGER", [(i,) for i in range(10)])
    _table(database, "small", "k INTEGER", [(i,) for i in range(2)])
    monkeypatch.setattr(embedded_module, "LINK_CHECK_MAX_KEYS", 3)
    agent = await _agent(tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database, name="Wide"))
        check = await agent.check_link(
            source.id, "big", ("k",), source.id, "small", ("k",)
        )
    finally:
        await agent.close()

    assert check.partial
    assert check.from_truncated and not check.to_truncated
    assert check.from_keys == 3


async def test_folder_and_tabular_sources_are_read_the_same_way(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "files"
    folder.mkdir()
    (folder / "regions.csv").write_text(
        "region\nnorth\nsouth\nnowhere\n", encoding="utf-8"
    )
    managers = _table(
        tmp_path / "managers.sqlite",
        "managers",
        "region TEXT, manager TEXT",
        [("north", "Ada"), ("south", "Grace"), ("east", "Edsger"), ("west", "Barbara")],
    )
    agent = await _agent(
        tmp_path,
        config=AgentConfig(source_types=(MEMORY_TYPE,)),
        clock=lambda: MEMORY_NOW,
    )
    try:
        files = await agent.attach(FolderSource(folder, name="Files"))
        database = await agent.attach(SQLiteSource(managers, name="Managers"))
        memory = await agent.attach(MemorySource("Regions"))
        from_file = await agent.check_link(
            files.id, "regions.csv", ("region",), database.id, "managers", ("region",)
        )
        to_memory = await agent.check_link(
            database.id, "managers", ("region",), memory.id, "regions", ("region",)
        )
    finally:
        await agent.close()

    assert from_file.error_code is None and from_file.match_rate == pytest.approx(2 / 3)
    assert to_memory.error_code is None and to_memory.match_rate == 1.0
    assert to_memory.to_keys == 4


async def test_a_table_outside_a_narrowed_scope_is_refused_without_a_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "scoped.sqlite"
    _table(database, "open_table", "k INTEGER", [(1,)])
    _table(database, "secret", "k INTEGER", [(1,)])
    agent = await _agent(tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database, name="Scoped"))
        (allowed,) = [
            item
            for item in await agent.list_catalog_resources(source_id=source.id)
            if item.name == "open_table"
        ]
        preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="selected",
            read_resource_ids=(allowed.id,),
            relational_write_scopes={},
        )
        await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=preview.confirmation_fingerprint,
        )
        backend = agent._embedded._source_backends[SourceOperation.SQL_READ][
            source.adapter_id
        ]
        reads: list[object] = []
        real = backend.execute_read

        async def counting(**arguments: Any) -> Any:
            reads.append(arguments)
            return await real(**arguments)

        monkeypatch.setattr(backend, "execute_read", counting)
        refused = await agent.check_link(
            source.id, "open_table", ("k",), source.id, "secret", ("k",)
        )
        permitted = await agent.check_link(
            source.id, "open_table", ("k",), source.id, "open_table", ("k",)
        )
    finally:
        await agent.close()

    assert refused.error_code is not None and refused.match_rate is None
    assert permitted.match_rate == 1.0
    # The refused check read nothing; the permitted one read each side.
    assert len(reads) == 2


async def test_failures_are_results_and_never_carry_driver_text(
    tmp_path: Path,
) -> None:
    database = tmp_path / "plain.sqlite"
    _table(database, "t", "k INTEGER", [(1,)])
    agent = await _agent(tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database, name="Plain"))
        missing = await agent.check_link(
            source.id, "t", ("k",), source.id, "no_such_table", ("k",)
        )
        bad_column = await agent.check_link(
            source.id, "t", ("nope",), source.id, "t", ("k",)
        )
        with pytest.raises(ValueError):
            await agent.check_link(
                "source-unknown", "t", ("k",), source.id, "t", ("k",)
            )
        with pytest.raises(ValueError):
            await agent.check_link(source.id, "t", ("k",), source.id, "t", ("k", "k"))
        with pytest.raises(ValueError):
            await agent.check_catalog_link("link-unknown")
    finally:
        await agent.close()

    for check in (missing, bad_column):
        assert check.error_code is not None
        assert check.match_rate is None and check.from_keys is None
        assert "no such" not in repr(check) and "no_such_table" not in str(
            check.error_code
        )


class _BrokenCatalogStore:
    def __getattr__(self, name: str) -> object:
        raise CatalogStoreError(f"catalog store is unavailable: {name}")


async def test_the_check_needs_no_catalog_and_a_declared_link_checks_by_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "linked.sqlite"
    _table(database, "orders", "customer_id INTEGER", [(1,), (2,), (3,), (4,)])
    _table(database, "customers", "id INTEGER", [(1,), (2,), (3,)])
    agent = await _agent(tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database, name="Linked"))
        resources = {
            item.name: item
            for item in await agent.list_catalog_resources(source_id=source.id)
        }
        link = await agent.declare_catalog_link(
            resources["orders"].id, resources["customers"].id, (("customer_id", "id"),)
        )
        by_id = await agent.check_catalog_link(link.id)

        for owner in (agent._embedded._catalog_service, agent._embedded._data_view):
            monkeypatch.setattr(owner, "_store", _BrokenCatalogStore())
        without_catalog = await agent.check_link(
            source.id, "orders", ("customer_id",), source.id, "customers", ("id",)
        )
    finally:
        await agent.close()

    assert by_id.error_code is None
    assert (by_id.from_table, by_id.to_table) == ("main.orders", "main.customers")
    assert by_id.match_rate == 0.75
    assert without_catalog.match_rate == 0.75


async def test_a_tabular_table_outside_a_narrowed_scope_is_refused_without_a_read(
    tmp_path: Path,
) -> None:
    agent = await _agent(
        tmp_path,
        config=AgentConfig(source_types=(MEMORY_TYPE,)),
        clock=lambda: MEMORY_NOW,
    )
    try:
        memory = await agent.attach(MemorySource("Regions"))
        (regions,) = await agent.list_catalog_resources(source_id=memory.id)
        preview = await agent.preview_source_permissions(
            source_id=memory.id,
            read_mode="selected",
            read_resource_ids=(regions.id,),
            relational_write_scopes={},
        )
        await agent.apply_source_permissions(
            source_id=memory.id,
            confirmation_fingerprint=preview.confirmation_fingerprint,
        )
        backend = agent._embedded._source_backends[SourceOperation.TABULAR_READ][
            memory.adapter_id
        ]
        refused = await agent.check_link(
            memory.id, "other", ("region",), memory.id, "regions", ("region",)
        )
        permitted = await agent.check_link(
            memory.id, "regions", ("region",), memory.id, "regions", ("region",)
        )
        reads = len(backend.requests)
    finally:
        await agent.close()

    assert refused.error_code is not None and refused.match_rate is None
    assert permitted.match_rate == 1.0
    assert reads == 2  # only the permitted check read


def test_postgresql_key_reads_validate_and_respect_a_narrowed_scope() -> None:
    endpoint = KeyEndpoint("source-pg", "sales.orders", ("customer_id",))
    read = checked_key_read(endpoint, "postgresql", access=None, schemas=("sales",))

    assert read is not None
    assert [name for _, name in read.resource_names] == ["sales.orders"]
    with pytest.raises(KeyReadError) as refused:
        checked_key_read(
            endpoint, "postgresql", access=frozenset({"other-id"}), schemas=("sales",)
        )
    assert refused.value.error_code == "key_read_out_of_scope"
