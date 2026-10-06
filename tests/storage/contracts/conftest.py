"""Run portable contracts against each implemented durable backend."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from daita.storage.protocols import StateStore
from daita.storage.sqlite import SQLiteStateStore
from tests.storage._support import StateStoreFactory
from tests.support.graph import GRAPH_NOW


def pytest_generate_tests(metafunc):
    if "state_store_backend" in metafunc.fixturenames:
        backends = (
            ["sqlite", "postgres"]
            if metafunc.config.getoption("--postgres")
            else ["sqlite"]
        )
        metafunc.parametrize("state_store_backend", backends)


@pytest.fixture
def state_store_factory(
    request: pytest.FixtureRequest, tmp_path: Path, state_store_backend: str
) -> StateStoreFactory:
    postgres_config = (
        request.getfixturevalue("postgres_config")
        if state_store_backend == "postgres"
        else None
    )

    @asynccontextmanager
    async def open_store() -> AsyncIterator[StateStore]:
        if postgres_config is not None:
            from daita.storage.postgres import PostgresStateStore

            store: StateStore = await PostgresStateStore.open(
                postgres_config, clock=lambda: GRAPH_NOW
            )
        else:
            store = await SQLiteStateStore.open(
                tmp_path / "state.db", clock=lambda: GRAPH_NOW
            )
        try:
            yield store
        finally:
            await store.close()

    return open_store


@pytest.fixture
async def state_store(
    state_store_factory: StateStoreFactory,
) -> AsyncIterator[StateStore]:
    async with state_store_factory() as store:
        yield store
