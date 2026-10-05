"""Run portable contracts against each implemented durable backend."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from daita.storage.protocols import StateStore
from daita.storage.sqlite import SQLiteStateStore
from tests.storage._support import StateStoreFactory
from tests.support.graph import GRAPH_NOW


@pytest.fixture(params=["sqlite"])
def state_store_factory(
    request: pytest.FixtureRequest, tmp_path: Path
) -> StateStoreFactory:
    # Add Postgres only with a real disposable database implementation. Missing
    # prerequisites must fail qualification, never silently skip that backend.
    assert request.param == "sqlite"

    @asynccontextmanager
    async def open_store() -> AsyncIterator[StateStore]:
        store: StateStore = await SQLiteStateStore.open(
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
