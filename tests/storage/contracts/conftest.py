"""Run portable contracts against each implemented durable backend."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from daita.storage.protocols import StateStore
from daita.storage.sqlite import SQLiteStateStore
from tests.artifacts.byte_storage_support import MemoryByteStorage
from tests.storage._support import StateStoreFactory
from tests.support.graph import GRAPH_NOW


@pytest.fixture
def state_store_factory(
    request: pytest.FixtureRequest, tmp_path: Path
) -> StateStoreFactory:
    # Downstream suites can indirectly parameterize this fixture with the name
    # of their own factory fixture, without changing or copying these contracts.
    backend_fixture = getattr(request, "param", None)
    if backend_fixture is not None:
        return request.getfixturevalue(backend_fixture)

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


@pytest.fixture
def artifact_byte_storage(request: pytest.FixtureRequest):
    backend_fixture = getattr(request, "param", None)
    if backend_fixture is not None:
        return request.getfixturevalue(backend_fixture)
    return MemoryByteStorage()
