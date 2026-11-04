"""Offline source and native-pool fixtures for PostgreSQL read tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from daita.adapters import postgresql_query as postgresql_query_module
from daita.adapters.models import SourceRegistration
from daita.security import EmptySecretProvider
from daita.storage.sqlite_records import SourceReadMode, SourceReadScope


class _SourceStore:
    def __init__(self, registration: SourceRegistration) -> None:
        self.registration = registration
        self.read_mode = SourceReadMode.ALL

    async def load_source_read_scope(
        self, agent_id: str, source_id: str
    ) -> SourceReadScope:
        return SourceReadScope(agent_id, source_id, self.read_mode)

    async def register_source(
        self,
        registration: SourceRegistration,
    ) -> SourceRegistration:
        self.registration = registration
        return registration

    async def load_source(
        self,
        agent_id: str,
        source_id: str,
    ) -> SourceRegistration | None:
        if self.registration.agent_id == agent_id and self.registration.id == source_id:
            return self.registration
        return None

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]:
        return (self.registration,) if self.registration.agent_id == agent_id else ()

    async def detach_source(
        self,
        agent_id: str,
        source_id: str,
        detached_at: datetime,
    ) -> SourceRegistration:
        assert self.registration.agent_id == agent_id
        assert self.registration.id == source_id
        self.registration = self.registration.detach(detached_at)
        return self.registration


def _backend() -> tuple[
    postgresql_query_module.PostgreSQLQueryBackend,
    SourceRegistration,
]:
    agent_id = "agent-postgresql"
    registration = SourceRegistration.build(
        agent_id=agent_id,
        adapter_id="postgresql",
        native_identity="offline-query-contract",
        display_name="Offline PostgreSQL",
        configuration={"schemas": ("public",)},
        attached_at=datetime(2026, 8, 5, tzinfo=UTC),
    )
    return (
        postgresql_query_module.PostgreSQLQueryBackend(
            _SourceStore(registration),
            EmptySecretProvider(),
        ),
        registration,
    )


class _Pool:
    def __init__(self, connect, maximum, *, fail_init=False):
        self.connect = connect
        self.maximum = maximum
        self.slots = asyncio.Semaphore(maximum)
        self.idle = []
        self.connections = []
        self.active = set()
        self.closed = False
        self.fail_init = fail_init
        self.terminations = 0

    def __await__(self):
        async def initialize():
            if self.fail_init:
                raise ValueError("private-init-diagnostic")
            return self

        return initialize().__await__()

    async def acquire(self, *, timeout):
        async with asyncio.timeout(timeout):
            await self.slots.acquire()
            try:
                connection = next((c for c in self.idle if not c.closed), None)
                self.idle = [
                    c for c in self.idle if c is not connection and not c.closed
                ]
                if connection is None:
                    connection = await self.connect()
                    self.connections.append(connection)
                self.active.add(connection)
                return connection
            except BaseException:
                self.slots.release()
                raise

    async def release(self, connection, *, timeout):
        try:
            if not connection.closed:
                async with asyncio.timeout(timeout):
                    await connection.reset()
                self.idle.append(connection)
        except BaseException:
            connection.terminate()
            raise
        finally:
            self.active.discard(connection)
            self.slots.release()

    async def close(self):
        if self.fail_init:
            raise RuntimeError("uninitialized pool")
        self.closed = True
        self.terminate()

    def terminate(self):
        self.closed = True
        self.terminations += 1
        for connection in self.connections:
            connection.terminate()
