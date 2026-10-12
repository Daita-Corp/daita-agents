"""Offline source, catalog and native-pool fixtures for PostgreSQL read tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from daita.adapters import postgresql_query as postgresql_query_module
from daita.adapters.models import SourceRegistration
from daita.domains.data.sql import ResourceSchema
from daita.security import EmptySecretProvider


class _SourceStore:
    def __init__(self, registration: SourceRegistration) -> None:
        self.registration = registration

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


class _CatalogSchemas:
    def __init__(self, resources: tuple[ResourceSchema, ...]) -> None:
        self.resources = resources

    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]:
        del agent_id
        return tuple(item for item in self.resources if item.source_id == source_id)

    async def readable_resource_ids(self, agent_id: str, source_ids=()):
        del agent_id
        return frozenset(
            item.resource_id
            for item in self.resources
            if not source_ids or item.source_id in source_ids
        )


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
        configuration={},
        attached_at=datetime(2026, 8, 5, tzinfo=UTC),
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
    )
    return (
        postgresql_query_module.PostgreSQLQueryBackend(
            _SourceStore(registration),
            _CatalogSchemas((resource,)),
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
