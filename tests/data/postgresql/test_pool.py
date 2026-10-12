"""Native-pool ownership boundaries using injected offline transports."""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from daita import AgentConfig, PostgreSQLReadLimits
from daita.adapters import postgresql as source_module, postgresql_query as module
from daita.adapters.models import SourceRegistration
from daita.security import SecretReference
from tests.data.postgresql._query_support import (
    _backend,
    _CatalogSchemas,
    _Pool,
    _SourceStore,
)


class _Secret:
    value = "first-private-password"

    async def resolve(self, reference):
        assert reference == SecretReference.environment("POOL_TEST_PASSWORD")
        return self.value


class _Transaction:
    def __init__(self, connection):
        self.connection = connection
        self.started = self.committed = self.aborted = False

    async def start(self):
        assert not self.connection.transaction_active
        self.connection.transaction_active = True
        self.started = True

    async def commit(self):
        assert self.connection.transaction_active
        self.connection.transaction_active = False
        self.committed = True


class _Connection:
    def __init__(self):
        self.closed = False
        self.transaction_active = False
        self.transactions = []
        self.settings = []
        self.resets = 0
        self.fail_reset = False

    def transaction(self, **kwargs):
        assert kwargs == {"isolation": "repeatable_read", "readonly": True}
        transaction = _Transaction(self)
        self.transactions.append(transaction)
        return transaction

    async def execute(self, sql, *arguments):
        assert self.transaction_active
        self.settings.append((sql, arguments))

    def terminate(self):
        self.closed = True
        self.transaction_active = False
        for transaction in self.transactions:
            if transaction.started and not transaction.committed:
                transaction.aborted = True

    async def reset(self):
        assert not self.transaction_active
        if self.fail_reset:
            raise ValueError("private-reset-diagnostic")
        self.resets += 1


@pytest.fixture
def transport(monkeypatch):
    created, pools, structures, queries, passwords = [], [], [], [], []
    state = SimpleNamespace(query=None, structure=None, connect=None, fail_init=False)

    async def connect(registration, provider):
        if state.connect is not None:
            await state.connect()
        reference = registration.configuration.get("credential_ref")
        if reference:
            passwords.append(await provider.resolve(SecretReference.parse(reference)))
        connection = _Connection()
        created.append(connection)
        return connection

    def create_pool(**kwargs):
        assert kwargs["min_size"] == 0
        pool = _Pool(kwargs["connect"], kwargs["max_size"], fail_init=state.fail_init)
        pools.append(pool)
        return pool

    async def structure(connection, registration, **kwargs):
        assert connection.transaction_active
        structures.append(connection)
        if state.structure is not None:
            return await state.structure(connection, registration)
        return SimpleNamespace(source_revision="sha256:" + "3" * 64)

    async def query(connection, sql, parameters, **kwargs):
        assert connection.transaction_active
        queries.append(connection)
        if state.query is not None:
            await state.query(connection)
        return ("amount",), ({"amount": 42},), False

    async def export(connection, sql, parameters, **kwargs):
        assert connection.transaction_active
        queries.append(connection)
        return b'"amount"\r\n42\r\n', ("amount",), 1

    monkeypatch.setattr(module, "_connect", connect)
    monkeypatch.setattr(
        source_module, "_load_asyncpg", lambda: SimpleNamespace(create_pool=create_pool)
    )
    monkeypatch.setattr(module, "_load_structure", structure)
    monkeypatch.setattr(module, "_execute_user_query", query)
    monkeypatch.setattr(module, "_execute_exact_tabular_query", export)
    state.created, state.pools, state.structures, state.queries, state.passwords = (
        created,
        pools,
        structures,
        queries,
        passwords,
    )
    return state


def _pooled(
    *, limits=None, source_capacity=2, secrets=None, deadline=None, observer=None
):
    original, registration = _backend()
    backend = module.PostgreSQLQueryBackend(
        original._sources,
        original._catalog,
        secrets,
        read_limits=limits or PostgreSQLReadLimits(pool_enabled=True),
        source_capacity=source_capacity,
        deadline_reader=deadline,
        observer=observer,
    )
    return backend, registration


async def _read(backend, registration):
    return await backend.execute_read(
        agent_id=registration.agent_id,
        source_id=registration.id,
        sql="SELECT amount FROM public.orders",
        parameters=(),
        max_rows=10,
        max_bytes=16384,
    )


async def test_repeated_queries_and_exports_share_connections_and_fresh_transactions(
    transport,
):
    backend, registration = _pooled()
    for _ in range(10):
        result = await _read(backend, registration)
        assert tuple(dict(row) for row in result.projection.rows) == ({"amount": 42},)
        exported = await backend.execute_exact_tabular(
            agent_id=registration.agent_id,
            source_id=registration.id,
            sql="SELECT amount FROM public.orders",
            parameters=(),
            format_name="csv",
            parameters_sha256="sha256:" + "0" * 64,
            created_at=datetime.now(UTC),
            max_rows=100,
            max_columns=10,
            max_bytes=16384,
            timeout_seconds=5,
        )
        assert exported.row_count == 1
    assert len(transport.created) == 1
    assert len(transport.structures) == len(transport.queries) == 20
    assert len(transport.created[0].transactions) == 20
    assert all(t.committed for t in transport.created[0].transactions)
    assert transport.created[0].resets == 20
    assert len(transport.created[0].settings) == 40
    metrics = backend.diagnostics()
    assert metrics.active == 0 and metrics.connections_created == 1
    assert metrics.structural_seconds >= 0 and metrics.query_seconds >= 0
    await backend.close()
    await backend.close()
    assert (
        backend.diagnostics().cached_sources
        == backend.diagnostics().reserved_connections
        == 0
    )
    assert all(connection.closed for connection in transport.created)


@pytest.mark.parametrize("capacity", (1, 2, 4, 8))
async def test_checkout_overlap_is_clamped_to_source_capacity(transport, capacity):
    backend, registration = _pooled(
        limits=PostgreSQLReadLimits(pool_enabled=True, pool_size_per_source=8),
        source_capacity=capacity,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def blocked(connection):
        nonlocal count
        count += 1
        if count == capacity:
            entered.set()
        await release.wait()

    transport.query = blocked
    tasks = [
        asyncio.create_task(_read(backend, registration)) for _ in range(capacity + 1)
    ]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert len(transport.created) == count == capacity
        assert backend.diagnostics().active == capacity
        assert backend.diagnostics().pool_size_per_source == capacity
        release.set()
        await asyncio.gather(*tasks)
        assert len(transport.created) == capacity
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await backend.close()


async def test_reused_connection_detects_schema_drift_before_query(transport):
    backend, registration = _pooled()
    await _read(backend, registration)

    async def changed(connection, registration):
        return SimpleNamespace(source_revision="sha256:" + "9" * 64)

    transport.structure = changed
    with pytest.raises(module.PostgreSQLQueryError, match="structure changed"):
        await _read(backend, registration)
    assert len(transport.created) == 1 and len(transport.queries) == 1
    assert transport.created[0].closed
    assert transport.created[0].transactions[-1].aborted
    await backend.close()


@pytest.mark.parametrize("failure", ("query", "reset"))
async def test_poisoned_connection_is_discarded_without_query_replay(
    transport, failure
):
    backend, registration = _pooled()

    async def faulty(connection):
        if failure == "query":
            raise ValueError("private-query-diagnostic")
        connection.fail_reset = True

    transport.query = faulty
    if failure == "query":
        with pytest.raises(module.PostgreSQLQueryError) as caught:
            await _read(backend, registration)
        assert "private-query-diagnostic" not in "".join(
            traceback.format_exception(caught.value)
        )
    else:
        await _read(backend, registration)
    assert len(transport.queries) == 1 and transport.created[0].closed
    transport.query = None
    await _read(backend, registration)
    assert len(transport.created) == len(transport.queries) == 2
    assert backend.diagnostics().discarded == 1
    await backend.close()


@pytest.mark.parametrize(
    "change", ("credential", "registration", "permission", "detach", "idle")
)
async def test_exact_source_and_credential_admission_retire_idle_pools(
    transport, monkeypatch, change
):
    secrets = _Secret()
    backend, registration = _pooled(secrets=secrets)
    store = backend._sources
    assert isinstance(store, _SourceStore)
    registration = replace(
        registration, configuration={"credential_ref": "env:POOL_TEST_PASSWORD"}
    )
    store.registration = registration
    await _read(backend, registration)
    if change == "credential":
        secrets.value = "second-private-password"
    elif change == "registration":
        store.registration = replace(registration, display_name="Changed admission")
    elif change == "permission":

        async def denied(*args):
            return frozenset()

        monkeypatch.setattr(backend._catalog, "readable_resource_ids", denied)
    elif change == "detach":
        await store.detach_source(
            registration.agent_id, registration.id, datetime.now(UTC)
        )
    else:
        next(iter(backend._owned_pools)).last_used -= 100
    if change in {"permission", "detach"}:
        with pytest.raises(module.PostgreSQLQueryError):
            await _read(backend, registration)
    else:
        await _read(backend, store.registration)
    assert transport.pools[0].closed and transport.created[0].closed
    assert backend.diagnostics().pools_retired >= 1
    if change == "credential":
        assert transport.passwords == [
            "first-private-password",
            "second-private-password",
        ]
        assert "private-password" not in repr(backend.diagnostics())
    await backend.close()


async def test_rotation_during_checkout_blocks_dispatch(transport):
    secrets = _Secret()
    backend, registration = _pooled(secrets=secrets, source_capacity=1)
    store = backend._sources
    assert isinstance(store, _SourceStore)
    registration = replace(
        registration, configuration={"credential_ref": "env:POOL_TEST_PASSWORD"}
    )
    store.registration = registration
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(connection):
        entered.set()
        await release.wait()

    transport.query = blocked
    first = asyncio.create_task(_read(backend, registration))
    await entered.wait()
    second = asyncio.create_task(_read(backend, registration))
    while backend.diagnostics().waiting != 1:
        await asyncio.sleep(0)
    secrets.value = "rotated-private-password"
    release.set()
    await first
    with pytest.raises(module.PostgreSQLQueryError, match="admission changed"):
        await second
    assert len(transport.queries) == 1
    transport.query = None
    await _read(backend, registration)
    assert len(transport.queries) == 2 and len(transport.created) == 2
    await backend.close()


@pytest.mark.parametrize("interruption", ("cancel", "timeout"))
async def test_bounded_checkout_and_cancel_leave_no_lease(transport, interruption):
    backend, registration = _pooled(
        source_capacity=1,
        limits=PostgreSQLReadLimits(pool_enabled=True, acquire_timeout_seconds=0.1),
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(connection):
        entered.set()
        await release.wait()

    transport.query = blocked
    first = asyncio.create_task(_read(backend, registration))
    await entered.wait()
    second = asyncio.create_task(_read(backend, registration))
    while backend.diagnostics().waiting != 1:
        await asyncio.sleep(0)
    if interruption == "cancel":
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert backend.diagnostics().cancellations == 1
    else:
        with pytest.raises(module.PostgreSQLQueryError):
            await second
        assert backend.diagnostics().timeouts == 1
    release.set()
    await first
    assert len(transport.queries) == 1 and backend.diagnostics().waiting == 0
    await backend.close()
    assert backend.diagnostics().active == 0


async def test_pool_initialization_and_connection_failure_cleanup(transport):
    backend, registration = _pooled()
    transport.fail_init = True
    with pytest.raises(module.PostgreSQLQueryError) as caught:
        await _read(backend, registration)
    assert "private-init" not in "".join(traceback.format_exception(caught.value))
    assert transport.pools[0].terminations == 1
    assert backend.diagnostics().reserved_connections == 0
    transport.fail_init = False

    async def failed_connect():
        raise ValueError("private-connect-diagnostic")

    transport.connect = failed_connect
    with pytest.raises(module.PostgreSQLQueryError):
        await _read(backend, registration)
    assert not transport.created
    transport.connect = None
    await _read(backend, registration)
    assert len(transport.queries) == 1
    await backend.close()


async def test_deadline_bounds_connect_and_redacted_observer_is_optional(transport):
    events = []

    def observe(event):
        events.append(event)
        raise RuntimeError("observer cannot fail reads")

    backend, registration = _pooled(
        deadline=lambda: asyncio.get_running_loop().time() + 0.03, observer=observe
    )

    async def slow_connect():
        await asyncio.Event().wait()

    transport.connect = slow_connect
    with pytest.raises(module.PostgreSQLQueryError):
        await asyncio.wait_for(_read(backend, registration), 1)
    assert backend.diagnostics().timeouts == 1 and backend.diagnostics().active == 0
    transport.connect = None
    await _read(backend, registration)
    assert len(events) == 1 and "source-" not in repr(events[0].data)
    await backend.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("pool_enabled", 1),
        ("max_connections", True),
        ("pool_size_per_source", 9),
        ("max_cached_sources", 0),
        ("acquire_timeout_seconds", float("inf")),
        ("idle_timeout_seconds", float("nan")),
        ("statement_timeout_seconds", -1),
        ("cleanup_timeout_seconds", False),
    ],
)
def test_pool_configuration_is_runtime_only_and_strict(field, value):
    with pytest.raises((ValueError, TypeError)):
        PostgreSQLReadLimits(**{field: value})
    assert not AgentConfig().postgresql_reads.pool_enabled


async def test_host_connection_budget_retires_idle_sources(transport):
    backend, first = _pooled(
        limits=PostgreSQLReadLimits(
            pool_enabled=True,
            pool_size_per_source=1,
            max_connections=2,
            max_cached_sources=2,
        )
    )
    registrations = [
        first,
        *[
            SourceRegistration.build(
                agent_id=first.agent_id,
                adapter_id="postgresql",
                native_identity=f"source-{i}",
                display_name=f"Source {i}",
                configuration={},
                attached_at=datetime.now(UTC),
            )
            for i in (2, 3)
        ],
    ]
    resources = [
        replace(
            backend._catalog.resources[0],
            source_id=registration.id,
            resource_id=f"resource-{i}",
        )
        for i, registration in enumerate(registrations)
    ]

    class Sources(_SourceStore):
        async def load_source(self, agent_id, source_id):
            return next(
                (
                    r
                    for r in registrations
                    if r.agent_id == agent_id and r.id == source_id
                ),
                None,
            )

    backend._sources = Sources(first)
    backend._catalog = _CatalogSchemas(tuple(resources))
    entered, release = asyncio.Event(), asyncio.Event()
    active = peak = count = 0

    async def blocked(connection):
        nonlocal active, peak, count
        active += 1
        peak = max(peak, active)
        count += 1
        if count == 2:
            entered.set()
        try:
            await release.wait()
        finally:
            active -= 1

    transport.query = blocked
    tasks = [asyncio.create_task(_read(backend, r)) for r in registrations[:2]]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        tasks.append(asyncio.create_task(_read(backend, registrations[2])))
        while backend.diagnostics().waiting != 1:
            await asyncio.sleep(0)
        assert count == 2 and backend.diagnostics().reserved_connections == 2
        release.set()
        await asyncio.gather(*tasks)
        assert len(transport.created) == count == 3 and peak == 2
        assert (
            backend.diagnostics().cached_sources
            == backend.diagnostics().reserved_connections
            == 2
        )
        assert backend.diagnostics().high_water_reserved_connections == 2
        assert transport.pools[0].closed or transport.pools[1].closed
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await backend.close()


async def test_stale_idle_connection_replaced_before_new_query(transport):
    backend, registration = _pooled()
    await _read(backend, registration)
    transport.created[0].terminate()
    await _read(backend, registration)
    assert len(transport.created) == len(transport.queries) == 2
    assert backend.diagnostics().pools_created == 1
    await backend.close()


async def test_cancel_after_query_dispatch_discards_and_settles(transport):
    backend, registration = _pooled()
    entered, settled = asyncio.Event(), asyncio.Event()

    async def blocked(connection):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            settled.set()

    transport.query = blocked
    task = asyncio.create_task(_read(backend, registration))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert settled.is_set() and transport.created[0].closed
    assert backend.diagnostics().active == 0 and len(transport.queries) == 1
    await backend.close()


async def test_cancel_during_reset_retains_lease_until_cleanup_settles(
    transport, monkeypatch
):
    backend, registration = _pooled()
    entered, release = asyncio.Event(), asyncio.Event()
    reset = _Connection.reset

    async def blocked(self):
        entered.set()
        await release.wait()
        await reset(self)

    monkeypatch.setattr(_Connection, "reset", blocked)
    task = asyncio.create_task(_read(backend, registration))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and backend.diagnostics().active == 1
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.diagnostics().active == 0
    assert len(transport.queries) == 1
    await backend.close()
