import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from daita import (
    Agent,
    AgentConfig,
    FolderSource,
    HostCapacityLimits,
    PostgreSQLSource,
    SQLiteSource,
)
from daita.adapters.models import DiscoveryResult, SourceRegistration
from daita.adapters.sqlite import SQLiteResourceAdapter
from daita.catalog.models import CatalogSync, CatalogSyncStatus, SourceCatalogSnapshot
from daita.hosting import embedded as embedded_module
from daita.hosting.embedded import AgentHomeError, HostActiveError
from daita.hosting.execution_governor import PermitKind, WorkloadClass
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from daita.llm.providers.mock import MockModelProvider
from daita.security import SecretReference
from daita.storage.sqlite_records import SourceReadMode
from tests.support.source_refresh import settled
from tests.support.workspace import workspace_for


class _BlockingProvider:
    provider_id = "mock:blocking-host-lifecycle"

    def __init__(self) -> None:
        self.model_profile = ModelProfile(
            id=self.provider_id,
            context_window_tokens=128_000,
            max_output_tokens=8_192,
            supports_tools=True,
            supports_parallel_tools=True,
        )
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def supports_request_policy(self, request: ModelRequest) -> bool:
        del request
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        del request
        self.started.set()
        await self.release.wait()
        return ModelResponse(finish_reason=FinishReason.STOP, text="done")


def _sqlite_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO records VALUES (1, 'one')")


def _assert_wal_drained(database: Path) -> None:
    wal = database.with_name("state.db-wal")
    assert not wal.exists() or wal.stat().st_size == 0


async def test_read_only_reopen_close_preserves_database_and_drains_wal(tmp_path):
    name = "read-only-close"
    agent = await Agent.create(
        name,
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    home = agent._embedded.home
    await agent.close()

    database = home / "state.db"
    before = database.read_bytes()
    _assert_wal_drained(database)

    reopened = await Agent.open(
        name,
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    await reopened.list_sources()
    await reopened.catalog_summary()
    await reopened.close()

    assert database.read_bytes() == before
    _assert_wal_drained(database)


async def test_close_retains_writer_lock_until_blocked_run_terminalizes(tmp_path):
    provider = _BlockingProvider()
    agent = await Agent.create(
        "close-drains-run",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        workspace=workspace_for(tmp_path),
    )
    run = asyncio.create_task(agent.run("answer without tools"))
    started = asyncio.create_task(provider.started.wait())
    # Only an upper bound: a loaded runner can take seconds to reach the model.
    done, _pending = await asyncio.wait(
        (run, started), timeout=30, return_when=asyncio.FIRST_COMPLETED
    )
    assert started in done, (
        repr(run.exception() or run.result())
        if run.done()
        else repr(
            [
                (frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno)
                for frame in run.get_stack()
            ]
        )
    )
    coordinator = agent._embedded._admission_coordinator
    diagnostics = coordinator.diagnostics()
    assert diagnostics.active_leases == 1
    provider_permits = sum(
        active
        for (kind, _key), active in coordinator._permit_active.items()
        if kind is PermitKind.PROVIDER
    )
    assert provider_permits == 1
    closing = asyncio.create_task(agent.close())
    await asyncio.sleep(0)

    with pytest.raises(HostActiveError):
        await asyncio.wait_for(
            Agent.open(
                "close-drains-run",
                root=tmp_path,
                model=_BlockingProvider(),
                model_profile=provider.model_profile,
                workspace=workspace_for(tmp_path),
            ),
            timeout=1,
        )
    assert not closing.done()

    provider.release.set()
    result = await asyncio.wait_for(run, timeout=1)
    await asyncio.wait_for(closing, timeout=1)

    reopened_provider = _BlockingProvider()
    reopened = await asyncio.wait_for(
        Agent.open(
            "close-drains-run",
            root=tmp_path,
            model=reopened_provider,
            model_profile=reopened_provider.model_profile,
            workspace=workspace_for(tmp_path),
        ),
        timeout=1,
    )
    try:
        persisted = await reopened.conversation_runs(result.conversation_id)
        assert persisted[-1].result == result
    finally:
        await reopened.close()


async def test_close_rejects_a_queued_run_before_releasing_writer_ownership(tmp_path):
    provider = _BlockingProvider()
    agent = await Agent.create(
        "close-rejects-queued-run",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        workspace=workspace_for(tmp_path),
    )
    coordinator = agent._embedded._admission_coordinator
    background_leases = [
        await coordinator.admit_execution(
            WorkloadClass.SYSTEM,
            f"saturating-background-{index}",
            absolute_deadline=asyncio.get_running_loop().time() + 2,
        )
        for index in range(4)
    ]
    active = asyncio.create_task(agent.run("active run"))
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    queued = asyncio.create_task(agent.run("queued run"))
    await asyncio.sleep(0)
    closing = asyncio.create_task(agent.close())
    await asyncio.sleep(0)

    provider.release.set()
    await asyncio.wait_for(active, timeout=1)
    with pytest.raises(AgentHomeError, match="closed"):
        await asyncio.wait_for(queued, timeout=1)
    for lease in background_leases:
        await lease.release()
    await asyncio.wait_for(closing, timeout=1)

    reopened_provider = _BlockingProvider()
    reopened = await Agent.open(
        "close-rejects-queued-run",
        root=tmp_path,
        model=reopened_provider,
        model_profile=reopened_provider.model_profile,
        workspace=workspace_for(tmp_path),
    )
    await reopened.close()


async def test_close_drains_admission_before_store_and_writer_lock_close(
    tmp_path,
    monkeypatch,
):
    provider = _BlockingProvider()
    agent = await Agent.create(
        "close-dependency-order",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        workspace=workspace_for(tmp_path),
    )
    embedded = agent._embedded
    lease = await embedded._admission_coordinator.admit_execution(
        WorkloadClass.SYSTEM,
        "held-test-lease",
        absolute_deadline=asyncio.get_running_loop().time() + 1,
    )
    events: list[str] = []
    original_store_close = embedded._store.close
    assert embedded._writer_lock is not None
    original_writer_release = embedded._writer_lock.release

    async def close_store():
        events.append("store")
        await original_store_close()

    def release_writer():
        events.append("writer")
        original_writer_release()

    monkeypatch.setattr(embedded._store, "close", close_store)
    monkeypatch.setattr(embedded._writer_lock, "release", release_writer)
    closing = asyncio.create_task(agent.close())
    await asyncio.sleep(0)

    assert events == []
    assert not closing.done()
    await lease.release()
    await asyncio.wait_for(closing, timeout=1)
    assert events[-2:] == ["store", "writer"]


async def test_same_conversation_turns_bind_predecessor_and_include_prior_result(
    tmp_path,
):
    class OrderedProvider(_BlockingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[ModelRequest] = []

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse(finish_reason=FinishReason.STOP, text="seed")
            if len(self.requests) == 2:
                self.started.set()
                await self.release.wait()
                return ModelResponse(
                    finish_reason=FinishReason.STOP,
                    text="first result",
                )
            return ModelResponse(finish_reason=FinishReason.STOP, text="second result")

    provider = OrderedProvider()
    agent = await Agent.create(
        "same-conversation-order",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        seed = await agent.run("seed question")
        first = asyncio.create_task(
            agent.run("first follow-up", conversation_id=seed.conversation_id)
        )
        await provider.started.wait()
        second = asyncio.create_task(
            agent.run("second follow-up", conversation_id=seed.conversation_id)
        )
        await asyncio.sleep(0)
        assert (
            agent._embedded._admission_coordinator.diagnostics().waiting_admissions == 1
        )

        provider.release.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert first_result.final_text == "first result"
        assert second_result.final_text == "second result"
        assert any(
            isinstance(block, TextBlock) and block.text == "first result"
            for message in provider.requests[2].messages
            for block in message.content
        )
        conversation = await agent.conversation_runs(seed.conversation_id)
        assert tuple(item.turn_index for item in conversation) == (0, 1, 2)
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "operation",
    ("detach", "refresh", "apply_permissions"),
)
async def test_foreground_run_does_not_hold_a_broad_host_mutation_lock(
    tmp_path,
    operation: str,
):
    database = tmp_path / f"{operation}.sqlite"
    _sqlite_database(database)
    provider = _BlockingProvider()
    agent = await Agent.create(
        f"run-mutation-{operation}",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        workspace=workspace_for(tmp_path),
    )
    source = await agent.attach_sqlite(database)
    permission_preview = None
    if operation == "apply_permissions":
        permission_preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode=SourceReadMode.NONE,
            read_resource_ids=(),
            relational_write_scopes={},
        )

    run = asyncio.create_task(
        agent.run("answer without tools", source_scope_ids=(source.id,))
    )
    await asyncio.sleep(0)
    assert not run.done(), repr(run.exception()) if run.done() else ""
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    mutation: asyncio.Task[Any]
    if operation == "detach":
        mutation = asyncio.create_task(agent.detach(source.id))
    elif operation == "refresh":
        mutation = asyncio.create_task(agent.refresh_source(source.id))
    else:
        assert permission_preview is not None
        mutation = asyncio.create_task(
            agent.apply_source_permissions(
                source_id=source.id,
                confirmation_fingerprint=(permission_preview.confirmation_fingerprint),
            )
        )
    mutation_result = await asyncio.wait_for(mutation, timeout=1)
    listed_sources = await asyncio.wait_for(agent.list_sources(), timeout=1)
    if operation == "detach":
        assert listed_sources == (mutation_result,)
        assert not listed_sources[0].active
    else:
        assert listed_sources == (source,)
    assert await asyncio.wait_for(
        agent.list_catalog_resources(source_id=source.id), timeout=1
    )

    provider.release.set()
    await asyncio.wait_for(run, timeout=1)
    await asyncio.wait_for(agent.close(), timeout=1)


async def test_a_refresh_stuck_in_discovery_blocks_neither_work_nor_close(
    tmp_path, monkeypatch
):
    database = tmp_path / "shop.sqlite"
    _sqlite_database(database)
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "describe",
                        "catalog_describe",
                        {"source": "Shop", "name": "invoices"},
                    ),
                    ToolCall(
                        "read",
                        "data_query",
                        {"source": "Shop", "sql": "SELECT count(*) AS n FROM records"},
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Done."),
        )
    )
    agent = await Agent.create(
        "stuck-refresh",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        workspace=workspace_for(tmp_path),
    )
    source = await agent.attach(SQLiteSource(database, name="Shop"))
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE invoices (id INTEGER)")
    entered = asyncio.Event()
    original = SQLiteResourceAdapter.discover

    async def stuck(self, request):
        entered.set()
        await asyncio.Event().wait()
        return await original(self, request)

    monkeypatch.setattr(SQLiteResourceAdapter, "discover", stuck)

    run = await asyncio.wait_for(agent.run("Look at invoices."), timeout=5)
    (_, describe), (_, read) = await agent.tool_evidence(run.run_id)
    assert describe is not None and not describe.is_error
    assert read is not None and not read.is_error
    data: Any = read.output["data"]
    assert [dict(row) for row in data["rows"]] == [{"n": 1}]
    await asyncio.wait_for(entered.wait(), timeout=5)
    refresh = agent._embedded._source_refreshes[source.id]
    assert not refresh.done()

    preview = await asyncio.wait_for(
        agent.preview_source_permissions(
            source_id=source.id,
            read_mode=SourceReadMode.NONE,
            read_resource_ids=(),
            relational_write_scopes={},
        ),
        timeout=5,
    )
    await asyncio.wait_for(
        agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=preview.confirmation_fingerprint,
        ),
        timeout=5,
    )
    await asyncio.wait_for(agent.catalog_summary(), timeout=5)

    await asyncio.wait_for(agent.close(), timeout=5)
    assert refresh.cancelled()
    agent._embedded.request_source_refresh(source.id)
    assert not agent._embedded._source_refreshes


async def test_a_connection_check_answers_while_a_discovery_is_stuck(
    tmp_path, monkeypatch
):
    database = tmp_path / "shop.sqlite"
    _sqlite_database(database)
    agent = await Agent.create(
        "check-while-stuck", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await agent.attach(SQLiteSource(database, name="Shop"))
    entered = asyncio.Event()

    async def stuck(self, request):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(SQLiteResourceAdapter, "discover", stuck)
    agent._embedded.request_source_refresh(source.id)
    await asyncio.wait_for(entered.wait(), timeout=5)

    health = await asyncio.wait_for(agent.check_source(source.id), timeout=5)

    assert health.healthy
    assert health.details["read_only"] is True
    await asyncio.wait_for(agent.close(), timeout=5)


async def test_a_connection_check_reports_a_missing_source_instead_of_raising(
    tmp_path,
):
    database = tmp_path / "shop.sqlite"
    _sqlite_database(database)
    folder = tmp_path / "drop"
    folder.mkdir()
    root = tmp_path / "state"
    agent = await Agent.create(
        "check-missing", root=root, workspace=workspace_for(root)
    )
    try:
        sqlite_source = await agent.attach(SQLiteSource(database, name="Shop"))
        folder_source = await agent.attach(FolderSource(folder, name="Drop"))
        for source in (sqlite_source, folder_source):
            health = await agent.check_source(source.id)
            assert health.healthy and health.details["read_only"] is True
            assert "role" not in health.details

        database.unlink()
        folder.rmdir()

        for source in (sqlite_source, folder_source):
            health = await agent.check_source(source.id)
            assert not health.healthy and health.error_code
            assert (health.source_id, health.agent_id) == (source.id, agent.id)
        with pytest.raises(ValueError, match="unknown active source"):
            await agent.check_source("source:sha256:" + "0" * 64)
    finally:
        await agent.close()


async def test_a_failed_postgresql_check_gives_a_code_and_no_connection_text(
    tmp_path, monkeypatch
):
    password = "database-password-do-not-expose"
    diagnostic = f"postgresql://reader:{password}@db.example.test/shop refused"

    class Transaction:
        async def start(self) -> None: ...

        async def commit(self) -> None: ...

        async def rollback(self) -> None: ...

    class Connection:
        def transaction(self, **_: object) -> Transaction:
            return Transaction()

        async def execute(self, *_: object) -> None: ...

        async def fetch(self, *_: object) -> list[object]:
            return []

        async def close(self) -> None: ...

    class Driver:
        reachable = True

        async def connect(self, **_: object) -> Connection:
            if not self.reachable:
                raise OSError(diagnostic)
            return Connection()

    class Secrets:
        async def resolve(self, reference: object) -> str:
            return password

    driver = Driver()
    monkeypatch.setattr("daita.adapters.postgresql._load_asyncpg", lambda: driver)
    agent = await Agent.create(
        "check-postgresql",
        root=tmp_path,
        secret_provider=Secrets(),
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach(
            PostgreSQLSource(
                host="db.example.test",
                database="shop",
                username="reader",
                credential=SecretReference.keychain("check-postgresql:password"),
                schemas=("public",),
                ssl_mode="disable",
                secret_provider=Secrets(),
            )
        )
        driver.reachable = False

        health = await agent.check_source(source.id)

        assert not health.healthy
        assert health.error_code == "postgresql_connect_failed"
        assert dict(health.details) == {}
        assert password not in repr(health) and "db.example.test" not in repr(health)
    finally:
        await agent.close()


async def test_a_refresh_for_qualified_names_rereads_only_their_schemas(
    tmp_path, monkeypatch
):
    agent = await Agent.create(
        "schema-refresh", root=tmp_path, workspace=workspace_for(tmp_path)
    )

    class FixtureAdapter:
        def __init__(self, registration):
            self.registration = registration

        async def discover(self, request):
            now = datetime.now(UTC)
            sync = CatalogSync(
                id=request.sync_id,
                agent_id=request.agent_id,
                source_id=request.source_id,
                adapter_id="postgresql",
                status=CatalogSyncStatus.SUCCEEDED,
                started_at=request.requested_at,
                completed_at=now,
                source_revision="catalog:sha256:" + "3" * 64,
            )
            return DiscoveryResult(
                request=request,
                snapshot=SourceCatalogSnapshot(sync=sync, resources=(), revisions=()),
                completed_at=now,
            )

        async def inspect(self, resource):
            raise AssertionError("empty fixture has no resources")

        async def health(self):
            raise AssertionError("refresh must not require a health call")

        async def close(self):
            pass

    class FixtureSource:
        async def open(self, *, agent_id, attached_at, clock):
            return FixtureAdapter(
                SourceRegistration.build(
                    agent_id=agent_id,
                    adapter_id="postgresql",
                    native_identity="offline-schema-refresh",
                    display_name="Offline fixture",
                    configuration={
                        "database": "shop",
                        "host": "localhost",
                        "port": 5432,
                        "schemas": ("public", "sales"),
                        "ssl_mode": "require",
                        "username": "reader",
                    },
                    attached_at=attached_at,
                )
            )

    try:
        source = await agent.attach(FixtureSource())
        requested = []

        async def record(opened, **keywords):
            requested.append((keywords["refresh_schemas"], keywords["previous"]))
            return keywords["persisted_registration"]

        monkeypatch.setattr(agent._embedded, "_attach_source_locked", record)
        monkeypatch.setattr(embedded_module, "_SOURCE_REFRESH_MIN_SECONDS", 0)
        for names in (
            ("sales.orders",),
            ("sales.orders", "public.users", "sales.items"),
            ("orders",),
            ("archive.orders",),
            ("sales.orders", "orders"),
            (),
        ):
            agent._embedded.request_source_refresh(source.id, names)
            await settled(agent)

        schemas = [item[0] for item in requested]
        assert schemas == [("sales",), ("sales", "public"), (), (), (), ()]
        previous = [item[1] for item in requested]
        assert all(item.sync.source_id == source.id for item in previous[:2])
        assert previous[2:] == [None] * 4
    finally:
        await agent.close()


@pytest.mark.parametrize("capacity", (1, 4, 8))
async def test_public_capacity_uses_all_configured_lanes_and_reopens_without_persistence(
    tmp_path, capacity
):
    class BarrierProvider(_BlockingProvider):
        active = 0
        peak = 0

        async def generate(self, request):
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.active == capacity:
                self.started.set()
            try:
                await self.release.wait()
                return ModelResponse(finish_reason=FinishReason.STOP, text="done")
            finally:
                self.active -= 1

    limits = HostCapacityLimits(
        execution_capacity=capacity,
        foreground_execution_reserve=0,
        provider_capacity=capacity,
        foreground_provider_reserve=0,
    )
    config = AgentConfig(host_capacity=limits)
    provider = BarrierProvider()
    agent = await Agent.create(
        "capacity",
        root=tmp_path,
        config=config,
        model=provider,
        model_profile=provider.model_profile,
        workspace=workspace_for(tmp_path),
    )
    runs = [
        asyncio.create_task(agent.run(f"request {index}"))
        for index in range(capacity + 1)
    ]
    try:
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        assert provider.active == capacity
        assert agent.admission_diagnostics().active_leases == capacity
        assert agent.admission_diagnostics().waiting_admissions == 1
        provider.release.set()
        results = await asyncio.gather(*runs)
        assert all(result.final_text == "done" for result in results)
        assert len({result.run_id for result in results}) == capacity + 1
        assert provider.peak == capacity
    finally:
        provider.release.set()
        await asyncio.gather(*runs, return_exceptions=True)
        await agent.close()
    assert (
        agent.admission_diagnostics().active_leases
        == agent.admission_diagnostics().active_permits
        == 0
    )
    assert not (agent._embedded.home / "config.json").exists()
    reopened = await Agent.open(
        "capacity", root=tmp_path, config=config, workspace=workspace_for(tmp_path)
    )
    try:
        assert (
            dict(reopened.admission_diagnostics().capacities)["execution"] == capacity
        )
    finally:
        await reopened.close()


async def test_postgresql_capacity_policy_applies_on_attach_and_reopen(tmp_path):
    config = AgentConfig(host_capacity=HostCapacityLimits(postgresql_read_capacity=4))
    agent = await Agent.create(
        "source-capacity",
        root=tmp_path,
        config=config,
        workspace=workspace_for(tmp_path),
    )

    class FixtureAdapter:
        def __init__(self, registration):
            self.registration = registration

        async def discover(self, request):
            now = datetime.now(UTC)
            sync = CatalogSync(
                id=request.sync_id,
                agent_id=request.agent_id,
                source_id=request.source_id,
                adapter_id="postgresql",
                status=CatalogSyncStatus.SUCCEEDED,
                started_at=request.requested_at,
                completed_at=now,
                source_revision="catalog:sha256:" + "3" * 64,
            )
            return DiscoveryResult(
                request=request,
                snapshot=SourceCatalogSnapshot(sync=sync, resources=(), revisions=()),
                completed_at=now,
            )

        async def inspect(self, resource):
            raise AssertionError("empty fixture has no resources")

        async def health(self):
            raise AssertionError("attachment must not require a health call")

        async def close(self):
            pass

    class FixtureSource:
        async def open(self, *, agent_id, attached_at, clock):
            return FixtureAdapter(
                SourceRegistration.build(
                    agent_id=agent_id,
                    adapter_id="postgresql",
                    native_identity="offline-capacity-source",
                    display_name="Offline fixture",
                    configuration={},
                    attached_at=attached_at,
                )
            )

    source = await agent.attach(FixtureSource())
    try:
        for opened in range(2):
            coordinator = agent._embedded._admission_coordinator
            permits = [
                await coordinator.source_resource_permit(f"source:{source.id}")
                for _ in range(4)
            ]
            waiting = asyncio.create_task(
                coordinator.source_resource_permit(f"source:{source.id}")
            )
            await asyncio.sleep(0)
            assert not waiting.done()
            await permits.pop().release()
            permits.append(await waiting)
            for permit in permits:
                await permit.release()
            assert await agent.list_catalog_resources(source_id=source.id) == ()
            await agent.close()
            if opened == 0:
                agent = await Agent.open(
                    "source-capacity",
                    root=tmp_path,
                    config=config,
                    workspace=workspace_for(tmp_path),
                )
    finally:
        await agent.close()


async def test_public_pool_policy_is_applied_closed_and_not_persisted(tmp_path):
    from daita import AgentConfig, HostCapacityLimits, PostgreSQLReadLimits

    config = AgentConfig(
        host_capacity=HostCapacityLimits(postgresql_read_capacity=4),
        postgresql_reads=PostgreSQLReadLimits(
            pool_enabled=True,
            pool_size_per_source=8,
            max_connections=3,
            max_cached_sources=2,
        ),
    )
    agent = await Agent.create(
        "pool-policy", root=tmp_path, config=config, workspace=workspace_for(tmp_path)
    )
    metrics = agent.postgresql_read_diagnostics()
    assert (
        metrics.limits == config.postgresql_reads and metrics.pool_size_per_source == 3
    )
    await agent.close()
    assert agent.postgresql_read_diagnostics().cached_sources == 0
    reopened = await Agent.open(
        "pool-policy", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        assert not reopened.postgresql_read_diagnostics().limits.pool_enabled
    finally:
        await reopened.close()


@pytest.mark.parametrize("failure", ("backend", "composition"))
async def test_pool_backend_construction_owns_no_unclosed_resources(
    tmp_path, monkeypatch, failure
):
    from daita.adapters.postgresql_query import PostgreSQLQueryBackend
    from daita.hosting.execution_governor import RunAdmissionCoordinator

    closed = []
    original_close = PostgreSQLQueryBackend.close
    original_init = PostgreSQLQueryBackend.__init__

    async def close(self):
        closed.append(self)
        assert self.diagnostics().active == 0
        await original_close(self)

    def backend(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        assert self.diagnostics().reserved_connections == 0
        raise RuntimeError("Injected backend construction failure")

    def coordinator(self, *args, **kwargs):
        raise RuntimeError("Injected composition failure")

    monkeypatch.setattr(PostgreSQLQueryBackend, "close", close)
    with monkeypatch.context() as patch:
        if failure == "backend":
            patch.setattr(PostgreSQLQueryBackend, "__init__", backend)
        else:
            patch.setattr(RunAdmissionCoordinator, "__init__", coordinator)
        with pytest.raises(RuntimeError, match="Injected"):
            await Agent.create(
                "failed-pool", root=tmp_path, workspace=workspace_for(tmp_path)
            )
    assert len(closed) == (1 if failure == "composition" else 0)
    agent = await Agent.create(
        "recovered-pool", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    await agent.close()
