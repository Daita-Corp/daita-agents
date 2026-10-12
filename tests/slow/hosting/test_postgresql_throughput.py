"""Injected PostgreSQL transport measurements, distinct from live databases."""

from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import pytest

from daita import PostgreSQLReadLimits
from daita._json import canonical_json
from daita.adapters import postgresql_query as module
from daita.adapters.models import SourceRegistration
from daita.domains.data.sql import ResourceSchema
from daita.security import EmptySecretProvider
from tests.support.job_benchmarks import p95_bootstrap_interval

pytestmark = pytest.mark.slow


class _Sources:
    def __init__(self, registration):
        self.registration = registration

    async def register_source(self, registration):
        self.registration = registration
        return registration

    async def load_source(self, agent_id, source_id):
        if agent_id == self.registration.agent_id and source_id == self.registration.id:
            return self.registration
        return None

    async def list_sources(self, agent_id):
        return (self.registration,)

    async def detach_source(self, agent_id, source_id, detached_at):
        self.registration = self.registration.detach(detached_at)
        return self.registration


class _Catalog:
    def __init__(self, registration):
        self.resource = ResourceSchema(
            resource_id="resource-fixture",
            source_id=registration.id,
            name="records",
            aliases=("public.records",),
            columns=("amount",),
            revision="sha256:" + "2" * 64,
            source_revision="sha256:" + "3" * 64,
            resource_kind="table",
        )

    async def resource_schemas(self, agent_id, source_id):
        return (self.resource,)

    async def readable_resource_ids(self, agent_id, source_ids=()):
        return frozenset({self.resource.resource_id})


class _Transaction:
    async def start(self):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass


class _Statement:
    def get_attributes(self):
        return (SimpleNamespace(name="amount"),)

    async def cursor(self, *args, **kwargs):
        yield (42, True)


class _Connection:
    def __init__(self):
        self.transactions = 0
        self.closed = False

    def transaction(self, **kwargs):
        assert kwargs == {"isolation": "repeatable_read", "readonly": True}
        self.transactions += 1
        return _Transaction()

    async def execute(self, *args):
        pass

    async def prepare(self, *args, **kwargs):
        return _Statement()

    async def reset(self):
        pass

    async def close(self, **kwargs):
        self.closed = True

    def terminate(self):
        self.closed = True


async def test_postgresql_transport_component_campaign(monkeypatch):
    if os.environ.get("DAITA_RUN_THROUGHPUT_COMPONENT") != "1":
        pytest.skip("choose the bounded offline component campaign explicitly")
    directory = Path(os.environ["DAITA_THROUGHPUT_REPORT_DIR"])
    assert directory.is_absolute() and not directory.is_relative_to(Path.cwd())
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    campaign = os.environ.get("DAITA_THROUGHPUT_CAMPAIGN", "baseline")
    count = int(os.environ.get("DAITA_THROUGHPUT_SAMPLES", "30"))
    assert 30 <= count <= 100
    registration = SourceRegistration.build(
        agent_id="agent-fixture",
        adapter_id="postgresql",
        native_identity="fixture",
        display_name="Injected fixture",
        configuration={},
        attached_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    connections: list[_Connection] = []
    structure_seconds = 0.0

    async def connect(*args):
        await asyncio.sleep(0.002)
        connection = _Connection()
        connections.append(connection)
        return connection

    async def structure(*args, **kwargs):
        nonlocal structure_seconds
        at = perf_counter()
        await asyncio.sleep(0.001)
        structure_seconds += perf_counter() - at
        return SimpleNamespace(source_revision="sha256:" + "3" * 64)

    monkeypatch.setattr(module, "_connect", connect)
    monkeypatch.setattr(module, "_load_structure", structure)
    backend = module.PostgreSQLQueryBackend(
        _Sources(registration), _Catalog(registration), EmptySecretProvider()
    )
    samples = []
    async with asyncio.timeout(30):
        for _ in range(count):
            at = perf_counter()
            result = await backend.execute_read(
                agent_id=registration.agent_id,
                source_id=registration.id,
                sql="SELECT amount FROM public.records",
                parameters=(),
                max_rows=100,
                max_bytes=16384,
            )
            samples.append(perf_counter() - at)
            assert json.loads(canonical_json(result.projection.rows)) == [
                {"amount": 42}
            ]
    assert len(connections) == count
    assert all(
        connection.closed and connection.transactions == 1 for connection in connections
    )
    report = {
        "campaign": campaign,
        "evidence_kind": "PostgreSQL backend with injected transport; no live database",
        "fixture_sha256": sha256(
            b"amount:int64=42;connect-delay=.002;structure-delay=.001;v1"
        ).hexdigest(),
        "configuration": {
            "pool_enabled": False,
            "statement_timeout_seconds": 5,
            "cleanup_timeout_seconds": 1,
        },
        "samples": count,
        "latency_seconds": samples,
        "p50_seconds": statistics.median(samples),
        "p95_seconds": sorted(samples)[math.ceil(0.95 * count) - 1],
        "p95_bootstrap_95_percent": p95_bootstrap_interval(samples),
        "completed_tasks_per_minute": count * 60 / sum(samples),
        "connections_created": len(connections),
        "fresh_transactions": sum(
            connection.transactions for connection in connections
        ),
        "fixture_structure_seconds": structure_seconds,
        "model_requests": 0,
        "tokens": None,
        "cost_usd": None,
        "usage_status": "unknown",
        "correctness": "exact independent amount oracle; no live type qualification",
        "cleanup": {
            "connections_closed": all(connection.closed for connection in connections)
        },
    }
    path = directory / f"{campaign}-postgresql-fixture.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Injected transport report: {path}")

    if campaign != "baseline":
        from daita.adapters import postgresql as source_module
        from tests.data.postgresql._query_support import _Pool

        pools = []

        def create_pool(**kwargs):
            assert kwargs["min_size"] == 0
            pool = _Pool(kwargs["connect"], kwargs["max_size"])
            pools.append(pool)
            return pool

        monkeypatch.setattr(
            source_module,
            "_load_asyncpg",
            lambda: SimpleNamespace(create_pool=create_pool),
        )
        connections.clear()
        structure_seconds = 0.0
        backend = module.PostgreSQLQueryBackend(
            _Sources(registration),
            _Catalog(registration),
            EmptySecretProvider(),
            read_limits=PostgreSQLReadLimits(pool_enabled=True),
        )
        samples = []
        try:
            async with asyncio.timeout(30):
                for _ in range(count):
                    at = perf_counter()
                    result = await backend.execute_read(
                        agent_id=registration.agent_id,
                        source_id=registration.id,
                        sql="SELECT amount FROM public.records",
                        parameters=(),
                        max_rows=100,
                        max_bytes=16384,
                    )
                    samples.append(perf_counter() - at)
                    assert json.loads(canonical_json(result.projection.rows)) == [
                        {"amount": 42}
                    ]
            assert len(connections) == 1 and connections[0].transactions == count
        finally:
            await backend.close()
        from dataclasses import asdict

        report.update(
            {
                "configuration": asdict(PostgreSQLReadLimits(pool_enabled=True)),
                "latency_seconds": samples,
                "p50_seconds": statistics.median(samples),
                "p95_seconds": sorted(samples)[math.ceil(0.95 * count) - 1],
                "p95_bootstrap_95_percent": p95_bootstrap_interval(samples),
                "completed_tasks_per_minute": count * 60 / sum(samples),
                "connections_created": len(connections),
                "fresh_transactions": sum(
                    connection.transactions for connection in connections
                ),
                "fixture_structure_seconds": structure_seconds,
                "pool_metrics": asdict(backend.diagnostics()),
                "cleanup": {
                    "connections_closed": all(
                        connection.closed for connection in connections
                    ),
                    "active_leases": backend.diagnostics().active,
                    "reserved_connections": backend.diagnostics().reserved_connections,
                },
            }
        )
        path = directory / f"{campaign}-postgresql-reused-fixture.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Injected reused transport report: {path}")
