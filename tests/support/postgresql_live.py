"""Shared helpers extracted from ``test_write_release.py``."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from daita import Agent, ApprovalDecision, ApprovalRequest, MCPToolSelection
from daita.adapters import postgresql_write as native
from daita.adapters.mcp import SDKMCPClientFactory
from daita.capabilities import EffectEvidenceBasis, EffectOutcome
from daita.llm.models import ModelSensitivity
from daita.security import SecretReference
from daita.storage.sqlite_records import EffectReceipt
from tests.support.mcp import conformance_identities, mock_transport
from tests.support.native_writes import ScriptedResearchModel
from tests.support.postgresql_write_release import DriverProbe, WriteModel
from tests.support.workspace import workspace_for

_TABLE = "write_acceptance.companies"
_ROLE = "daita_large_write_tester"
_LOCK_ID = 7_346_218_930


async def _reset(admin: Any) -> None:
    await admin.execute(f"ALTER TABLE {_TABLE} DROP COLUMN IF EXISTS release_drift")
    await admin.execute(f"GRANT SELECT, UPDATE ON {_TABLE} TO {_ROLE}")
    await admin.execute(f"REVOKE DELETE ON {_TABLE} FROM {_ROLE}")
    await admin.execute(
        f"GRANT INSERT (domain, name, evidence_url, notes) ON {_TABLE} TO {_ROLE}"
    )
    await admin.execute(
        f"TRUNCATE ONLY {_TABLE}, ONLY write_acceptance.cells RESTART IDENTITY"
    )


class WriteTesterSecrets:
    async def resolve(self, reference: SecretReference) -> str:
        assert reference == SecretReference.keychain(
            "fixture:postgres-large:write-release"
        )
        return os.environ.get(
            "DAITA_LARGE_POSTGRES_WRITE_TESTER_PASSWORD",
            "daita_large_write_tester_fixture_password",
        )


@dataclass
class Fixture:
    agent: Agent
    model: WriteModel
    admin: Any
    probe: DriverProbe
    source_id: str
    resource_id: str
    cells_resource_id: str
    root: Path
    approvals: list[ApprovalRequest]
    approval_hook: Callable[[], Awaitable[None]] | None = None
    approval_decision: ApprovalDecision = ApprovalDecision.APPROVE

    async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        self.approvals.append(request)
        if self.approval_hook is not None:
            await self.approval_hook()
        return self.approval_decision

    async def permissions(
        self, operation: str = "upsert", max_rows: int = 1000
    ) -> None:
        scopes = {}
        if operation != "none":
            scopes[_TABLE] = {
                "allowed_operations": {
                    "upsert": ("insert", "update"),
                    "all": ("insert", "update", "delete"),
                }.get(operation, (operation,)),
                "allowed_insert_columns": (
                    ("domain", "name", "evidence_url", "notes")
                    if operation in {"upsert", "all"}
                    else ()
                ),
                "allowed_update_columns": ("name", "evidence_url", "notes"),
                "max_rows": max_rows,
            }
        preview = await self.agent.preview_source_permissions(
            source_id=self.source_id,
            read_mode="all",
            read_resource_ids=(),
            relational_write_scopes=scopes,
        )
        await self.agent.apply_source_permissions(
            source_id=self.source_id,
            confirmation_fingerprint=preview.confirmation_fingerprint,
        )

    def upsert(self, rows: list[dict[str, object]], *, notes: bool = False) -> None:
        columns = ["domain", "name", "evidence_url", *(["notes"] if notes else [])]
        self.model.configure(
            batch_upsert(self.source_id, _TABLE, columns, "domain", rows)
        )

    def update(self) -> None:
        self.model.configure(
            {
                "source_id": self.source_id,
                "sql": f"UPDATE {_TABLE} SET name = $1 WHERE id = $2",
                "parameters": ["Updated", 1],
            }
        )

    async def run(self) -> None:
        before = len(self.model.requests)
        async with asyncio.timeout(60):
            result = await self.agent.run("Apply the exact fixture canary once.")
        assert result.reason == "completed", result
        requests = len(self.model.requests) - before
        assert requests in {3, 4}
        assert result.usage.total_tokens == 12 * requests
        assert self.model.results["load"].is_error is False

    async def rows(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in await self.admin.fetch(
                f"SELECT id, domain, name, evidence_url, notes FROM {_TABLE} ORDER BY domain"
            )
        ]

    async def seed(self) -> None:
        await self.admin.execute(
            f"INSERT INTO {_TABLE} (domain, name, evidence_url, notes) VALUES "
            "('existing.test', 'Before', 'https://evidence.test/existing', 'Retained')"
        )

    async def receipt(self, outcome: EffectOutcome) -> EffectReceipt:
        receipts = await self.agent.list_effects()
        assert len(receipts) == 1, {
            call_id: block.output
            for call_id, block in self.model.results.items()
            if block.is_error
        }
        receipt = receipts[0]
        assert receipt.outcome is outcome
        assert receipt.evidence_basis is EffectEvidenceBasis.ADAPTER_VERIFIED
        assert receipt.finished_at is not None and receipt.payload is not None
        assert await self.agent.inspect_effect(receipt.receipt_id) == receipt
        return receipt

    async def reopen(self) -> None:
        await self.agent.close()
        self.agent = await Agent.open(
            "postgres-write-release",
            root=self.root,
            model=self.model,
            model_profile=self.model.model_profile,
            secret_provider=WriteTesterSecrets(),
            workspace=workspace_for(self.root),
            approval_handler=self.approve,
        )


async def _admin_connection() -> Any:
    import asyncpg  # type: ignore[import-untyped]

    # No configurable host/database: this harness must never target a customer DB.
    return await asyncpg.connect(
        host="127.0.0.1",
        port=int(os.environ.get("DAITA_LARGE_POSTGRES_PORT", "55433")),
        database="daita_large_fixture",
        user="postgres",
        password=os.environ.get(
            "DAITA_LARGE_POSTGRES_ADMIN_PASSWORD", "fixture_admin_password"
        ),
        ssl=False,
        timeout=5,
        command_timeout=10,
        server_settings={"application_name": "daita-write-release-control"},
    )


@dataclass
class LiveWriteSource:
    """A home whose one writable source is the real postgres-large fixture."""

    admin: Any
    source_id: str
    research: Any
    research_factory: SDKMCPClientFactory
    parameters: tuple[str, str, str]

    async def rows(self) -> Mapping[str, Mapping[str, Any]]:
        return {
            row["domain"]: dict(row)
            for row in await self.admin.fetch(f"SELECT * FROM {_TABLE}")
        }


@contextlib.asynccontextmanager
async def live_write_source(root: Path, agent: str) -> AsyncIterator[LiveWriteSource]:
    """Prepare and close one agent home; live job and routine runs then open it."""

    admin = await _admin_connection()
    locked = await admin.fetchval("SELECT pg_try_advisory_lock($1)", _LOCK_ID)
    assert locked, "Another write-release suite owns this fixture; run serially."
    try:
        await _reset(admin)
        research, _ = conformance_identities()
        research.results["lookup"] = {
            "content": [
                {
                    "type": "text",
                    "text": "New Co lists Acme as a customer. "
                    "https://source.test/new; one source, coverage is incomplete.",
                }
            ],
            "structuredContent": {
                "answer": "New Co; https://source.test/new; model must assess the claim"
            },
            "isError": False,
        }
        factory = SDKMCPClientFactory(http_transport=mock_transport(research))
        model = ScriptedResearchModel()
        home = await Agent.create(
            agent,
            root=root,
            model=model,
            model_profile=model.model_profile,
            secret_provider=WriteTesterSecrets(),
            workspace=workspace_for(root),
            mcp_client_factory=factory,
        )
        try:
            source = await home.attach_postgresql(
                host="127.0.0.1",
                port=int(os.environ.get("DAITA_LARGE_POSTGRES_PORT", "55433")),
                database="daita_large_fixture",
                username=_ROLE,
                credential=SecretReference.keychain(
                    "fixture:postgres-large:write-release"
                ),
                schemas=("write_acceptance",),
                ssl_mode="disable",
            )
            preview = await home.preview_source_permissions(
                source_id=source.id,
                read_mode="all",
                read_resource_ids=(),
                relational_write_scopes={
                    _TABLE: {
                        "allowed_operations": ("insert", "update"),
                        "allowed_insert_columns": ("domain", "name", "evidence_url"),
                        "allowed_update_columns": ("name", "evidence_url"),
                        "max_rows": 10,
                    }
                },
            )
            await home.apply_source_permissions(
                source_id=source.id,
                confirmation_fingerprint=preview.confirmation_fingerprint,
            )
            await home.attach_mcp_server(
                endpoint=research.endpoint,
                selections=(
                    MCPToolSelection(
                        remote_name="lookup",
                        local_alias="research",
                        description="Research companies and return cited evidence.",
                        # A higher class would lift the request above this source's.
                        result_sensitivity=ModelSensitivity.INTERNAL,
                    ),
                ),
                maximum_outbound_sensitivity=ModelSensitivity.INTERNAL,
            )
        finally:
            await home.close()
        yield LiveWriteSource(
            admin,
            source.id,
            research,
            factory,
            ("new.test", "New Co", "https://source.test/new"),
        )
    finally:
        await _reset(admin)
        await admin.fetchval("SELECT pg_advisory_unlock($1)", _LOCK_ID)
        await admin.close()


@pytest.fixture
async def database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Fixture]:
    admin = await _admin_connection()
    locked = False
    fixture = None
    probe = DriverProbe()
    agent = None
    try:
        assert await admin.fetchval("SELECT ready FROM private.fixture_status") is True
        assert (
            await admin.fetchval("SELECT current_database()") == "daita_large_fixture"
        )
        assert (
            await admin.fetchval("SELECT to_regclass($1)::text", _TABLE) == _TABLE
        ), "Recreate the disposable postgres-large fixture with the current init.sql."
        assert (
            await admin.fetchval("SELECT to_regclass('write_acceptance.cells')")
            is not None
        )
        locked = await admin.fetchval("SELECT pg_try_advisory_lock($1)", _LOCK_ID)
        assert locked, "Another write-release suite owns this fixture; run serially."
        await _reset(admin)
        model = WriteModel()
        agent = await Agent.create(
            "postgres-write-release",
            root=tmp_path,
            model=model,
            model_profile=model.model_profile,
            secret_provider=WriteTesterSecrets(),
            workspace=workspace_for(tmp_path),
            approval_handler=lambda request: approve(request),
        )

        async def approve(request: ApprovalRequest) -> ApprovalDecision:
            assert fixture is not None
            return await fixture.approve(request)

        source = await agent.attach_postgresql(
            host="127.0.0.1",
            port=int(os.environ.get("DAITA_LARGE_POSTGRES_PORT", "55433")),
            database="daita_large_fixture",
            username=_ROLE,
            credential=SecretReference.keychain("fixture:postgres-large:write-release"),
            schemas=("write_acceptance",),
            ssl_mode="disable",
        )
        inspection = await agent.inspect_source_permissions(source.id)
        resources = {
            item.display_name: item.resource_id for item in inspection.resources
        }
        assert set(resources) == {_TABLE, "write_acceptance.cells"}
        connect = native._connect

        async def observed_connect(*args: Any, **kwargs: Any) -> Any:
            return probe.wrap(await connect(*args, **kwargs))

        monkeypatch.setattr(native, "_connect", observed_connect)
        fixture = Fixture(
            agent,
            model,
            admin,
            probe,
            source.id,
            resources[_TABLE],
            resources["write_acceptance.cells"],
            tmp_path,
            [],
        )
        await fixture.permissions()
        yield fixture
    finally:
        try:
            if fixture is not None:
                await fixture.agent.close()
            elif agent is not None:
                await agent.close()
        finally:
            # Verify cleanup instead of hiding leaked transaction connections.
            leaked = [
                connection
                for connection in probe.connections
                if not connection.is_closed()
            ]
            for connection in leaked:
                connection.terminate()
            try:
                if locked:
                    await _reset(admin)
                    await admin.fetchval("SELECT pg_advisory_unlock($1)", _LOCK_ID)
            finally:
                await admin.close()
            assert not leaked, "native write left an asyncpg connection open"


def batch_upsert(
    source_id: str,
    table: str,
    columns: list[str],
    key: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """One INSERT ... VALUES ... ON CONFLICT with bound parameters."""

    values = ", ".join(
        "("
        + ", ".join(f"${i * len(columns) + j + 1}" for j in range(len(columns)))
        + ")"
        for i in range(len(rows))
    )
    updates = ", ".join(
        f"{column} = EXCLUDED.{column}" for column in columns if column != key
    )
    return {
        "source_id": source_id,
        "sql": (
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES {values} "
            f"ON CONFLICT ({key}) DO UPDATE SET {updates}"
        ),
        "parameters": [item.get(column) for item in rows for column in columns],
    }


def row(domain: str = "new.test", name: str = "New") -> dict[str, object]:
    return {
        "domain": domain,
        "name": name,
        "evidence_url": f"https://evidence.test/{domain}",
    }
