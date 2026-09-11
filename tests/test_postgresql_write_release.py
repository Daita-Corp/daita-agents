"""Opt-in release checks against the existing disposable postgres-large fixture.

Real Agent, catalog discovery, permissions, runtime, SQL, asyncpg, PostgreSQL
transactions and SQLite receipts; scripted model only. Never starts Docker.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import pytest
from _postgresql_write_release_support import DriverProbe, WriteModel
from _workspace_support import workspace_for

from daita import Agent, ApprovalDecision, ApprovalRequest
from daita._json import thaw_json
from daita.adapters import postgresql_write as native
from daita.capabilities import EffectEvidenceBasis, EffectOutcome
from daita.security import SecretReference
from daita.storage.sqlite_records import EffectReceipt, EffectResolutionDecision

pytestmark = [
    pytest.mark.integration,
    pytest.mark.acceptance,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get("DAITA_RUN_POSTGRES_WRITE_RELEASE") != "1",
        reason="set DAITA_RUN_POSTGRES_WRITE_RELEASE=1 for the disposable postgres-large fixture",
    ),
]

_TABLE = "write_acceptance.companies"
_ROLE = "daita_large_write_tester"
_LOCK_ID = 7_346_218_930


class _Secrets:
    async def resolve(self, reference: SecretReference) -> str:
        assert reference == SecretReference.keychain(
            "fixture:postgres-large:write-release"
        )
        return os.environ.get(
            "DAITA_LARGE_POSTGRES_WRITE_TESTER_PASSWORD",
            "daita_large_write_tester_fixture_password",
        )


async def _reset(admin: Any) -> None:
    # These are the only objects this suite mutates. Never reset the large
    # reader workload, support.tickets, an arbitrary schema or an agent home.
    await admin.execute(f"ALTER TABLE {_TABLE} DROP COLUMN IF EXISTS release_drift")
    await admin.execute(f"GRANT SELECT, UPDATE ON {_TABLE} TO {_ROLE}")
    await admin.execute(
        f"GRANT INSERT (domain, name, evidence_url, notes) ON {_TABLE} TO {_ROLE}"
    )
    await admin.execute(
        f"TRUNCATE ONLY {_TABLE}, ONLY write_acceptance.cells RESTART IDENTITY"
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
            scopes[self.resource_id] = {
                "allowed_operations": (operation,),
                "allowed_insert_columns": (
                    ("domain", "name", "evidence_url", "notes")
                    if operation == "upsert"
                    else ()
                ),
                "allowed_update_columns": ("name", "evidence_url", "notes"),
                "key_columns": ("domain",) if operation == "upsert" else ("id",),
                "generated_identity_columns": ("id",) if operation == "upsert" else (),
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
        self.model.configure(
            "upsert",
            {
                "source_id": self.source_id,
                "resource_id": self.resource_id,
                "key_columns": ("domain",),
                "insert_columns": (
                    ("domain", "name", "evidence_url", "notes")
                    if notes
                    else ("domain", "name", "evidence_url")
                ),
                "update_columns": (
                    ("name", "evidence_url", "notes")
                    if notes
                    else ("name", "evidence_url")
                ),
                "rows": rows,
            },
        )

    def update(self) -> None:
        self.model.configure(
            "update",
            {
                "source_id": self.source_id,
                "resource_id": self.resource_id,
                "where": [{"column": "id", "operator": "eq", "value": 1}],
                "assignments": [{"column": "name", "value": "Updated"}],
            },
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
            secret_provider=_Secrets(),
            workspace=workspace_for(self.root),
            approval_handler=self.approve,
        )


@pytest.fixture
async def database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Fixture]:
    import asyncpg  # type: ignore[import-untyped]

    # No configurable host/database: this harness must never target a customer DB.
    admin = await asyncpg.connect(
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
            secret_provider=_Secrets(),
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


def row(domain: str = "new.test", name: str = "New") -> dict[str, object]:
    return {
        "domain": domain,
        "name": name,
        "evidence_url": f"https://evidence.test/{domain}",
    }


async def test_fixture_role_is_restricted_and_cannot_delete_or_create(
    database: Fixture,
) -> None:
    facts = await database.admin.fetchrow(
        "SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls "
        "FROM pg_roles WHERE rolname = $1",
        _ROLE,
    )
    assert facts is not None and not any(facts.values())
    for privilege in ("DELETE", "TRUNCATE", "TRIGGER", "REFERENCES"):
        assert not await database.admin.fetchval(
            "SELECT has_table_privilege($1, $2, $3)", _ROLE, _TABLE, privilege
        )
    assert not await database.admin.fetchval(
        "SELECT has_schema_privilege($1, 'write_acceptance', 'CREATE')", _ROLE
    )


async def test_mixed_upsert_generated_identity_exact_receipt_and_unchanged_repeat(
    database: Fixture,
) -> None:
    await database.seed()
    await database.admin.execute(
        f"INSERT INTO {_TABLE} (domain, name, evidence_url) "
        "VALUES ('stable.test', 'Stable', 'https://evidence.test/stable.test')"
    )
    rows: list[dict[str, object]] = [
        {
            "domain": "existing.test",
            "name": "After",
            "evidence_url": "https://evidence.test/existing",
        },
        row(),
        row("stable.test", "Stable"),
    ]
    database.upsert(rows)
    await database.run()
    receipt = await database.receipt(EffectOutcome.SUCCEEDED)
    assert receipt.payload is not None
    assert (
        receipt.payload["inserted_count"],
        receipt.payload["updated_count"],
        receipt.payload["unchanged_count"],
    ) == (1, 1, 1)
    actual = await database.rows()
    assert [(item["domain"], item["name"]) for item in actual] == [
        ("existing.test", "After"),
        ("new.test", "New"),
        ("stable.test", "Stable"),
    ]
    assert actual[0]["id"] == 1 and actual[0]["notes"] == "Retained"
    assert actual[1]["id"] > 1 and actual[1]["notes"] is None
    output = database.model.results["write"].output["data"]
    assert isinstance(output, Mapping)
    assert thaw_json(output["generated_identities"]) == [
        {"key": {"domain": "new.test"}, "values": {"id": actual[1]["id"]}},
    ]
    assert database.probe.mutations == 2 and database.probe.server_commits == 1
    assert len(database.approvals) == 1
    approved_arguments = database.approvals[0].arguments["arguments"]
    assert isinstance(approved_arguments, Mapping)
    assert (
        approved_arguments["preview_fingerprint"]
        == receipt.payload["preview_fingerprint"]
    )
    database.upsert(rows)
    await database.run()
    receipts = await database.agent.list_effects()
    assert len(receipts) == 2
    repeated = next(item for item in receipts if item.receipt_id != receipt.receipt_id)
    assert repeated.outcome is EffectOutcome.SUCCEEDED and repeated.payload is not None
    assert (
        repeated.payload["inserted_count"],
        repeated.payload["updated_count"],
        repeated.payload["unchanged_count"],
    ) == (0, 0, 3)
    assert database.probe.mutations == 2 and database.probe.server_commits == 2
    assert await database.rows() == actual
    await database.reopen()
    assert await database.agent.inspect_effect(receipt.receipt_id) == receipt
    assert len(await database.agent.list_effects()) == 2
    assert await database.rows() == actual


async def test_explicit_null_clears_while_omitted_value_is_retained(
    database: Fixture,
) -> None:
    await database.seed()
    database.upsert(
        [
            {
                "domain": "existing.test",
                "name": "Before",
                "evidence_url": "https://evidence.test/existing",
                "notes": None,
            }
        ],
        notes=True,
    )
    await database.run()
    await database.receipt(EffectOutcome.SUCCEEDED)
    assert (await database.rows())[0]["notes"] is None


@pytest.mark.parametrize("operation", ["update", "upsert"])
async def test_denied_approval_has_no_mutation_or_receipt(
    database: Fixture, operation: str
) -> None:
    await database.permissions(operation)
    await database.seed()
    before = await database.rows()
    database.approval_decision = ApprovalDecision.DENY
    database.update() if operation == "update" else database.upsert([row()])
    await database.run()
    assert len(database.approvals) == 1
    assert database.model.results["write"].is_error
    assert not await database.agent.list_effects()
    assert database.probe.mutations == 0 and await database.rows() == before


@pytest.mark.parametrize("operation", ["update", "upsert"])
@pytest.mark.parametrize("change", ["values", "structure", "database_permission"])
async def test_change_during_exact_approval_prevents_mutation(
    database: Fixture, operation: str, change: str
) -> None:
    await database.permissions(operation)
    await database.seed()
    (
        database.update()
        if operation == "update"
        else database.upsert(
            [
                {
                    "domain": "existing.test",
                    "name": "Updated",
                    "evidence_url": "https://evidence.test/existing",
                }
            ]
        )
    )

    async def change_after_preview() -> None:
        if change == "values":
            await database.admin.execute(
                f"UPDATE {_TABLE} SET name = 'Concurrent writer'"
            )
        elif change == "structure":
            await database.admin.execute(
                f"ALTER TABLE {_TABLE} ADD COLUMN release_drift text"
            )
        elif change == "database_permission":
            await database.admin.execute(f"REVOKE UPDATE ON {_TABLE} FROM {_ROLE}")

    database.approval_hook = change_after_preview
    await database.run()
    assert len(database.approvals) == 1 and database.model.results["write"].is_error
    assert database.probe.mutations == 0
    assert [item["name"] for item in await database.rows()] == [
        "Concurrent writer" if change == "values" else "Before"
    ]
    # The runtime repeats preflight after approval, before reserving a receipt.
    # These changes must be rejected there, without entering a transaction.
    assert not await database.agent.list_effects()
    assert database.probe.write_pid is None
    error = database.model.results["write"].output["error"]
    assert isinstance(error, Mapping) and error["code"] == "state_changed"


@pytest.mark.parametrize("collision", ["within_batch", "after_preview"])
async def test_second_insert_unique_violation_rolls_back_entire_batch(
    database: Fixture,
    collision: str,
) -> None:
    # A second unique key is real server enforcement, independent of the
    # selected domain conflict key. The first insertion must also roll back.
    rows = [
        row("a.test"),
        {**row("b.test"), "evidence_url": "https://evidence.test/a.test"},
    ]
    expected_rows = []
    if collision == "after_preview":
        rows = [row("a.test"), row("b.test")]

        async def occupy_evidence_url() -> None:
            await database.admin.execute(
                f"INSERT INTO {_TABLE} (domain, name, evidence_url) VALUES "
                "('collision.test', 'Concurrent writer', 'https://evidence.test/b.test')"
            )

        database.approval_hook = occupy_evidence_url
        expected_rows = [
            {
                "id": 1,
                "domain": "collision.test",
                "name": "Concurrent writer",
                "evidence_url": "https://evidence.test/b.test",
                "notes": None,
            }
        ]
    database.upsert(rows)
    await database.run()
    receipt = await database.receipt(EffectOutcome.NOT_APPLIED)
    assert receipt.payload is not None
    assert receipt.payload["normalized_error_code"] == "write_constraint_violation"
    assert (
        receipt.payload["inserted_count"] == 0 and receipt.payload["updated_count"] == 0
    )
    assert receipt.payload["identity_sequence_gaps_possible"] is True
    assert database.probe.mutations == 1 and database.probe.commit_attempts == 0
    assert await database.rows() == expected_rows
    # Sequences are intentionally not a rollback guarantee.
    assert (
        await database.admin.fetchval(
            "SELECT last_value FROM write_acceptance.companies_id_seq"
        )
        >= 2
    )


@pytest.mark.parametrize("operation", ["update", "upsert"])
async def test_wrong_reported_update_count_rolls_back_real_mutation(
    database: Fixture, operation: str
) -> None:
    await database.permissions(operation)
    await database.seed()
    before = await database.rows()
    database.probe.mode = "wrong_update_count"
    (
        database.update()
        if operation == "update"
        else database.upsert(
            [
                {
                    "domain": "existing.test",
                    "name": "Updated",
                    "evidence_url": "https://evidence.test/existing",
                }
            ]
        )
    )
    await database.run()
    receipt = await database.receipt(EffectOutcome.NOT_APPLIED)
    assert receipt.payload is not None
    assert receipt.payload["normalized_error_code"] == "write_affected_rows_mismatch"
    assert database.probe.mutations == 1 and database.probe.commit_attempts == 0
    assert await database.rows() == before


@pytest.mark.parametrize(
    "invalid", ["duplicate_key", "null_key", "row_limit", "update_only", "identity"]
)
async def test_invalid_batch_is_rejected_before_native_dispatch(
    database: Fixture, invalid: str
) -> None:
    await database.permissions(
        "update" if invalid == "update_only" else "upsert", max_rows=2
    )
    rows = [row()]
    if invalid == "duplicate_key":
        rows = [row(), row()]
    elif invalid == "null_key":
        rows = [{**row(), "domain": None}]
    elif invalid == "row_limit":
        rows = [row(f"row-{i}.test") for i in range(3)]
    database.upsert(rows)
    if invalid == "identity":
        database.model.arguments["insert_columns"] = (
            "id",
            "domain",
            "name",
            "evidence_url",
        )
        database.model.arguments["rows"] = [{**row(), "id": 42}]
    # Some invalid scopes reject toolbox load; still submit the exact preview
    # to establish a structured denial, without accepting an empty script.
    result = await database.agent.run("Preview only the requested fixture batch.")
    assert result.reason == "completed"
    assert database.model.results["preview"].is_error
    assert "write" not in database.model.results
    assert not database.approvals and not await database.agent.list_effects()
    assert not database.probe.connections and not await database.rows()


@pytest.mark.parametrize("operation", ["update", "upsert"])
async def test_revoked_daita_permission_stays_revoked_after_reopen(
    database: Fixture, operation: str
) -> None:
    await database.permissions(operation)
    await database.seed()
    before = await database.rows()
    database.update() if operation == "update" else database.upsert([row()])
    database.model.preview_only = True
    await database.run()
    assert not database.model.results["preview"].is_error
    # Owner controls serialize with runs. Exercise revocation between runs,
    # rather than trying to re-enter the run lock from an approval callback.
    await database.permissions("none")
    await database.reopen()
    connections = len(database.probe.connections)
    database.update() if operation == "update" else database.upsert([row()])
    result = await database.agent.run("Try the old fixture write permission.")
    assert result.reason == "completed"
    assert database.model.results["preview"].is_error
    assert not database.approvals and not await database.agent.list_effects()
    assert len(database.probe.connections) == connections
    assert await database.rows() == before


async def _wait_for_lock(database: Fixture) -> None:
    async with asyncio.timeout(10):
        while True:
            if database.probe.write_pid is not None:
                await database.admin.execute("SELECT pg_stat_clear_snapshot()")
                waiting = await database.admin.fetchval(
                    "SELECT wait_event_type = 'Lock' FROM pg_stat_activity WHERE pid = $1",
                    database.probe.write_pid,
                )
                if waiting:
                    return
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("operation", ["update", "upsert"])
async def test_real_lock_timeout_is_bounded_and_does_not_mutate(
    database: Fixture, operation: str
) -> None:
    await database.permissions(operation)
    await database.seed()
    before = await database.rows()
    database.update() if operation == "update" else database.upsert([row()])
    transaction = database.admin.transaction()
    await transaction.start()
    # Blocks row writers and EXCLUSIVE, but permits the read-only preview.
    await database.admin.execute(f"LOCK TABLE {_TABLE} IN SHARE MODE")
    started = monotonic()
    try:
        await database.run()
    finally:
        await transaction.rollback()
    if operation == "update":
        # PostgreSQL EXPLAIN acquires the UPDATE statement's table lock even
        # though the preview transaction is read-only. No effect is reserved.
        assert not await database.agent.list_effects()
        assert database.probe.write_pid is None
        error = database.model.results["preview"].output["error"]
        assert isinstance(error, Mapping) and error["code"] == "write_lock_timeout"
    else:
        receipt = await database.receipt(EffectOutcome.NOT_APPLIED)
        assert receipt.payload is not None
        assert receipt.payload["normalized_error_code"] == "write_lock_timeout"
    assert monotonic() - started < 15
    assert database.probe.mutations == 0 and await database.rows() == before


async def test_concurrent_insert_while_waiting_is_detected_after_exclusive_lock(
    database: Fixture,
) -> None:
    database.upsert([row()])
    transaction = database.admin.transaction()
    await transaction.start()
    await database.admin.execute(f"LOCK TABLE {_TABLE} IN ROW EXCLUSIVE MODE")
    task = asyncio.create_task(database.run())
    try:
        await _wait_for_lock(database)
        await database.admin.execute(
            f"INSERT INTO {_TABLE} (domain, name, evidence_url) VALUES ('new.test', 'Concurrent writer', 'https://evidence.test/new.test')"
        )
        await transaction.commit()
        await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if database.admin.is_in_transaction():
            await transaction.rollback()
    receipt = await database.receipt(EffectOutcome.NOT_APPLIED)
    assert (
        receipt.payload is not None
        and receipt.payload["normalized_error_code"] == "write_state_changed"
    )
    assert database.probe.mutations == 0
    assert [item["name"] for item in await database.rows()] == ["Concurrent writer"]


@pytest.mark.parametrize("operation", ["update", "upsert"])
@pytest.mark.parametrize("stage", ["pause_after_mutation", "pause_after_commit"])
async def test_driver_cancellation_preserves_database_truth_and_terminal_receipt(
    database: Fixture, operation: str, stage: str
) -> None:
    await database.permissions(operation)
    await database.seed()
    before = await database.rows()
    database.probe.mode = stage
    (
        database.update()
        if operation == "update"
        else database.upsert([row("a.test"), row("b.test")])
    )
    task = asyncio.create_task(
        database.agent.run("Apply the exact fixture canary once.")
    )
    try:
        await asyncio.wait_for(database.probe.reached.wait(), timeout=15)
        # Inject cancellation at the asyncpg boundary. Cancelling the outer
        # Agent.run instead deliberately drains its already-started effect.
        assert database.probe.worker is not None
        database.probe.worker.cancel()
        result = await asyncio.wait_for(task, timeout=10)
        assert result.reason == "tool_batch_interrupted" and result.final_text is None
        assert len(database.model.requests) == 3 and result.usage.total_tokens == 36
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    outcome = (
        EffectOutcome.NOT_APPLIED
        if stage == "pause_after_mutation"
        else EffectOutcome.UNCERTAIN
    )
    receipt = await database.receipt(outcome)
    if stage == "pause_after_mutation":
        assert await database.rows() == before
        assert database.probe.mutations == 1 and database.probe.server_commits == 0
    else:
        assert await database.rows() != before and database.probe.server_commits == 1
    await database.reopen()
    assert await database.agent.inspect_effect(receipt.receipt_id) == receipt


@pytest.mark.parametrize("operation", ["update", "upsert"])
@pytest.mark.parametrize(
    "stage", ["disconnect_before_commit", "disconnect_after_commit"]
)
@pytest.mark.parametrize("decision", list(EffectResolutionDecision))
async def test_lost_commit_confirmation_blocks_replay_and_recovery_performs_no_action(
    database: Fixture, operation: str, stage: str, decision: EffectResolutionDecision
) -> None:
    await database.permissions(operation)
    await database.seed()
    before = await database.rows()
    database.probe.mode = stage
    database.update() if operation == "update" else database.upsert([row()])
    await database.run()
    receipt = await database.receipt(EffectOutcome.UNCERTAIN)
    assert database.model.results["write"].is_error
    assert database.probe.commit_attempts == 1
    actual = await database.rows()
    assert (actual == before) is (stage == "disconnect_before_commit")
    assert database.probe.server_commits == int(stage == "disconnect_after_commit")
    await database.reopen()
    assert await database.agent.inspect_effect(receipt.receipt_id) == receipt
    mutations = database.probe.mutations
    # New intent/run/call identity still cannot bypass the unresolved receipt.
    database.probe.mode = None
    (
        database.update()
        if operation == "update"
        else database.upsert([row("different.test")])
    )
    await database.run()
    assert database.model.results["write"].is_error
    assert len(await database.agent.list_effects()) == 1
    assert database.probe.mutations == mutations and await database.rows() == actual
    # Recovery is allowed even after connector authority has been revoked.
    await database.permissions("none")
    requests = len(database.model.requests)
    resolved = await database.agent.resolve_effect(
        receipt.receipt_id,
        expected_digest=receipt.receipt_digest,
        decision=decision,
        note="Fixture administrator independently inspected table state; no retry requested.",
    )
    assert resolved.resolution is not None and resolved.resolution.decision is decision
    assert (
        resolved.outcome is EffectOutcome.UNCERTAIN
        and resolved.payload == receipt.payload
    )
    assert len(database.model.requests) == requests
    assert database.probe.mutations == mutations and database.probe.commit_attempts == 1
    assert await database.rows() == actual
    assert not (
        await database.agent.inspect_source_permissions(database.source_id)
    ).state.relational_write_scopes
    await database.reopen()
    assert await database.agent.inspect_effect(receipt.receipt_id) == resolved


@pytest.mark.parametrize("count", [1, 100, 1000])
async def test_bounded_batch_size_records_real_transaction_duration(
    database: Fixture, count: int, record_property: Callable[[str, object], None]
) -> None:
    preview = await database.agent.preview_source_permissions(
        source_id=database.source_id,
        read_mode="all",
        read_resource_ids=(),
        relational_write_scopes={
            database.cells_resource_id: {
                "allowed_operations": ("upsert",),
                "allowed_insert_columns": ("k", "v"),
                "allowed_update_columns": ("v",),
                "key_columns": ("k",),
                "generated_identity_columns": (),
                "max_rows": 1000,
            }
        },
    )
    await database.agent.apply_source_permissions(
        source_id=database.source_id,
        confirmation_fingerprint=preview.confirmation_fingerprint,
    )
    database.model.configure(
        "upsert",
        {
            "source_id": database.source_id,
            "resource_id": database.cells_resource_id,
            "key_columns": ("k",),
            "insert_columns": ("k", "v"),
            "update_columns": ("v",),
            "rows": [{"k": i, "v": i} for i in range(count)],
        },
    )
    started = monotonic()
    await database.run()
    elapsed = monotonic() - started
    receipt = await database.receipt(EffectOutcome.SUCCEEDED)
    assert receipt.payload is not None
    assert (
        receipt.payload["input_count"],
        receipt.payload["inserted_count"],
        receipt.payload["updated_count"],
        receipt.payload["unchanged_count"],
    ) == (count, count, 0, 0)
    actual = await database.admin.fetch(
        "SELECT k, v FROM write_acceptance.cells ORDER BY k"
    )
    assert [tuple(item) for item in actual] == [(i, i) for i in range(count)]
    assert database.probe.mutations == count and database.probe.server_commits == 1
    record_property("batch_rows", count)
    record_property("scripted_agent_wall_seconds", round(elapsed, 6))
    record_property(
        "postgresql_version", await database.admin.fetchval("SHOW server_version")
    )


async def test_wide_batch_cannot_bypass_byte_and_exact_approval_limits(
    database: Fixture,
) -> None:
    database.upsert([row(f"company-{i:04d}.test") for i in range(1000)])
    await database.run()
    error = database.model.results["preview"].output["error"]
    assert isinstance(error, Mapping) and error["code"] == "upsert_invalid_batch"
    assert "64 KiB" in str(error["message"])
    assert not database.probe.connections and not database.approvals
    assert not await database.agent.list_effects() and not await database.rows()


@pytest.mark.parametrize("operation", ["update", "upsert"])
async def test_foreground_cancellation_drains_started_write_and_retains_commit(
    database: Fixture, operation: str
) -> None:
    await database.permissions(operation)
    await database.seed()
    database.probe.mode = "pause_after_mutation"
    database.update() if operation == "update" else database.upsert([row()])
    task = asyncio.create_task(database.agent.run("Apply the fixture canary once."))
    try:
        await asyncio.wait_for(database.probe.reached.wait(), timeout=15)
        task.cancel()
        # Let the shielded native operation finish normally. User cancellation
        # must not pretend an already-started external effect was rolled back.
        database.probe.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
    finally:
        if not task.done():
            task.cancel()
        database.probe.release.set()
        await asyncio.gather(task, return_exceptions=True)
    receipt = await database.receipt(EffectOutcome.SUCCEEDED)
    assert database.probe.mutations == 1 and database.probe.server_commits == 1
    actual = await database.rows()
    if operation == "update":
        assert [item["name"] for item in actual] == ["Updated"]
    else:
        assert len(actual) == 2
    await database.reopen()
    assert await database.agent.inspect_effect(receipt.receipt_id) == receipt
    assert await database.rows() == actual
    assert database.probe.mutations == 1
