"""Opt-in real-PostgreSQL certification for the Phase 4 update contract.

This suite never starts or stops the fixture. The operator must explicitly start
the disposable tmpfs-backed fixture and authorize this exact test command.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from daita import Agent, ApprovalDecision, ApprovalRequest
from daita.adapters.models import SourceRegistration
from daita.catalog.models import CatalogResource
from daita.domains.data.capabilities import (
    POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME,
    POSTGRESQL_UPDATE_TOOL_NAME,
)
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import LoopExitKind, Transcript
from daita.security import EnvironmentSecretProvider, SecretReference
from daita.storage.sqlite import DatabaseWriteOutcome

_AUTHORIZATION = "DAITA_RUN_POSTGRESQL_UPDATE_CERTIFICATION"
_WRITER_PASSWORD = "DAITA_FIXTURE_POSTGRES_WRITER_PASSWORD"
_ADMIN_PASSWORD = "DAITA_FIXTURE_POSTGRES_ADMIN_PASSWORD"
_PORT = "DAITA_FIXTURE_POSTGRES_PORT"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing the "
            "disposable PostgreSQL certification fixture"
        ),
    ),
]


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        pytest.fail(f"{name} is required for authorized certification")
    return value


async def _database_connection(
    *,
    admin: bool = False,
    database: str = "daita_fixture",
):
    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError:
        pytest.fail("asyncpg must be available in the complete application install")
    return await asyncpg.connect(
        host="127.0.0.1",
        port=int(os.environ.get(_PORT, "55432")),
        database=database,
        user="postgres" if admin else "daita_writer",
        password=_required_environment(_ADMIN_PASSWORD if admin else _WRITER_PASSWORD),
        ssl=False,
        server_settings={"search_path": "pg_catalog"},
    )


async def _reset_canary() -> None:
    connection = await _database_connection(admin=True)
    try:
        await connection.execute("""
            UPDATE write_canary.accounts
            SET status = 'active',
                external_key = 'canary-42',
                region_code = 'NA',
                note = 'phase-4 canary',
                counter = 0,
                updated_at = timestamptz '2026-08-10 00:00:00+00'
            WHERE account_id = 42
            """)
        await connection.execute("""
            UPDATE write_canary.accounts
            SET status = 'inactive',
                external_key = 'canary-43',
                region_code = 'EU',
                note = 'constraint peer',
                counter = 1,
                updated_at = timestamptz '2026-08-10 00:00:00+00'
            WHERE account_id = 43
            """)
        await connection.execute(
            "ALTER TABLE write_canary.accounts "
            "DROP COLUMN IF EXISTS phase4_stale_probe"
        )
    finally:
        await connection.close()


async def _set_writer_status_update_grant(*, enabled: bool) -> None:
    connection = await _database_connection(admin=True)
    action = "GRANT" if enabled else "REVOKE"
    preposition = "TO" if enabled else "FROM"
    try:
        await connection.execute(
            f"{action} UPDATE (status) ON write_canary.accounts "
            f"{preposition} daita_writer"
        )
    finally:
        await connection.close()


def _tool_results(
    messages: Sequence[CanonicalMessage],
) -> dict[str, ToolResultBlock]:
    return {
        block.call_id: block
        for message in messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }


def _tool_error_code(result: ToolResultBlock) -> object:
    error = result.output["error"]
    assert isinstance(error, Mapping)
    return error["code"]


class _ScriptedUpdateProvider:
    """Deterministic model boundary for real adapter/runtime certification."""

    provider_id = "openai:phase4-fixture-model"

    def __init__(self) -> None:
        self.source_id = ""
        self.resource_id = ""
        self.assignments: tuple[dict[str, object], ...] = ()
        self.requests: list[ModelRequest] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        results = tuple(
            block
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
        if not results:
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="preview-call",
                        name=POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME,
                        arguments={
                            "source_id": self.source_id,
                            "resource_id": self.resource_id,
                            "match": [{"column": "account_id", "value": 42}],
                            "assignments": self.assignments,
                        },
                    ),
                ),
                provider_id=self.provider_id,
            )
        preview = results[-1]
        if preview.call_id == "preview-call" and not preview.is_error:
            data = preview.output["data"]
            assert isinstance(data, Mapping)
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="update-call",
                        name=POSTGRESQL_UPDATE_TOOL_NAME,
                        arguments={
                            "source_id": self.source_id,
                            "resource_id": self.resource_id,
                            "match": [{"column": "account_id", "value": 42}],
                            "assignments": self.assignments,
                            "preview_fingerprint": data["preview_fingerprint"],
                            "max_affected_rows": 1,
                        },
                    ),
                ),
                provider_id=self.provider_id,
            )
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            text=(
                "The exact update committed."
                if not results[-1].is_error
                else "The exact update was not committed."
            ),
            provider_id=self.provider_id,
        )


async def _attached_agent(
    tmp_path: Path,
    *,
    name: str,
    table: str,
    assignments: tuple[dict[str, object], ...],
    approval_handler: Any,
) -> tuple[Agent, _ScriptedUpdateProvider, SourceRegistration, CatalogResource]:
    provider = _ScriptedUpdateProvider()
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=128_000,
        max_output_tokens=1_024,
        supports_tools=True,
    )
    agent = await Agent.create(
        name,
        root=tmp_path,
        model=provider,
        model_profile=profile,
        secret_provider=EnvironmentSecretProvider(),
        approval_handler=approval_handler,
    )
    source = await agent.attach_postgresql(
        host="127.0.0.1",
        port=int(os.environ.get(_PORT, "55432")),
        database="daita_fixture",
        username="daita_writer",
        credential=SecretReference.environment(_WRITER_PASSWORD),
        schemas=("write_canary",),
        ssl_mode="disable",
        name="Disposable Phase 4 PostgreSQL",
    )
    assert source.configuration["write_access"] is False
    resources = await agent.list_catalog_resources(source_id=source.id)
    resource = next(
        item for item in resources if item.native_identity == f"write_canary.{table}"
    )
    provider.source_id = source.id
    provider.resource_id = resource.id
    provider.assignments = assignments
    return agent, provider, source, resource


async def _run_update(
    tmp_path: Path,
    *,
    name: str,
    table: str = "accounts",
    assignments: tuple[dict[str, object], ...],
    approval_handler: Any,
):
    agent, provider, source, resource = await _attached_agent(
        tmp_path,
        name=name,
        table=table,
        assignments=assignments,
        approval_handler=approval_handler,
    )
    try:
        readiness_before = await agent.postgresql_update_readiness(
            source.id,
            resource.id,
            tuple(str(item["column"]) for item in assignments),
        )
        assert readiness_before.rejection_codes == ("write_access_not_enabled",)
        await agent.set_source_write_access(source.id, True)
        readiness = await agent.postgresql_update_readiness(
            source.id,
            resource.id,
            tuple(str(item["column"]) for item in assignments),
        )
        exit = await agent.run(
            "Execute the scripted exact update.", source_id=source.id
        )
        transcript = await agent.transcript(exit.run_id)
        receipt = await agent._embedded._store.load_database_write_receipt_for_call(
            agent.id,
            exit.run_id,
            "update-call",
        )
        return agent, provider, source, resource, readiness, exit, transcript, receipt
    except BaseException:
        await agent.close()
        raise


async def test_real_success_result_receipt_readiness_and_changed_type_encoding(
    tmp_path: Path,
) -> None:
    await _reset_canary()
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    assignments: tuple[dict[str, object], ...] = (
        {"column": "status", "value": "inactive"},
        {"column": "counter", "value": 2},
        {"column": "updated_at", "value": "2026-08-10T12:30:00Z"},
    )
    values = None
    agent, provider, source, resource, readiness, exit, transcript, receipt = (
        await _run_update(
            tmp_path,
            name="phase4-live-success",
            assignments=assignments,
            approval_handler=approve,
        )
    )
    try:
        connection = await _database_connection(admin=True)
        try:
            values = await connection.fetchrow(
                "SELECT status, counter, updated_at FROM write_canary.accounts "
                "WHERE account_id = 42"
            )
        finally:
            await connection.close()
    finally:
        await agent.close()
        await _reset_canary()

    assert readiness.ready_for_preview is True
    assert readiness.proves_execution is False
    assert exit.kind is LoopExitKind.COMPLETED, exit.reason
    results = _tool_results(transcript.messages)
    assert results["preview-call"].is_error is False
    assert results["update-call"].is_error is False
    result = results["update-call"].output["data"]
    assert isinstance(result, Mapping)
    assert result["outcome"] == "committed"
    assert result["affected_rows"] == 1
    returned = {
        item["column"]: item["value"]
        for item in result["returned"]
        if isinstance(item, Mapping)
    }
    assert returned == {
        "account_id": 42,
        "status": "inactive",
        "counter": 2,
        "updated_at": "2026-08-10T12:30:00+00:00",
    }
    assert receipt is not None
    assert receipt.outcome is DatabaseWriteOutcome.COMMITTED
    assert receipt.preview_fingerprint == result["preview_fingerprint"]
    assert values is not None
    assert tuple(values)[:2] == ("inactive", 2)
    assert values["updated_at"].isoformat() == "2026-08-10T12:30:00+00:00"
    assert len(approvals) == 1
    update_call = next(
        call
        for message in transcript.messages
        for call in message.tool_calls
        if call.name == POSTGRESQL_UPDATE_TOOL_NAME
    )
    assert approvals[0].arguments == update_call.arguments
    assert source.id == readiness.source_id
    assert resource.id == readiness.resource_id


async def test_writer_role_cannot_connect_to_an_unselected_database() -> None:
    with pytest.raises(Exception) as denied:
        await _database_connection(database="postgres")

    assert getattr(denied.value, "sqlstate", None) == "42501"


@pytest.mark.parametrize(
    ("assignment", "expected_code"),
    (
        ({"column": "status", "value": "invalid"}, "write_constraint_violation"),
        (
            {"column": "external_key", "value": "canary-43"},
            "write_constraint_violation",
        ),
        ({"column": "region_code", "value": "MISSING"}, "write_constraint_violation"),
    ),
)
async def test_real_constraints_roll_back_and_record_not_committed(
    tmp_path: Path,
    assignment: dict[str, object],
    expected_code: str,
) -> None:
    await _reset_canary()

    async def approve(_request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.APPROVE

    agent, _provider, _source, _resource, readiness, exit, transcript, receipt = (
        await _run_update(
            tmp_path,
            name="phase4-live-constraint-" + str(assignment["column"]),
            assignments=(assignment,),
            approval_handler=approve,
        )
    )
    try:
        connection = await _database_connection(admin=True)
        try:
            values = await connection.fetchrow(
                "SELECT status, external_key, region_code "
                "FROM write_canary.accounts WHERE account_id = 42"
            )
        finally:
            await connection.close()
    finally:
        await agent.close()
        await _reset_canary()

    result = _tool_results(transcript.messages)["update-call"]
    assert readiness.ready_for_preview is True
    assert exit.kind is LoopExitKind.COMPLETED
    assert result.is_error is True
    assert _tool_error_code(result) == expected_code
    assert receipt is not None
    assert receipt.outcome is DatabaseWriteOutcome.NOT_COMMITTED
    assert receipt.affected_rows == 0
    assert tuple(values) == ("active", "canary-42", "NA")


async def test_real_permissions_and_relation_guardrails_match_readiness(
    tmp_path: Path,
) -> None:
    await _reset_canary()

    async def deny(_request: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError("rejected readiness scopes must not request approval")

    cases = (
        (
            "permission_denied",
            ("status",),
            "write_privilege_column_update_missing",
            "write_guardrail_rejected",
        ),
        (
            "no_primary_key",
            ("status",),
            "write_primary_key_required",
            "write_primary_key_required",
        ),
        (
            "composite_primary_key",
            ("status",),
            "write_primary_key_required",
            "write_primary_key_required",
        ),
        (
            "rls_accounts",
            ("status",),
            "write_relation_rls_enabled",
            "write_guardrail_rejected",
        ),
        (
            "trigger_accounts",
            ("status",),
            "write_relation_user_triggers",
            "write_guardrail_rejected",
        ),
    )
    for index, (
        table,
        columns,
        readiness_code,
        preview_code,
    ) in enumerate(cases):
        agent, _provider, source, resource = await _attached_agent(
            tmp_path,
            name=f"phase4-live-guardrail-{index}",
            table=table,
            assignments=({"column": "status", "value": "inactive"},),
            approval_handler=deny,
        )
        try:
            await agent.set_source_write_access(source.id, True)
            readiness = await agent.postgresql_update_readiness(
                source.id,
                resource.id,
                columns,
            )
            assert readiness.ready_for_preview is False
            assert readiness_code in readiness.rejection_codes
            exit = await agent.run(
                "Attempt the scripted preview for this rejected scope.",
                source_id=source.id,
            )
            transcript = await agent.transcript(exit.run_id)
            result = _tool_results(transcript.messages)["preview-call"]
            assert result.is_error is True
            assert _tool_error_code(result) == preview_code
            assert "update-call" not in _tool_results(transcript.messages)
        finally:
            await agent.close()


async def test_real_permission_revocation_after_preview_records_not_committed_receipt(
    tmp_path: Path,
) -> None:
    await _reset_canary()

    async def approve(_request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.APPROVE

    agent = None
    try:
        assignments: tuple[dict[str, object], ...] = (
            {"column": "status", "value": "inactive"},
        )
        agent, _provider, source, resource = await _attached_agent(
            tmp_path,
            name="phase4-live-permission-revoked",
            table="accounts",
            assignments=assignments,
            approval_handler=approve,
        )
        await agent.set_source_write_access(source.id, True)
        readiness = await agent.postgresql_update_readiness(
            source.id,
            resource.id,
            ("status",),
        )
        store = agent._embedded._store
        start_receipt = store.start_database_write_receipt

        async def start_receipt_then_revoke(receipt: Any):
            started = await start_receipt(receipt)
            await _set_writer_status_update_grant(enabled=False)
            return started

        store.start_database_write_receipt = start_receipt_then_revoke  # type: ignore[method-assign]
        exit = await agent.run(
            "Execute the scripted exact update.",
            source_id=source.id,
        )
        transcript = await agent.transcript(exit.run_id)
        receipt = await store.load_database_write_receipt_for_call(
            agent.id,
            exit.run_id,
            "update-call",
        )
        result = _tool_results(transcript.messages)["update-call"]
        connection = await _database_connection(admin=True)
        try:
            status = await connection.fetchval(
                "SELECT status FROM write_canary.accounts WHERE account_id = 42"
            )
        finally:
            await connection.close()
    finally:
        await _set_writer_status_update_grant(enabled=True)
        if agent is not None:
            await agent.close()
        await _reset_canary()

    assert readiness.ready_for_preview is True
    assert exit.kind is LoopExitKind.COMPLETED
    assert result.is_error is True
    assert _tool_error_code(result) == "write_guardrail_rejected"
    assert receipt is not None
    assert receipt.outcome is DatabaseWriteOutcome.NOT_COMMITTED
    assert receipt.affected_rows == 0
    assert status == "active"


async def test_real_concurrent_change_after_preview_fails_before_receipt(
    tmp_path: Path,
) -> None:
    await _reset_canary()

    async def change_then_approve(_request: ApprovalRequest) -> ApprovalDecision:
        connection = await _database_connection(admin=True)
        try:
            await connection.execute(
                "UPDATE write_canary.accounts SET note = 'concurrent change' "
                "WHERE account_id = 42"
            )
        finally:
            await connection.close()
        return ApprovalDecision.APPROVE

    agent, _provider, _source, _resource, _readiness, _exit, transcript, receipt = (
        await _run_update(
            tmp_path,
            name="phase4-live-concurrent",
            assignments=({"column": "status", "value": "inactive"},),
            approval_handler=change_then_approve,
        )
    )
    try:
        result = _tool_results(transcript.messages)["update-call"]
    finally:
        await agent.close()
        await _reset_canary()

    assert result.is_error is True
    assert _tool_error_code(result) == "state_changed"
    assert receipt is None


async def test_real_row_lock_timeout_rolls_back_and_records_receipt(
    tmp_path: Path,
) -> None:
    await _reset_canary()
    lock_connection = await _database_connection(admin=True)
    transaction = lock_connection.transaction()
    await transaction.start()
    await lock_connection.fetchrow(
        "SELECT account_id FROM write_canary.accounts "
        "WHERE account_id = 42 FOR UPDATE"
    )

    async def approve(_request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.APPROVE

    try:
        agent, _provider, _source, _resource, _readiness, _exit, transcript, receipt = (
            await _run_update(
                tmp_path,
                name="phase4-live-lock-timeout",
                assignments=({"column": "status", "value": "inactive"},),
                approval_handler=approve,
            )
        )
    finally:
        await transaction.rollback()
        await lock_connection.close()
    try:
        result = _tool_results(transcript.messages)["update-call"]
    finally:
        await agent.close()
        await _reset_canary()

    assert result.is_error is True
    assert _tool_error_code(result) == "write_lock_timeout"
    assert receipt is not None
    assert receipt.outcome is DatabaseWriteOutcome.NOT_COMMITTED


async def test_real_stale_schema_is_rejected_until_catalog_refresh(
    tmp_path: Path,
) -> None:
    await _reset_canary()

    async def deny(_request: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError("stale schema must be rejected before approval")

    agent, _provider, source, resource = await _attached_agent(
        tmp_path,
        name="phase4-live-stale-schema",
        table="accounts",
        assignments=({"column": "status", "value": "inactive"},),
        approval_handler=deny,
    )
    connection = await _database_connection(admin=True)
    try:
        await agent.set_source_write_access(source.id, True)
        await connection.execute(
            "ALTER TABLE write_canary.accounts " "ADD COLUMN phase4_stale_probe text"
        )
        stale = await agent.postgresql_update_readiness(
            source.id,
            resource.id,
            ("status",),
        )
        assert stale.ready_for_preview is False
        assert "write_resource_not_writable" in stale.rejection_codes
        exit = await agent.run(
            "Attempt the scripted preview against the stale catalog.",
            source_id=source.id,
        )
        transcript = await agent.transcript(exit.run_id)
        result = _tool_results(transcript.messages)["preview-call"]
        assert result.is_error is True
        assert _tool_error_code(result) == "write_resource_not_writable"
        await connection.execute(
            "ALTER TABLE write_canary.accounts DROP COLUMN phase4_stale_probe"
        )
        await agent.refresh_source(source.id)
        current = await agent.postgresql_update_readiness(
            source.id,
            resource.id,
            ("status",),
        )
        assert current.ready_for_preview is True
    finally:
        await connection.execute(
            "ALTER TABLE write_canary.accounts "
            "DROP COLUMN IF EXISTS phase4_stale_probe"
        )
        await connection.close()
        await agent.close()
        await _reset_canary()


async def test_real_precommit_connection_loss_is_not_committed_and_not_retried(
    tmp_path: Path,
) -> None:
    await _reset_canary()
    lock_connection = await _database_connection(admin=True)
    lock_transaction = lock_connection.transaction()
    await lock_transaction.start()
    await lock_connection.fetchrow(
        "SELECT account_id FROM write_canary.accounts "
        "WHERE account_id = 42 FOR UPDATE"
    )

    async def approve(_request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.APPROVE

    run_task = asyncio.create_task(
        _run_update(
            tmp_path,
            name="phase4-live-connection-loss",
            assignments=({"column": "status", "value": "inactive"},),
            approval_handler=approve,
        )
    )
    admin = await _database_connection(admin=True)
    terminated = False
    try:
        for _attempt in range(80):
            pid = await admin.fetchval("""
                SELECT pid
                FROM pg_catalog.pg_stat_activity
                WHERE usename = 'daita_writer'
                  AND wait_event_type = 'Lock'
                  AND query LIKE '/* daita:postgresql.update_preview_row */%'
                ORDER BY query_start DESC
                LIMIT 1
                """)
            if pid is not None:
                terminated = bool(
                    await admin.fetchval(
                        "SELECT pg_catalog.pg_terminate_backend($1)", pid
                    )
                )
                break
            await asyncio.sleep(0.025)
        assert terminated, "fixture did not expose the blocked writer session"
        agent, provider, _source, _resource, _readiness, _exit, transcript, receipt = (
            await run_task
        )
    finally:
        await lock_transaction.rollback()
        await lock_connection.close()
        await admin.close()
    try:
        result = _tool_results(transcript.messages)["update-call"]
    finally:
        await agent.close()
        await _reset_canary()

    assert result.is_error is True
    assert receipt is not None
    assert receipt.outcome is DatabaseWriteOutcome.NOT_COMMITTED
    assert (
        len(
            [
                call
                for request in provider.requests
                for message in request.messages
                for call in message.tool_calls
                if call.name == POSTGRESQL_UPDATE_TOOL_NAME
            ]
        )
        == 1
    )
