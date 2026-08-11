"""Single opt-in paid-model plus disposable-PostgreSQL Phase 4 acceptance."""

from __future__ import annotations

import os
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from daita import Agent, LoopLimits, create_llm_provider
from daita._json import canonical_json
from daita.capabilities import ApprovalDecision, ApprovalRequest
from daita.domains.data.capabilities import (
    POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME,
    POSTGRESQL_UPDATE_TOOL_NAME,
)
from daita.llm.models import ModelRequest, ModelResponse, ToolResultBlock
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import ModelProvider, provider_has_complete_pricing
from daita.loop.models import LoopExitKind
from daita.security import EnvironmentSecretProvider, SecretReference
from daita.storage.sqlite import DatabaseWriteOutcome

_AUTHORIZATION = "DAITA_RUN_LIVE_POSTGRESQL_UPDATE_ACCEPTANCE"
_MODEL_ID = "DAITA_POSTGRESQL_UPDATE_ACCEPTANCE_MODEL_ID"
_MAX_COST = "DAITA_POSTGRESQL_UPDATE_ACCEPTANCE_MAX_COST_USD"
_MODEL_KEY = "OPENAI_API_KEY"
_WRITER_PASSWORD = "DAITA_FIXTURE_POSTGRES_WRITER_PASSWORD"
_ADMIN_PASSWORD = "DAITA_FIXTURE_POSTGRES_ADMIN_PASSWORD"
_PORT = "DAITA_FIXTURE_POSTGRES_PORT"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_ALLOWED_TOOLS = frozenset(
    {
        "catalog_search",
        "catalog_schema",
        "catalog_inspect",
        "catalog_traverse",
        POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME,
        POSTGRESQL_UPDATE_TOOL_NAME,
    }
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_db,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after the DB-only certification passes "
            "and one paid disposable-database run is explicitly authorized"
        ),
    ),
]


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        pytest.fail(f"{name} is required for authorized acceptance")
    return value


def _cost_limit() -> Decimal:
    raw = os.environ.get(_MAX_COST, "0.20")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    if not value.is_finite() or value <= 0:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    return value


async def _admin_connection():
    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError:
        pytest.fail("asyncpg must be available in the complete application install")
    return await asyncpg.connect(
        host="127.0.0.1",
        port=int(os.environ.get(_PORT, "55432")),
        database="daita_fixture",
        user="postgres",
        password=_required_environment(_ADMIN_PASSWORD),
        ssl=False,
        server_settings={"search_path": "pg_catalog"},
    )


async def _reset_canary() -> None:
    connection = await _admin_connection()
    try:
        await connection.execute("""
            UPDATE write_canary.accounts
            SET status = 'active', note = 'phase-4 canary'
            WHERE account_id = 42
            """)
    finally:
        await connection.close()


class _GuardedProvider:
    def __init__(self, delegate: ModelProvider) -> None:
        self._delegate = delegate
        self.requests: list[ModelRequest] = []

    @property
    def provider_id(self) -> str:
        return self._delegate.provider_id

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return self._delegate.supports_request_policy(request)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return provider_has_complete_pricing(self._delegate, request)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        projected = {definition.name for definition in request.tools}
        assert POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME in projected
        assert POSTGRESQL_UPDATE_TOOL_NAME in projected
        assert "set_source_write_access" not in projected
        self.requests.append(request)
        response = await self._delegate.generate(request)
        unexpected = {call.name for call in response.tool_calls} - _ALLOWED_TOOLS
        if unexpected:
            raise AssertionError(f"model selected unexpected tools: {unexpected}")
        return response


async def test_public_agent_model_and_real_postgresql_update_vertical_slice(
    tmp_path: Path,
) -> None:
    await _reset_canary()
    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    if not model_id.startswith("openai:"):
        pytest.fail(f"{_MODEL_ID} must name an OpenAI provider model")
    profile = reviewed_model_profile(model_id)
    if profile is None:
        pytest.fail(f"{_MODEL_ID} must name one release-reviewed model")
    provider = _GuardedProvider(
        create_llm_provider(
            model_id,
            api_key=_required_environment(_MODEL_KEY),
            max_output_tokens=min(profile.max_output_tokens, 2_048),
        )
    )
    approvals: list[ApprovalRequest] = []
    expected_source_id = ""
    expected_resource_id = ""

    async def approve_exact(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        arguments = request.arguments.to_dict()
        fingerprint = arguments.pop("preview_fingerprint", None)
        expected = {
            "source_id": expected_source_id,
            "resource_id": expected_resource_id,
            "match": [{"column": "account_id", "value": 42}],
            "assignments": [{"column": "status", "value": "inactive"}],
            "max_affected_rows": 1,
        }
        eligible = (
            request.tool_name == POSTGRESQL_UPDATE_TOOL_NAME
            and canonical_json(arguments) == canonical_json(expected)
            and isinstance(fingerprint, str)
            and len(fingerprint) == 71
        )
        return ApprovalDecision.APPROVE if eligible else ApprovalDecision.DENY

    agent = await Agent.create(
        "phase4-paid-postgresql-acceptance",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        secret_provider=EnvironmentSecretProvider(),
        approval_handler=approve_exact,
        limits=LoopLimits(
            max_steps=10,
            max_total_tokens=24_000,
            max_wall_time_seconds=120,
            max_estimated_cost_usd=_cost_limit(),
        ),
    )
    try:
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
            item
            for item in resources
            if item.native_identity == "write_canary.accounts"
        )
        expected_source_id = source.id
        expected_resource_id = resource.id
        readiness = await agent.postgresql_update_readiness(
            source.id,
            resource.id,
            ("status",),
        )
        assert readiness.rejection_codes == ("write_access_not_enabled",)
        await agent.set_source_write_access(source.id, True)
        exit = await agent.run(
            "Update exactly write_canary.accounts account_id 42 so status is the "
            "literal text inactive. First use the PostgreSQL update preview. Then "
            "call the PostgreSQL update exactly once with the returned fingerprint "
            "and max_affected_rows 1. Do not use SQL or retry. Claim success only "
            "after a committed tool result with a receipt.",
            source_id=source.id,
        )
        transcript = await agent.transcript(exit.run_id)
        calls = tuple(
            call for message in transcript.messages for call in message.tool_calls
        )
        results = {
            block.call_id: block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        }
        update_calls = tuple(
            call for call in calls if call.name == POSTGRESQL_UPDATE_TOOL_NAME
        )
        receipt = (
            await agent._embedded._store.load_database_write_receipt_for_call(
                agent.id,
                exit.run_id,
                update_calls[0].id,
            )
            if len(update_calls) == 1
            else None
        )
        await agent.set_source_write_access(source.id, False)
        disabled = next(
            item for item in await agent.list_sources() if item.id == source.id
        )
    finally:
        await agent.close()

    connection = await _admin_connection()
    try:
        status = await connection.fetchval(
            "SELECT status FROM write_canary.accounts WHERE account_id = 42"
        )
    finally:
        await connection.close()
        await _reset_canary()

    assert exit.kind is LoopExitKind.COMPLETED
    assert exit.final_text is not None and "commit" in exit.final_text.casefold()
    assert provider.requests
    assert len(approvals) == 1
    assert len(update_calls) == 1
    update_result = results[update_calls[0].id]
    assert update_result.is_error is False
    update_data = update_result.output["data"]
    assert isinstance(update_data, Mapping)
    assert update_data["outcome"] == "committed"
    assert update_data["affected_rows"] == 1
    assert receipt is not None
    assert receipt.outcome is DatabaseWriteOutcome.COMMITTED
    assert receipt.preview_fingerprint == update_data["preview_fingerprint"]
    assert status == "inactive"
    assert disabled.configuration["write_access"] is False
