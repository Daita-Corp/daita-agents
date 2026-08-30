from __future__ import annotations

import os
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from importlib import import_module
from pathlib import Path

import pytest
from _workspace_support import workspace_for

from daita import Agent, LoopLimits, create_llm_provider
from daita.capabilities import ApprovalDecision, ApprovalRequest
from daita.domains.data.capabilities import (
    POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME,
    POSTGRESQL_UPDATE_TOOL_NAME,
)
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import ModelProvider, provider_has_complete_pricing
from daita.loop.models import LoopExitKind, Transcript
from daita.security import EnvironmentSecretProvider, SecretReference

_AUTHORIZATION = "DAITA_RUN_LIVE_DATABASE_WRITE_GUARD"
_MODEL_ID = "DAITA_DATABASE_WRITE_GUARD_MODEL_ID"
_MODEL_KEY = "OPENAI_API_KEY"
_MAX_COST = "DAITA_DATABASE_WRITE_GUARD_MAX_COST_USD"
_DATABASE_PASSWORD = "DAITA_FIXTURE_POSTGRES_PASSWORD"
_DATABASE_PORT = "DAITA_FIXTURE_POSTGRES_PORT"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_FORBIDDEN_MODEL_TOOLS = frozenset(
    {
        "data_preview_sqlite_update",
        "data_update_sqlite",
        "preview_source_permissions",
        "apply_source_permissions",
    }
)
_ALLOWED_MODEL_CALLS = frozenset(
    {
        "toolbox_search",
        "toolbox_load",
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
    pytest.mark.requires_llm,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after starting "
            "tests/fixtures/postgresql/compose.yaml and explicitly authorizing "
            "one live OpenAI call"
        ),
    ),
]


class _GuardedRecordingProvider:
    """Capture real requests and constrain calls to the write-validation path."""

    def __init__(self, delegate: ModelProvider) -> None:
        self._delegate = delegate
        self.requests: list[ModelRequest] = []
        self.bootstrap_count = 0

    @property
    def provider_id(self) -> str:
        return self._delegate.provider_id

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return self._delegate.supports_request_policy(request)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return provider_has_complete_pricing(self._delegate, request)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        projected = {definition.name for definition in request.tools}
        forbidden = sorted(projected & _FORBIDDEN_MODEL_TOOLS)
        if forbidden:
            raise AssertionError(
                f"an unrelated mutation surface was projected to the model: {forbidden}"
            )
        required_update_tools = {
            POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME,
            POSTGRESQL_UPDATE_TOOL_NAME,
        }
        if not required_update_tools.issubset(projected):
            if self.bootstrap_count != 0:
                raise AssertionError("guard update-tool bootstrap did not take effect")
            if "toolbox_load" not in projected:
                raise AssertionError("toolbox_load is unavailable for guard bootstrap")
            self.bootstrap_count = 1
            self.requests.append(request)
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="guard-load-postgresql-update-tools",
                        name="toolbox_load",
                        arguments={"tool_names": sorted(required_update_tools)},
                    ),
                ),
                provider_id=self.provider_id,
                usage=ModelUsage(
                    cost_estimate=CostEstimate.complete(Decimal("0")),
                ),
            )
        calls = {
            call.id: call.name
            for message in request.messages
            for call in message.tool_calls
        }
        if any(
            isinstance(block, ToolResultBlock)
            and not block.is_error
            and calls.get(block.call_id) == "data_query_postgresql"
            for message in request.messages
            for block in message.content
        ):
            raise AssertionError(
                "live write guard refuses to send PostgreSQL row results to OpenAI"
            )
        self.requests.append(request)
        response = await self._delegate.generate(request)
        unexpected = sorted(
            {call.name for call in response.tool_calls} - _ALLOWED_MODEL_CALLS
        )
        if unexpected:
            raise AssertionError(
                f"live write guard selected an unexpected tool: {unexpected}"
            )
        return response


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        pytest.fail(f"{name} must be set for the authorized live test")
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


async def _customer_snapshot(
    *, password: str, port: int
) -> tuple[tuple[object, ...], ...]:
    try:
        asyncpg = import_module("asyncpg")
    except ImportError:
        raise ImportError(
            "PostgreSQL support is unavailable; repair with: "
            "pipx reinstall daita-agents"
        ) from None

    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=port,
        database="daita_fixture",
        user="daita_reader",
        password=password,
        ssl=False,
    )
    try:
        rows = await connection.fetch(
            """SELECT customer_id, customer_name, email, region_code, segment,
                      signed_up_at, is_active
               FROM analytics.customers
               ORDER BY customer_id"""
        )
        return tuple(tuple(row) for row in rows)
    finally:
        await connection.close()


def _tool_calls(transcript: Transcript) -> tuple[ToolCall, ...]:
    return tuple(call for message in transcript.messages for call in message.tool_calls)


def _tool_results(
    messages: Sequence[CanonicalMessage],
) -> dict[str, ToolResultBlock]:
    return {
        block.call_id: block
        for message in messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }


async def test_live_openai_cannot_write_with_read_only_database_role(
    tmp_path: Path,
) -> None:
    api_key = _required_environment(_MODEL_KEY)
    database_password = _required_environment(_DATABASE_PASSWORD)
    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    if not model_id.startswith("openai:"):
        pytest.fail(f"{_MODEL_ID} must name an OpenAI provider model")
    profile = reviewed_model_profile(model_id)
    if profile is None:
        pytest.fail(f"{_MODEL_ID} must name one release-reviewed tool-capable model")

    delegate = create_llm_provider(
        model_id,
        api_key=api_key,
        max_output_tokens=min(profile.max_output_tokens, 2_048),
    )
    provider = _GuardedRecordingProvider(delegate)
    port = int(os.environ.get(_DATABASE_PORT, "55432"))
    before = await _customer_snapshot(password=database_password, port=port)
    if not before:
        pytest.fail("fixture customer snapshot is missing its write target")

    approvals: list[ApprovalRequest] = []

    async def reject_unexpected_approval(
        request: ApprovalRequest,
    ) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.DENY

    agent = await Agent.create(
        "live-database-write-guard",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        secret_provider=EnvironmentSecretProvider(),
        approval_handler=reject_unexpected_approval,
        limits=LoopLimits(
            max_steps=12,
            max_total_tokens=20_000,
            max_wall_time_seconds=120,
            max_estimated_cost_usd=_cost_limit(),
        ),
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach_postgresql(
            host="127.0.0.1",
            port=port,
            database="daita_fixture",
            username="daita_reader",
            credential=SecretReference.environment(_DATABASE_PASSWORD),
            schemas=("analytics",),
            ssl_mode="disable",
            name="Fixture PostgreSQL",
        )
        resource = next(
            item
            for item in await agent.list_catalog_resources(source_id=source.id)
            if item.native_identity == "analytics.customers"
        )
        permission_preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="all",
            read_resource_ids=(),
            postgresql_update_scopes={resource.id: ["is_active"]},
        )
        await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=permission_preview.confirmation_fingerprint,
        )

        exit = await agent.run(
            "Set analytics.customers customer_id = 1 is_active to the confirmed "
            "exact boolean value false. Load the PostgreSQL update-preview and "
            "PostgreSQL update tools, then use only those update tools; do not call "
            "a query tool, merely show SQL, or claim success without a committed "
            "update result. If the database role cannot pass the write guardrails, "
            "explain that the change was not committed.",
            source_id=source.id,
        )
        transcript = await agent.transcript(exit.run_id)
        update_receipts = []
        for call in _tool_calls(transcript):
            if call.name == POSTGRESQL_UPDATE_TOOL_NAME:
                update_receipts.append(
                    await agent._embedded._store.load_database_write_receipt_for_call(
                        agent.id,
                        exit.run_id,
                        call.id,
                    )
                )
    finally:
        await agent.close()

    after = await _customer_snapshot(password=database_password, port=port)
    assert exit.kind is LoopExitKind.COMPLETED
    assert provider.requests
    assert provider.bootstrap_count == 1
    assert all(
        _FORBIDDEN_MODEL_TOOLS.isdisjoint(
            definition.name for definition in request.tools
        )
        for request in provider.requests
    )
    projected_names = tuple(
        {definition.name for definition in request.tools}
        for request in provider.requests
    )
    assert any(
        POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME in names for names in projected_names
    )
    assert any(POSTGRESQL_UPDATE_TOOL_NAME in names for names in projected_names)
    results = _tool_results(transcript.messages)
    write_calls = tuple(
        call
        for call in _tool_calls(transcript)
        if call.name
        in {POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME, POSTGRESQL_UPDATE_TOOL_NAME}
    )
    assert write_calls
    assert all(results[call.id].is_error is True for call in write_calls)
    assert approvals == []
    assert all(receipt is None for receipt in update_receipts)
    assert after == before
