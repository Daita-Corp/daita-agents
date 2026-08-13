"""Opt-in live-model confidence gate for PostgreSQL update preview.

Run from the repository root after loading ``OPENAI_API_KEY`` into the
environment::

    DAITA_RUN_LIVE_POSTGRESQL_UPDATE_PREVIEW=1 \
      .venv/bin/python -m pytest \
      tests/live/test_postgresql_update_preview_live.py -v -s

The model call is live. PostgreSQL is a deterministic in-process fake so this
test cannot mutate an external database and does not require ``requires_db``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path

import pytest

from daita import Agent, LoopLimits, create_llm_provider
from daita._json import canonical_json
from daita.adapters import (
    postgresql as postgresql_module,
    postgresql_write as write_module,
)
from daita.adapters.models import (
    DiscoveryRequest,
    SourceRegistration,
)
from daita.catalog.models import ResourceKind, TabularColumn, catalog_resource_id
from daita.domains.data.capabilities import (
    POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME,
    POSTGRESQL_UPDATE_TOOL_NAME,
)
from daita.domains.data.controller import POSTGRESQL_UPDATE_PREVIEW_EVIDENCE_KIND
from daita.llm.models import (
    CanonicalMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import ModelProvider, provider_has_complete_pricing
from daita.loop.models import LoopExitKind, Transcript

_AUTHORIZATION = "DAITA_RUN_LIVE_POSTGRESQL_UPDATE_PREVIEW"
_MODEL_ID = "DAITA_POSTGRESQL_UPDATE_PREVIEW_MODEL_ID"
_MODEL_KEY = "OPENAI_API_KEY"
_MAX_COST = "DAITA_POSTGRESQL_UPDATE_PREVIEW_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
_AGENT_ID = "agent-live-postgresql-update-preview"
_NATIVE_IDENTITY = "postgresql:live-model-preview"
_FORBIDDEN_MODEL_TOOLS = frozenset(
    {
        "data_preview_sqlite_update",
        "data_update_sqlite",
        "preview_source_permissions",
        "apply_source_permissions",
    }
)
_ALLOWED_CATALOG_TOOLS = frozenset(
    {
        "catalog_search",
        "catalog_schema",
        "catalog_inspect",
        "catalog_traverse",
    }
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing "
            "one live OpenAI preview confidence run"
        ),
    ),
]


class _ProjectionGuardProvider:
    """Record paid requests and reject an unexpected model-facing surface."""

    def __init__(self, delegate: ModelProvider) -> None:
        self._delegate = delegate
        self.requests: list[ModelRequest] = []
        self.preview_expected = False

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
                f"database-mutation surface was projected to the model: {forbidden}"
            )
        assert (
            POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME in projected
        ) is self.preview_expected
        assert (POSTGRESQL_UPDATE_TOOL_NAME in projected) is self.preview_expected
        self.requests.append(request)
        response = await self._delegate.generate(request)
        if response.tool_calls:
            expected_calls = set(_ALLOWED_CATALOG_TOOLS)
            if self.preview_expected:
                expected_calls.add(POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME)
            unexpected = sorted(
                {call.name for call in response.tool_calls} - expected_calls
            )
            if unexpected:
                raise AssertionError(
                    f"live model selected an unexpected tool: {unexpected}"
                )
        return response


class _Transaction:
    def __init__(self, log: list[tuple[object, ...]]) -> None:
        self._log = log

    async def start(self) -> None:
        self._log.append(("transaction.start",))

    async def commit(self) -> None:
        self._log.append(("transaction.commit",))

    async def rollback(self) -> None:
        self._log.append(("transaction.rollback",))


class _ReadOnlyPreviewConnection:
    def __init__(self) -> None:
        self.log: list[tuple[object, ...]] = []
        self._transaction = _Transaction(self.log)

    def transaction(self, **kwargs: object) -> _Transaction:
        self.log.append(("transaction", kwargs))
        return self._transaction

    async def execute(self, sql: str, *parameters: object) -> str:
        self.log.append(("execute", sql, parameters))
        return "SELECT 1"

    async def fetchrow(
        self,
        sql: str,
        *parameters: object,
        **kwargs: object,
    ) -> Mapping[str, object]:
        self.log.append(("fetchrow", sql, parameters, kwargs))
        assert "update_preview_guardrails" in sql
        return {
            "relation_oid": "16384",
            "relation_kind": "r",
            "is_partition": False,
            "row_level_security": False,
            "force_row_level_security": False,
            "has_inheritance": False,
            "has_user_triggers": False,
            "has_rewrite_rules": False,
            "role_superuser": False,
            "role_bypass_rls": False,
            "role_create_database": False,
            "role_create_role": False,
            "role_replication": False,
            "can_connect": True,
            "can_use_schema": True,
            "can_select_table": True,
            "can_update_columns": True,
        }

    async def fetch(
        self,
        sql: str,
        *parameters: object,
        **kwargs: object,
    ) -> tuple[Mapping[str, object], ...]:
        self.log.append(("fetch", sql, parameters, kwargs))
        if sql.startswith("EXPLAIN"):
            return ({"QUERY PLAN": ()},)
        assert "update_preview_row" in sql
        return (
            {
                "__daita_primary_key_0": 42,
                "__daita_before_0": True,
                "__daita_within_preview_limit": True,
                "__daita_tableoid": "16384",
                "__daita_ctid": "(0,1)",
                "__daita_xmin": "751",
            },
        )

    async def close(self) -> None:
        self.log.append(("close",))

    def terminate(self) -> None:
        self.log.append(("terminate",))


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


def _registration() -> SourceRegistration:
    return SourceRegistration.build(
        agent_id=_AGENT_ID,
        adapter_id="postgresql",
        native_identity=_NATIVE_IDENTITY,
        display_name="Synthetic Preview PostgreSQL",
        configuration={
            "database": "synthetic_preview",
            "host": "synthetic.invalid",
            "port": 5432,
            "schemas": ("public",),
            "ssl_mode": "require",
            "username": "synthetic_preview_role",
        },
        attached_at=_NOW,
    )


def _structure() -> postgresql_module.PostgreSQLStructure:
    table = postgresql_module._TableStructure(
        schema="public",
        name="accounts",
        kind=ResourceKind.TABLE,
        columns=(
            TabularColumn(
                name="account_id",
                native_type="bigint",
                native_type_namespace="pg_catalog",
                native_type_name="int8",
                ordinal=0,
                nullable=False,
                primary_key_ordinal=1,
                updatable=True,
            ),
            TabularColumn(
                name="is_active",
                native_type="boolean",
                native_type_namespace="pg_catalog",
                native_type_name="bool",
                ordinal=1,
                nullable=False,
                updatable=True,
            ),
        ),
        indexes=(),
    )
    encoded = canonical_json(
        {
            "relationships": (),
            "tables": (table.payload(),),
        }
    ).encode("utf-8")
    return postgresql_module.PostgreSQLStructure(
        tables=(table,),
        relationships=(),
        source_revision="catalog:sha256:" + sha256(encoded).hexdigest(),
    )


def _id_factory():
    counters: defaultdict[str, int] = defaultdict(int)

    def next_id(prefix: str) -> str:
        if prefix == "agent":
            return _AGENT_ID
        counters[prefix] += 1
        return f"{prefix}-live-preview-{counters[prefix]}"

    return next_id


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


def _patch_preview_io(
    monkeypatch: pytest.MonkeyPatch,
    connection: _ReadOnlyPreviewConnection,
    structure: postgresql_module.PostgreSQLStructure,
) -> None:
    async def connect(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return connection

    async def load_structure(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return structure

    monkeypatch.setattr(write_module, "_connect", connect)
    monkeypatch.setattr(write_module, "_load_structure", load_structure)


async def test_live_model_uses_only_enabled_read_only_postgresql_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = _required_environment(_MODEL_KEY)
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
    provider = _ProjectionGuardProvider(delegate)
    registration = _registration()
    structure = _structure()
    resource_id = catalog_resource_id(
        registration.id,
        ResourceKind.TABLE,
        "public.accounts",
    )
    connection = _ReadOnlyPreviewConnection()
    _patch_preview_io(monkeypatch, connection, structure)

    agent = await Agent.create(
        "live-postgresql-update-preview",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        clock=lambda: _NOW,
        id_factory=_id_factory(),
        limits=LoopLimits(
            max_steps=6,
            max_total_tokens=20_000,
            max_wall_time_seconds=120,
            max_estimated_cost_usd=_cost_limit(),
        ),
    )
    snapshot = postgresql_module._catalog_snapshot(
        registration,
        DiscoveryRequest(
            agent_id=agent.id,
            source_id=registration.id,
            sync_id="catalog-sync-live-preview",
            requested_at=_NOW,
        ),
        structure,
        _NOW,
    )
    await agent._embedded._store.register_source(registration)
    await agent._embedded._store.commit_snapshot(snapshot)
    resource = next(
        item for item in snapshot.resources if item.native_identity == "public.accounts"
    )
    try:
        disabled_exit = await agent.run(
            "Can you preview changing public.accounts account_id 42 is_active to "
            "false? The exact typed boolean value is confirmed. Do not query the "
            "row and do not provide SQL. If the exact PostgreSQL update-preview "
            "tool is unavailable, explain that preview is not enabled."
        )
        disabled_request_count = len(provider.requests)
        disabled_transcript = await agent.transcript(disabled_exit.run_id)

        permission_preview = await agent.preview_source_permissions(
            source_id=registration.id,
            read_mode="all",
            read_resource_ids=(),
            postgresql_update_scopes={resource.id: ["is_active"]},
        )
        await agent.apply_source_permissions(
            source_id=registration.id,
            confirmation_fingerprint=permission_preview.confirmation_fingerprint,
        )
        provider.preview_expected = True
        preview_exit = await agent.run(
            "Use the PostgreSQL update-preview tool now. Preview exactly one "
            "structured change to public.accounts: match account_id 42 and assign "
            "is_active to the confirmed exact boolean value false. This message is "
            "the user's confirmation of that literal. Use the exact source and "
            "resource identities from the catalog. Do not query with SQL and do not "
            "execute a database mutation. After the preview, state exactly 'No "
            "database mutation was performed' and summarize the before and after "
            "values."
        )
        preview_transcript = await agent.transcript(preview_exit.run_id)
        preview_receipts = {
            call.id: await agent._embedded._store.load_database_write_receipt_for_call(
                agent.id,
                preview_exit.run_id,
                call.id,
            )
            for call in _tool_calls(preview_transcript)
            if call.name == POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME
        }
    finally:
        await agent.close()

    assert disabled_exit.kind is LoopExitKind.COMPLETED
    assert all(
        call.name != POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME
        for call in _tool_calls(disabled_transcript)
    )
    assert all(
        POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME
        not in {definition.name for definition in request.tools}
        for request in provider.requests[:disabled_request_count]
    )

    assert preview_exit.kind is LoopExitKind.COMPLETED
    assert preview_exit.final_text is not None
    assert "no database mutation was performed" in preview_exit.final_text.casefold()
    enabled_requests = provider.requests[disabled_request_count:]
    assert enabled_requests
    assert all(
        POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME
        in {definition.name for definition in request.tools}
        for request in enabled_requests
    )
    assert all(
        POSTGRESQL_UPDATE_TOOL_NAME in {definition.name for definition in request.tools}
        for request in enabled_requests
    )
    assert all(
        call.name != POSTGRESQL_UPDATE_TOOL_NAME
        for call in _tool_calls(preview_transcript)
    )
    assert all(
        _FORBIDDEN_MODEL_TOOLS.isdisjoint(
            definition.name for definition in request.tools
        )
        for request in provider.requests
    )

    results = _tool_results(preview_transcript.messages)
    preview_calls = tuple(
        call
        for call in _tool_calls(preview_transcript)
        if call.name == POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME
    )
    successful_calls = tuple(
        call
        for call in preview_calls
        if call.id in results and not results[call.id].is_error
    )
    assert len(successful_calls) == 1
    successful_call = successful_calls[0]
    assert canonical_json(successful_call.arguments) == canonical_json(
        {
            "source_id": registration.id,
            "resource_id": resource_id,
            "match": [{"column": "account_id", "value": 42}],
            "assignments": [{"column": "is_active", "value": False}],
        }
    )
    output = results[successful_call.id].output
    assert output["kind"] == POSTGRESQL_UPDATE_PREVIEW_EVIDENCE_KIND
    data = output["data"]
    assert isinstance(data, Mapping)
    assert data["would_affect"] == 1
    assert canonical_json(data["before"]) == canonical_json(
        [{"column": "is_active", "value": True}]
    )
    assert canonical_json(data["after"]) == canonical_json(
        [{"column": "is_active", "value": False}]
    )
    assert preview_receipts
    assert all(receipt is None for receipt in preview_receipts.values())

    assert connection.log[0] == (
        "transaction",
        {"isolation": "repeatable_read", "readonly": True},
    )
    sql_calls = tuple(
        entry
        for entry in connection.log
        if entry[0] in {"execute", "fetch", "fetchrow"}
    )
    rendered_sql = tuple(str(entry[1]) for entry in sql_calls)
    assert not any(
        sql.lstrip()
        .upper()
        .startswith(("INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "ALTER", "DROP"))
        for sql in rendered_sql
    )
    explain = next(sql for sql in rendered_sql if sql.startswith("EXPLAIN"))
    assert "ANALYZE" not in explain
    assert any("update_preview_row" in sql and "LIMIT 2" in sql for sql in rendered_sql)
    assert ("transaction.commit",) in connection.log
    assert connection.log[-1] == ("close",)
