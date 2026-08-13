"""Opt-in live-model confidence gate for one PostgreSQL update.

Run from the repository root after loading ``OPENAI_API_KEY`` into the
environment::

    DAITA_RUN_LIVE_POSTGRESQL_UPDATE_MODEL=1 \
      .venv/bin/python -m pytest \
      tests/live/test_postgresql_update_live.py -v -s

The model call is live. PostgreSQL behavior is a deterministic in-process fake,
so this test cannot mutate an external database and does not require
``requires_db``. It certifies that a release-reviewed model can complete the
public preview -> approval -> update -> committed-result transcript protocol.
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
from daita.adapters.models import DiscoveryRequest, SourceRegistration
from daita.capabilities import ApprovalDecision, ApprovalRequest
from daita.catalog.models import ResourceKind, TabularColumn, catalog_resource_id
from daita.domains.data.capabilities import (
    POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME,
    POSTGRESQL_UPDATE_TOOL_NAME,
)
from daita.domains.data.controller import (
    POSTGRESQL_UPDATE_CAPABILITY_ID,
    POSTGRESQL_UPDATE_EVIDENCE_KIND,
    POSTGRESQL_UPDATE_PREVIEW_EVIDENCE_KIND,
)
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
from daita.storage.sqlite import DatabaseWriteOutcome

_AUTHORIZATION = "DAITA_RUN_LIVE_POSTGRESQL_UPDATE_MODEL"
_MODEL_ID = "DAITA_POSTGRESQL_UPDATE_MODEL_ID"
_MODEL_KEY = "OPENAI_API_KEY"
_MAX_COST = "DAITA_POSTGRESQL_UPDATE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
_AGENT_ID = "agent-live-postgresql-update"
_NATIVE_IDENTITY = "postgresql:live-model-update"
_ALLOWED_MODEL_CALLS = frozenset(
    {
        "catalog_search",
        "catalog_schema",
        "catalog_inspect",
        "catalog_traverse",
        POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME,
        POSTGRESQL_UPDATE_TOOL_NAME,
    }
)
_FORBIDDEN_MODEL_TOOLS = frozenset(
    {
        "data_preview_sqlite_update",
        "data_update_sqlite",
        "preview_source_permissions",
        "apply_source_permissions",
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
            "one live OpenAI PostgreSQL update-protocol confidence run"
        ),
    ),
]


class _ProjectionGuardProvider:
    """Record paid requests and admit only the intended model tool path."""

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
        forbidden = sorted(projected & _FORBIDDEN_MODEL_TOOLS)
        if forbidden:
            raise AssertionError(
                f"an unrelated mutation surface was projected: {forbidden}"
            )
        self.requests.append(request)
        response = await self._delegate.generate(request)
        unexpected = sorted(
            {call.name for call in response.tool_calls} - _ALLOWED_MODEL_CALLS
        )
        if unexpected:
            raise AssertionError(
                f"live update model selected an unexpected tool: {unexpected}"
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


class _Cursor:
    def __init__(self, log: list[tuple[object, ...]]) -> None:
        self._log = log

    async def fetch(self, count: int) -> tuple[Mapping[str, object], ...]:
        self._log.append(("cursor.fetch", count))
        return ({"account_id": 42, "is_active": False},)[:count]


class _UpdateConnection:
    def __init__(self) -> None:
        self.log: list[tuple[object, ...]] = []
        self.update_count = 0

    def transaction(self, **kwargs: object) -> _Transaction:
        self.log.append(("transaction", kwargs))
        return _Transaction(self.log)

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

    async def cursor(self, sql: str, *parameters: object) -> _Cursor:
        self.update_count += 1
        self.log.append(("cursor", sql, parameters))
        return _Cursor(self.log)

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
        display_name="Synthetic Update PostgreSQL",
        configuration={
            "database": "synthetic_update",
            "host": "synthetic.invalid",
            "port": 5432,
            "schemas": ("public",),
            "ssl_mode": "require",
            "username": "synthetic_update_role",
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
        return f"{prefix}-live-update-{counters[prefix]}"

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


def _patch_update_io(
    monkeypatch: pytest.MonkeyPatch,
    connection: _UpdateConnection,
    structure: postgresql_module.PostgreSQLStructure,
) -> None:
    async def connect(*args: object, **kwargs: object) -> object:
        del args, kwargs
        connection.log.append(("connect",))
        return connection

    async def load_structure(*args: object, **kwargs: object) -> object:
        del args, kwargs
        connection.log.append(("structure",))
        return structure

    monkeypatch.setattr(write_module, "_connect", connect)
    monkeypatch.setattr(write_module, "_load_structure", load_structure)


async def test_live_model_previews_approves_and_commits_exact_postgresql_update(
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
    expected_intent = {
        "source_id": registration.id,
        "resource_id": resource_id,
        "match": [{"column": "account_id", "value": 42}],
        "assignments": [{"column": "is_active", "value": False}],
    }
    connection = _UpdateConnection()
    _patch_update_io(monkeypatch, connection, structure)
    approvals: list[ApprovalRequest] = []

    async def approve_exact_update(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        arguments = dict(request.arguments)
        fingerprint = arguments.pop("preview_fingerprint", None)
        expected = {**expected_intent, "max_affected_rows": 1}
        eligible = (
            request.tool_name == POSTGRESQL_UPDATE_TOOL_NAME
            and request.capability_id == POSTGRESQL_UPDATE_CAPABILITY_ID
            and canonical_json(arguments) == canonical_json(expected)
            and isinstance(fingerprint, str)
            and fingerprint.startswith("sha256:")
            and len(fingerprint) == 71
        )
        return ApprovalDecision.APPROVE if eligible else ApprovalDecision.DENY

    agent = await Agent.create(
        "live-postgresql-update",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        approval_handler=approve_exact_update,
        clock=lambda: _NOW,
        id_factory=_id_factory(),
        limits=LoopLimits(
            max_steps=8,
            max_total_tokens=24_000,
            max_wall_time_seconds=120,
            max_estimated_cost_usd=_cost_limit(),
        ),
    )
    snapshot = postgresql_module._catalog_snapshot(
        registration,
        DiscoveryRequest(
            agent_id=agent.id,
            source_id=registration.id,
            sync_id="catalog-sync-live-update",
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
        exit = await agent.run(
            "Perform exactly one PostgreSQL update to public.accounts. Match the "
            "single primary-key row account_id 42 and assign is_active to the "
            "confirmed exact boolean value false. First call the PostgreSQL "
            "update-preview tool exactly once. Then copy its exact current preview "
            "fingerprint into the PostgreSQL update tool with max_affected_rows 1. "
            "Do not call a SQL query tool, do not invent SQL, and do not retry the "
            "update. Only say the change committed after the update tool returns a "
            "committed outcome and receipt.",
            source_id=registration.id,
        )
        transcript = await agent.transcript(exit.run_id)
        calls = _tool_calls(transcript)
        results = _tool_results(transcript.messages)
        preview_calls = tuple(
            call for call in calls if call.name == POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME
        )
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
    finally:
        await agent.close()

    assert exit.kind is LoopExitKind.COMPLETED, exit.reason
    assert exit.final_text is not None
    assert "committed" in exit.final_text.casefold()
    assert provider.requests
    assert len(preview_calls) == 1
    assert len(update_calls) == 1
    preview_call = preview_calls[0]
    update_call = update_calls[0]
    assert calls.index(preview_call) < calls.index(update_call)
    assert canonical_json(preview_call.arguments) == canonical_json(expected_intent)
    preview_result = results[preview_call.id]
    assert preview_result.is_error is False
    assert preview_result.output["kind"] == POSTGRESQL_UPDATE_PREVIEW_EVIDENCE_KIND
    preview_data = preview_result.output["data"]
    assert isinstance(preview_data, Mapping)
    expected_update = {
        **expected_intent,
        "preview_fingerprint": preview_data["preview_fingerprint"],
        "max_affected_rows": 1,
    }
    assert canonical_json(update_call.arguments) == canonical_json(expected_update)
    assert len(approvals) == 1
    assert approvals[0].run_id == exit.run_id
    assert approvals[0].call_id == update_call.id
    assert canonical_json(approvals[0].arguments) == canonical_json(expected_update)
    update_result = results[update_call.id]
    assert update_result.is_error is False
    assert update_result.output["kind"] == POSTGRESQL_UPDATE_EVIDENCE_KIND
    update_data = update_result.output["data"]
    assert isinstance(update_data, Mapping)
    assert update_data["outcome"] == "committed"
    assert update_data["affected_rows"] == 1
    assert canonical_json(update_data["returned"]) == canonical_json(
        [
            {"column": "account_id", "value": 42},
            {"column": "is_active", "value": False},
        ]
    )
    assert receipt is not None
    assert receipt.outcome is DatabaseWriteOutcome.COMMITTED
    assert receipt.call_id == update_call.id
    assert receipt.preview_fingerprint == preview_data["preview_fingerprint"]

    transaction_calls = tuple(
        entry for entry in connection.log if entry[0] == "transaction"
    )
    assert (
        transaction_calls.count(
            ("transaction", {"isolation": "repeatable_read", "readonly": True})
        )
        == 3
    )
    assert (
        transaction_calls.count(("transaction", {"isolation": "repeatable_read"})) == 1
    )
    locked_selects = tuple(
        entry
        for entry in connection.log
        if entry[0] == "fetch" and "FOR UPDATE" in str(entry[1])
    )
    assert len(locked_selects) == 1
    assert locked_selects[0][2] == (42,)
    update_entries = tuple(entry for entry in connection.log if entry[0] == "cursor")
    assert len(update_entries) == 1
    assert update_entries[0][1] == (
        'UPDATE ONLY "public"."accounts" SET "is_active" = $1 '
        'WHERE "account_id" = $2 RETURNING "account_id", "is_active"'
    )
    assert update_entries[0][2] == (False, 42)
    assert connection.update_count == 1
    assert ("cursor.fetch", 2) in connection.log
    assert connection.log.count(("transaction.commit",)) == 4
    assert ("transaction.rollback",) not in connection.log
    serialized = canonical_json(update_result.output)
    assert "synthetic.invalid" not in serialized
    assert "synthetic_update_role" not in serialized
