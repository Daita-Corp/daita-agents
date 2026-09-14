"""Authorized live-model acceptance coverage for material clarification.

Only the model provider is live. SQLite and MCP use deterministic local fixtures so
these tests measure the agent's target-selection behavior without depending on live
data services or dispatching an external action.
"""

from __future__ import annotations

import os
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from daita import (
    Agent,
    LoopLimits,
    MCPAuthentication,
    MCPToolSelection,
    SQLiteSource,
    create_llm_provider,
)
from daita.adapters.mcp import StreamableHTTPMCPClientFactory
from daita.llm.models import ModelProfile, ToolCall, ToolResultBlock
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import ManagedModelProvider
from daita.loop.models import (
    LoopExit,
    LoopExitKind,
    Transcript,
    validate_completed_transcript,
)
from daita.security import SecretReference
from tests.support.mcp import (
    MappingSecretProvider,
    conformance_identities,
    mock_transport,
)
from tests.support.workspace import workspace_for

_AUTHORIZATION = "DAITA_RUN_LIVE_CLARIFICATION"
_MODEL_ID = "DAITA_CLARIFICATION_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_CLARIFICATION_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_CLARIFICATION_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_KEY_ENVIRONMENT = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "grok": "XAI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_EAST_TOKEN = "CLARIFICATION_EAST_8D3C21"
_WEST_TOKEN = "CLARIFICATION_WEST_5A7E94"
_UNRELATED_TOKEN = "ORDERS_UNRELATED_OK"

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing five live "
            f"Agent.run interactions, each capped by {_MAX_COST}"
        ),
    ),
]


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        pytest.fail(f"{name} must be set for the authorized live test")
    return value


def _cost_limit() -> Decimal:
    raw = os.environ.get(_MAX_COST, "0.50")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    if not value.is_finite() or value <= 0:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    return value


def _live_provider() -> tuple[ModelProfile, ManagedModelProvider]:
    if os.environ.get(_AUTHORIZATION) != "1":
        pytest.fail(f"{_AUTHORIZATION}=1 is required before constructing a provider")
    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    provider_name = model_id.partition(":")[0]
    key_environment = _API_KEY_ENVIRONMENT.get(provider_name)
    if key_environment is None:
        pytest.fail(f"{_MODEL_ID} must name one API-backed reviewed model")
    profile = reviewed_model_profile(model_id)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{_MODEL_ID} must name one release-reviewed tool-capable model")
    if not profile.supports_streaming:
        pytest.fail(f"{_MODEL_ID} must name a model with reviewed streaming support")
    api_key = os.environ.get(_MODEL_KEY) or _required_environment(key_environment)
    provider = create_llm_provider(
        model_id,
        api_key=api_key,
        max_output_tokens=min(profile.max_output_tokens, 1_536),
    )
    return profile, provider


def _limits() -> LoopLimits:
    return LoopLimits(max_estimated_cost_usd=_cost_limit())


def _orders_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(f"""
            CREATE TABLE orders_east (
                order_id INTEGER PRIMARY KEY,
                verification_token TEXT NOT NULL,
                amount INTEGER NOT NULL
            );
            INSERT INTO orders_east VALUES (1, '{_EAST_TOKEN}', 41);

            CREATE TABLE orders_west (
                order_id INTEGER PRIMARY KEY,
                verification_token TEXT NOT NULL,
                amount INTEGER NOT NULL
            );
            INSERT INTO orders_west VALUES (1, '{_WEST_TOKEN}', 73);
            """)


def _tool_calls(transcript: Transcript) -> tuple[ToolCall, ...]:
    return tuple(call for call, _result in transcript.tool_pairs)


def _successful_calls(
    transcript: Transcript,
    tool_name: str,
) -> tuple[ToolCall, ...]:
    return tuple(
        call
        for call, result in transcript.tool_pairs
        if call.name == tool_name
        and isinstance(result, ToolResultBlock)
        and not result.is_error
    )


def _assert_completed(result: LoopExit, transcript: Transcript) -> str:
    assert result.kind is LoopExitKind.COMPLETED, result
    assert result.final_text is not None
    validate_completed_transcript(transcript, result)
    return result.final_text


def _assert_actionable_clarification(
    text: str,
    *,
    candidate_terms: tuple[str, str],
) -> None:
    lowered = text.casefold()
    assert all(term.casefold() in lowered for term in candidate_terms), text
    assert "?" in text or any(
        cue in lowered for cue in ("which", "choose", "select", "specify", "clarify")
    ), text


async def test_live_catalog_clarification_then_selection_and_control_cases(
    tmp_path: Path,
) -> None:
    profile, provider = _live_provider()
    state_root = tmp_path / "catalog-clarification-state"
    database = tmp_path / "orders.sqlite"
    _orders_database(database)
    agent = await Agent.create(
        "live-catalog-clarification",
        root=state_root,
        workspace=workspace_for(state_root),
        model=provider,
        model_profile=profile,
        limits=_limits(),
    )
    try:
        await agent.attach(SQLiteSource(database, name="Regional orders"))

        ambiguous = await agent.run(
            "Read the only order and return its exact verification_token and "
            "amount. The catalog contains two plausible orders resources, but I "
            "have not selected one. Do not guess or choose a resource for me."
        )
        ambiguous_transcript = await agent.transcript(ambiguous.run_id)
        ambiguous_text = _assert_completed(ambiguous, ambiguous_transcript)
        _assert_actionable_clarification(
            ambiguous_text,
            candidate_terms=("east", "west"),
        )
        assert not any(
            call.name.startswith("data_") for call in _tool_calls(ambiguous_transcript)
        )

        selected = await agent.run(
            "Use orders_east. Now read its only row and return the exact "
            "verification_token and amount.",
            conversation_id=ambiguous.conversation_id,
        )
        selected_transcript = await agent.transcript(selected.run_id)
        selected_text = _assert_completed(selected, selected_transcript)
        assert _successful_calls(selected_transcript, "data_query")
        assert _EAST_TOKEN in selected_text
        assert "41" in selected_text
        assert _WEST_TOKEN not in selected_text

        comparison = await agent.run(
            "Compare the explicit set orders_east and orders_west. Read both and "
            "return each resource name, exact verification_token, and amount; no "
            "other comparison is required."
        )
        comparison_transcript = await agent.transcript(comparison.run_id)
        comparison_text = _assert_completed(comparison, comparison_transcript)
        assert _successful_calls(comparison_transcript, "data_query")
        assert _EAST_TOKEN in comparison_text
        assert _WEST_TOKEN in comparison_text
        assert "41" in comparison_text
        assert "73" in comparison_text

        unrelated = await agent.run(
            "The word orders is writing content only, not a request to inspect "
            f"data. Without using any tool, reply exactly: {_UNRELATED_TOKEN}"
        )
        unrelated_transcript = await agent.transcript(unrelated.run_id)
        unrelated_text = _assert_completed(unrelated, unrelated_transcript)
        assert unrelated_text.strip() == _UNRELATED_TOKEN
        assert not _tool_calls(unrelated_transcript)
    finally:
        try:
            await agent.close()
        finally:
            await provider.close()


async def test_live_mcp_ambiguity_clarifies_without_remote_dispatch(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "mcp-clarification-state"
    workspace = workspace_for(state_root)
    alpha, beta = conformance_identities()
    secrets = MappingSecretProvider({"env:BETA_TOKEN": "fixture-beta-secret"})
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(alpha, beta))

    seed = await Agent.create(
        "live-mcp-clarification",
        root=state_root,
        workspace=workspace,
        secret_provider=secrets,
        mcp_client_factory=factory,
    )
    try:
        alpha_status = await seed.attach_mcp_server(
            endpoint=alpha.endpoint,
            selections=(
                MCPToolSelection(
                    remote_name="lookup",
                    local_alias="alpha_lookup",
                    description="Read the admitted alpha fixture value.",
                ),
            ),
        )
        beta_status = await seed.attach_mcp_server(
            endpoint=beta.endpoint,
            authentication=MCPAuthentication.bearer(
                SecretReference.environment("BETA_TOKEN")
            ),
            selections=(
                MCPToolSelection(
                    remote_name="lookup",
                    local_alias="beta_lookup",
                    description="Read the admitted beta fixture value.",
                ),
            ),
        )
        assert alpha_status.reopen_required
        assert beta_status.reopen_required
    finally:
        await seed.close()

    profile, provider = _live_provider()
    agent = await Agent.open(
        "live-mcp-clarification",
        root=state_root,
        workspace=workspace,
        model=provider,
        model_profile=profile,
        limits=_limits(),
        secret_provider=secrets,
        mcp_client_factory=factory,
    )
    try:
        result = await agent.run(
            "Use the connected lookup service to retrieve the fixture value. The "
            "request intentionally does not identify which connected lookup "
            "should be used."
        )
        transcript = await agent.transcript(result.run_id)
    finally:
        try:
            await agent.close()
        finally:
            await provider.close()

    text = _assert_completed(result, transcript)
    _assert_actionable_clarification(text, candidate_terms=("alpha", "beta"))
    assert alpha.calls == []
    assert beta.calls == []
