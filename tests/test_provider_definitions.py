from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from _workspace_support import workspace_for

from daita import Agent
from daita.llm.factory import create_llm_provider
from daita.llm.models import FinishReason, ModelResponse, ToolCall
from daita.llm.provider_definitions import (
    BUILTIN_PROVIDER_IDS,
    PROVIDER_DEFINITIONS,
    PROVIDER_PRESENTATION,
    SUBSCRIPTION_CLIENTS,
    SUBSCRIPTION_PROVIDER_IDS,
    admit_model_selection,
    provider_definition,
)
from daita.llm.providers.mock import MockModelProvider
from daita.llm.providers.openai_compatible import OpenAICompatibleProvider
from daita.security import SecretReference
from daita.tui.models import PROVIDERS


class _Keychain:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def resolve(self, reference: SecretReference) -> str:
        return self.values[reference.name]

    async def set(self, reference: SecretReference, value: str) -> None:
        self.values[reference.name] = value

    async def delete(self, reference: SecretReference) -> None:
        self.values.pop(reference.name, None)


def test_provider_definitions_are_unique_and_feed_tui_presentation():
    identifiers = tuple(definition.id for definition in PROVIDER_DEFINITIONS)

    assert len(identifiers) == len(set(identifiers))
    assert BUILTIN_PROVIDER_IDS == frozenset(identifiers)
    assert PROVIDERS == (
        *PROVIDER_PRESENTATION,
        ("custom", "Custom API (OpenAI-compatible)"),
    )
    assert SUBSCRIPTION_PROVIDER_IDS == frozenset(SUBSCRIPTION_CLIENTS)
    assert all(
        provider_definition(identifier) is not None for identifier in identifiers
    )


def test_definition_and_tui_imports_do_not_import_native_sdks():
    script = """
import sys
import daita.llm.provider_definitions
import daita.tui.models

loaded = sorted(
    name for name in ("anthropic", "google", "google.genai", "openai")
    if name in sys.modules
)
if loaded:
    raise AssertionError(f"native SDKs imported while listing providers: {loaded}")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("provider", BUILTIN_PROVIDER_IDS - {"ollama"})
def test_fixed_builtin_endpoints_reject_overrides(provider):
    with pytest.raises(ValueError, match="fixed endpoint"):
        admit_model_selection(provider, "model", "https://override.invalid")


def test_local_and_custom_endpoint_rules_come_from_definitions():
    assert admit_model_selection("ollama", "model", None) == (
        "ollama",
        "model",
        None,
        False,
    )
    with pytest.raises(ValueError, match="explicit base URL"):
        admit_model_selection("synthetic-compatible", "model", None)


async def test_synthetic_compatible_provider_flows_through_host_and_factory(
    tmp_path: Path,
):
    provider_id = "synthetic-compatible:test-model"
    keychain = _Keychain()
    validator = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="validation-call",
                        name="daita_validate_tool_support",
                        arguments={},
                    ),
                ),
                provider_id=provider_id,
            ),
        ),
        provider_id=provider_id,
    )
    agent = await Agent.create(
        "provider-definitions",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        keychain=keychain,
        model_validator=validator,
    )
    try:
        route = await agent.configure_model(
            provider="synthetic-compatible",
            model="test-model",
            base_url="https://synthetic.invalid/v1",
            api_key="secret",
            context_window_tokens=8_192,
            max_output_tokens=512,
        )
    finally:
        await agent.close()

    candidate = route.candidates[0]
    assert candidate.provider_id == provider_id
    assert candidate.base_url == "https://synthetic.invalid/v1"
    assert candidate.secret_reference is not None

    constructed = create_llm_provider(
        provider_id,
        api_key="not-dispatched",
        base_url=candidate.base_url,
        max_output_tokens=512,
    )
    try:
        assert isinstance(constructed, OpenAICompatibleProvider)
        assert constructed.provider_id == provider_id
    finally:
        await constructed.close()
