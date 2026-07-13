import json
from unittest.mock import AsyncMock

import pytest

from daita.db.llm_service import (
    DbLLMConfig,
    DbLLMService,
    redact_db_llm_private_diagnostic,
)


class _StructuredProvider:
    def __init__(self):
        self.calls = []

    def structured_output_options(self, schema, *, name):
        return {"native_schema": {"name": name, "schema": schema}}

    async def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return '{"status":"finish","actions":[]}'


async def test_db_llm_service_closes_created_provider_once_and_clears_it():
    service = DbLLMService(DbLLMConfig(provider="openai", model="test-model"))
    provider = AsyncMock()
    service._provider = provider

    await service.aclose()
    await service.aclose()

    provider.aclose.assert_awaited_once_with()
    assert service._provider is None


async def test_db_llm_service_close_before_first_use_stays_lazy():
    service = DbLLMService(DbLLMConfig(provider="openai", model="test-model"))

    await service.aclose()

    assert service._provider is None


async def test_db_llm_service_surfaces_close_failure_and_clears_provider():
    service = DbLLMService(DbLLMConfig(provider="openai", model="test-model"))
    provider = AsyncMock()
    provider.aclose.side_effect = RuntimeError("provider close failed")
    service._provider = provider

    with pytest.raises(RuntimeError, match="provider close failed"):
        await service.aclose()

    assert service._provider is None
    await service.aclose()
    provider.aclose.assert_awaited_once_with()


async def test_db_llm_service_clears_compatibility_provider_without_close_hook():
    service = DbLLMService(DbLLMConfig(provider="openai", model="test-model"))
    provider = object()
    service._provider = provider

    await service.aclose()

    assert service._provider is None


async def test_db_llm_service_uses_provider_native_schema_options_when_supported():
    service = DbLLMService(DbLLMConfig(provider="fake", model="test-model"))
    provider = _StructuredProvider()
    service._provider = provider
    schema = {"type": "object", "properties": {"status": {"type": "string"}}}

    response = await service.generate_json(
        [{"role": "user", "content": "plan"}],
        response_schema=schema,
        schema_name="db_planner_decision",
    )

    assert response.content == '{"status":"finish","actions":[]}'
    assert provider.calls[0][1]["native_schema"] == {
        "name": "db_planner_decision",
        "schema": schema,
    }
    assert response.diagnostics["structured_output"] == {
        "schema_name": "db_planner_decision",
        "provider_native": True,
    }


def test_private_llm_diagnostics_are_bounded_and_redact_content_and_pii():
    secret = "sk-test-secret-123456"
    value = {
        "status": "finish",
        "actions": [
            {
                "kind": "execute_validated_read",
                "input": {
                    "sql": "select * from customers where email='ada@example.com'",
                    "rows": [{"full_name": "Ada Lovelace"}],
                },
            }
        ],
        "rationale": f"Email ada@example.com with {secret}." + ("x" * 2000),
    }

    redacted = redact_db_llm_private_diagnostic(value, max_chars=512)
    dumped = json.dumps(redacted, sort_keys=True)

    assert len(dumped) < 700
    assert secret not in dumped
    assert "ada@example.com" not in dumped
    assert "Ada Lovelace" not in dumped
    assert "[REDACTED_PRIVATE]" in dumped
    assert "[REDACTED_EMAIL]" in dumped
    assert "truncated" in dumped


def test_private_llm_diagnostics_redact_free_text_identity_and_payment_pii():
    raw_response = json.dumps(
        {
            "status": "failed",
            "actions": [],
            "rationale": (
                "SSN 123-45-6789; card 4111 1111 1111 1111; "
                "phones 312-555-0199 and +44 20 7946 0958."
            ),
        }
    )

    redacted = redact_db_llm_private_diagnostic(raw_response)

    assert "123-45-6789" not in redacted
    assert "4111 1111 1111 1111" not in redacted
    assert "312-555-0199" not in redacted
    assert "+44 20 7946 0958" not in redacted
    assert "[REDACTED_SSN]" in redacted
    assert "[REDACTED_CREDIT_CARD]" in redacted
    assert redacted.count("[REDACTED_PHONE]") == 2


def test_private_llm_diagnostics_preserve_non_card_numeric_identifiers():
    value = "order 1234 5678 9012 3456"

    assert redact_db_llm_private_diagnostic(value) == value
