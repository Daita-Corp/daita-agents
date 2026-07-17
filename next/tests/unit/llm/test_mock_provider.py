from __future__ import annotations

import pytest

from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
)
from daita.llm.providers.mock import MockModelProvider, MockScriptExhausted


def _request(turn_id: str = "turn-1") -> ModelRequest:
    return ModelRequest(
        operation_id="op-1",
        turn_id=turn_id,
        messages=(
            CanonicalMessage(
                agent_id="agent-1",
                operation_id="op-1",
                turn_id=turn_id,
                role=MessageRole.USER,
                content=(TextBlock("hello"),),
            ),
        ),
    )


async def test_mock_provider_captures_requests_and_fails_on_script_exhaustion() -> None:
    response = ModelResponse(text="hello back", finish_reason=FinishReason.STOP)
    provider = MockModelProvider((response,))

    with pytest.raises(AssertionError, match="unconsumed"):
        provider.assert_consumed()

    assert await provider.generate(_request()) is response
    assert provider.requests == (_request(),)
    provider.assert_consumed()

    with pytest.raises(MockScriptExhausted, match="exhausted"):
        await provider.generate(_request("turn-2"))


async def test_mock_provider_raises_scripted_errors_without_coercion() -> None:
    error = TimeoutError("scripted timeout")
    provider = MockModelProvider((error,))

    with pytest.raises(TimeoutError, match="scripted timeout") as raised:
        await provider.generate(_request())

    assert raised.value is error
    assert provider.requests == (_request(),)
    provider.assert_consumed()
