from daita._json import FrozenJsonObject
from daita.llm.providers.anthropic import AnthropicMessagesProvider


def test_anthropic_preserves_signed_omitted_thinking_blocks():
    provider = AnthropicMessagesProvider("claude-opus-5-5")

    response = provider._decode_response(
        {
            "type": "message",
            "role": "assistant",
            "id": "message-1",
            "model": "claude-opus-5-5",
            "stop_reason": "end_turn",
            "content": (
                {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "signed-opaque-thinking",
                },
                {"type": "text", "text": "Done."},
            ),
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
            },
        }
    )

    continuation = response.provider_metadata["anthropic_continuation"]

    assert isinstance(continuation, FrozenJsonObject)
    assert continuation.to_dict()["content_blocks"] == [
        {
            "type": "thinking",
            "thinking": "",
            "signature": "signed-opaque-thinking",
        }
    ]
