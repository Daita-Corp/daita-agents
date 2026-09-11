"""Serialize canonical messages for Chat Completions protocols."""

from __future__ import annotations

from collections.abc import Mapping

from ...._json import canonical_json
from ...errors import ModelProviderError, ProviderErrorCode
from ...models import CanonicalMessage, MessageRole, TextBlock, ToolResultBlock

_CONTINUATION_KEY = "openai_compatible_continuation"


def _chat_messages(
    messages: tuple[CanonicalMessage, ...],
    provider_id: str,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    call_ids: dict[str, str] = {}
    for message in messages:
        text = "\n".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        ).strip()
        if message.role is MessageRole.ASSISTANT:
            same_origin = _same_origin(message, provider_id)
            native_calls: list[dict[str, object]] = []
            for call in message.tool_calls:
                native_id = (
                    call.provider_call_id
                    if same_origin and call.provider_call_id is not None
                    else call.id
                )
                call_ids[call.id] = native_id
                native_calls.append(
                    {
                        "id": native_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": canonical_json(call.arguments),
                        },
                    }
                )
            native_message: dict[str, object] = {
                "role": "assistant",
                "content": text or None,
            }
            if native_calls:
                native_message["tool_calls"] = native_calls
            result.append(native_message)
            continue
        if message.role is MessageRole.TOOL:
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    raise ValueError("tool message contains a non-tool result")
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_ids.get(block.call_id, block.call_id),
                        "content": canonical_json(
                            {"is_error": block.is_error, "output": block.output}
                        ),
                    }
                )
            continue
        if not text:
            raise ValueError("canonical text message produced no content")
        result.append({"role": message.role.value, "content": text})
    if not result:
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "canonical request produced no compatible chat messages",
        )
    return result


def _same_origin(message: CanonicalMessage, provider_id: str) -> bool:
    if message.provider_id != provider_id:
        return False
    continuation = message.provider_metadata.get(_CONTINUATION_KEY)
    if continuation is None:
        return True
    if (
        not isinstance(continuation, Mapping)
        or continuation.get("provider_id") != provider_id
    ):
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "compatible continuation origin does not match canonical provider",
        )
    return True
