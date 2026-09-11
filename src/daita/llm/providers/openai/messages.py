"""Serialize canonical messages for OpenAI Responses."""

from __future__ import annotations

from ...._json import FrozenJsonObject, canonical_json, thaw_json
from ...errors import ModelProviderError, ProviderErrorCode
from ...models import CanonicalMessage, MessageRole, TextBlock, ToolResultBlock


def _response_input(
    messages: tuple[CanonicalMessage, ...],
    provider_id: str,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    provider_call_ids: dict[str, str] = {}
    for message in messages:
        same_origin = message.provider_id == provider_id
        if same_origin:
            metadata = FrozenJsonObject.from_mapping(message.provider_metadata)
            replay_value = metadata.get("openai_replay_items")
            if replay_value is not None:
                replay_items = thaw_json(replay_value)
                if not isinstance(replay_items, list):
                    raise ModelProviderError(
                        ProviderErrorCode.INVALID_REQUEST,
                        "OpenAI replay metadata must contain JSON objects",
                    )
                decoded_replay_items: list[dict[str, object]] = []
                for replay_item in replay_items:
                    if not isinstance(replay_item, dict):
                        raise ModelProviderError(
                            ProviderErrorCode.INVALID_REQUEST,
                            "OpenAI replay metadata must contain JSON objects",
                        )
                    decoded_replay_items.append(replay_item)
                items.extend(decoded_replay_items)
        text = "\n".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        ).strip()
        if text:
            items.append({"role": message.role.value, "content": text})
        if message.role is MessageRole.ASSISTANT:
            for call in message.tool_calls:
                provider_call_id = (
                    call.provider_call_id if same_origin else None
                ) or call.id
                provider_call_ids[call.id] = provider_call_id
                items.append(
                    {
                        "type": "function_call",
                        "call_id": provider_call_id,
                        "name": call.name,
                        "arguments": canonical_json(call.arguments),
                    }
                )
        if message.role is MessageRole.TOOL:
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    continue
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": provider_call_ids.get(block.call_id, block.call_id),
                        "output": canonical_json(
                            {"is_error": block.is_error, "output": block.output}
                        ),
                    }
                )
    if not items:
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "canonical request produced no OpenAI input items",
        )
    return items
