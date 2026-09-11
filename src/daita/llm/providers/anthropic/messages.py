"""Serialize canonical messages and continuations for Anthropic Messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from ...._json import FrozenJsonObject, canonical_json
from ...errors import ModelProviderError, ProviderErrorCode
from ...models import CanonicalMessage, MessageRole, TextBlock, ToolResultBlock
from .._fields import field as _field, required_text as _required_text

_CONTINUATION_KEY = "anthropic_continuation"


_OPAQUE_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


def _message_input(
    messages: tuple[CanonicalMessage, ...],
    provider_id: str,
) -> tuple[str | None, list[dict[str, object]]]:
    system_parts: list[str] = []
    provider_messages: list[dict[str, object]] = []
    provider_call_ids: dict[str, str] = {}
    for message in messages:
        if message.role is MessageRole.SYSTEM:
            system_parts.extend(
                block.text for block in message.content if isinstance(block, TextBlock)
            )
            continue

        content: list[dict[str, object]] = []
        if message.role is MessageRole.ASSISTANT:
            content.extend(_continuation_blocks(message, provider_id))
        content.extend(
            {"type": "text", "text": block.text}
            for block in message.content
            if isinstance(block, TextBlock)
        )
        if message.role is MessageRole.ASSISTANT:
            same_origin = _same_origin(message, provider_id)
            for call in message.tool_calls:
                provider_call_id = (
                    call.provider_call_id
                    if same_origin and call.provider_call_id is not None
                    else call.id
                )
                provider_call_ids[call.id] = provider_call_id
                content.append(
                    {
                        "type": "tool_use",
                        "id": provider_call_id,
                        "name": call.name,
                        "input": FrozenJsonObject.from_mapping(
                            call.arguments
                        ).to_dict(),
                    }
                )
        elif message.role is MessageRole.TOOL:
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    continue
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": provider_call_ids.get(
                            block.call_id,
                            block.call_id,
                        ),
                        "content": canonical_json(block.output),
                        "is_error": block.is_error,
                    }
                )
        if not content:
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "canonical message produced no Anthropic content blocks",
            )
        role = "assistant" if message.role is MessageRole.ASSISTANT else "user"
        provider_messages.append({"role": role, "content": content})

    if not provider_messages:
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "canonical request produced no Anthropic messages",
        )
    system = "\n\n".join(part for part in system_parts if part.strip()).strip()
    return system or None, provider_messages


def _continuation_blocks(
    message: CanonicalMessage,
    provider_id: str,
) -> list[dict[str, object]]:
    if message.provider_id != provider_id:
        return []
    continuation_value = message.provider_metadata.get(_CONTINUATION_KEY)
    if continuation_value is None:
        return []
    if not isinstance(continuation_value, FrozenJsonObject):
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "Anthropic continuation metadata must be an object",
        )
    continuation = continuation_value.to_dict()
    origin = continuation.get("provider_id")
    if not isinstance(origin, str) or not origin.strip():
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "Anthropic continuation metadata requires a provider origin",
        )
    if origin != provider_id:
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "Anthropic continuation origin does not match canonical provider",
        )
    blocks = continuation.get("content_blocks")
    if not isinstance(blocks, list):
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "Anthropic continuation metadata requires content blocks",
        )
    decoded: list[dict[str, object]] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "Anthropic continuation content blocks must be objects",
            )
        block_type = block.get("type")
        if block_type not in _OPAQUE_BLOCK_TYPES:
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "Anthropic continuation contains an unsupported content block",
            )
        try:
            normalized = FrozenJsonObject.from_mapping(block).to_dict()
            _validate_opaque_block(normalized, cast(str, block_type))
        except (TypeError, ValueError) as error:
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "Anthropic continuation contains a malformed content block",
            ) from error
        decoded.append(normalized)
    return decoded


def _same_origin(message: CanonicalMessage, provider_id: str) -> bool:
    return message.provider_id == provider_id


def _plain_opaque_block(
    block: object,
    block_type: str,
) -> dict[str, object]:
    plain: dict[str, object]
    if isinstance(block, FrozenJsonObject):
        plain = block.to_dict()
    elif isinstance(block, Mapping):
        plain = FrozenJsonObject.from_mapping(block).to_dict()
    else:
        model_dump = getattr(block, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(mode="json", exclude_none=True)
            if not isinstance(dumped, Mapping):
                raise ValueError("opaque content block dump must be an object")
            plain = FrozenJsonObject.from_mapping(dumped).to_dict()
        else:
            plain = {"type": block_type}
            field_names = (
                ("thinking", "signature") if block_type == "thinking" else ("data",)
            )
            for name in field_names:
                value = _field(block, name, None)
                if value is not None:
                    plain[name] = value
            plain = FrozenJsonObject.from_mapping(plain).to_dict()
    _validate_opaque_block(plain, block_type)
    return plain


def _validate_opaque_block(block: Mapping[str, object], block_type: str) -> None:
    if block.get("type") != block_type:
        raise ValueError("opaque content block type changed during normalization")
    if block_type == "thinking":
        _required_text(block.get("thinking"), "thinking content")
        _required_text(block.get("signature"), "thinking signature")
    elif block_type == "redacted_thinking":
        _required_text(block.get("data"), "redacted thinking data")
    else:
        raise ValueError("unsupported opaque content block")
