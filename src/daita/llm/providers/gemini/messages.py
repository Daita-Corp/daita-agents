"""Serialize canonical Gemini contents, counting payloads, and continuations."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import cast

from ...._json import FrozenJsonObject
from ...errors import ModelProviderError, ProviderErrorCode
from ...models import CanonicalMessage, MessageRole, TextBlock, ToolResultBlock
from .._fields import optional_text as _optional_text, required_text as _required_text


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return value


_CONTINUATION_KEY = "gemini_continuation"


_PORTABLE_FUNCTION_CALL_SIGNATURE = b"skip_thought_signature_validator"


def _count_request(arguments: Mapping[str, object]) -> dict[str, object]:
    """Project prepared SDK arguments into countTokens' full REST input shape.

    Only the field names of the admitted SDK surface are translated; arbitrary
    function arguments, results and JSON schemas remain untouched. Actual-SDK
    tests compare this body with generation for every supported content shape.
    """
    contents = []
    for content in cast(list[dict[str, object]], arguments["contents"]):
        parts = []
        for part in cast(list[dict[str, object]], content["parts"]):
            native: dict[str, object] = {}
            for key, value in part.items():
                field = {
                    "text": "text",
                    "thought": "thought",
                    "function_call": "functionCall",
                    "function_response": "functionResponse",
                    "thought_signature": "thoughtSignature",
                }[key]
                native[field] = (
                    base64.b64encode(value).decode("ascii")
                    if key == "thought_signature" and isinstance(value, bytes)
                    else value
                )
            parts.append(native)
        contents.append({"role": content["role"], "parts": parts})
    model = cast(str, arguments["model"])
    body: dict[str, object] = {
        "model": model if model.startswith("models/") else f"models/{model}",
        "contents": contents,
    }
    config = cast(dict[str, object], arguments["config"])
    if set(config) - {
        "http_options",
        "max_output_tokens",
        "system_instruction",
        "tools",
        "response_mime_type",
        "response_json_schema",
    }:
        raise ValueError("uncounted Gemini configuration")
    if "system_instruction" in config:
        body["systemInstruction"] = {
            "role": "user",
            "parts": [{"text": config["system_instruction"]}],
        }
    if "tools" in config:
        body["tools"] = [
            {"functionDeclarations": tool["function_declarations"]}
            for tool in cast(list[dict[str, object]], config["tools"])
        ]
    generation = {
        native: config[key]
        for key, native in (
            ("response_mime_type", "responseMimeType"),
            ("response_json_schema", "responseJsonSchema"),
        )
        if key in config
    }
    if generation:
        body["generationConfig"] = generation
    return body


def _gemini_contents(
    messages: tuple[CanonicalMessage, ...],
    provider_id: str,
) -> tuple[list[dict[str, object]], str | None]:
    contents: list[dict[str, object]] = []
    system_parts: list[str] = []
    call_ids: dict[str, tuple[str | None, str]] = {}
    for message in messages:
        text = "\n".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        ).strip()
        if message.role is MessageRole.SYSTEM:
            if not text:
                raise ValueError("system message produced no text")
            system_parts.append(text)
            continue
        if message.role is MessageRole.ASSISTANT:
            continuation = _same_origin_continuation(
                message,
                provider_id,
            )
            replay_value = (
                None if continuation is None else continuation.get("content_parts")
            )
            if replay_value is not None:
                parts = _replay_content_parts(replay_value, message)
            else:
                parts = []
                if text:
                    parts.append({"text": text})
            for call_index, call in enumerate(message.tool_calls):
                native_id = (
                    call.provider_call_id
                    if continuation is not None and call.provider_call_id is not None
                    else (None if continuation is not None else call.id)
                )
                call_ids[call.id] = (native_id, call.name)
                if replay_value is None:
                    native_call: dict[str, object] = {
                        "name": call.name,
                        "args": FrozenJsonObject.from_mapping(call.arguments).to_dict(),
                    }
                    if native_id is not None:
                        native_call["id"] = native_id
                    native_part: dict[str, object] = {"function_call": native_call}
                    if continuation is None and call_index == 0:
                        native_part["thought_signature"] = (
                            _PORTABLE_FUNCTION_CALL_SIGNATURE
                        )
                    parts.append(native_part)
            if not parts:
                raise ValueError("assistant message produced no Gemini parts")
            contents.append({"role": "model", "parts": parts})
            continue
        if message.role is MessageRole.TOOL:
            parts = []
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    raise ValueError("tool message contains a non-tool result")
                try:
                    native_id, name = call_ids[block.call_id]
                except KeyError as error:
                    raise ValueError(
                        "tool result has no preceding Gemini function call"
                    ) from error
                function_response: dict[str, object] = {
                    "name": name,
                    "response": {
                        "is_error": block.is_error,
                        "output": FrozenJsonObject.from_mapping(block.output).to_dict(),
                    },
                }
                if native_id is not None:
                    function_response["id"] = native_id
                parts.append(
                    {
                        "function_response": function_response,
                    }
                )
            contents.append({"role": "user", "parts": parts})
            continue
        if not text:
            raise ValueError("user message produced no text")
        contents.append({"role": "user", "parts": [{"text": text}]})
    if not contents:
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "canonical request produced no Gemini contents",
        )
    return contents, "\n".join(system_parts) or None


def _same_origin_continuation(
    message: CanonicalMessage,
    provider_id: str,
) -> Mapping[str, object] | None:
    if message.provider_id != provider_id:
        return None
    value = message.provider_metadata.get(_CONTINUATION_KEY)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Gemini continuation metadata must be an object")
    origin = _required_text(
        value.get("provider_id"),
        "Gemini continuation provider origin",
    )
    if origin != provider_id:
        raise ValueError("Gemini continuation origin does not match canonical provider")
    return value


def _replay_content_parts(
    value: object,
    message: CanonicalMessage,
) -> list[dict[str, object]]:
    raw_parts = _sequence(value, "Gemini continuation content parts")
    replay_parts: list[dict[str, object]] = []
    replay_text: list[str] = []
    replay_calls: list[tuple[str | None, str, FrozenJsonObject]] = []
    saw_signature = False
    allowed_keys = frozenset({"function_call", "text", "thought", "thought_signature"})
    for value_part in raw_parts:
        if not isinstance(value_part, Mapping):
            raise ValueError("Gemini continuation parts must be objects")
        if set(value_part) - allowed_keys:
            raise ValueError("Gemini continuation part has unsupported fields")
        replay_part: dict[str, object] = {}
        signature = value_part.get("thought_signature")
        if signature is not None:
            replay_part["thought_signature"] = _decode_signature(signature)
            saw_signature = True
        thought = value_part.get("thought", False)
        if thought is not False and thought is not True:
            raise ValueError("Gemini continuation thought flag must be boolean")
        if thought:
            replay_part["thought"] = True
        part_text = value_part.get("text")
        if part_text is not None:
            if not isinstance(part_text, str):
                raise ValueError("Gemini continuation text must be text")
            replay_part["text"] = part_text
            if not thought and part_text.strip():
                replay_text.append(part_text)
        function_call = value_part.get("function_call")
        if function_call is not None:
            if part_text is not None or not isinstance(function_call, Mapping):
                raise ValueError("Gemini continuation function call is malformed")
            name = _required_text(
                function_call.get("name"),
                "Gemini continuation function name",
            )
            arguments = function_call.get("args")
            if not isinstance(arguments, Mapping):
                raise ValueError("Gemini continuation arguments must be an object")
            native_id = _optional_text(
                function_call.get("id"),
                "Gemini continuation function id",
            )
            native_call: dict[str, object] = {
                "name": name,
                "args": FrozenJsonObject.from_mapping(arguments).to_dict(),
            }
            if native_id is not None:
                native_call["id"] = native_id
            replay_part["function_call"] = native_call
            replay_calls.append(
                (native_id, name, FrozenJsonObject.from_mapping(arguments))
            )
        if not replay_part:
            raise ValueError("Gemini continuation contains an empty part")
        replay_parts.append(replay_part)
    if not replay_parts or not saw_signature:
        raise ValueError("Gemini continuation requires signed content parts")

    canonical_text = "\n".join(
        block.text for block in message.content if isinstance(block, TextBlock)
    ).strip()
    if "\n".join(replay_text).strip() != canonical_text:
        raise ValueError("Gemini continuation text does not match canonical content")
    if len(replay_calls) != len(message.tool_calls):
        raise ValueError("Gemini continuation calls do not match canonical calls")
    for replay_call, canonical_call in zip(
        replay_calls,
        message.tool_calls,
        strict=True,
    ):
        native_id, name, arguments = replay_call
        if (
            native_id != canonical_call.provider_call_id
            or name != canonical_call.name
            or arguments != FrozenJsonObject.from_mapping(canonical_call.arguments)
        ):
            raise ValueError(
                "Gemini continuation call does not match canonical content"
            )
    return replay_parts


def _encode_signature(value: object) -> str:
    if isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, bytes):
        raw = value
    else:
        raise ValueError("thought signature must be bytes or text")
    if not raw:
        raise ValueError("thought signature must not be empty")
    return base64.b64encode(raw).decode("ascii")


def _decode_signature(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("encoded thought signature must be text")
    try:
        return base64.b64decode(value, validate=True)
    except Exception as error:
        raise ValueError("encoded thought signature is malformed") from error
