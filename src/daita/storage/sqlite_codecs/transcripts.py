"""Encode and decode run inputs, messages, usage, loop exits, and outcomes."""

from __future__ import annotations

from ...llm.errors import ProviderFailureDiagnostic, ProviderFailurePhase
from ...llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelSensitivity,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from ...llm.pricing import (
    CostBasis,
    CostComponent,
    CostEstimate,
    CostEstimateStatus,
    PricingModifier,
    PricingUsageRange,
)
from ...loop.models import (
    InstructionAuthority,
    LoopExit,
    LoopExitKind,
    RunInput,
    RunOrigin,
    RunStartEnvelope,
)
from .artifacts import (
    decode_artifact_ref,
    decode_delivery_receipt,
    encode_artifact_ref,
    encode_delivery_receipt,
)
from .common import (
    JsonValue,
    boolean,
    datetime_decode,
    datetime_encode,
    decimal_decode,
    decimal_encode,
    dump_payload,
    enum_decode,
    enum_encode,
    integer,
    load_payload,
    mapping,
    optional_decimal_decode,
    optional_decimal_encode,
    optional_integer,
    optional_text,
    plain_decode,
    plain_encode,
    record,
    record_fields,
    sequence,
    text,
)
from .execution_scope import decode_execution_scope, encode_execution_scope


def encode_run_input(value: RunInput) -> str:
    if not isinstance(value, RunInput):
        raise TypeError("run codec requires RunInput")
    return dump_payload(_encode_run_input(value))


def decode_run_input(value: str) -> RunInput:
    return _decode_run_input(load_payload(value))


def encode_message(value: CanonicalMessage) -> str:
    if not isinstance(value, CanonicalMessage):
        raise TypeError("message codec requires CanonicalMessage")
    return dump_payload(_encode_message(value))


def decode_message(value: str) -> CanonicalMessage:
    return _decode_message(load_payload(value))


def encode_loop_exit(value: LoopExit) -> str:
    if not isinstance(value, LoopExit):
        raise TypeError("loop-exit codec requires LoopExit")
    return dump_payload(_encode_loop_exit(value))


def decode_loop_exit(value: str) -> LoopExit:
    return _decode_loop_exit(load_payload(value))


def _encode_run_input(value: RunInput) -> dict[str, JsonValue]:
    return record(
        "RunInput",
        {
            "id": value.id,
            "agent_id": value.agent_id,
            "message": value.message,
            "created_at": datetime_encode(value.created_at),
            "conversation_id": value.conversation_id,
            "source_id": value.source_id,
            "conversation_source_id": value.conversation_source_id,
            "start": _encode_run_start(value.start),
        },
    )


def _decode_run_input(value: JsonValue) -> RunInput:
    fields = record_fields(
        value,
        "RunInput",
        (
            "id",
            "agent_id",
            "message",
            "created_at",
            "conversation_id",
            "source_id",
            "conversation_source_id",
            "start",
        ),
    )
    return RunInput(
        id=text(fields["id"], "run id"),
        agent_id=text(fields["agent_id"], "run agent_id"),
        message=text(fields["message"], "run message"),
        created_at=datetime_decode(fields["created_at"]),
        conversation_id=optional_text(fields["conversation_id"], "conversation id"),
        source_id=optional_text(fields["source_id"], "run source_id"),
        conversation_source_id=optional_text(
            fields["conversation_source_id"], "conversation source_id"
        ),
        start=_decode_run_start(fields["start"]),
    )


def _encode_run_start(value: RunStartEnvelope | None):
    if not isinstance(value, RunStartEnvelope):
        raise TypeError("normalized run input requires RunStartEnvelope")
    return record(
        "RunStartEnvelope",
        {
            "origin": value.origin.value,
            "instruction_authority": (
                None
                if value.instruction_authority is None
                else value.instruction_authority.value
            ),
            "user_message": value.user_message,
            "trusted_instruction_id": value.trusted_instruction_id,
            "trusted_instruction": value.trusted_instruction,
            "instruction_digest": value.instruction_digest,
            "untrusted_payload": plain_encode(value.untrusted_payload),
            "payload_digest": value.payload_digest,
            "execution_scope": (
                None
                if value.execution_scope is None
                else encode_execution_scope(value.execution_scope)
            ),
        },
    )


def _decode_run_start(value) -> RunStartEnvelope:
    fields = record_fields(
        value,
        "RunStartEnvelope",
        (
            "origin",
            "instruction_authority",
            "user_message",
            "trusted_instruction_id",
            "trusted_instruction",
            "instruction_digest",
            "untrusted_payload",
            "payload_digest",
            "execution_scope",
        ),
    )
    try:
        origin = RunOrigin(text(fields["origin"], "run start origin"))
        instruction_authority = (
            None
            if fields["instruction_authority"] is None
            else InstructionAuthority(
                text(fields["instruction_authority"], "run instruction authority")
            )
        )
    except ValueError:
        raise ValueError("stored run start origin is invalid") from None
    payload = plain_decode(fields["untrusted_payload"])
    if not isinstance(payload, dict):
        raise ValueError("stored run start payload must be an object")
    scope = fields["execution_scope"]
    return RunStartEnvelope(
        origin=origin,
        instruction_authority=instruction_authority,
        user_message=optional_text(fields["user_message"], "run start user message"),
        trusted_instruction_id=optional_text(
            fields["trusted_instruction_id"],
            "run start trusted instruction id",
        ),
        trusted_instruction=optional_text(
            fields["trusted_instruction"],
            "run start trusted instruction",
        ),
        instruction_digest=optional_text(
            fields["instruction_digest"],
            "run start instruction digest",
        ),
        untrusted_payload=payload,
        payload_digest=optional_text(
            fields["payload_digest"],
            "run start payload digest",
        ),
        execution_scope=None if scope is None else decode_execution_scope(scope),
    )


def _encode_text_block(value: TextBlock) -> dict[str, JsonValue]:
    return record("TextBlock", {"text": value.text})


def _decode_text_block(value: JsonValue) -> TextBlock:
    fields = record_fields(value, "TextBlock", ("text",))
    return TextBlock(text(fields["text"], "message text"))


def _encode_tool_call(value: ToolCall) -> dict[str, JsonValue]:
    return record(
        "ToolCall",
        {
            "id": value.id,
            "name": value.name,
            "arguments": plain_encode(value.arguments),
            "provider_call_id": value.provider_call_id,
        },
    )


def _decode_tool_call(value: JsonValue) -> ToolCall:
    fields = record_fields(
        value,
        "ToolCall",
        ("id", "name", "arguments", "provider_call_id"),
    )
    arguments = plain_decode(mapping(fields["arguments"], "tool-call arguments"))
    if not isinstance(arguments, dict):
        raise ValueError("stored tool-call arguments are invalid")
    return ToolCall(
        id=text(fields["id"], "tool-call id"),
        name=text(fields["name"], "tool-call name"),
        arguments=arguments,
        provider_call_id=optional_text(fields["provider_call_id"], "provider call id"),
    )


def _encode_tool_result(value: ToolResultBlock) -> dict[str, JsonValue]:
    return record(
        "ToolResultBlock",
        {
            "call_id": value.call_id,
            "output": plain_encode(value.output),
            "is_error": value.is_error,
            "sensitivity": (
                None
                if value.sensitivity is None
                else enum_encode(value.sensitivity, "ModelSensitivity")
            ),
            "sensitivity_provenance": plain_encode(value.sensitivity_provenance),
            "capability_id": value.capability_id,
            "executor_id": value.executor_id,
        },
    )


def _decode_tool_result(value: JsonValue) -> ToolResultBlock:
    fields = record_fields(
        value,
        "ToolResultBlock",
        (
            "call_id",
            "output",
            "is_error",
            "sensitivity",
            "sensitivity_provenance",
            "capability_id",
            "executor_id",
        ),
    )
    output = plain_decode(mapping(fields["output"], "tool-result output"))
    if not isinstance(output, dict):
        raise ValueError("stored tool-result output is invalid")
    provenance = plain_decode(
        mapping(fields["sensitivity_provenance"], "tool-result provenance")
    )
    if not isinstance(provenance, dict):
        raise ValueError("stored tool-result provenance is invalid")
    return ToolResultBlock(
        call_id=text(fields["call_id"], "tool-result call_id"),
        output=output,
        is_error=boolean(fields["is_error"], "tool-result is_error"),
        sensitivity=(
            None
            if fields["sensitivity"] is None
            else enum_decode(
                fields["sensitivity"], ModelSensitivity, "ModelSensitivity"
            )
        ),
        sensitivity_provenance=provenance,
        capability_id=optional_text(
            fields["capability_id"],
            "tool-result capability id",
        ),
        executor_id=optional_text(
            fields["executor_id"],
            "tool-result executor id",
        ),
    )


def _encode_message(value: CanonicalMessage) -> dict[str, JsonValue]:
    content: list[JsonValue] = []
    for block in value.content:
        if isinstance(block, TextBlock):
            content.append(_encode_text_block(block))
        elif isinstance(block, ToolResultBlock):
            content.append(_encode_tool_result(block))
        else:
            raise TypeError("message contains an unsupported stored block")
    return record(
        "CanonicalMessage",
        {
            "role": enum_encode(value.role, "MessageRole"),
            "content": content,
            "tool_calls": [_encode_tool_call(item) for item in value.tool_calls],
            "provider_id": value.provider_id,
            "provider_metadata": plain_encode(value.provider_metadata),
        },
    )


def _decode_message(value: JsonValue) -> CanonicalMessage:
    fields = record_fields(
        value,
        "CanonicalMessage",
        ("role", "content", "tool_calls", "provider_id", "provider_metadata"),
    )
    content: list[TextBlock | ToolResultBlock] = []
    for item in sequence(fields["content"], "message content"):
        if isinstance(item, dict) and item.get("__record__") == "TextBlock":
            content.append(_decode_text_block(item))
        elif isinstance(item, dict) and item.get("__record__") == "ToolResultBlock":
            content.append(_decode_tool_result(item))
        else:
            raise ValueError("stored message content block is unsupported")
    metadata = plain_decode(
        mapping(fields["provider_metadata"], "message provider metadata")
    )
    if not isinstance(metadata, dict):
        raise ValueError("stored message provider metadata is invalid")
    return CanonicalMessage(
        role=enum_decode(fields["role"], MessageRole, "MessageRole"),
        content=tuple(content),
        tool_calls=tuple(
            _decode_tool_call(item)
            for item in sequence(fields["tool_calls"], "message tool_calls")
        ),
        provider_id=optional_text(fields["provider_id"], "message provider_id"),
        provider_metadata=metadata,
    )


def _encode_usage_range(value: PricingUsageRange) -> dict[str, JsonValue]:
    return record(
        "PricingUsageRange",
        {
            "metric": value.metric,
            "minimum_inclusive": value.minimum_inclusive,
            "maximum_inclusive": value.maximum_inclusive,
        },
    )


def _decode_usage_range(value: JsonValue) -> PricingUsageRange:
    fields = record_fields(
        value,
        "PricingUsageRange",
        ("metric", "minimum_inclusive", "maximum_inclusive"),
    )
    return PricingUsageRange(
        metric=text(fields["metric"], "pricing usage metric"),
        minimum_inclusive=optional_integer(
            fields["minimum_inclusive"], "pricing minimum"
        ),
        maximum_inclusive=optional_integer(
            fields["maximum_inclusive"], "pricing maximum"
        ),
    )


def _encode_modifier(value: PricingModifier) -> dict[str, JsonValue]:
    return record(
        "PricingModifier",
        {"name": value.name, "multiplier": decimal_encode(value.multiplier)},
    )


def _decode_modifier(value: JsonValue) -> PricingModifier:
    fields = record_fields(value, "PricingModifier", ("name", "multiplier"))
    return PricingModifier(
        name=text(fields["name"], "pricing modifier name"),
        multiplier=decimal_decode(fields["multiplier"]),
    )


def _encode_cost_component(value: CostComponent) -> dict[str, JsonValue]:
    return record(
        "CostComponent",
        {
            "name": value.name,
            "amount_usd": decimal_encode(value.amount_usd),
            "basis": (
                None if value.basis is None else enum_encode(value.basis, "CostBasis")
            ),
            "rate_schedule_id": value.rate_schedule_id,
            "metric": value.metric,
            "quantity": optional_decimal_encode(value.quantity),
            "unit": value.unit,
            "unit_size": value.unit_size,
            "rate_usd": optional_decimal_encode(value.rate_usd),
            "usage_range": (
                None
                if value.usage_range is None
                else _encode_usage_range(value.usage_range)
            ),
            "modifiers": [_encode_modifier(item) for item in value.modifiers],
        },
    )


def _decode_cost_component(value: JsonValue) -> CostComponent:
    fields = record_fields(
        value,
        "CostComponent",
        (
            "name",
            "amount_usd",
            "basis",
            "rate_schedule_id",
            "metric",
            "quantity",
            "unit",
            "unit_size",
            "rate_usd",
            "usage_range",
            "modifiers",
        ),
    )
    return CostComponent(
        name=text(fields["name"], "cost component name"),
        amount_usd=decimal_decode(fields["amount_usd"]),
        basis=(
            None
            if fields["basis"] is None
            else enum_decode(fields["basis"], CostBasis, "CostBasis")
        ),
        rate_schedule_id=optional_text(
            fields["rate_schedule_id"], "cost rate schedule id"
        ),
        metric=optional_text(fields["metric"], "cost metric"),
        quantity=optional_decimal_decode(fields["quantity"]),
        unit=optional_text(fields["unit"], "cost unit"),
        unit_size=optional_integer(fields["unit_size"], "cost unit size"),
        rate_usd=optional_decimal_decode(fields["rate_usd"]),
        usage_range=(
            None
            if fields["usage_range"] is None
            else _decode_usage_range(fields["usage_range"])
        ),
        modifiers=tuple(
            _decode_modifier(item)
            for item in sequence(fields["modifiers"], "pricing modifiers")
        ),
    )


def _encode_cost_estimate(value: CostEstimate) -> dict[str, JsonValue]:
    return record(
        "CostEstimate",
        {
            "amount_usd": optional_decimal_encode(value.amount_usd),
            "status": enum_encode(value.status, "CostEstimateStatus"),
            "basis": (
                None if value.basis is None else enum_encode(value.basis, "CostBasis")
            ),
            "rate_schedule_id": value.rate_schedule_id,
            "components": [_encode_cost_component(item) for item in value.components],
            "code": value.code,
        },
    )


def _decode_cost_estimate(value: JsonValue) -> CostEstimate:
    fields = record_fields(
        value,
        "CostEstimate",
        (
            "amount_usd",
            "status",
            "basis",
            "rate_schedule_id",
            "components",
            "code",
        ),
    )
    return CostEstimate(
        amount_usd=optional_decimal_decode(fields["amount_usd"]),
        status=enum_decode(fields["status"], CostEstimateStatus, "CostEstimateStatus"),
        basis=(
            None
            if fields["basis"] is None
            else enum_decode(fields["basis"], CostBasis, "CostBasis")
        ),
        rate_schedule_id=optional_text(
            fields["rate_schedule_id"], "estimate rate schedule id"
        ),
        components=tuple(
            _decode_cost_component(item)
            for item in sequence(fields["components"], "cost components")
        ),
        code=optional_text(fields["code"], "cost estimate code"),
    )


def _encode_model_usage(value: ModelUsage) -> dict[str, JsonValue]:
    return record(
        "ModelUsage",
        {
            "input_tokens": value.input_tokens,
            "output_tokens": value.output_tokens,
            "reasoning_tokens": value.reasoning_tokens,
            "cache_read_tokens": value.cache_read_tokens,
            "cache_write_tokens": value.cache_write_tokens,
            "cost_estimate": _encode_cost_estimate(value.cost_estimate),
        },
    )


def _decode_model_usage(value: JsonValue) -> ModelUsage:
    fields = record_fields(
        value,
        "ModelUsage",
        (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cost_estimate",
        ),
    )
    return ModelUsage(
        input_tokens=integer(fields["input_tokens"], "input tokens"),
        output_tokens=integer(fields["output_tokens"], "output tokens"),
        reasoning_tokens=integer(fields["reasoning_tokens"], "reasoning tokens"),
        cache_read_tokens=integer(fields["cache_read_tokens"], "cache read tokens"),
        cache_write_tokens=integer(fields["cache_write_tokens"], "cache write tokens"),
        cost_estimate=_decode_cost_estimate(fields["cost_estimate"]),
    )


def _encode_loop_exit(value: LoopExit) -> dict[str, JsonValue]:
    return record(
        "LoopExit",
        {
            "run_id": value.run_id,
            "conversation_id": value.conversation_id,
            "kind": enum_encode(value.kind, "LoopExitKind"),
            "reason": value.reason,
            "created_at": datetime_encode(value.created_at),
            "final_text": value.final_text,
            "steps": value.steps,
            "usage": _encode_model_usage(value.usage),
            "provider_id": value.provider_id,
            "provider_failure": (
                None
                if value.provider_failure is None
                else _encode_provider_failure(value.provider_failure)
            ),
            "artifacts": [encode_artifact_ref(item) for item in value.artifacts],
            "artifact_deliveries": [
                encode_delivery_receipt(item) for item in value.artifact_deliveries
            ],
        },
    )


def _decode_loop_exit(value: JsonValue) -> LoopExit:
    fields = record_fields(
        value,
        "LoopExit",
        (
            "run_id",
            "conversation_id",
            "kind",
            "reason",
            "created_at",
            "final_text",
            "steps",
            "usage",
            "provider_id",
            "provider_failure",
            "artifacts",
            "artifact_deliveries",
        ),
    )
    return LoopExit(
        run_id=text(fields["run_id"], "loop-exit run_id"),
        conversation_id=text(fields["conversation_id"], "loop-exit conversation_id"),
        kind=enum_decode(fields["kind"], LoopExitKind, "LoopExitKind"),
        reason=text(fields["reason"], "loop-exit reason"),
        created_at=datetime_decode(fields["created_at"]),
        final_text=optional_text(fields["final_text"], "loop-exit final text"),
        steps=integer(fields["steps"], "loop-exit steps"),
        usage=_decode_model_usage(fields["usage"]),
        provider_id=optional_text(fields["provider_id"], "loop-exit provider id"),
        provider_failure=(
            None
            if fields["provider_failure"] is None
            else _decode_provider_failure(fields["provider_failure"])
        ),
        artifacts=tuple(
            decode_artifact_ref(item)
            for item in sequence(fields["artifacts"], "loop-exit artifacts")
        ),
        artifact_deliveries=tuple(
            decode_delivery_receipt(item)
            for item in sequence(
                fields["artifact_deliveries"], "loop-exit artifact deliveries"
            )
        ),
    )


def _encode_provider_failure(
    value: ProviderFailureDiagnostic,
) -> dict[str, JsonValue]:
    return record(
        "ProviderFailureDiagnostic",
        {
            "phase": enum_encode(value.phase, "ProviderFailurePhase"),
            "code": value.code,
            "event_type": value.event_type,
            "terminal_status": value.terminal_status,
            "output_item_types": list(value.output_item_types),
            "response_id_digest": value.response_id_digest,
        },
    )


def _decode_provider_failure(value: JsonValue) -> ProviderFailureDiagnostic:
    fields = record_fields(
        value,
        "ProviderFailureDiagnostic",
        (
            "phase",
            "code",
            "event_type",
            "terminal_status",
            "output_item_types",
            "response_id_digest",
        ),
    )
    return ProviderFailureDiagnostic(
        phase=enum_decode(
            fields["phase"],
            ProviderFailurePhase,
            "ProviderFailurePhase",
        ),
        code=text(fields["code"], "provider failure code"),
        event_type=optional_text(fields["event_type"], "provider failure event type"),
        terminal_status=optional_text(
            fields["terminal_status"], "provider failure terminal status"
        ),
        output_item_types=tuple(
            text(item, "provider failure output item type")
            for item in sequence(
                fields["output_item_types"],
                "provider failure output item types",
            )
        ),
        response_id_digest=optional_text(
            fields["response_id_digest"],
            "provider failure response ID digest",
        ),
    )
