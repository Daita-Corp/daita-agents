from __future__ import annotations

import pytest

from daita.context import (
    ContextBlock,
    ContextKind,
    ContextMessageGroup,
    ContextProvenance,
    ContextTrust,
)
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)


def _message(text: str, *, operation_id: str = "operation-1") -> CanonicalMessage:
    return CanonicalMessage(
        agent_id="agent-1",
        operation_id=operation_id,
        role=MessageRole.USER,
        content=(TextBlock(text),),
    )


def test_context_block_freezes_attribution_trust_and_indivisible_groups() -> None:
    provenance_items = [
        ContextProvenance(
            kind="catalog.resource",
            reference_id="resource-1",
            revision="sha256:abc",
        )
    ]
    groups = [ContextMessageGroup(id="catalog.group", messages=(_message("orders"),))]

    block = ContextBlock(
        id="catalog.selection",
        owner="catalog",
        kind=ContextKind.CATALOG,
        trust=ContextTrust.UNTRUSTED_EXTERNAL,
        provenance=provenance_items,  # type: ignore[arg-type]
        groups=groups,  # type: ignore[arg-type]
        priority=70,
        required=False,
    )
    provenance_items.clear()
    groups.clear()

    assert block.priority == 70
    assert block.provenance[0].reference_id == "resource-1"
    assert block.messages[0].content == (TextBlock("orders"),)
    assert len(block.groups) == 1


def test_context_message_group_requires_complete_tool_exchange() -> None:
    call = ToolCall(id="call-1", name="catalog_search", arguments={"q": "orders"})
    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        role=MessageRole.ASSISTANT,
        tool_calls=(call,),
    )
    result = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        role=MessageRole.TOOL,
        content=(ToolResultBlock(call_id=call.id, output={"matches": []}),),
    )

    group = ContextMessageGroup(
        id="operation.tool-exchange",
        messages=(assistant, result),
    )
    assert group.messages == (assistant, result)

    with pytest.raises(ValueError, match="together"):
        ContextMessageGroup(id="operation.split", messages=(assistant,))


def test_context_block_rejects_missing_provenance_and_cross_operation_groups() -> None:
    first = ContextMessageGroup(id="first", messages=(_message("one"),))
    second = ContextMessageGroup(
        id="second",
        messages=(_message("two", operation_id="operation-2"),),
    )
    provenance = (ContextProvenance(kind="runtime.operation", reference_id="op"),)

    with pytest.raises(ValueError, match="provenance"):
        ContextBlock(
            id="operation.objective",
            owner="operations",
            kind=ContextKind.OPERATION,
            trust=ContextTrust.TRUSTED_RUNTIME,
            provenance=(),
            groups=(first,),
            required=True,
        )
    with pytest.raises(ValueError, match="one agent and operation"):
        ContextBlock(
            id="operation.objective",
            owner="operations",
            kind=ContextKind.OPERATION,
            trust=ContextTrust.TRUSTED_RUNTIME,
            provenance=provenance,
            groups=(first, second),
            required=True,
        )


def test_context_block_priority_and_required_are_strict_types() -> None:
    group = ContextMessageGroup(id="intent", messages=(_message("question"),))
    provenance = (ContextProvenance(kind="trigger.user", reference_id="trigger-1"),)

    with pytest.raises(ValueError, match="priority"):
        ContextBlock(
            id="intent.current",
            owner="loop",
            kind=ContextKind.INTENT,
            trust=ContextTrust.UNTRUSTED_EXTERNAL,
            provenance=provenance,
            groups=(group,),
            priority=-1,
            required=True,
        )
    with pytest.raises(TypeError, match="required"):
        ContextBlock(
            id="intent.current",
            owner="loop",
            kind=ContextKind.INTENT,
            trust=ContextTrust.UNTRUSTED_EXTERNAL,
            provenance=provenance,
            groups=(group,),
            required=1,  # type: ignore[arg-type]
        )
