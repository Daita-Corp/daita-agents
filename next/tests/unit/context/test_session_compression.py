from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json

import pytest

from daita._json import canonical_json
from daita.context import (
    ContextKind,
    ContextTrust,
    RequiredContextOverflow,
    SessionCompressionIntegrityError,
    SessionCompressionPolicy,
    SessionCompressionScopeError,
    SessionCompressionService,
    SessionOperationFacts,
    estimate_context_block_tokens,
    select_context_blocks,
)
from daita.context.session import (
    SessionApprovalStateFact,
    SessionResourceScopeFact,
)
from daita.llm.models import (
    CanonicalMessage,
    ContentBlock,
    MessageRole,
    ModelProfile,
    ModelSensitivity,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.sessions import Session, SessionCompressionCheckpoint, SessionTranscript
from daita.operations.governance import ApprovalStatus

NOW = datetime(2026, 7, 18, 22, 0, tzinfo=timezone.utc)
AGENT_ID = "agent-1"
SESSION_ID = "session-1"


def _message(
    operation_id: str,
    role: MessageRole,
    text: str | None = None,
    *,
    tool_calls: tuple[ToolCall, ...] = (),
    tool_result: ToolResultBlock | None = None,
) -> CanonicalMessage:
    content: tuple[ContentBlock, ...]
    if tool_result is not None:
        content = (tool_result,)
    elif text is not None:
        content = (TextBlock(text),)
    else:
        content = ()
    return CanonicalMessage(
        agent_id=AGENT_ID,
        operation_id=operation_id,
        session_id=SESSION_ID,
        turn_id=f"turn-{operation_id}",
        role=role,
        content=content,
        tool_calls=tool_calls,
        provider_metadata=(
            {"opaque": operation_id} if role is MessageRole.ASSISTANT else {}
        ),
    )


def _plain_exchange(
    operation_id: str, user: str, answer: str
) -> tuple[CanonicalMessage, ...]:
    return (
        _message(operation_id, MessageRole.USER, user),
        _message(operation_id, MessageRole.ASSISTANT, answer),
    )


def _tool_exchange(operation_id: str) -> tuple[CanonicalMessage, ...]:
    call = ToolCall(id=f"call-{operation_id}", name="catalog_search", arguments={})
    return (
        _message(operation_id, MessageRole.USER, "find customers"),
        _message(operation_id, MessageRole.ASSISTANT, tool_calls=(call,)),
        _message(
            operation_id,
            MessageRole.TOOL,
            tool_result=ToolResultBlock(
                call_id=call.id,
                output={"resource_id": "resource-customers"},
            ),
        ),
        _message(operation_id, MessageRole.ASSISTANT, "Customers found."),
    )


def _facts(
    operation_id: str,
    *,
    evidence_ids: tuple[str, ...] = (),
    approval_ids: tuple[str, ...] = (),
    resource_ids: tuple[str, ...] = (),
    objective: str | None = None,
    terminal_reason: str | None = None,
    approval_state_facts: tuple[SessionApprovalStateFact, ...] = (),
    resource_scope_facts: tuple[SessionResourceScopeFact, ...] = (),
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL,
) -> SessionOperationFacts:
    return SessionOperationFacts(
        operation_id=operation_id,
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        revision=f"revision-{operation_id}",
        status="succeeded" if operation_id != "op-current" else "running",
        sensitivity=sensitivity,
        evidence_ids=evidence_ids,
        approval_ids=approval_ids,
        resource_ids=resource_ids,
        final_text=None if operation_id == "op-current" else f"answer {operation_id}",
        objective=objective,
        terminal_reason=terminal_reason,
        approval_state_facts=approval_state_facts,
        resource_scope_facts=resource_scope_facts,
    )


def _transcript(
    operation_ids: tuple[str, ...],
    messages: tuple[CanonicalMessage, ...],
) -> SessionTranscript:
    return SessionTranscript(
        session=Session(
            id=SESSION_ID,
            agent_id=AGENT_ID,
            title="Compression test",
            created_at=NOW,
            updated_at=NOW,
        ),
        operation_ids=operation_ids,
        messages=messages,
    )


@dataclass
class MemorySessionBackend:
    transcript: SessionTranscript
    facts: dict[str, SessionOperationFacts]
    checkpoint: SessionCompressionCheckpoint | None = None
    commits: int = 0
    expected_versions: tuple[int, ...] = ()

    async def load_session(
        self,
        agent_id: str,
        session_id: str,
    ) -> SessionTranscript | None:
        return self.transcript

    async def load_session_compression(
        self,
        agent_id: str,
        session_id: str,
    ) -> SessionCompressionCheckpoint | None:
        return self.checkpoint

    async def load_session_operation(
        self,
        operation_id: str,
    ) -> SessionOperationFacts | None:
        return self.facts.get(operation_id)

    async def commit_session_compression(
        self,
        checkpoint: SessionCompressionCheckpoint,
        *,
        expected_version: int,
    ) -> SessionCompressionCheckpoint:
        actual = 0 if self.checkpoint is None else self.checkpoint.version
        if actual != expected_version:
            raise RuntimeError("CAS conflict")
        self.checkpoint = checkpoint
        self.commits += 1
        self.expected_versions = (*self.expected_versions, expected_version)
        return checkpoint


def _service(
    backend: MemorySessionBackend,
    *,
    threshold: int,
    retain: int = 1,
    max_summary_characters: int = 8_000,
    max_excerpt_characters: int = 256,
) -> SessionCompressionService:
    return SessionCompressionService(
        transcripts=backend,
        checkpoints=backend,
        operations=backend,
        committer=backend,
        policy=SessionCompressionPolicy(
            compression_threshold_tokens=threshold,
            retain_latest_operations=retain,
            max_summary_characters=max_summary_characters,
            max_excerpt_characters=max_excerpt_characters,
        ),
        clock=lambda: NOW,
        id_factory=lambda prefix: f"{prefix}-1",
    )


def _profile() -> ModelProfile:
    return ModelProfile(
        id="mock:compression",
        context_window_tokens=20_000,
        max_output_tokens=1_000,
    )


def test_session_compression_policy_is_versioned_and_profile_default_is_explicit() -> (
    None
):
    policy = SessionCompressionPolicy()

    assert policy.schema_version == 1
    assert policy.compression_threshold_tokens is None

    with pytest.raises(ValueError, match="schema_version"):
        SessionCompressionPolicy(schema_version=2)
    with pytest.raises(ValueError, match="positive integer or None"):
        SessionCompressionPolicy(compression_threshold_tokens=False)


async def test_default_compression_threshold_is_derived_from_truthful_profile() -> None:
    history = _plain_exchange("op-history", "prior", "answer")
    current = _plain_exchange("op-current", "current", "unused")
    backend = MemorySessionBackend(
        _transcript(("op-history", "op-current"), (*history, *current)),
        {
            "op-history": _facts("op-history"),
            "op-current": _facts("op-current"),
        },
    )
    service = SessionCompressionService(
        transcripts=backend,
        checkpoints=backend,
        operations=backend,
        committer=backend,
        policy=SessionCompressionPolicy(),
        clock=lambda: NOW,
        id_factory=lambda prefix: f"{prefix}-1",
    )

    projection = await service.project(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        current_operation_id="op-current",
        profile=_profile(),
        maximum_projection_tokens=_profile().maximum_input_tokens,
    )

    assert projection.threshold_tokens == _profile().maximum_input_tokens * 3 // 4


async def test_low_threshold_compresses_history_shorter_than_retention_cap() -> None:
    history = _plain_exchange(
        "op-history",
        "question-" + "x" * 3_000,
        "answer-" + "y" * 3_000,
    )
    current = _plain_exchange("op-current", "continue", "unused")
    transcript = _transcript(("op-history", "op-current"), (*history, *current))
    facts = {
        "op-history": _facts(
            "op-history",
            evidence_ids=("evidence-history",),
            objective="Retain the historical objective",
        ),
        "op-current": _facts("op-current"),
    }
    backend = MemorySessionBackend(transcript, facts)

    projection = await _service(
        backend,
        threshold=1,
        retain=4,
    ).project(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        current_operation_id="op-current",
        profile=_profile(),
        maximum_projection_tokens=_profile().maximum_input_tokens,
    )

    assert projection.compressed_now
    assert projection.checkpoint is not None
    assert projection.checkpoint.operation_ids == ("op-history",)
    assert projection.checkpoint.through_position == 0
    assert projection.checkpoint.evidence_ids == ("evidence-history",)
    assert [block.kind for block in projection.blocks] == [ContextKind.SESSION_SUMMARY]
    assert backend.commits == 1
    assert backend.transcript == transcript
    assert backend.facts == facts


async def test_low_threshold_advances_short_checkpoint_when_new_history_arrives() -> (
    None
):
    first_history = _plain_exchange(
        "op-0",
        "question-zero-" + "x" * 3_000,
        "answer-zero-" + "y" * 3_000,
    )
    next_history = _plain_exchange(
        "op-1",
        "question-one-" + "x" * 3_000,
        "answer-one-" + "y" * 3_000,
    )
    backend = MemorySessionBackend(
        _transcript(("op-0", "op-1"), (*first_history, *next_history)),
        {
            "op-0": _facts("op-0", evidence_ids=("evidence-0",)),
            "op-1": _facts("op-1", evidence_ids=("evidence-1",)),
        },
    )
    service = _service(backend, threshold=1_000, retain=4)

    first = await service.project(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        current_operation_id="op-1",
        profile=_profile(),
        maximum_projection_tokens=_profile().maximum_input_tokens,
    )

    assert first.compressed_now
    assert first.checkpoint is not None
    assert first.checkpoint.operation_ids == ("op-0",)
    assert first.estimated_tokens <= first.threshold_tokens

    current = _plain_exchange("op-current", "continue", "unused")
    backend.transcript = _transcript(
        ("op-0", "op-1", "op-current"),
        (*first_history, *next_history, *current),
    )
    backend.facts["op-current"] = _facts("op-current")

    advanced = await service.project(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        current_operation_id="op-current",
        profile=_profile(),
        maximum_projection_tokens=_profile().maximum_input_tokens,
    )
    reused = await service.project(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        current_operation_id="op-current",
        profile=_profile(),
        maximum_projection_tokens=_profile().maximum_input_tokens,
    )

    assert advanced.compressed_now
    assert advanced.checkpoint is not None
    assert advanced.checkpoint.operation_ids == ("op-0", "op-1")
    assert advanced.checkpoint.evidence_ids == ("evidence-0", "evidence-1")
    assert advanced.estimated_tokens <= advanced.threshold_tokens
    assert reused.checkpoint == advanced.checkpoint
    assert not reused.compressed_now
    assert backend.commits == 2


async def test_projection_reduces_recent_history_until_it_fits_explicit_residual() -> (
    None
):
    historical_ids = tuple(f"op-{index}" for index in range(6))
    operation_ids = (*historical_ids, "op-current")
    messages = tuple(
        message
        for operation_id in historical_ids
        for message in _plain_exchange(
            operation_id,
            f"question-{operation_id}-" + "x" * 3_000,
            f"answer-{operation_id}-" + "y" * 3_000,
        )
    ) + _plain_exchange("op-current", "continue", "unused")

    def backend() -> MemorySessionBackend:
        return MemorySessionBackend(
            _transcript(operation_ids, messages),
            {operation_id: _facts(operation_id) for operation_id in operation_ids},
        )

    one_recent_backend = backend()
    one_recent = await _service(
        one_recent_backend,
        threshold=3_000,
        retain=1,
    ).project(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        current_operation_id="op-current",
        profile=_profile(),
        maximum_projection_tokens=_profile().maximum_input_tokens,
    )
    residual = one_recent.estimated_tokens

    constrained_backend = backend()
    constrained = await _service(
        constrained_backend,
        threshold=_profile().maximum_input_tokens,
        retain=4,
    ).project(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        current_operation_id="op-current",
        profile=_profile(),
        maximum_projection_tokens=residual,
    )

    assert constrained.compressed_now
    assert constrained.estimated_tokens <= residual
    assert constrained.checkpoint is not None
    assert constrained.checkpoint.operation_ids == historical_ids[:-1]
    assert [block.kind for block in constrained.blocks] == [
        ContextKind.SESSION_SUMMARY,
        ContextKind.SESSION_RECENT,
    ]
    assert constrained.blocks[-1].provenance[0].reference_id == historical_ids[-1]
    assert constrained_backend.commits == 1


async def test_impossible_session_residual_fails_typed_without_mutating_history() -> (
    None
):
    transcript = _transcript(
        ("op-0", "op-1", "op-current"),
        (
            *_plain_exchange("op-0", "old zero", "answer zero"),
            *_plain_exchange("op-1", "old one", "answer one"),
            *_plain_exchange("op-current", "continue", "unused"),
        ),
    )
    facts = {
        operation_id: _facts(operation_id) for operation_id in transcript.operation_ids
    }
    backend = MemorySessionBackend(transcript, facts)

    with pytest.raises(RequiredContextOverflow) as raised:
        await _service(
            backend,
            threshold=_profile().maximum_input_tokens,
            retain=4,
        ).project(
            agent_id=AGENT_ID,
            session_id=SESSION_ID,
            current_operation_id="op-current",
            profile=_profile(),
            maximum_projection_tokens=0,
        )

    assert raised.value.available_tokens == 0
    assert raised.value.minimum_session_tokens > 0
    assert raised.value.projected_session_tokens > 0
    # The minimum safe *shape* is the all-history summary. For very short raw
    # messages that structured summary can legitimately cost more tokens.
    assert raised.value.projected_session_tokens != raised.value.minimum_session_tokens
    assert backend.commits == 0
    assert backend.checkpoint is None
    assert backend.transcript == transcript
    assert backend.facts == facts


async def test_small_history_rebinds_and_keeps_tool_exchange_indivisible() -> None:
    history = _tool_exchange("op-tool")
    current = _plain_exchange("op-current", "current request", "not yet used")
    transcript = _transcript(("op-tool", "op-current"), (*history, *current))
    backend = MemorySessionBackend(
        transcript,
        {
            "op-tool": _facts(
                "op-tool",
                sensitivity=ModelSensitivity.RESTRICTED,
            ),
            "op-current": _facts("op-current"),
        },
    )

    projection = await _service(backend, threshold=10_000).project(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        current_operation_id="op-current",
        profile=_profile(),
        maximum_projection_tokens=_profile().maximum_input_tokens,
    )

    assert not projection.compressed_now
    assert projection.checkpoint is None
    assert backend.commits == 0
    assert projection.historical_operation_ids == ("op-tool",)
    assert projection.sensitivity is ModelSensitivity.RESTRICTED
    assert len(projection.blocks) == 1
    block = projection.blocks[0]
    assert block.kind is ContextKind.SESSION_RECENT
    assert block.required
    assert block.trust is ContextTrust.UNTRUSTED_EXTERNAL
    assert [len(group.messages) for group in block.groups] == [1, 2, 1]
    assert all(message.operation_id == "op-current" for message in block.messages)
    assert all(message.turn_id is None for message in block.messages)
    assert all(not message.provider_metadata for message in block.messages)
    assert "current request" not in repr(projection.blocks)

    required_tokens = estimate_context_block_tokens(block)
    with pytest.raises(RequiredContextOverflow):
        select_context_blocks(
            projection.blocks,
            ModelProfile(
                id="mock:recent-overflow",
                context_window_tokens=required_tokens,
                max_output_tokens=1,
            ),
        )


async def test_compression_summarizes_only_old_prefix_and_preserves_references() -> (
    None
):
    op0 = _plain_exchange(
        "op-0",
        "Correction: revenue means net revenue after refunds.",
        "Understood." + "x" * 3_000,
    )
    op1 = _plain_exchange(
        "op-1",
        "inspect orders",
        "Orders inspected." + "y" * 3_000,
    )
    op2 = _plain_exchange("op-2", "latest question", "Latest answer.")
    current = _plain_exchange("op-current", "continue", "not projected")
    transcript = _transcript(
        ("op-0", "op-1", "op-2", "op-current"),
        (*op0, *op1, *op2, *current),
    )
    original_messages = transcript.messages
    backend = MemorySessionBackend(
        transcript,
        {
            "op-0": _facts(
                "op-0",
                evidence_ids=("evidence-0",),
                approval_ids=("approval-0",),
                resource_ids=("resource-orders",),
                objective="Reconcile the revenue definition",
                terminal_reason="completed",
                approval_state_facts=(
                    SessionApprovalStateFact(
                        approval_id="approval-0",
                        status=ApprovalStatus.APPROVED,
                    ),
                ),
                resource_scope_facts=(
                    SessionResourceScopeFact(
                        source_id="source-sales",
                        resource_id="resource-orders",
                        source_revision="schema:orders-v4",
                        resource_revision="revision:orders-v4",
                    ),
                ),
            ),
            "op-1": _facts(
                "op-1",
                evidence_ids=("evidence-1",),
                resource_ids=("resource-customers",),
                objective="Inspect current orders",
                terminal_reason="completed",
                resource_scope_facts=(
                    SessionResourceScopeFact(
                        source_id="source-sales",
                        resource_id="resource-customers",
                        source_revision="schema:orders-v4",
                        resource_revision="revision:customers-v2",
                    ),
                ),
            ),
            "op-2": _facts("op-2"),
            "op-current": _facts("op-current"),
        },
    )

    projection = await _service(backend, threshold=1_000, retain=1).project(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        current_operation_id="op-current",
        profile=_profile(),
        maximum_projection_tokens=_profile().maximum_input_tokens,
    )

    checkpoint = projection.checkpoint
    assert checkpoint is not None
    assert projection.compressed_now
    assert checkpoint.id == "session-compression-1"
    assert checkpoint.version == 1
    assert checkpoint.operation_ids == ("op-0", "op-1")
    assert checkpoint.through_position == 1
    assert checkpoint.evidence_ids == ("evidence-0", "evidence-1")
    assert checkpoint.approval_ids == ("approval-0",)
    assert checkpoint.resource_ids == (
        "resource-orders",
        "resource-customers",
    )
    assert checkpoint.source_fingerprint.startswith("sha256:")
    summary = json.loads(checkpoint.summary)
    assert summary["operation_ids"] == ["op-0", "op-1"]
    assert summary["user_corrections"] == [
        {
            "operation_id": "op-0",
            "text": "Correction: revenue means net revenue after refunds.",
        }
    ]
    assert summary["active_objective"] == {
        "operation_id": "op-1",
        "text": "Inspect current orders",
    }
    assert summary["operation_states"] == [
        {
            "operation_id": "op-0",
            "status": "succeeded",
            "terminal_reason": "completed",
        },
        {
            "operation_id": "op-1",
            "status": "succeeded",
            "terminal_reason": "completed",
        },
    ]
    assert summary["approvals"] == [
        {
            "approval_id": "approval-0",
            "operation_id": "op-0",
            "state": "approved",
        }
    ]
    assert summary["evidence"] == [
        {"evidence_id": "evidence-0", "operation_id": "op-0"},
        {"evidence_id": "evidence-1", "operation_id": "op-1"},
    ]
    assert summary["resource_scope"] == [
        {
            "operation_id": "op-0",
            "resource_id": "resource-orders",
            "resource_revision": "revision:orders-v4",
            "source_id": "source-sales",
            "source_revision": "schema:orders-v4",
        },
        {
            "operation_id": "op-1",
            "resource_id": "resource-customers",
            "resource_revision": "revision:customers-v2",
            "source_id": "source-sales",
            "source_revision": "schema:orders-v4",
        },
    ]
    assert summary["recent_intent"] == {
        "operation_id": "op-1",
        "text": "inspect orders",
    }
    assert [block.kind for block in projection.blocks] == [
        ContextKind.SESSION_SUMMARY,
        ContextKind.SESSION_RECENT,
    ]
    assert all(block.required for block in projection.blocks)
    assert all(
        block.trust is ContextTrust.UNTRUSTED_EXTERNAL for block in projection.blocks
    )
    assert projection.blocks[-1].provenance[0].reference_id == "op-2"
    assert backend.expected_versions == (0,)
    assert backend.transcript.messages == original_messages


async def test_matching_checkpoint_is_reused_without_rewriting_history() -> None:
    messages = (
        *_plain_exchange("op-0", "old zero" * 20, "answer zero" * 20),
        *_plain_exchange("op-1", "old one" * 20, "answer one" * 20),
        *_plain_exchange("op-2", "recent" * 20, "recent answer" * 20),
        *_plain_exchange("op-current", "continue", "unused"),
    )
    transcript = _transcript(
        ("op-0", "op-1", "op-2", "op-current"),
        messages,
    )
    backend = MemorySessionBackend(
        transcript,
        {
            operation_id: _facts(operation_id)
            for operation_id in transcript.operation_ids
        },
    )
    service = _service(backend, threshold=1, retain=1)

    first = await service.project(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        current_operation_id="op-current",
        profile=_profile(),
        maximum_projection_tokens=_profile().maximum_input_tokens,
    )
    second = await service.project(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        current_operation_id="op-current",
        profile=_profile(),
        maximum_projection_tokens=_profile().maximum_input_tokens,
    )

    assert first.checkpoint == second.checkpoint
    assert first.compressed_now
    assert not second.compressed_now
    assert backend.commits == 1


async def test_terminal_incomplete_tool_exchange_uses_required_factual_fallback() -> (
    None
):
    call = ToolCall(id="call-broken", name="catalog_search", arguments={})
    history = (
        _message(
            "op-broken",
            MessageRole.USER,
            "Correction: inspect the approved customer resource.",
        ),
        _message(
            "op-broken",
            MessageRole.ASSISTANT,
            tool_calls=(call,),
        ),
    )
    current = _plain_exchange("op-current", "continue", "unused")
    transcript = _transcript(
        ("op-broken", "op-current"),
        (*history, *current),
    )
    backend = MemorySessionBackend(
        transcript,
        {
            "op-broken": _facts(
                "op-broken",
                evidence_ids=("evidence-broken",),
                approval_ids=("approval-broken",),
                resource_ids=("resource-customers",),
                objective="Inspect the customer resource",
                terminal_reason="task_failed",
                approval_state_facts=(
                    SessionApprovalStateFact(
                        approval_id="approval-broken",
                        status=ApprovalStatus.APPROVED,
                    ),
                ),
                resource_scope_facts=(
                    SessionResourceScopeFact(
                        source_id="source-crm",
                        resource_id="resource-customers",
                        source_revision="schema:crm-v7",
                        resource_revision="revision:customers-v5",
                    ),
                ),
            ),
            "op-current": _facts("op-current"),
        },
    )
    backend.facts["op-broken"] = replace(
        backend.facts["op-broken"],
        status="failed",
    )

    projection = await _service(backend, threshold=10_000).project(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        current_operation_id="op-current",
        profile=_profile(),
        maximum_projection_tokens=_profile().maximum_input_tokens,
    )

    assert len(projection.blocks) == 1
    block = projection.blocks[0]
    assert block.kind is ContextKind.SESSION_RECENT
    assert block.required
    assert len(block.messages) == 1
    assert block.messages[0].role is MessageRole.USER
    assert not block.messages[0].tool_calls
    text = block.messages[0].content[0]
    assert isinstance(text, TextBlock)
    prefix = "UNTRUSTED_SESSION_OPERATION_FACTS="
    assert text.text.startswith(prefix)
    payload = json.loads(text.text.removeprefix(prefix))
    assert payload["status"] == "failed"
    assert payload["terminal_reason"] == "task_failed"
    assert payload["objective"] == "Inspect the customer resource"
    assert payload["approvals"][0]["state"] == "approved"
    assert payload["evidence_ids"] == ["evidence-broken"]
    assert payload["resource_scope"][0] == {
        "operation_id": "op-broken",
        "resource_id": "resource-customers",
        "resource_revision": "revision:customers-v5",
        "source_id": "source-crm",
        "source_revision": "schema:crm-v7",
    }
    assert payload["recent_intent"] == {
        "operation_id": "op-broken",
        "text": "Correction: inspect the approved customer resource.",
    }


async def test_required_summary_fails_closed_instead_of_dropping_facts() -> None:
    transcript = _transcript(
        ("op-0", "op-1", "op-2", "op-current"),
        (
            *_plain_exchange("op-0", "old zero", "answer zero"),
            *_plain_exchange("op-1", "old one", "answer one"),
            *_plain_exchange("op-2", "recent", "recent answer"),
            *_plain_exchange("op-current", "continue", "unused"),
        ),
    )
    backend = MemorySessionBackend(
        transcript,
        {
            operation_id: _facts(operation_id)
            for operation_id in transcript.operation_ids
        },
    )

    with pytest.raises(
        SessionCompressionIntegrityError,
        match="required summary references",
    ):
        await _service(
            backend,
            threshold=1,
            retain=1,
            max_summary_characters=256,
            max_excerpt_characters=32,
        ).project(
            agent_id=AGENT_ID,
            session_id=SESSION_ID,
            current_operation_id="op-current",
            profile=_profile(),
            maximum_projection_tokens=_profile().maximum_input_tokens,
        )

    assert backend.commits == 0


async def test_checkpoint_source_or_frontier_mismatch_fails_closed() -> None:
    messages = (
        *_plain_exchange("op-0", "old", "answer"),
        *_plain_exchange("op-1", "recent", "answer"),
        *_plain_exchange("op-current", "continue", "unused"),
    )
    transcript = _transcript(("op-0", "op-1", "op-current"), messages)
    facts = {
        operation_id: _facts(operation_id) for operation_id in transcript.operation_ids
    }
    backend = MemorySessionBackend(transcript, facts)
    service = _service(backend, threshold=1, retain=1)
    await service.project(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        current_operation_id="op-current",
        profile=_profile(),
        maximum_projection_tokens=_profile().maximum_input_tokens,
    )
    assert backend.checkpoint is not None

    changed_messages = (
        *_plain_exchange("op-0", "tampered", "answer"),
        *_plain_exchange("op-1", "recent", "answer"),
        *_plain_exchange("op-current", "continue", "unused"),
    )
    backend.transcript = _transcript(
        ("op-0", "op-1", "op-current"),
        changed_messages,
    )
    with pytest.raises(SessionCompressionIntegrityError, match="fingerprint"):
        await service.project(
            agent_id=AGENT_ID,
            session_id=SESSION_ID,
            current_operation_id="op-current",
            profile=_profile(),
            maximum_projection_tokens=_profile().maximum_input_tokens,
        )

    backend.transcript = transcript
    checkpoint = backend.checkpoint
    drifted_summary = json.loads(checkpoint.summary)
    drifted_summary["recent_intent"] = {
        "operation_id": "op-0",
        "text": "different intent",
    }
    backend.checkpoint = SessionCompressionCheckpoint(
        id=checkpoint.id,
        agent_id=checkpoint.agent_id,
        session_id=checkpoint.session_id,
        version=checkpoint.version,
        through_position=checkpoint.through_position,
        through_operation_id=checkpoint.through_operation_id,
        source_fingerprint=checkpoint.source_fingerprint,
        summary=canonical_json(drifted_summary),
        operation_ids=checkpoint.operation_ids,
        created_at=checkpoint.created_at,
        evidence_ids=checkpoint.evidence_ids,
        approval_ids=checkpoint.approval_ids,
        resource_ids=checkpoint.resource_ids,
    )
    with pytest.raises(SessionCompressionIntegrityError, match="summary drifted"):
        await service.project(
            agent_id=AGENT_ID,
            session_id=SESSION_ID,
            current_operation_id="op-current",
            profile=_profile(),
            maximum_projection_tokens=_profile().maximum_input_tokens,
        )

    backend.checkpoint = SessionCompressionCheckpoint(
        id=checkpoint.id,
        agent_id=checkpoint.agent_id,
        session_id=checkpoint.session_id,
        version=checkpoint.version,
        through_position=1,
        through_operation_id="op-1",
        source_fingerprint=checkpoint.source_fingerprint,
        summary=checkpoint.summary,
        operation_ids=("op-0", "op-1"),
        created_at=checkpoint.created_at,
        evidence_ids=checkpoint.evidence_ids,
        approval_ids=checkpoint.approval_ids,
        resource_ids=checkpoint.resource_ids,
    )
    with pytest.raises(SessionCompressionScopeError, match="frontier"):
        await service.project(
            agent_id=AGENT_ID,
            session_id=SESSION_ID,
            current_operation_id="op-1",
            profile=_profile(),
            maximum_projection_tokens=_profile().maximum_input_tokens,
        )
