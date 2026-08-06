"""Process-local state and bridges for Daita's terminal presentation."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..capabilities import ApprovalDecision, ApprovalRequest
from ..llm.models import MessageRole, ToolCall, ToolResultBlock
from ..llm.pricing import CostEstimate, format_cost_estimate
from ..loop.models import Transcript
from ..observation import AgentEvent, AgentEventKind
from ..terminal_transcript import (
    PresentationBlockId,
    TranscriptDocument,
    TranscriptFollowState,
    TranscriptProjection,
    TranscriptSelection,
    TranscriptViewport,
)
from .text import (
    MAX_RENDER_CHARACTERS as _MAX_RENDER_CHARACTERS,
    sanitize_terminal_text as _sanitize_terminal_text,
)
from .tool_view import (
    _CAPABILITY_LABELS,
    ToolCardState,
    _project_tool_details,
    _tool_result_error_code,
)

MAX_QUEUED_EVENTS = 4_096
MAX_EVENT_COUNTER = 999_999_999_999
MAX_TRACKED_CONTEXT_CONVERSATIONS = 64
_MAX_QUEUED_EVENTS = MAX_QUEUED_EVENTS
_MAX_EVENT_COUNTER = MAX_EVENT_COUNTER
_MAX_TRACKED_CONTEXT_CONVERSATIONS = MAX_TRACKED_CONTEXT_CONVERSATIONS


@dataclass(frozen=True, slots=True)
class TerminalStartupInfo:
    """Safe runtime facts rendered once when the focused shell opens."""

    version: str
    provider_label: str
    model_status: str
    agent_home: str
    source_count: int
    resource_count: int
    relationship_count: int
    read_capabilities: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("source_count", "resource_count", "relationship_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"startup {name} must be non-negative")
        for name in ("read_capabilities", "warnings"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) for value in values):
                raise TypeError(f"startup {name} must contain strings")
            object.__setattr__(self, name, values)


@dataclass(slots=True)
class TerminalBlock:
    """One disposable transcript block shown in the current process."""

    kind: str
    text: str
    tool_card: ToolCardState | None = None
    presentation_id: PresentationBlockId | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(slots=True)
class ApprovalPanelState:
    """One focused, bounded, process-local exact-approval projection."""

    tool_name: str
    capability_id: str
    arguments_text: str
    cursor_line: int = 0
    rendered_line_count: int = 1

    def move(self, amount: int) -> None:
        last_line = max(0, self.rendered_line_count - 1)
        self.cursor_line = min(last_line, max(0, self.cursor_line + amount))


class TerminalApprovalBridge:
    """Route the one existing approval callback to the active terminal surface."""

    def __init__(
        self,
        fallback: Callable[[ApprovalRequest], Awaitable[ApprovalDecision]],
    ) -> None:
        if not callable(fallback):
            raise TypeError("approval fallback must be callable")
        self._fallback = fallback
        self._presenter: (
            Callable[[ApprovalRequest], Awaitable[ApprovalDecision]] | None
        ) = None

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        presenter = self._presenter
        decision = await (
            self._fallback(request) if presenter is None else presenter(request)
        )
        if not isinstance(decision, ApprovalDecision):
            raise TypeError("approval presenter must return ApprovalDecision")
        return decision

    def install(
        self,
        presenter: Callable[[ApprovalRequest], Awaitable[ApprovalDecision]],
    ) -> Callable[[ApprovalRequest], Awaitable[ApprovalDecision]] | None:
        if not callable(presenter):
            raise TypeError("approval presenter must be callable")
        previous = self._presenter
        self._presenter = presenter
        return previous

    def restore(
        self,
        previous: Callable[[ApprovalRequest], Awaitable[ApprovalDecision]] | None,
    ) -> None:
        self._presenter = previous


class TerminalObserverBridge:
    """Best-effort enqueue-only bridge from execution into terminal state."""

    def __init__(self) -> None:
        self._events: deque[AgentEvent] = deque(maxlen=_MAX_QUEUED_EVENTS)

    def __call__(self, event: AgentEvent, /) -> None:
        try:
            self._events.append(event)
        except Exception:
            pass

    def drain(self) -> tuple[AgentEvent, ...]:
        pending: list[AgentEvent] = []
        while True:
            try:
                pending.append(self._events.popleft())
            except IndexError:
                return tuple(pending)


@dataclass(slots=True)
class TerminalViewState:
    """Bounded, process-local state for the focused terminal shell."""

    agent_label: str
    model_label: str
    source_summary: str
    conversation_id: str | None = None
    context_capacity_tokens: int | None = None
    conversation_context_tokens: int | None = None
    startup: TerminalStartupInfo | None = None
    blocks: list[TerminalBlock] = field(default_factory=list)
    running: bool = False
    notice: str = ""
    transient_selection_hint: str = ""
    steps: int = 0
    total_tokens: int = 0
    estimated_cost: str = "$0"
    active_task: asyncio.Task[Any] | None = None
    active_run_id: str | None = None
    run_status: str = "ready"
    run_duration_ms: int | None = None
    model_duration_ms: int | None = None
    run_input_tokens: int = 0
    run_output_tokens: int = 0
    animation_frame: int = 0
    tool_cards: dict[str, ToolCardState] = field(default_factory=dict)
    tool_history_run_id: str | None = None
    approval_panel: ApprovalPanelState | None = None
    transcript_document: TranscriptDocument = field(
        default_factory=TranscriptDocument,
        repr=False,
        compare=False,
    )
    transcript_viewport: TranscriptViewport = field(
        default_factory=TranscriptViewport,
        repr=False,
        compare=False,
    )
    transcript_selection: TranscriptSelection = field(
        default_factory=TranscriptSelection,
        repr=False,
        compare=False,
    )
    _transcript_render_generation: int = field(
        default=0,
        init=False,
        repr=False,
        compare=False,
    )
    _context_by_conversation: dict[str, int] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _partial_assistant_block_id: PresentationBlockId | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _partial_assistant_run_id: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _partial_model_call_index: int | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _partial_counted_unseen: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )
    _unrecorded_partial_run_id: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.context_capacity_tokens, "context_capacity_tokens"),
            (self.conversation_context_tokens, "conversation_context_tokens"),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.context_capacity_tokens == 0:
            raise ValueError("context_capacity_tokens must be positive")
        if (
            self.conversation_id is not None
            and self.conversation_context_tokens is not None
        ):
            self._remember_conversation_context(
                self.conversation_id,
                self.conversation_context_tokens,
            )

    def select_conversation(self, conversation_id: str | None) -> None:
        """Select one conversation and project its process-local context usage."""

        if conversation_id == self.conversation_id:
            return
        self.conversation_id = conversation_id
        self.conversation_context_tokens = (
            None
            if conversation_id is None
            else self._context_by_conversation.get(conversation_id)
        )

    @property
    def transcript_projection(self) -> TranscriptProjection | None:
        """Expose the viewport-owned cached projection to the renderer."""

        return self.transcript_viewport.projection

    @property
    def transcript_render_generation(self) -> int:
        """Return the disposable content generation used by the TUI render cache."""

        return self._transcript_render_generation

    def _mark_transcript_dirty(self) -> None:
        self._transcript_render_generation += 1

    def reconcile_transcript_selection(self) -> bool:
        """Clear a selection rather than copying text no longer projected."""

        had_state = self.transcript_selection.has_state
        survived = self.transcript_selection.reconcile(self.transcript_document)
        if had_state and not survived:
            self.transient_selection_hint = ""
            self.notice = (
                "Transcript selection cleared because visible content changed."
            )
        return survived

    def _remember_conversation_context(
        self,
        conversation_id: str,
        tokens: int,
    ) -> None:
        if conversation_id in self._context_by_conversation:
            del self._context_by_conversation[conversation_id]
        self._context_by_conversation[conversation_id] = tokens
        while len(self._context_by_conversation) > _MAX_TRACKED_CONTEXT_CONVERSATIONS:
            oldest = next(iter(self._context_by_conversation))
            del self._context_by_conversation[oldest]

    def append_plain(
        self,
        kind: str,
        value: object,
        *,
        maximum: int | None = _MAX_RENDER_CHARACTERS,
    ) -> None:
        safe = _sanitize_terminal_text(
            value,
            maximum=maximum,
            preserve_lines=True,
            fallback="",
        )
        if safe:
            self._append_block(TerminalBlock(kind, safe))

    def append_user(self, message: str) -> None:
        self.append_plain("user", message, maximum=None)

    def apply_model_text_delta(
        self,
        run_id: str,
        model_call_index: int,
        text: str,
    ) -> None:
        """Append one coalesced fragment batch to the sole disposable partial."""

        if not isinstance(run_id, str) or not run_id:
            return
        if (
            not isinstance(model_call_index, int)
            or isinstance(model_call_index, bool)
            or model_call_index < 1
        ):
            return
        safe = _sanitize_terminal_text(
            text,
            maximum=None,
            preserve_lines=True,
            fallback="",
        )
        if not safe:
            return
        block = self._partial_assistant_block()
        if block is None or (
            self._partial_assistant_run_id,
            self._partial_model_call_index,
        ) != (run_id, model_call_index):
            self._remove_partial_assistant()
            counted_unseen = (
                self.transcript_viewport.state is TranscriptFollowState.REVIEWING
            )
            block = TerminalBlock("assistant.partial", safe)
            self._append_block(block)
            self._partial_assistant_block_id = block.presentation_id
            self._partial_assistant_run_id = run_id
            self._partial_model_call_index = model_call_index
            self._partial_counted_unseen = counted_unseen
            self._unrecorded_partial_run_id = None
            return
        prior_text = block.text
        block.text += safe
        assert block.presentation_id is not None
        if self.transcript_document.text(block.presentation_id) == prior_text:
            self.transcript_document.replace(block.presentation_id, block.text)
            self.reconcile_transcript_selection()
        self._mark_transcript_dirty()

    def append_local(self, presentation: str, value: object) -> None:
        kind = (
            f"local.{presentation}"
            if presentation in {"status", "sources", "catalog", "settings"}
            else "local"
        )
        self.append_plain(kind, value)

    def apply_result(self, result: Any) -> None:
        previous_conversation = self.conversation_id
        candidate_conversation = getattr(result, "conversation_id", None)
        if isinstance(candidate_conversation, str) and candidate_conversation:
            self.select_conversation(candidate_conversation)
            if previous_conversation is None:
                self.append_plain(
                    "metadata",
                    f"Conversation  {candidate_conversation}",
                )

        result_run_id = getattr(result, "run_id", None)
        final_text = getattr(result, "final_text", None)
        if final_text is not None:
            safe_answer = _sanitize_terminal_text(
                final_text,
                maximum=None,
                preserve_lines=True,
                fallback="(empty response)",
            )
        else:
            kind = getattr(getattr(result, "kind", None), "value", None)
            reason = _sanitize_terminal_text(
                getattr(result, "reason", None),
                maximum=256,
                preserve_lines=False,
                fallback="failed",
            )
            safe_answer = f"{kind or 'failed'}: {reason}"
        if final_text is not None and self._reconcile_partial_assistant(
            result_run_id,
            safe_answer,
        ):
            pass
        else:
            if final_text is None and isinstance(result_run_id, str):
                self._remove_partial_assistant(
                    run_id=result_run_id,
                    unrecorded=True,
                )
            self._append_block(TerminalBlock("assistant", safe_answer))
        for receipt in tuple(getattr(result, "artifact_deliveries", ())):
            filename = getattr(receipt, "filename", None)
            saved_path = getattr(receipt, "saved_path", None)
            if not isinstance(filename, str) or not isinstance(saved_path, str):
                continue
            self.append_plain(
                "artifact.delivery",
                f"Saved {filename} to {saved_path}",
                maximum=None,
            )

        steps = getattr(result, "steps", 0)
        usage = getattr(result, "usage", None)
        total_tokens = getattr(usage, "total_tokens", 0)
        cost_estimate = getattr(usage, "cost_estimate", None)
        self.steps = steps if isinstance(steps, int) and steps >= 0 else 0
        self.total_tokens = (
            total_tokens if isinstance(total_tokens, int) and total_tokens >= 0 else 0
        )
        self.estimated_cost = _sanitize_terminal_text(
            (
                format_cost_estimate(cost_estimate)
                if isinstance(cost_estimate, CostEstimate)
                else "cost unavailable"
            ),
            maximum=96,
            preserve_lines=False,
            fallback="cost unavailable",
        )
        kind = getattr(getattr(result, "kind", None), "value", None)
        self.run_status = (
            "ready"
            if final_text is not None or kind == "completed"
            else _sanitize_terminal_text(
                kind,
                maximum=32,
                preserve_lines=False,
                fallback="failed",
            )
        )
        if (
            isinstance(result_run_id, str)
            and result_run_id
            and self.active_run_id == result_run_id
        ):
            result_kind = (
                kind
                if isinstance(kind, str) and kind
                else ("completed" if final_text is not None else "failed")
            )
            result_reason = _sanitize_terminal_text(
                getattr(result, "reason", None),
                maximum=128,
                preserve_lines=False,
                fallback=result_kind,
            )
            self._settle_run_cards(result_run_id, result_kind, result_reason)
            self.active_run_id = None
            self._mark_transcript_dirty()
        if (
            isinstance(result_run_id, str)
            and self._unrecorded_partial_run_id == result_run_id
        ):
            self.notice = (
                "Partial assistant output was interrupted and was not recorded."
            )
            self._unrecorded_partial_run_id = None
        else:
            self.notice = ""

    def hydrate_transcript(
        self,
        transcript: Transcript,
        *,
        run_id: str,
        initial: bool = False,
    ) -> None:
        """Hydrate and canonically reorder one completed run's tool cards."""

        if not isinstance(transcript, Transcript):
            raise TypeError("completed transcript must be a Transcript")
        if transcript.run.id != run_id:
            raise ValueError("completed transcript belongs to a different run")

        pairs = _completed_tool_pairs(transcript)
        canonical_ids = {call.id for call, _result in pairs}
        prior_cards = {
            call_id: card
            for call_id, card in self.tool_cards.items()
            if card.run_id == run_id
        }
        canonical_cards: list[ToolCardState] = []
        for call, result in pairs:
            card = prior_cards.get(call.id)
            capability_id = None if card is None else card.capability_id
            label = (
                _CAPABILITY_LABELS.get(capability_id or "")
                if capability_id is not None
                else None
            )
            if label is None:
                label = _sanitize_terminal_text(
                    call.name,
                    maximum=128,
                    preserve_lines=False,
                    fallback="Tool call",
                )
            if card is None:
                card = ToolCardState(
                    run_id=run_id,
                    call_id=call.id,
                    capability_id=capability_id,
                    label=label,
                )
            else:
                card.label = label

            if result is not None:
                already_hydrated = card.details is not None
                card.details = _project_tool_details(call, result)
                if result.is_error:
                    card.state = "failed"
                    card.error_code = _tool_result_error_code(result)
                    if not already_hydrated:
                        card.expanded = True
                else:
                    card.state = "succeeded"
                    card.error_code = None
                    if not already_hydrated:
                        card.expanded = False
            canonical_cards.append(card)

        if self.tool_history_run_id == run_id:
            if canonical_cards:
                for card in canonical_cards:
                    card.expanded = True
            else:
                self.tool_history_run_id = None

        target_indexes = [
            index
            for index, block in enumerate(self.blocks)
            if block.kind == "tool"
            and block.tool_card is not None
            and block.tool_card.run_id == run_id
        ]
        insertion_index = min(target_indexes, default=len(self.blocks))
        retained: list[TerminalBlock] = []
        retained_before_insertion = 0
        for index, block in enumerate(self.blocks):
            is_target = (
                block.kind == "tool"
                and block.tool_card is not None
                and block.tool_card.run_id == run_id
            )
            if is_target:
                continue
            if index < insertion_index:
                retained_before_insertion += 1
            retained.append(block)
        prior_presentation_ids = {
            block.tool_card.call_id: block.presentation_id
            for block in self.blocks
            if block.kind == "tool"
            and block.tool_card is not None
            and block.tool_card.run_id == run_id
        }
        canonical_blocks = [
            TerminalBlock(
                "tool",
                card.call_id,
                tool_card=card,
                presentation_id=prior_presentation_ids.get(card.call_id),
            )
            for card in canonical_cards
        ]
        delivery_failures = _artifact_delivery_messages(pairs)
        canonical_blocks.extend(
            TerminalBlock("artifact.delivery", message) for message in delivery_failures
        )
        self.blocks = [
            *retained[:retained_before_insertion],
            *canonical_blocks,
            *retained[retained_before_insertion:],
        ]

        for call_id, card in tuple(self.tool_cards.items()):
            if card.run_id == run_id and call_id not in canonical_ids:
                del self.tool_cards[call_id]
        for card in canonical_cards:
            self.tool_cards[card.call_id] = card
        self._sync_transcript_document(
            tuple(
                (
                    self.transcript_document.text(block.presentation_id)
                    if block.presentation_id is not None
                    and self.transcript_document.contains(block.presentation_id)
                    else block.text
                )
                for block in self.blocks
            ),
            count_as_unseen=not initial,
        )
        self._mark_transcript_dirty()

    def active_tool_activity(self) -> tuple[ToolCardState, int] | None:
        """Return the newest active card and bounded concurrent activity count."""

        active = tuple(
            card
            for card in self.tool_cards.values()
            if card.run_id == self.active_run_id
            and card.state in {"running", "approval"}
        )
        return None if not active else (active[-1], len(active))

    def toggle_tool_history(self) -> bool:
        """Show or hide every recorded card from the latest completed tool run."""

        if self.running:
            return False
        target_run_id = next(
            (
                card.run_id
                for block in reversed(self.blocks)
                if block.kind == "tool"
                and (card := block.tool_card) is not None
                and card.state in {"succeeded", "failed"}
            ),
            None,
        )
        if target_run_id is None:
            return False
        if self.tool_history_run_id == target_run_id:
            self.tool_history_run_id = None
            self.transcript_viewport.follow_latest()
        else:
            self.tool_history_run_id = target_run_id
            for card in self.tool_cards.values():
                if card.run_id == target_run_id:
                    card.expanded = True
            first_block = next(
                (
                    block
                    for block in self.blocks
                    if block.kind == "tool"
                    and block.tool_card is not None
                    and block.tool_card.run_id == target_run_id
                ),
                None,
            )
            if (
                first_block is not None
                and first_block.presentation_id is not None
                and self.transcript_document.contains(first_block.presentation_id)
                and self.transcript_document.text(first_block.presentation_id)
            ):
                self.transcript_viewport.review_position(
                    self.transcript_document,
                    self.transcript_document.position(
                        first_block.presentation_id,
                        0,
                    ),
                )
        self.transcript_selection.clear()
        self.transient_selection_hint = ""
        self._mark_transcript_dirty()
        return True

    def apply_event(self, event: AgentEvent) -> None:
        """Project one bounded observation event into disposable view state."""

        if not isinstance(event, AgentEvent):
            raise TypeError("terminal event must be AgentEvent")
        if event.kind is AgentEventKind.RUN_STARTED:
            self._remove_partial_assistant()
            self._unrecorded_partial_run_id = None
            if self.tool_history_run_id is not None:
                self.tool_history_run_id = None
                self.transcript_viewport.follow_latest()
                self._mark_transcript_dirty()
            self.active_run_id = event.run_id
            self.running = True
            self.run_status = "working"
            self.run_duration_ms = None
            self.model_duration_ms = None
            self.run_input_tokens = 0
            self.run_output_tokens = 0
            self.steps = 0
            self.total_tokens = 0
            self.estimated_cost = "cost unavailable"
            self.animation_frame = 0
            return
        if event.kind is AgentEventKind.MODEL_TEXT_DELTA:
            fields = _model_text_event_fields(event)
            if fields is not None:
                model_call_index, text = fields
                self.apply_model_text_delta(event.run_id, model_call_index, text)
            return
        if event.kind is AgentEventKind.MODEL_COMPLETED:
            completed_model_call_index = _event_counter(
                event.data.get("model_call_index")
            )
            if (
                completed_model_call_index is not None
                and self._partial_assistant_run_id == event.run_id
                and (
                    self._partial_model_call_index != completed_model_call_index
                    or event.data.get("has_tool_calls") is True
                    or event.data.get("has_text") is not True
                )
            ):
                self._remove_partial_assistant(run_id=event.run_id)
            self.model_duration_ms = _event_counter(event.data.get("duration_ms"))
            context_input_tokens = _event_counter(
                event.data.get("context_input_tokens")
            )
            if context_input_tokens is None:
                context_input_tokens = _event_counter(event.data.get("input_tokens"))
            if context_input_tokens is not None:
                self._remember_conversation_context(
                    event.conversation_id,
                    context_input_tokens,
                )
                if self.conversation_id in {None, event.conversation_id}:
                    self.conversation_context_tokens = context_input_tokens
            self.run_input_tokens = min(
                _MAX_EVENT_COUNTER,
                self.run_input_tokens
                + (_event_counter(event.data.get("input_tokens")) or 0),
            )
            self.run_output_tokens = min(
                _MAX_EVENT_COUNTER,
                self.run_output_tokens
                + (_event_counter(event.data.get("output_tokens")) or 0),
            )
            self.total_tokens = min(
                _MAX_EVENT_COUNTER,
                self.run_input_tokens + self.run_output_tokens,
            )
            if self.running:
                self.run_status = "working"
            return
        if event.kind is AgentEventKind.TOOL_STARTED:
            self._remove_partial_assistant(run_id=event.run_id)
            card = self._card_for_event(event)
            if card is None:
                return
            card.state = "running"
            card.duration_ms = None
            card.error_code = None
            self.run_status = "querying"
            self._mark_transcript_dirty()
            return
        if event.kind is AgentEventKind.APPROVAL_REQUESTED:
            card = self._card_for_event(event)
            if card is None:
                return
            card.state = "approval"
            card.approval_outcome = None
            card.expanded = True
            self.run_status = "approval"
            self._mark_transcript_dirty()
            return
        if event.kind is AgentEventKind.APPROVAL_DECIDED:
            card = self._card_for_event(event)
            if card is None:
                return
            outcome = _event_text(
                event.data.get("outcome"),
                maximum=32,
                fallback="failed",
            )
            card.approval_outcome = outcome
            if outcome == "approved":
                card.state = "running"
                card.expanded = False
                self.run_status = "querying"
            else:
                card.state = "failed"
                card.expanded = True
                card.error_code = (
                    "approval_denied" if outcome == "denied" else "approval_failed"
                )
                self.run_status = "working"
            self._mark_transcript_dirty()
            return
        if event.kind is AgentEventKind.TOOL_COMPLETED:
            card = self._card_for_event(event)
            if card is None:
                return
            card.duration_ms = _event_counter(event.data.get("duration_ms"))
            success = event.data.get("success")
            if success is True:
                card.state = "succeeded"
                card.error_code = None
                card.expanded = False
            else:
                card.state = "failed"
                card.expanded = True
                card.error_code = _event_text(
                    event.data.get("error_code"),
                    maximum=128,
                    fallback="tool_failed",
                )
            self.run_status = "working"
            self._mark_transcript_dirty()
            return
        if event.kind is AgentEventKind.RUN_COMPLETED:
            self.run_duration_ms = _event_counter(event.data.get("duration_ms"))
            self.steps = _event_counter(event.data.get("steps")) or 0
            self.run_input_tokens = _event_counter(event.data.get("input_tokens")) or 0
            self.run_output_tokens = (
                _event_counter(event.data.get("output_tokens")) or 0
            )
            self.total_tokens = _event_counter(event.data.get("total_tokens")) or 0
            self.estimated_cost = _event_text(
                event.data.get("cost_display"),
                maximum=96,
                fallback="cost unavailable",
            )
            exit_kind = _event_text(
                event.data.get("exit_kind"),
                maximum=32,
                fallback="failed",
            )
            reason = _event_text(
                event.data.get("reason"),
                maximum=128,
                fallback=exit_kind,
            )
            if exit_kind != "completed":
                self._remove_partial_assistant(
                    run_id=event.run_id,
                    unrecorded=True,
                )
            self._settle_run_cards(event.run_id, exit_kind, reason)
            if self.active_run_id == event.run_id:
                self.active_run_id = None
            self.running = False
            self.run_status = "ready" if exit_kind == "completed" else exit_kind
            self._mark_transcript_dirty()

    def settle_cancelled_run(self) -> bool:
        removed_partial = self._remove_partial_assistant(
            run_id=self.active_run_id,
            unrecorded=True,
        )
        had_unrecorded_partial = (
            removed_partial or self._unrecorded_partial_run_id is not None
        )
        run_id = self.active_run_id
        if run_id is not None:
            self._settle_run_cards(run_id, "interrupted", "cancelled")
            self.active_run_id = None
            self.run_status = "interrupted"
        self.running = False
        return had_unrecorded_partial

    def _partial_assistant_block(self) -> TerminalBlock | None:
        block_id = self._partial_assistant_block_id
        if block_id is None:
            return None
        for block in self.blocks:
            if block.presentation_id == block_id and block.kind == "assistant.partial":
                return block
        self._clear_partial_assistant_identity()
        return None

    def _remove_partial_assistant(
        self,
        *,
        run_id: str | None = None,
        unrecorded: bool = False,
    ) -> bool:
        if run_id is not None and self._partial_assistant_run_id != run_id:
            return False
        block = self._partial_assistant_block()
        partial_run_id = self._partial_assistant_run_id
        if block is None:
            return False
        self.blocks.remove(block)
        if block.presentation_id is not None and self.transcript_document.contains(
            block.presentation_id
        ):
            self.transcript_document.remove(block.presentation_id)
            self.reconcile_transcript_selection()
        if self._partial_counted_unseen:
            self.transcript_viewport.record_removed()
        self._clear_partial_assistant_identity()
        if unrecorded:
            self._unrecorded_partial_run_id = partial_run_id
        self._mark_transcript_dirty()
        return True

    def _reconcile_partial_assistant(
        self,
        run_id: object,
        finalized_text: str,
    ) -> bool:
        if not isinstance(run_id, str) or self._partial_assistant_run_id != run_id:
            return False
        block = self._partial_assistant_block()
        if block is None:
            return False
        prior_text = block.text
        block.kind = "assistant"
        block.text = finalized_text
        assert block.presentation_id is not None
        if self.transcript_document.text(block.presentation_id) == prior_text:
            self.transcript_document.replace(block.presentation_id, finalized_text)
            self.reconcile_transcript_selection()
        self._clear_partial_assistant_identity()
        self._unrecorded_partial_run_id = None
        self._mark_transcript_dirty()
        return True

    def _clear_partial_assistant_identity(self) -> None:
        self._partial_assistant_block_id = None
        self._partial_assistant_run_id = None
        self._partial_model_call_index = None
        self._partial_counted_unseen = False

    def _card_for_event(self, event: AgentEvent) -> ToolCardState | None:
        call_id = _event_text(
            event.data.get("call_id"),
            maximum=256,
            fallback="",
        )
        if not call_id:
            return None
        capability_id = _optional_event_text(
            event.data.get("capability_id"),
            maximum=256,
        )
        tool_name = _event_text(
            event.data.get("tool_name"),
            maximum=128,
            fallback="Tool call",
        )
        label = _CAPABILITY_LABELS.get(capability_id or "", tool_name)
        label = _sanitize_terminal_text(
            label,
            maximum=128,
            preserve_lines=False,
            fallback="Tool call",
        )
        card = self.tool_cards.get(call_id)
        if card is None or card.run_id != event.run_id:
            card = ToolCardState(
                run_id=event.run_id,
                call_id=call_id,
                capability_id=capability_id,
                label=label,
            )
            self.tool_cards[call_id] = card
            self._append_block(
                TerminalBlock("tool", call_id, tool_card=card),
                count_as_unseen=False,
            )
        else:
            if capability_id is not None:
                card.capability_id = capability_id
            if capability_id is not None or card.label == "Tool call":
                card.label = label
        return card

    def _settle_run_cards(
        self,
        run_id: str,
        exit_kind: str,
        reason: str,
    ) -> None:
        changed = False
        for card in self.tool_cards.values():
            if card.run_id != run_id or card.state not in {
                "queued",
                "running",
                "approval",
            }:
                continue
            card.state = "failed"
            card.expanded = True
            card.error_code = (
                "cancelled"
                if exit_kind == "interrupted" or reason == "cancelled"
                else "observation_incomplete"
            )
            changed = True
        if changed:
            self._mark_transcript_dirty()

    def _append_block(
        self,
        block: TerminalBlock,
        *,
        count_as_unseen: bool = True,
    ) -> None:
        """Admit one block to both disposable views under one stable identity."""

        snapshot = self.transcript_document.append(block.text)
        block.presentation_id = snapshot.id
        self.blocks.append(block)
        if count_as_unseen:
            self.transcript_viewport.record_appended()
        self._mark_transcript_dirty()

    def _sync_transcript_document(
        self,
        selectable_texts: Sequence[str],
        *,
        width: int | None = None,
        count_as_unseen: bool = True,
    ) -> None:
        """Reconcile renderer-owned selectable projections without persisting them."""

        if len(selectable_texts) != len(self.blocks):
            raise ValueError("each terminal block requires one selectable projection")
        prior_ids = set(self.transcript_document.presentation_ids)
        current_ids: list[PresentationBlockId] = []
        seen: set[PresentationBlockId] = set()
        for block, selectable_text in zip(
            self.blocks,
            selectable_texts,
            strict=True,
        ):
            block_id = block.presentation_id
            if (
                block_id is None
                or block_id in seen
                or not self.transcript_document.contains(block_id)
            ):
                block_id = self.transcript_document.append(selectable_text).id
                block.presentation_id = block_id
            else:
                self.transcript_document.replace(block_id, selectable_text)
            current_ids.append(block_id)
            seen.add(block_id)
        for block_id in tuple(self.transcript_document.presentation_ids):
            if block_id not in seen:
                self.transcript_document.remove(block_id)
        self.transcript_document.reorder(tuple(current_ids))
        self.reconcile_transcript_selection()
        if count_as_unseen:
            self.transcript_viewport.record_appended(
                sum(block_id not in prior_ids for block_id in current_ids)
            )
        if width is not None:
            self.transcript_viewport.projection_for(
                self.transcript_document,
                width=width,
            )


def _completed_tool_pairs(
    transcript: Transcript,
) -> tuple[tuple[ToolCall, ToolResultBlock | None], ...]:
    calls: list[ToolCall] = []
    call_ids: set[str] = set()
    results: dict[str, ToolResultBlock] = {}
    for message in transcript.messages:
        if message.role is MessageRole.ASSISTANT:
            for call in message.tool_calls:
                if call.id in call_ids:
                    raise ValueError("completed transcript repeats a tool call ID")
                call_ids.add(call.id)
                calls.append(call)
        elif message.role is MessageRole.TOOL:
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    raise TypeError("tool transcript message contains non-tool content")
                if block.call_id in results:
                    raise ValueError("completed transcript repeats a tool result ID")
                results[block.call_id] = block
    if not set(results).issubset(call_ids):
        raise ValueError("completed transcript contains an unmatched tool result")
    return tuple((call, results.get(call.id)) for call in calls)


def _artifact_delivery_messages(
    pairs: tuple[tuple[ToolCall, ToolResultBlock | None], ...],
) -> tuple[str, ...]:
    messages: list[str] = []
    created_artifact_ids: list[str] = []
    delivery_attempts: set[str] = set()
    for call, result in pairs:
        if result is not None and not result.is_error:
            artifact = result.output.get("artifact")
            if isinstance(artifact, Mapping):
                artifact_id = artifact.get("artifact_id")
                if (
                    isinstance(artifact_id, str)
                    and artifact_id not in created_artifact_ids
                ):
                    created_artifact_ids.append(artifact_id)
        if call.name != "artifact_save_local" or result is None:
            continue
        artifact_id = call.arguments.get("artifact_id")
        if isinstance(artifact_id, str):
            delivery_attempts.add(artifact_id)
        if not result.is_error:
            continue
        error = result.output.get("error")
        if not isinstance(artifact_id, str) or not isinstance(error, Mapping):
            continue
        code = _sanitize_terminal_text(
            error.get("code"),
            maximum=128,
            preserve_lines=False,
            fallback="artifact_delivery_failed",
        )
        detail = _sanitize_terminal_text(
            error.get("message"),
            maximum=512,
            preserve_lines=False,
            fallback="The artifact was not saved locally.",
        )
        safe_id = _sanitize_terminal_text(
            artifact_id,
            maximum=64,
            preserve_lines=False,
            fallback="the internal artifact",
        )
        messages.append(
            f"Artifact {safe_id} remains available; local delivery failed: "
            f"{code}: {detail}"
        )
    for artifact_id in created_artifact_ids:
        if artifact_id in delivery_attempts:
            continue
        safe_id = _sanitize_terminal_text(
            artifact_id,
            maximum=64,
            preserve_lines=False,
            fallback="The internal artifact",
        )
        messages.append(
            f"Artifact {safe_id} was created internally but was not saved locally; "
            "no delivery completed."
        )
    return tuple(messages)


def _event_counter(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return min(value, _MAX_EVENT_COUNTER)


def _model_text_event_fields(event: AgentEvent) -> tuple[int, str] | None:
    if event.kind is not AgentEventKind.MODEL_TEXT_DELTA:
        return None
    model_call_index = _event_counter(event.data.get("model_call_index"))
    text = event.data.get("text")
    if (
        model_call_index is None
        or model_call_index < 1
        or not isinstance(text, str)
        or not text
    ):
        return None
    return model_call_index, text


def _event_text(value: object, *, maximum: int, fallback: str) -> str:
    return _sanitize_terminal_text(
        value,
        maximum=maximum,
        preserve_lines=False,
        fallback=fallback,
    )


def _optional_event_text(value: object, *, maximum: int) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return _event_text(value, maximum=maximum, fallback="") or None
