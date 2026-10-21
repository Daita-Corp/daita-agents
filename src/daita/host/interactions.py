"""Hold questions for a person as host-side records any attached client can answer."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from uuid import uuid4

from ..capabilities import ApprovalDecision, ApprovalHandler, ApprovalRequest
from ..hosting.sources import SourceEditConfirmationHandler, SourceEditPreview
from .events import EventBus
from .protocol import EventType, InteractionResolvedError

DEFAULT_EXPIRY = timedelta(hours=24)


class InteractionKind(str, Enum):
    TOOL_APPROVAL = "tool_approval"
    ROUTINE_CONFIRMATION = "routine_confirmation"
    SOURCE_EDIT_CONFIRMATION = "source_edit_confirmation"
    MODEL_LOGIN = "model_login"


@dataclass(slots=True)
class Interaction:
    id: str
    agent: str
    kind: InteractionKind
    request: dict[str, object]
    created_at: datetime
    expires_at: datetime | None
    run_ref: str | None
    state: str = "pending"
    answer: dict[str, object] | None = None
    future: asyncio.Future[dict[str, object]] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    owner: asyncio.Task[object] | None = None  # cancelled on dismiss

    def view(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "state": self.state,
            "run_ref": self.run_ref,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "request": self.request,
            "answer": self.answer,
        }


class InteractionHub:
    """First valid answer wins; expiry and dismissal interrupt, never deny."""

    def __init__(
        self,
        bus: EventBus,
        *,
        expiry: timedelta = DEFAULT_EXPIRY,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        on_open: Callable[[str, InteractionKind], None] | None = None,
    ) -> None:
        self._bus = bus
        self._expiry = expiry
        self._clock = clock
        # Told of every new question, for example to notify when nobody watches.
        self._on_open = on_open
        self._open: dict[str, dict[str, Interaction]] = {}

    async def ask(
        self,
        agent: str,
        kind: InteractionKind,
        request: dict[str, object],
        *,
        run_ref: str | None = None,
    ) -> dict[str, object]:
        """Wait for the first valid answer; raise CancelledError on expiry/dismiss."""

        interaction = self._start(agent, kind, request, run_ref, self._expiry)
        try:
            async with asyncio.timeout(self._expiry.total_seconds()):
                return await interaction.future
        except TimeoutError:
            self._finish(interaction, "expired", op="expired")
            raise asyncio.CancelledError from None
        finally:
            # The waiting call itself was cancelled, for example a cancelled run.
            self._finish(interaction, "cancelled")

    def show(
        self,
        agent: str,
        kind: InteractionKind,
        request: dict[str, object],
        *,
        owner: asyncio.Task[object],
    ) -> Interaction:
        """Show an answerless prompt; dismissing it cancels its owner task."""

        interaction = self._start(agent, kind, request, None, None)
        interaction.owner = owner
        return interaction

    def complete(self, interaction: Interaction) -> None:
        self._finish(interaction, "completed")

    def update(self, interaction: Interaction, request: dict[str, object]) -> None:
        if interaction.state == "pending":
            interaction.request = {**interaction.request, **request}
            self._publish(interaction, "updated")

    def answer(
        self, agent: str, interaction_id: str, payload: Mapping[str, object]
    ) -> dict[str, object]:
        interaction = self._pending(agent, interaction_id)
        answer = _validated_answer(interaction.kind, payload)
        interaction.answer = answer
        interaction.future.set_result(answer)
        self._finish(interaction, "answered")
        return interaction.view()

    def dismiss(self, agent: str, interaction_id: str) -> dict[str, object]:
        interaction = self._pending(agent, interaction_id)
        if interaction.owner is not None:
            interaction.owner.cancel()
        else:
            interaction.future.cancel()
        self._finish(interaction, "dismissed")
        return interaction.view()

    def pending(self, agent: str) -> tuple[dict[str, object], ...]:
        return tuple(item.view() for item in self._open.get(agent, {}).values())

    def close(self) -> None:
        """Interrupt every pending interaction, for example at shutdown."""

        for agent in tuple(self._open):
            for interaction_id in tuple(self._open.get(agent, {})):
                self.dismiss(agent, interaction_id)

    def _start(
        self,
        agent: str,
        kind: InteractionKind,
        request: dict[str, object],
        run_ref: str | None,
        expiry: timedelta | None,
    ) -> Interaction:
        now = self._clock()
        interaction = Interaction(
            id=f"interaction-{uuid4().hex}",
            agent=agent,
            kind=kind,
            request=request,
            created_at=now,
            expires_at=None if expiry is None else now + expiry,
            run_ref=run_ref,
        )
        self._open.setdefault(agent, {})[interaction.id] = interaction
        self._publish(interaction, "opened")
        if self._on_open is not None:
            self._on_open(agent, kind)
        return interaction

    def _pending(self, agent: str, interaction_id: str) -> Interaction:
        interaction = self._open.get(agent, {}).get(interaction_id)
        if interaction is None:
            raise InteractionResolvedError(
                f"interaction is not pending for this agent: {interaction_id}"
            )
        return interaction

    def _finish(
        self, interaction: Interaction, state: str, op: str = "resolved"
    ) -> None:
        if interaction.state != "pending":
            return
        interaction.state = state
        self._open.get(interaction.agent, {}).pop(interaction.id, None)
        self._publish(interaction, op)

    def _publish(self, interaction: Interaction, op: str) -> None:
        self._bus.publish(
            interaction.agent,
            EventType.INTERACTION,
            {"op": op, "interaction": interaction.view()},
        )


def approval_handler(
    hub: InteractionHub,
    agent: str,
    kind: InteractionKind,
    *,
    run_ref: Callable[[], str | None] = lambda: None,
    waiting: Callable[[str | None, bool], None] = lambda _ref, _waiting: None,
) -> ApprovalHandler:
    """Answer an engine approval callback through one host interaction."""

    from ..tui.projection import approval_review_document

    async def handle(request: ApprovalRequest) -> ApprovalDecision:
        document, reviewable = approval_review_document(
            tool_name=request.tool_name,
            capability_id=request.capability_id,
            arguments_text=request.render_arguments_for_review(),
            reason=request.reason,
        )
        ref = run_ref()
        waiting(ref, True)
        try:
            answer = await hub.ask(
                agent,
                kind,
                {
                    "approval": request,
                    "review_document": document,
                    "reviewable": reviewable,
                },
                run_ref=ref,
            )
        finally:
            waiting(ref, False)
        return ApprovalDecision(answer["decision"])

    return handle


def source_edit_handler(
    hub: InteractionHub, agent: str
) -> SourceEditConfirmationHandler:
    async def handle(preview: SourceEditPreview) -> bool:
        answer = await hub.ask(
            agent, InteractionKind.SOURCE_EDIT_CONFIRMATION, {"preview": preview}
        )
        return answer["accept"] is True

    return handle


def _validated_answer(
    kind: InteractionKind, payload: Mapping[str, object]
) -> dict[str, object]:
    answer = dict(payload)
    if kind in (InteractionKind.TOOL_APPROVAL, InteractionKind.ROUTINE_CONFIRMATION):
        if set(answer) != {"decision"} or answer["decision"] not in ("approve", "deny"):
            raise ValueError('answer must be {"decision": "approve" | "deny"}')
    elif kind is InteractionKind.SOURCE_EDIT_CONFIRMATION:
        if set(answer) != {"accept"} or type(answer["accept"]) is not bool:
            raise ValueError('answer must be {"accept": true | false}')
    else:
        raise ValueError("this interaction is display only; dismiss it to cancel")
    return answer


__all__ = [
    "DEFAULT_EXPIRY",
    "Interaction",
    "InteractionHub",
    "InteractionKind",
    "approval_handler",
    "source_edit_handler",
]
