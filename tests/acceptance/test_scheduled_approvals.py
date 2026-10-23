"""Scheduled runs ask a person at the time before they write (invariant 5)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from daita import (
    Agent,
    IntervalSchedule,
    OnceSchedule,
    RoutineIntent,
    WorkColumn,
    WorkKind,
    connect,
)
from daita.capabilities import EffectOutcome
from daita.distribution import OutcomeState
from daita.llm.models import (
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.routines.models import RoutineSchedule
from tests.support.host import (
    StartHost,
    bounded,
    pending_interaction,
    short_root,
    value,
)
from tests.support.native_writes import (
    NOW,
    ScriptedResearchModel,
    create_fixture,
    response,
)
from tests.support.workspace import workspace_for

pytestmark = [pytest.mark.acceptance, pytest.mark.integration]

NAME = "native-acceptance"
TITLE = "Daily totals"
INSTRUCTION = "Upsert today's totals into reporting.daily."
USAGE = ModelUsage(
    input_tokens=100,
    output_tokens=10,
    cost_estimate=CostEstimate.complete(Decimal("0")),
)
Step = ModelResponse | Callable[[ModelRequest], ModelResponse]


class Sent:
    """A notification sender that records instead of showing anything."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def __call__(self, title: str, text: str) -> None:
        self.items.append((title, text))

    def texts(self) -> list[str]:
        return [text for _title, text in self.items]


def _measured(step: Step) -> Step:
    """Scheduled runs need measured usage; wrap a scripted step to report it."""

    if isinstance(step, ModelResponse):
        return replace(step, usage=USAGE)
    return lambda request: replace(step(request), usage=USAGE)


def _tool_result(request: ModelRequest, call_id: str) -> ToolResultBlock:
    return next(
        block
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == call_id
    )


def _write_script(batch: Mapping[str, object]) -> list[Step]:
    """Preview, then apply, the fixture's one-row upsert; report what happened."""

    def apply(request: ModelRequest) -> ModelResponse:
        preview = _tool_result(request, "preview")
        assert not preview.is_error, preview.output
        data = preview.output["data"]
        assert isinstance(data, Mapping)
        return response(
            ToolCall(
                "write",
                "data_upsert_rows",
                {
                    **batch,
                    "mode": "apply",
                    "preview_fingerprint": data["preview_fingerprint"],
                },
            )
        )

    def report(request: ModelRequest) -> ModelResponse:
        write = _tool_result(request, "write")
        if not write.is_error:
            return response(text="Saved today's totals.")
        error = write.output["error"]
        assert isinstance(error, Mapping)
        return response(text=f"The write was refused ({error['code']}).")

    steps: list[Step] = [
        response(
            ToolCall("load", "toolbox_load", {"tool_names": ["data_upsert_rows"]})
        ),
        response(ToolCall("preview", "data_upsert_rows", {**batch, "mode": "preview"})),
        apply,
        report,
    ]
    return [_measured(step) for step in steps]


def _intent(source_id: str, schedule: RoutineSchedule) -> RoutineIntent:
    """A routine is a title, a prompt and a schedule: no grant is authored."""

    return RoutineIntent(
        title=TITLE, prompt=INSTRUCTION, schedule=schedule, sources=(source_id,)
    )


def _inserts(db: Any) -> int:
    return len(
        [item for item in db.log if item[0] == "fetch" and item[1].startswith("INSERT")]
    )


async def _scheduled_write(
    start_host: StartHost,
    monkeypatch: pytest.MonkeyPatch,
    **host_options: object,
) -> tuple[Any, ScriptedResearchModel, Any, Sent, str]:
    """A host whose agent's routine fires in seconds while nobody watches."""

    root = short_root()
    agent, provider, db, _alpha, _binding, _resource, _scope, batch, _clock, _ok = (
        await create_fixture(root, monkeypatch)
    )
    await agent.close()
    sent = Sent()
    harness = await start_host(
        model=provider, root=root, notify=sent, notify_interval=0.0, **host_options
    )
    remote = await connect(NAME, cwd=harness.work(), root=harness.root)
    try:
        provider.replace_script([_measured(response(text="Noted."))])
        origin = await remote.run("Keep today's totals in the companies table.")
        assert origin.conversation_id is not None
        provider.replace_script(_write_script(batch))
        routine = await remote.create_routine(
            _intent(
                str(batch["source_id"]),
                OnceSchedule(datetime.now(UTC) + timedelta(seconds=2)),
            ),
            conversation_id=origin.conversation_id,
        )
    finally:
        # Nobody watches from here on, so the question also notifies.
        await remote.close()
    return harness, provider, db, sent, routine.routine_id


async def _result(harness: Any, routine_id: str) -> tuple[Any, Any, Any]:
    """Wait for the occurrence's card on the work board and read its outcome."""

    remote = await connect(NAME, cwd=harness.work(), root=harness.root)

    async def probe() -> object:
        board = await remote.work_board()
        return next(
            (
                card
                for column in (WorkColumn.DONE, WorkColumn.NEEDS_YOU)
                for card in board.columns.get(column, ())
                if card.kind is WorkKind.ROUTINE_RUN
                and card.routine_id == routine_id
                and card.delivery_id is not None
            ),
            None,
        )

    try:
        card = await bounded(probe, timeout=30)
        inspection = await remote.inspect_delivery(card.delivery_id)
        assert inspection is not None
        return remote, card, inspection.delivery.outcome
    except BaseException:
        await remote.close()
        raise


async def test_a_scheduled_write_asks_notifies_and_commits_once_after_approval(
    start_host: StartHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness, provider, db, sent, routine_id = await _scheduled_write(
        start_host, monkeypatch
    )
    interaction = await pending_interaction(harness.http, NAME, timeout=30)
    assert interaction["kind"] == "tool_approval"
    approval = interaction["request"]["approval"]
    assert approval.capability_id == "data.upsert_rows"
    assert approval.requester == TITLE
    assert f"Asked by: {TITLE}" in interaction["request"]["review_document"]
    await bounded(sent.texts)
    # The notification names only the agent, never the routine or its data.
    assert sent.texts() == [f"{NAME} needs your approval"]
    assert not db.rows and _inserts(db) == 0

    answered = await harness.http.post(
        f"/v1/agents/{NAME}/interactions/{interaction['id']}",
        json={"decision": "approve"},
    )
    value(answered)
    remote, card, outcome = await _result(harness, routine_id)
    try:
        assert card.column is WorkColumn.DONE
        assert outcome.conclusion_state is OutcomeState.SUCCEEDED
        assert "Saved today's totals." in outcome.conclusion_preview
        assert db.rows["new.test"]["name"] == "New Co"
        assert _inserts(db) == 1
        (receipt,) = await remote.list_effects()
        assert receipt.outcome is EffectOutcome.SUCCEEDED
        assert receipt.routine_id == routine_id
        assert receipt.capability_grant_digest is None
        assert outcome.effect_receipt_ids == (receipt.receipt_id,)
        assert not provider.steps
    finally:
        await remote.close()


async def test_a_denied_scheduled_write_finishes_with_the_refusal_and_writes_nothing(
    start_host: StartHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness, provider, db, _sent, routine_id = await _scheduled_write(
        start_host, monkeypatch
    )
    interaction = await pending_interaction(harness.http, NAME, timeout=30)
    value(
        await harness.http.post(
            f"/v1/agents/{NAME}/interactions/{interaction['id']}",
            json={"decision": "deny"},
        )
    )
    remote, card, outcome = await _result(harness, routine_id)
    try:
        assert outcome.conclusion_state is OutcomeState.SUCCEEDED
        assert "refused (approval_denied)" in outcome.conclusion_preview
        assert not db.rows and _inserts(db) == 0
        assert await remote.list_effects() == ()
        assert not provider.steps
    finally:
        await remote.close()


async def test_an_unanswered_scheduled_write_expires_into_needs_you(
    start_host: StartHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness, provider, db, _sent, routine_id = await _scheduled_write(
        start_host, monkeypatch, interaction_expiry=timedelta(seconds=1)
    )
    await pending_interaction(harness.http, NAME, timeout=30)
    remote, card, outcome = await _result(harness, routine_id)
    try:
        # Expiry interrupts the run; it never counts as a denial.
        assert outcome.conclusion_state is OutcomeState.FAILED
        assert card.column is WorkColumn.NEEDS_YOU
        assert value(await harness.http.get(f"/v1/agents/{NAME}/interactions")) == ()
        assert not db.rows and _inserts(db) == 0
        assert await remote.list_effects() == ()
        # The model was never asked to continue after the expired question.
        assert len(provider.steps) == 1
    finally:
        await remote.close()


async def test_library_mode_without_an_approval_handler_refuses_a_scheduled_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        agent,
        provider,
        db,
        _alpha,
        _binding,
        _resource,
        _scope,
        batch,
        clock,
        approvals,
    ) = await create_fixture(tmp_path, monkeypatch)
    await agent.close()
    agent = await Agent.open(
        NAME,
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
        clock=lambda: clock[0],
    )
    try:
        provider.replace_script([_measured(response(text="Noted."))])
        origin = await agent.run("Keep today's totals in the companies table.")
        assert origin.conversation_id is not None
        provider.replace_script(_write_script(batch))
        routine = await agent.create_routine(
            _intent(
                str(batch["source_id"]),
                IntervalSchedule(3_600, NOW + timedelta(hours=1)),
            ),
            conversation_id=origin.conversation_id,
        )
        await agent.run_routine_now(
            routine.routine_id, expected_revision=routine.revision
        )

        async def delivered() -> object:
            deliveries = await agent.list_deliveries(
                conversation_id=origin.conversation_id
            )
            return deliveries[0] if deliveries else None

        delivery = await bounded(delivered, timeout=30)
        assert "refused (approval_required)" in delivery.conclusion_preview
        assert not db.rows and _inserts(db) == 0
        assert await agent.list_effects() == ()
        assert approvals == []
    finally:
        await agent.close()
