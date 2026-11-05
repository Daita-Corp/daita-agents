"""Scheduled runs ask a person at the time before they write (invariant 5)."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
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
from daita.host.paths import host_paths
from daita.host.supervise import host_status
from daita.routines.models import (
    ROUTINE_CLAIM_LEASE_SECONDS,
    RoutineSchedule,
    RoutineState,
)
from tests.support.host import (
    StartHost,
    bounded,
    client,
    pending_interaction,
    routine_approval,
    short_root,
    value,
    work_folder,
)
from tests.support.native_writes import (
    NOW,
    ScriptedResearchModel,
    create_fixture,
    measured_step,
    patch_fake_postgresql,
    response,
    scheduled_write_script,
)
from tests.support.workspace import workspace_for
from tests.support.write_database import Database

pytestmark = [pytest.mark.acceptance, pytest.mark.integration]

REPOSITORY = Path(__file__).resolve().parents[2]
NAME = "native-acceptance"
TITLE = "Daily totals"
INSTRUCTION = "Upsert today's totals into reporting.daily."


class Sent:
    """A notification sender that records instead of showing anything."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def __call__(self, title: str, text: str) -> None:
        self.items.append((title, text))

    def texts(self) -> list[str]:
        return [text for _title, text in self.items]


def _intent(source_id: str, schedule: RoutineSchedule) -> RoutineIntent:
    """A routine is a title, a prompt and a schedule: no grant is authored."""

    return RoutineIntent(
        title=TITLE, prompt=INSTRUCTION, schedule=schedule, sources=(source_id,)
    )


def _inserts(db: Any) -> int:
    return db.inserts()


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
        provider.replace_script([measured_step(response(text="Noted."))])
        origin = await remote.run("Keep today's totals in the companies table.")
        assert origin.conversation_id is not None
        provider.replace_script(scheduled_write_script(batch))
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
    interaction = await routine_approval(harness, NAME, routine_id, timeout=30)
    assert interaction["kind"] == "tool_approval"
    approval = interaction["request"]["approval"]
    assert approval.capability_id == "data.write"
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


async def test_waiting_for_an_approval_fails_at_once_when_the_run_ends_without_asking(
    start_host: StartHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness, provider, _db, _sent, routine_id = await _scheduled_write(
        start_host, monkeypatch
    )
    provider.replace_script([measured_step(response(text="Nothing to write."))])
    started = time.monotonic()
    # The poll can see the run before it finishes finalizing, so pin that it
    # ended cleanly, not which disposition it had reached.
    with pytest.raises(AssertionError, match="ended without asking.*failure_code None"):
        await routine_approval(harness, NAME, routine_id, timeout=60)
    assert time.monotonic() - started < 30


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


async def test_a_scheduled_write_pending_when_the_host_restarts_never_writes_unasked(
    start_host: StartHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness, provider, db, _sent, routine_id = await _scheduled_write(
        start_host, monkeypatch, drain_seconds=0.2
    )
    await pending_interaction(harness.http, NAME, timeout=30)
    # The host stops with the question open; the run cannot keep waiting.
    await harness.stop()
    assert not db.rows and _inserts(db) == 0
    steps_at_stop = len(provider.steps)

    restarted = await start_host(model=provider, root=harness.root)
    remote, card, outcome = await _result(restarted, routine_id)
    try:
        # Questions live in host memory: the restarted host has none, and the
        # interrupted occurrence needs a person rather than writing unasked.
        assert value(await restarted.http.get(f"/v1/agents/{NAME}/interactions")) == ()
        assert card.column is WorkColumn.NEEDS_YOU
        assert outcome.conclusion_state is OutcomeState.FAILED
        assert not db.rows and _inserts(db) == 0
        assert await remote.list_effects() == ()
        assert len(provider.steps) == steps_at_stop
    finally:
        await remote.close()


async def test_a_host_killed_after_a_scheduled_write_dispatches_never_replays_it(
    start_host: StartHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = short_root()
    agent, *_fixture, batch, _clock, _approvals = await create_fixture(
        root, monkeypatch
    )
    await agent.close()
    batch_file = root.parent / f"{root.name}-batch.json"
    batch_file.write_text(json.dumps(batch), encoding="utf-8")
    committing = root.parent / f"{root.name}-committing"
    paths = host_paths(root)
    # A real host process: its approved write hangs at commit and is killed there.
    host = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.support.killed_write_host",
            str(root),
            str(batch_file),
            str(committing),
        ],
        cwd=REPOSITORY,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    http = client(paths)
    try:
        await bounded(lambda: host_status(paths), timeout=30)
        remote = await connect(NAME, cwd=work_folder(root), root=root)
        try:
            origin = await remote.run("Keep today's totals in the companies table.")
            assert origin.conversation_id is not None
            routine = await remote.create_routine(
                _intent(
                    str(batch["source_id"]),
                    OnceSchedule(datetime.now(UTC) + timedelta(seconds=2)),
                ),
                conversation_id=origin.conversation_id,
            )
        finally:
            await remote.close()
        interaction = await pending_interaction(http, NAME, timeout=30)
        value(
            await http.post(
                f"/v1/agents/{NAME}/interactions/{interaction['id']}",
                json={"decision": "approve"},
            )
        )
        await bounded(committing.exists, timeout=30)
        host.send_signal(signal.SIGKILL)
        host.wait(timeout=20)
    finally:
        await http.aclose()
        if host.poll() is None:
            host.kill()
            host.wait(timeout=20)

    # Any replay after the restart would reach this database.
    replay = Database()
    patch_fake_postgresql(monkeypatch, replay)
    model = ScriptedResearchModel()
    restarted = await start_host(model=model, root=root)
    remote = await connect(NAME, cwd=restarted.work(), root=root)
    try:
        (receipt,) = await bounded(remote.list_effects, timeout=30)
        # Nobody knows whether the commit landed, so the outcome stays uncertain.
        assert receipt.outcome is EffectOutcome.UNCERTAIN
        assert receipt.routine_id == routine.routine_id

        async def needs_you() -> object:
            board = await remote.work_board()
            return next(
                (
                    card
                    for card in board.columns.get(WorkColumn.NEEDS_YOU, ())
                    if card.kind is WorkKind.ROUTINE_RUN
                    and card.routine_id == routine.routine_id
                ),
                None,
            )

        # The killed host's claim fences the occurrence until its lease ends;
        # then it surfaces as Needs you, with the routine paused.
        await bounded(needs_you, timeout=ROUTINE_CLAIM_LEASE_SECONDS + 30)
        inspection = await remote.inspect_routine(routine.routine_id)
        assert inspection is not None
        assert inspection.routine.state is RoutineState.PAUSED
        assert value(await restarted.http.get(f"/v1/agents/{NAME}/interactions")) == ()
        assert _inserts(replay) == 0 and not replay.rows
        assert model.requests == []
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
        provider.replace_script([measured_step(response(text="Noted."))])
        origin = await agent.run("Keep today's totals in the companies table.")
        assert origin.conversation_id is not None
        provider.replace_script(scheduled_write_script(batch))
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
