"""A real model's scheduled routine asks through the host and writes once.

Opt in with ``DAITA_RUN_LIVE_SCHEDULED_WRITE=1``; use
``DAITA_LIVE_BILLING=subscription`` to run on the ChatGPT subscription. The
routine fires on its schedule in a real in-process host while no client
watches, so the question also notifies; the test approves through the host's
interaction route. Denial, expiry, restart and crash mechanics are
deterministic tests in ``tests/acceptance/test_scheduled_approvals.py``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from daita import (
    LoopLimits,
    OnceSchedule,
    RoutineIntent,
    WorkColumn,
    WorkKind,
    connect,
)
from daita.adapters.mcp import SDKMCPClientFactory
from daita.capabilities import EffectOutcome
from daita.distribution import OutcomeState
from tests.support.host import bounded, hosts, pending_interaction, short_root, value
from tests.support.live_billing import cost_cap, live_provider
from tests.support.mcp import mock_transport
from tests.support.native_writes import create_fixture

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get("DAITA_RUN_LIVE_SCHEDULED_WRITE") != "1",
        reason="requires explicit live scheduled-write authorization",
    ),
]

NAME = "native-acceptance"
TITLE = "Record New Co"
_MODEL = os.environ.get("DAITA_LIVE_SCHEDULED_WRITE_MODEL_ID", "openai:gpt-5.6-terra")
_TIMEOUT = 10 * 60
_LIMITS = LoopLimits(
    max_steps=16,
    max_total_tokens=150_000,
    max_wall_time_seconds=_TIMEOUT,
    max_estimated_cost_usd=cost_cap(Decimal("0.50")),
)


def _api_key() -> str:
    key = os.environ.get("DAITA_LIVE_SCHEDULED_WRITE_LLM_API_KEY", "")
    if not key.strip():
        pytest.fail(
            "DAITA_LIVE_SCHEDULED_WRITE_LLM_API_KEY is required under API billing"
        )
    return key


async def test_live_scheduled_routine_asks_notifies_and_writes_once_after_approval(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    root = short_root()
    fixture, _scripted, db, alpha, *_rest, batch, _clock, _approvals = (
        await create_fixture(root, monkeypatch)
    )
    await fixture.close()
    profile, provider = live_provider(_MODEL, api_key=_api_key, max_output_tokens=4_096)
    (row,) = batch["rows"]
    prompt = (
        "Upsert one company into the companies table, keyed on domain: domain "
        f"{row['domain']}, name {row['name']}, evidence URL {row['evidence_url']}. "
        "Then report what you saved."
    )
    sent: list[str] = []

    def inserts() -> int:
        return len(
            [
                item
                for item in db.log
                if item[0] == "fetch" and item[1].startswith("INSERT")
            ]
        )

    async with hosts() as start:
        harness = await start(
            model=provider,
            root=root,
            notify=lambda _title, text: sent.append(text),
            notify_interval=0.0,
            agent_options={
                "model_profile": profile,
                "limits": _LIMITS,
                "mcp_client_factory": SDKMCPClientFactory(
                    http_transport=mock_transport(alpha)
                ),
            },
        )
        remote = await connect(NAME, cwd=harness.work(), root=root)
        try:
            routine = await remote.create_routine(
                RoutineIntent(
                    title=TITLE,
                    prompt=prompt,
                    schedule=OnceSchedule(datetime.now(UTC) + timedelta(seconds=5)),
                    sources=(str(batch["source_id"]),),
                )
            )
        finally:
            # Nobody watches from here on, so the question also notifies.
            await remote.close()

        interaction = await pending_interaction(harness.http, NAME, timeout=_TIMEOUT)
        approval = interaction["request"]["approval"]
        record_property("approval_capability", approval.capability_id)
        assert approval.capability_id == "data.upsert_rows"
        assert approval.requester == TITLE
        await bounded(lambda: sent, timeout=30)
        assert sent == [f"{NAME} needs your approval"]
        assert inserts() == 0 and not db.rows
        value(
            await harness.http.post(
                f"/v1/agents/{NAME}/interactions/{interaction['id']}",
                json={"decision": "approve"},
            )
        )

        remote = await connect(NAME, cwd=harness.work(), root=root)
        try:

            async def done() -> object:
                board = await remote.work_board()
                return next(
                    (
                        card
                        for column in (WorkColumn.DONE, WorkColumn.NEEDS_YOU)
                        for card in board.columns.get(column, ())
                        if card.kind is WorkKind.ROUTINE_RUN
                        and card.routine_id == routine.routine_id
                        and card.delivery_id is not None
                    ),
                    None,
                )

            card = await bounded(done, timeout=_TIMEOUT, interval=0.5)
            inspection = await remote.inspect_delivery(card.delivery_id)
            assert inspection is not None
            outcome = inspection.delivery.outcome
            record_property("conclusion", outcome.conclusion_preview)
            assert card.column is WorkColumn.DONE, outcome.conclusion_preview
            assert outcome.conclusion_state is OutcomeState.SUCCEEDED
            assert inserts() == 1
            assert db.rows[row["domain"]]["name"] == row["name"]
            (receipt,) = await remote.list_effects()
            assert receipt.outcome is EffectOutcome.SUCCEEDED
            assert receipt.routine_id == routine.routine_id
            assert outcome.effect_receipt_ids == (receipt.receipt_id,)
            # The one approval was the only question this run asked.
            assert (
                value(await harness.http.get(f"/v1/agents/{NAME}/interactions")) == ()
            )
        finally:
            await remote.close()
