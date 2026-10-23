"""Open one immutable whole-home fixture for every production revision."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from daita import Agent, SQLiteSource
from daita.catalog import CatalogLinkStaleReason
from daita.distribution import OutcomeState
from daita.errors import StateCompatibilityError
from daita.llm.models import FinishReason, ModelResponse, ModelUsage, TextBlock
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.routines.models import (
    IntervalSchedule,
    RoutineOccurrence,
    RoutineOccurrenceDisposition,
    RoutineSlotKind,
    RoutineState,
    ScheduledRoutine,
    text_digest,
)
from daita.routines.schedule import occurrence_id, scheduled_slot_key
from daita.storage.home_migrations import (
    CURRENT_HOME_REVISION,
    HOME_MIGRATIONS,
    MINIMUM_SUPPORTED_HOME_REVISION,
)
from tests.storage._routine_history import released_occurrence, released_routine
from tests.support.paths import REPO_ROOT
from tests.support.workspace import workspace_for

FIXTURES = REPO_ROOT / "tests/fixtures/agent-home-revisions"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_fixture(fixture: Path, home: Path) -> None:
    home.mkdir(parents=True)
    for source in fixture.iterdir():
        if source.name == "state.sql":
            with sqlite3.connect(home / "state.db") as connection:
                connection.executescript(source.read_text(encoding="utf-8"))
                if fixture.name in {"revision-2", "revision-3", "revision-4"}:
                    assert connection.execute(
                        "PRAGMA journal_mode = WAL"
                    ).fetchone() == ("wal",)
        elif source.is_dir():
            shutil.copytree(source, home / source.name)
        else:
            shutil.copyfile(source, home / source.name)
    if (home / "artifacts").is_dir():
        # Git cannot retain the empty directory required by the artifact layout.
        (home / "artifacts" / ".staging").mkdir(mode=0o700, exist_ok=True)


def test_every_production_home_revision_has_one_golden_fixture() -> None:
    assert {path.name for path in FIXTURES.iterdir() if path.is_dir()} == {
        f"revision-{migration.revision}" for migration in HOME_MIGRATIONS
    }


async def test_revision_1_golden_whole_home_upgrades_once_and_preserves_content(
    tmp_path: Path,
) -> None:
    fixture = FIXTURES / "revision-1"
    home = tmp_path / "agents/golden"
    _materialize_fixture(fixture, home)
    before = _sha256(home / "state.db")

    status = await Agent.inspect_home("golden", root=tmp_path)
    assert status.current_revision == CURRENT_HOME_REVISION
    assert status.found_revision == 1
    assert status.minimum_supported_revision == 1
    assert status.upgrade_required
    assert not status.recovery_required

    agent = await Agent.open(
        "golden",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    try:
        assert agent.id == "agent-golden-revision-1"
        assert await agent.read_memory() == "Golden durable memory.\n"
        assert await agent.read_user_profile() == "Golden durable user profile.\n"
        skill = await agent.read_skill("golden-skill")
        assert skill is not None
        assert skill.description == "A fixed revision-one procedure."
        assert skill.instructions == "Use current evidence only."
        transcript = await agent.transcript("run-golden-revision-1")
        assert transcript.messages[-1].content == (TextBlock("Golden answer."),)
    finally:
        await agent.close()

    assert _sha256(home / "state.db") != before
    assert len(tuple((home / ".home-rollbacks").iterdir())) == 1
    upgraded = _sha256(home / "state.db")
    reopened = await Agent.open(
        "golden",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    await reopened.close()
    assert _sha256(home / "state.db") == upgraded
    assert len(tuple((home / ".home-rollbacks").iterdir())) == 1


@pytest.mark.parametrize(
    "source_revision",
    tuple(
        migration.revision
        for migration in HOME_MIGRATIONS
        if migration.revision >= MINIMUM_SUPPORTED_HOME_REVISION
    ),
)
async def test_every_supported_golden_home_reaches_current_revision(
    tmp_path: Path,
    source_revision: int,
) -> None:
    fixture = FIXTURES / f"revision-{source_revision}"
    home = tmp_path / "agents/golden"
    _materialize_fixture(fixture, home)
    before = _sha256(home / "state.db")

    initial = await Agent.inspect_home("golden", root=tmp_path)
    assert initial.found_revision == source_revision
    assert initial.upgrade_required is (source_revision < CURRENT_HOME_REVISION)

    agent = await Agent.open(
        "golden",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    if source_revision >= 3:
        servers = await agent.list_mcp_servers()
        assert len(servers) == 1
        tool = servers[0].binding.tools[0]
        assert tool.raw_input_schema is not None
        assert (
            tool.raw_input_schema["$schema"]
            == "http://json-schema.org/draft-07/schema#"
        )
        assert "$schema" not in tool.input_schema
    if source_revision == 3:
        (ref,) = await agent._embedded._artifact_store.list_refs()
        assert (
            await agent.read_artifact(ref.artifact_id)
        ).content == b"Golden artifact 0."
        historical = await agent._embedded._store.result(ref.run_id)
        assert historical is not None
        assert len(historical.artifacts) == 2
        assert not await agent.delete_artifact(historical.artifacts[1].artifact_id)
    await agent.close()

    current = await Agent.inspect_home("golden", root=tmp_path)
    assert current.found_revision == CURRENT_HOME_REVISION
    assert not current.upgrade_required
    assert not current.recovery_required
    if source_revision == CURRENT_HOME_REVISION:
        assert _sha256(home / "state.db") == before
        assert not (home / ".home-rollbacks").exists()
    else:
        assert len(tuple((home / ".home-rollbacks").iterdir())) == 1


async def test_released_revision_3_home_upgrades_and_keeps_declared_links(
    tmp_path: Path,
) -> None:
    _materialize_fixture(FIXTURES / "revision-3", tmp_path / "agents/golden")
    database = tmp_path / "linked.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY)")
        connection.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER)")

    agent = await Agent.open("golden", root=tmp_path, workspace=workspace_for(tmp_path))
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = {
            resource.name: resource.id
            for resource in await agent.list_catalog_resources(source_id=source.id)
        }
        link = await agent.declare_catalog_link(
            resources["orders"], resources["customers"], (("customer_id", "id"),)
        )
    finally:
        await agent.close()

    reopened = await Agent.open(
        "golden", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        (status,) = await reopened.list_catalog_links()
        assert status.link == link
        assert status.current
    finally:
        await reopened.close()
    assert len(tuple((tmp_path / "agents/golden/.home-rollbacks").iterdir())) == 1


async def test_revision_4_golden_home_retains_its_declared_link(
    tmp_path: Path,
) -> None:
    _materialize_fixture(FIXTURES / "revision-4", tmp_path / "agents/golden")
    agent = await Agent.open("golden", root=tmp_path, workspace=workspace_for(tmp_path))
    try:
        (status,) = await agent.list_catalog_links()
        assert status.link.note == "Golden declared link."
        assert status.stale_reason is CatalogLinkStaleReason.SOURCE_INACTIVE
    finally:
        await agent.close()


_AGENT = "agent-golden-revision-1"
_CONVERSATION = "conversation-released"
_NAMED_SOURCE = "source:sha256:" + "b" * 64


def _microseconds(value: datetime) -> int:
    return (value - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)


def _released_routines(due: datetime) -> tuple[
    tuple[ScheduledRoutine, str],
    tuple[ScheduledRoutine, str],
    tuple[ScheduledRoutine, str],
    dict[str, str],
]:
    """A due recurring routine, one interrupted mid-precheck and one disabled
    routine naming a source, as released revisions stored them."""

    def routine(routine_id: str, prompt: str, **values: object) -> ScheduledRoutine:
        fields: dict[str, object] = dict(
            routine_id=routine_id,
            agent_id=_AGENT,
            conversation_id=_CONVERSATION,
            owner_principal_id=f"agent:{_AGENT}",
            title=routine_id.replace("-", " "),
            authorized_instruction=prompt,
            instruction_digest=text_digest(prompt),
            schedule=IntervalSchedule(3_600, due),
            schedule_interpreter_revision=1,
            source_ids=(),
            charged_tokens=0,
            charged_cost_usd=Decimal("0"),
            attempt_count=0,
            occurrence_count=0,
            consecutive_failures=0,
            next_due_at=due,
            active_occurrence_id=None,
            last_occurrence_id=None,
            last_delivery_ids=(),
            state=RoutineState.ACTIVE,
            revision=1,
            created_at=due - timedelta(days=1),
            updated_at=due - timedelta(hours=2),
        )
        fields.update(values)
        return ScheduledRoutine(**fields)  # type: ignore[arg-type]

    def occurrence(
        owner: ScheduledRoutine, scheduled_for: datetime
    ) -> RoutineOccurrence:
        slot = scheduled_slot_key(owner.routine_id, owner.revision, scheduled_for)
        return RoutineOccurrence(
            occurrence_id=occurrence_id(owner.routine_id, slot),
            agent_id=_AGENT,
            routine_id=owner.routine_id,
            routine_revision=owner.revision,
            slot_kind=RoutineSlotKind.SCHEDULED,
            slot_key=slot,
            scheduled_for=scheduled_for,
            claimed_at=scheduled_for,
            claim_token="released-claim",
            lease_expires_at=scheduled_for + timedelta(seconds=30),
            execution_scope=None,
            execution_scope_digest=None,
            reserved_run_id=None,
            charged_tokens=0,
            charged_cost_usd=Decimal("0"),
            run_bound_at=None,
            run_terminal_at=None,
            conclusion_digest=None,
            terminal_run_id=None,
            delivery_ids=(),
            attempt_count=1,
            failure_code=None,
            retry_at=None,
            disposition=RoutineOccurrenceDisposition.CLAIMED,
            created_at=scheduled_for,
            updated_at=scheduled_for,
        )

    earlier = due - timedelta(hours=1)
    skipped_slot = scheduled_slot_key("released-recurring", 1, earlier)
    recurring = routine(
        "released-recurring",
        "Report the paid invoice count.",
        charged_tokens=120,
        charged_cost_usd=Decimal("0.02"),
        attempt_count=1,
        occurrence_count=1,
        consecutive_failures=1,
        last_occurrence_id=occurrence_id("released-recurring", skipped_slot),
    )
    skipped = occurrence(recurring, earlier)
    interrupted_slot = scheduled_slot_key("released-interrupted", 1, earlier)
    interrupted = routine(
        "released-interrupted",
        "Report the open invoice count.",
        attempt_count=1,
        occurrence_count=1,
        next_due_at=due + timedelta(hours=1),
        active_occurrence_id=occurrence_id("released-interrupted", interrupted_slot),
    )
    prechecking = occurrence(interrupted, earlier)
    named = routine(
        "released-named",
        "Report the overdue invoice count.",
        source_ids=(_NAMED_SOURCE,),
        next_due_at=None,
        state=RoutineState.DISABLED,
    )
    return (
        (
            recurring,
            released_routine(
                recurring,
                capability_grants=(
                    {
                        "__record__": "CapabilityGrant",
                        "fields": {"capability_id": "data.upsert_rows"},
                    },
                ),
            ),
        ),
        (
            interrupted,
            released_routine(
                interrupted,
                precheck={
                    "__record__": "ResourceRevisionPrecheck",
                    "fields": {
                        "capability_id": "data.resource_revision_observation",
                        "contract_digest": "sha256:" + "3" * 64,
                        "source_id": _NAMED_SOURCE,
                        "resource_id": "catalog-resource:sha256:" + "d" * 64,
                    },
                },
            ),
        ),
        (named, released_routine(named)),
        {
            skipped.occurrence_id: released_occurrence(
                skipped, disposition="skipped_no_change"
            ),
            prechecking.occurrence_id: released_occurrence(
                prechecking, disposition="prechecking"
            ),
        },
    )


def _seed(
    home: Path, due: datetime
) -> tuple[ScheduledRoutine, ScheduledRoutine, ScheduledRoutine]:
    recurring, interrupted, named, occurrences = _released_routines(due)
    with sqlite3.connect(home / "state.db") as connection:
        # The fixture's connector tool is unrelated; the mock route cannot
        # represent it.
        connection.execute("DELETE FROM mcp_server_bindings")
        for routine, data in (recurring, interrupted, named):
            connection.execute(
                "INSERT INTO scheduled_routines VALUES (?, ?, ?, ?, ?, ?)",
                (
                    _AGENT,
                    routine.routine_id,
                    _CONVERSATION,
                    routine.state.value,
                    (
                        None
                        if routine.next_due_at is None
                        else _microseconds(routine.next_due_at)
                    ),
                    data,
                ),
            )
        for identity, data in occurrences.items():
            fields = json.loads(data)["fields"]
            lease = datetime.fromisoformat(
                fields["lease_expires_at"]["__datetime__"].replace("Z", "+00:00")
            )
            connection.execute(
                "INSERT INTO routine_occurrences VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _AGENT,
                    identity,
                    fields["routine_id"],
                    1,
                    fields["slot_key"],
                    fields["disposition"],
                    _microseconds(lease),
                    None,
                    data,
                ),
            )
    return recurring[0], interrupted[0], named[0]


@pytest.mark.parametrize(
    "source_revision",
    tuple(
        migration.revision
        for migration in HOME_MIGRATIONS
        if MINIMUM_SUPPORTED_HOME_REVISION <= migration.revision < 4
    ),
)
async def test_released_routines_upgrade_to_the_simple_shape_and_keep_running(
    tmp_path: Path, source_revision: int
) -> None:
    home = tmp_path / "agents/golden"
    _materialize_fixture(FIXTURES / f"revision-{source_revision}", home)
    due = datetime.now(UTC) - timedelta(seconds=30)
    recurring, interrupted, named = _seed(home, due)
    with sqlite3.connect(home / "state.db") as connection:
        released_rows = tuple(
            connection.execute("SELECT * FROM scheduled_routines ORDER BY routine_id")
        )

    # Without a model nothing runs, so the upgraded records are as stored.
    agent = await Agent.open("golden", root=tmp_path, workspace=workspace_for(tmp_path))
    try:
        upgraded = await agent.inspect_routine(recurring.routine_id)
        assert upgraded is not None and upgraded.routine == recurring
        (skipped,) = upgraded.recent_occurrences
        assert skipped.disposition is RoutineOccurrenceDisposition.COMPLETED
        resumable = await agent.inspect_routine(interrupted.routine_id)
        assert resumable is not None
        (claim,) = resumable.recent_occurrences
        # A claim interrupted mid-precheck is now an ordinary claim.
        assert claim.disposition is RoutineOccurrenceDisposition.CLAIMED
        # The routine's named sources become its optional source list.
        disabled = await agent.inspect_routine(named.routine_id)
        assert disabled is not None and disabled.routine == named
        assert disabled.routine.source_ids == (_NAMED_SOURCE,)
    finally:
        await agent.close()
    # The rollback copy keeps the released rows exactly.
    (rollback,) = tuple((home / ".home-rollbacks").iterdir())
    with sqlite3.connect(rollback / "state.db") as connection:
        assert (
            tuple(
                connection.execute(
                    "SELECT * FROM scheduled_routines ORDER BY routine_id"
                )
            )
            == released_rows
        )

    usage = ModelUsage(
        input_tokens=40,
        output_tokens=8,
        cost_estimate=CostEstimate.complete(Decimal("0")),
    )
    provider = MockModelProvider(
        tuple(
            ModelResponse(
                finish_reason=FinishReason.STOP, text="The count is 7.", usage=usage
            )
            for _ in range(2)
        ),
        provider_id="mock:released",
        complete_pricing=True,
    )
    running = await Agent.open(
        "golden",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
    )
    try:
        async with asyncio.timeout(30):
            while len(await running.list_deliveries(conversation_id=_CONVERSATION)) < 2:
                await asyncio.sleep(0.01)
        deliveries = await running.list_deliveries(conversation_id=_CONVERSATION)
        assert {item.conclusion_state for item in deliveries} == {
            OutcomeState.SUCCEEDED
        }, [item.failure_code for item in deliveries]
        ran = await running.inspect_routine(recurring.routine_id)
        assert ran is not None
        assert ran.routine.occurrence_count == recurring.occurrence_count + 1
        assert ran.routine.charged_tokens == recurring.charged_tokens + 48
        assert ran.routine.consecutive_failures == 0
        assert ran.routine.next_due_at == due + timedelta(hours=1)
        resumed = await running.inspect_routine(interrupted.routine_id)
        assert resumed is not None
        assert resumed.routine.last_occurrence_id == interrupted.active_occurrence_id
        assert resumed.recent_occurrences[0].disposition is (
            RoutineOccurrenceDisposition.COMPLETED
        )
    finally:
        await running.close()


async def test_a_malformed_released_routine_fails_closed_without_writing(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agents/golden"
    _materialize_fixture(FIXTURES / "revision-3", home)
    _seed(home, datetime.now(UTC) - timedelta(seconds=30))
    with sqlite3.connect(home / "state.db") as connection:
        (data,) = connection.execute(
            "SELECT data FROM scheduled_routines WHERE routine_id = ?",
            ("released-recurring",),
        ).fetchone()
        damaged = json.loads(data)
        del damaged["fields"]["allowed_source_ids"]
        connection.execute(
            "UPDATE scheduled_routines SET data = ? WHERE routine_id = ?",
            (json.dumps(damaged, sort_keys=True), "released-recurring"),
        )
    before = _sha256(home / "state.db")
    with pytest.raises(StateCompatibilityError):
        await Agent.open("golden", root=tmp_path, workspace=workspace_for(tmp_path))
    assert _sha256(home / "state.db") == before
    assert not (home / ".home-rollbacks").exists()
