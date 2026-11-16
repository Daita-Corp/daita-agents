"""Create the revision 3 to 4 state-directory upgrade fixture for storage adapters.

``before`` is the released revision-3 golden database plus a catalog snapshot, an
unfinished graph job, a routine and a saved model route, all in their released
revision-3 shapes. ``after`` is what ``upgrade_state_directory`` makes of it.

Run once from the repository root; the fixture is immutable afterwards:

    python -m scripts.create_revision_3_to_4_upgrade_fixture
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from daita import AgentConfig
from daita.adapters.models import SourceRegistration, source_registration_id
from daita.hosting.configuration import _model_route, _write_model_configuration
from daita.hosting.home_upgrade import upgrade_state_directory
from daita.routines.models import (
    IntervalSchedule,
    RoutineState,
    ScheduledRoutine,
    text_digest,
)
from daita.storage.sqlite_codecs import (
    encode_catalog_snapshot,
    encode_catalog_sync,
    encode_source,
    encode_source_read_scope,
)
from daita.storage.sqlite_graph import admit_graph
from daita.storage.sqlite_records import SourceReadMode, SourceReadScope
from tests.storage._routine_history import released_routine
from tests.support.catalog_records import SHOP, SHOP_REFERENCES, snapshot
from tests.support.graph import graph_admission

REPOSITORY = Path(__file__).resolve().parents[1]
GOLDEN = REPOSITORY / "tests/fixtures/agent-home-revisions/revision-3/state.sql"
DESTINATION = REPOSITORY / "tests/fixtures/state-upgrades/revision-3-to-4"
AGENT = "agent-golden-revision-1"
AT = datetime(2026, 11, 1, 9, 30, tzinfo=UTC)


def _microseconds(value: datetime) -> int:
    return (value - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)


def _seed(connection: sqlite3.Connection) -> None:
    source_id = source_registration_id(AGENT, "postgresql", "shop")
    connection.execute(
        "INSERT INTO sources VALUES (?, ?, ?)",
        (
            AGENT,
            source_id,
            encode_source(
                SourceRegistration(
                    id=source_id,
                    agent_id=AGENT,
                    adapter_id="postgresql",
                    native_identity="shop",
                    display_name="Shop",
                    configuration={},
                    attached_at=AT,
                )
            ),
        ),
    )
    connection.execute(
        "INSERT INTO source_read_scopes VALUES (?, ?, ?)",
        (
            AGENT,
            source_id,
            encode_source_read_scope(
                SourceReadScope(AGENT, source_id, SourceReadMode.ALL)
            ),
        ),
    )
    shop = snapshot(AGENT, source_id, "sync-shop", SHOP, SHOP_REFERENCES)
    # Revision-3 syncs predate the truncation flag.
    sync = json.loads(encode_catalog_sync(shop.sync))
    del sync["fields"]["truncated"]
    stored = json.loads(encode_catalog_snapshot(shop))
    del stored["fields"]["sync"]["fields"]["truncated"]
    connection.execute(
        "INSERT INTO syncs VALUES (?, ?, ?, ?)",
        (AGENT, "sync-shop", source_id, json.dumps(sync, sort_keys=True)),
    )
    connection.execute(
        "INSERT INTO snapshots VALUES (?, ?, ?, ?)",
        (AGENT, source_id, "sync-shop", json.dumps(stored, sort_keys=True)),
    )
    admit_graph(connection, graph_admission(job_id="job-unfinished", agent_id=AGENT))
    prompt = "Summarize yesterday's orders."
    routine = ScheduledRoutine(
        routine_id="routine-daily",
        agent_id=AGENT,
        conversation_id="conversation-routine",
        owner_principal_id=f"agent:{AGENT}",
        title="Daily orders",
        authorized_instruction=prompt,
        instruction_digest=text_digest(prompt),
        schedule=IntervalSchedule(86_400, AT),
        schedule_interpreter_revision=1,
        source_ids=(source_id,),
        charged_tokens=0,
        charged_cost_usd=Decimal("0"),
        attempt_count=0,
        occurrence_count=0,
        consecutive_failures=0,
        next_due_at=AT,
        active_occurrence_id=None,
        last_occurrence_id=None,
        last_delivery_ids=(),
        state=RoutineState.ACTIVE,
        revision=1,
        created_at=AT,
        updated_at=AT,
    )
    connection.execute(
        "INSERT INTO scheduled_routines VALUES (?, ?, ?, ?, ?, ?)",
        (
            AGENT,
            routine.routine_id,
            routine.conversation_id,
            routine.state.value,
            _microseconds(AT),
            released_routine(routine),
        ),
    )


def _write_revision_3_config(directory: Path) -> None:
    route = _model_route(
        "ollama",
        "fixture-model",
        base_url=None,
        secret_reference=None,
        context_window_tokens=8_192,
        max_output_tokens=1_024,
    )
    _write_model_configuration(directory, AgentConfig(model_route=route))
    path = directory / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    for candidate in config["model_route"]["candidates"]:
        del candidate["reasoning_effort"]
    path.write_text(
        json.dumps(config, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )


def _dump(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        statements = list(connection.iterdump())
    # SQLite keeps its sequence table after the last AUTOINCREMENT table is
    # dropped; a fresh database has none to clear.
    if not any("AUTOINCREMENT" in statement for statement in statements):
        statements.remove('DELETE FROM "sqlite_sequence";')
    return "\n".join(statements) + "\n"


def main() -> int:
    if DESTINATION.exists():
        raise RuntimeError("The revision 3 to 4 upgrade fixture already exists")
    with tempfile.TemporaryDirectory(prefix="daita-state-upgrade-") as temporary:
        directory = Path(temporary).resolve()
        with sqlite3.connect(directory / "state.db") as connection:
            connection.executescript(GOLDEN.read_text(encoding="utf-8"))
            _seed(connection)
        _write_revision_3_config(directory)
        before = (
            _dump(directory / "state.db"),
            (directory / "config.json").read_text(),
        )
        result = upgrade_state_directory(directory)
        if (result.source_revision, result.target_revision) != (3, 4):
            raise RuntimeError("The fixture did not upgrade from revision 3 to 4")
        after = (_dump(directory / "state.db"), (directory / "config.json").read_text())
    for name, (state, config) in (("before", before), ("after", after)):
        (DESTINATION / name).mkdir(parents=True)
        (DESTINATION / name / "state.sql").write_text(state, encoding="utf-8")
        (DESTINATION / name / "config.json").write_text(config, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
