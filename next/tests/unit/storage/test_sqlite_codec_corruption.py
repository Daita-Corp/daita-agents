from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
from typing import cast

import pytest

from daita.storage import sqlite as sqlite_owner
from daita.events.models import RuntimeEvent
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolResultBlock,
)
from daita.loop.models import LoopBudgets, LoopPhase, LoopState, Turn
from daita.operations.checkpoints import (
    ModelCall,
    ModelCallStatus,
    OperationSnapshot,
)
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    TriggerKind,
)
from daita.storage.sqlite import (
    SQLiteCompatibilityError,
    SQLiteCorruptionError,
    SQLiteOperationStore,
)

NOW = datetime(2026, 7, 17, 12, 34, 56, 789_012, tzinfo=timezone.utc)
OPERATION_ID = "operation-codec-corruption"


def _minimal_model_snapshot() -> OperationSnapshot:
    agent_id = "agent-codec-corruption"
    trigger = AgentTrigger(
        id="trigger-codec-corruption",
        agent_id=agent_id,
        kind=TriggerKind.USER,
        source_id="user-codec-corruption",
        payload={"prompt": "persist this model call"},
        created_at=NOW,
    )
    operation = Operation(
        id=OPERATION_ID,
        agent_id=agent_id,
        trigger_id=trigger.id,
        status=OperationStatus.RUNNING,
        created_at=NOW,
        updated_at=NOW,
    )
    turn = Turn(
        id="turn-codec-corruption",
        operation_id=OPERATION_ID,
        number=1,
        model_request_id="model-call-codec-corruption",
        model_response_id="model-call-codec-corruption",
        created_at=NOW,
    )
    request = ModelRequest(
        operation_id=OPERATION_ID,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id=agent_id,
                operation_id=OPERATION_ID,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Load the durable result."),),
            ),
            CanonicalMessage(
                agent_id=agent_id,
                operation_id=OPERATION_ID,
                turn_id=turn.id,
                role=MessageRole.TOOL,
                content=(
                    ToolResultBlock(
                        call_id="prior-call",
                        output={"found": True},
                    ),
                ),
            ),
        ),
        context_selection={
            "estimated_input_tokens": 23,
            "input_limit_tokens": 100,
            "omitted_blocks": [],
            "output_reserve_tokens": 20,
            "profile_id": "mock:codec-corruption",
            "schema_version": 1,
            "selected_blocks": [
                {
                    "estimated_tokens": 23,
                    "id": "data.intent",
                    "kind": "intent",
                    "owner": "data",
                    "priority": 1_000_000,
                    "provenance": [
                        {
                            "kind": "operation.trigger",
                            "reference_id": trigger.id,
                            "revision": None,
                        }
                    ],
                    "required": True,
                    "trust": "trusted_runtime",
                }
            ],
            "tool_tokens": 0,
        },
    )
    response = ModelResponse(
        finish_reason=FinishReason.STOP,
        text="The durable result loaded.",
        usage=ModelUsage(
            input_tokens=7,
            output_tokens=5,
            estimated_cost_usd=Decimal("0.00012"),
        ),
    )
    model_call = ModelCall(
        id="model-call-codec-corruption",
        operation_id=OPERATION_ID,
        turn_id=turn.id,
        provider_id="mock:codec-corruption",
        request=request,
        response=response,
        status=ModelCallStatus.COMPLETED,
        created_at=NOW,
        updated_at=NOW,
    )
    event = RuntimeEvent(
        id="event-codec-corruption",
        type="turn.created",
        agent_id=agent_id,
        operation_id=OPERATION_ID,
        turn_id=turn.id,
        payload={"number": 1},
        created_at=NOW,
    )
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(phase=LoopPhase.AWAITING_MODEL, turn_count=1),
        budgets=LoopBudgets(),
        turns=(turn,),
        model_calls=(model_call,),
        readiness=(),
        tasks=(),
        evidence=(),
        observations=(),
        events=(event,),
    )


async def _create_valid_database(path: Path) -> None:
    snapshot = _minimal_model_snapshot()
    store = await SQLiteOperationStore.open(path)
    try:
        created = await store.create(snapshot)
        assert await store.load(OPERATION_ID) == created.operation
    finally:
        await store.close()


def _execute(
    path: Path,
    statement: str,
    parameters: tuple[object, ...] = (),
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(statement, parameters)
        connection.commit()
    finally:
        connection.close()


def _edit_request_json(
    path: Path,
    edit: Callable[[dict[str, object]], None],
) -> None:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT request_json FROM model_calls WHERE operation_id = ?",
            (OPERATION_ID,),
        ).fetchone()
        assert row is not None
        request = cast(dict[str, object], json.loads(str(row[0])))
        edit(request)
        connection.execute(
            "UPDATE model_calls SET request_json = ? WHERE operation_id = ?",
            (
                json.dumps(
                    request,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                OPERATION_ID,
            ),
        )
        connection.commit()
    finally:
        connection.close()


async def _assert_typed_corruption(path: Path) -> None:
    store = await SQLiteOperationStore.open(path)
    try:
        with pytest.raises(SQLiteCorruptionError):
            await store.load(OPERATION_ID)
    finally:
        await store.close()


async def test_model_request_context_selection_roundtrips_and_decodes_v1(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    await _create_valid_database(path)

    store = await SQLiteOperationStore.open(path)
    try:
        loaded = await store.load(OPERATION_ID)
        context_selection = loaded.snapshot.model_calls[0].request.context_selection
        assert context_selection["schema_version"] == 1
        assert context_selection["profile_id"] == "mock:codec-corruption"
        assert context_selection["selected_blocks"]
    finally:
        await store.close()

    def downgrade_to_v1(request: dict[str, object]) -> None:
        request["codec_version"] = 1
        request.pop("context_selection")
        request.pop("response_schema")
        request.pop("sensitivity")

    _edit_request_json(path, downgrade_to_v1)
    legacy_store = await SQLiteOperationStore.open(path)
    try:
        legacy = await legacy_store.load(OPERATION_ID)
        assert not legacy.snapshot.model_calls[0].request.context_selection
    finally:
        await legacy_store.close()


@pytest.mark.parametrize(
    "corrupt_json",
    (
        '{"prompt":"one","prompt":"two"}',
        '{"prompt":NaN}',
        '["wrong-root"]',
    ),
    ids=("duplicate-key", "nonfinite", "wrong-root"),
)
async def test_load_rejects_corrupt_generic_json(
    tmp_path: Path,
    corrupt_json: str,
) -> None:
    path = tmp_path / "state.db"
    await _create_valid_database(path)
    _execute(
        path,
        "UPDATE triggers SET payload_json = ? WHERE id = ?",
        (corrupt_json, "trigger-codec-corruption"),
    )

    await _assert_typed_corruption(path)


def _set_unknown_request_codec(request: dict[str, object]) -> None:
    request["codec_version"] = 4


def _set_unknown_content_kind(request: dict[str, object]) -> None:
    messages = cast(list[dict[str, object]], request["messages"])
    content = cast(list[dict[str, object]], messages[0]["content"])
    content[0]["kind"] = "future_content"


def _set_unknown_message_role(request: dict[str, object]) -> None:
    messages = cast(list[dict[str, object]], request["messages"])
    messages[0]["role"] = "future_role"


@pytest.mark.parametrize(
    "edit",
    (
        _set_unknown_request_codec,
        _set_unknown_content_kind,
        _set_unknown_message_role,
    ),
    ids=("codec-version", "content-kind", "message-role-enum"),
)
async def test_load_rejects_unknown_model_codec_values(
    tmp_path: Path,
    edit: Callable[[dict[str, object]], None],
) -> None:
    path = tmp_path / "state.db"
    await _create_valid_database(path)
    _edit_request_json(path, edit)

    await _assert_typed_corruption(path)


def _set_non_boolean_tool_result_flag(request: dict[str, object]) -> None:
    messages = cast(list[dict[str, object]], request["messages"])
    content = cast(list[dict[str, object]], messages[1]["content"])
    content[0]["is_error"] = 2


async def test_load_rejects_integer_two_for_json_boolean(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    await _create_valid_database(path)
    _edit_request_json(path, _set_non_boolean_tool_result_flag)

    await _assert_typed_corruption(path)


async def test_load_rejects_naive_persisted_datetime(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    await _create_valid_database(path)
    _execute(
        path,
        "UPDATE operations SET created_at = ? WHERE id = ?",
        ("2026-07-17T12:34:56.789012", OPERATION_ID),
    )

    await _assert_typed_corruption(path)


async def test_load_rejects_invalid_persisted_decimal(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    await _create_valid_database(path)
    _execute(
        path,
        "UPDATE loop_state SET estimated_cost_usd = ? WHERE operation_id = ?",
        ("not-a-decimal", OPERATION_ID),
    )

    await _assert_typed_corruption(path)


async def test_load_rejects_wrong_sqlite_storage_class(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    await _create_valid_database(path)
    _execute(
        path,
        "UPDATE triggers SET source_id = CAST(source_id AS BLOB) WHERE id = ?",
        ("trigger-codec-corruption",),
    )

    await _assert_typed_corruption(path)


async def test_load_rejects_noncontiguous_collection_positions(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    await _create_valid_database(path)
    _execute(
        path,
        "UPDATE turns SET position = 10 WHERE operation_id = ?",
        (OPERATION_ID,),
    )

    await _assert_typed_corruption(path)


@pytest.mark.parametrize(
    "schema_mutation",
    (
        "DROP TABLE loop_state",
        "ALTER TABLE loop_state RENAME COLUMN phase TO lifecycle_phase",
        "CREATE TABLE rogue_state(value BLOB)",
    ),
    ids=("missing-table", "renamed-column", "unexpected-table"),
)
async def test_public_open_rejects_lifecycle_schema_drift_with_valid_history(
    tmp_path: Path,
    schema_mutation: str,
) -> None:
    path = tmp_path / "state.db"
    await _create_valid_database(path)
    _execute(path, schema_mutation)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall() == [(migration.version,) for migration in sqlite_owner._MIGRATIONS]
    finally:
        connection.close()

    reopened: SQLiteOperationStore | None = None
    try:
        with pytest.raises(SQLiteCompatibilityError, match="schema"):
            reopened = await SQLiteOperationStore.open(path)
    finally:
        if reopened is not None:
            await reopened.close()
