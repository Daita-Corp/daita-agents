"""A write's outcome is known, or stays uncertain; it is never retried."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from daita.adapters import postgresql_write as native
from daita.adapters.postgresql_write import (
    RelationalWriteCancelled,
    RelationalWriteError,
)
from daita.adapters.registry import SourceOperation
from daita.capabilities import ApprovalDecision, EffectOutcome
from daita.domains.data.sql import WriteFacts, WritePreview
from daita.llm.models import ToolCall
from tests.support.native_writes import UPSERT_SQL, create_fixture, response


class _ServerError(Exception):
    def __init__(self, sqlstate: str) -> None:
        super().__init__("server text that must not cross, with secret values")
        self.sqlstate = sqlstate


def _outcome(error: RelationalWriteError) -> EffectOutcome:
    assert error.effect_observation is not None
    return error.effect_observation.outcome


class _Execution:
    effect_receipt_id = "effect-receipt:sha256:" + "1" * 64
    capability_id = "data.write"


async def _prepared(tmp_path: Any, monkeypatch: Any) -> tuple[Any, ...]:
    agent, provider, db, *_rest, batch, _clock, _approvals = await create_fixture(
        tmp_path, monkeypatch
    )
    backend = agent._embedded._source_backends[SourceOperation.RELATIONAL_WRITE][
        "postgresql"
    ]
    write = backend.describe(
        UPSERT_SQL,
        source_id=batch["source_id"],
        parameters=batch["parameters"],
        schemas=("public",),
    )
    assert isinstance(write, WriteFacts)
    preview = await backend.preview(
        agent_id=agent.id, source_id=batch["source_id"], facts=write
    )
    db.log.clear()
    return agent, backend, db, batch["source_id"], write, preview


async def _execute(
    agent: Any,
    backend: Any,
    source_id: str,
    write: WriteFacts,
    preview: WritePreview,
) -> Any:
    return await backend.apply(
        agent_id=agent.id,
        source_id=source_id,
        execution=_Execution(),
        facts=write,
        preview_fingerprint=preview.preview_fingerprint,
        expected_affected_rows=preview.affected_rows,
    )


async def test_a_committed_write_reports_its_exact_evidence(tmp_path, monkeypatch):
    agent, backend, db, source_id, write, preview = await _prepared(
        tmp_path, monkeypatch
    )
    try:
        result = await _execute(agent, backend, source_id, write, preview)
        assert result.affected_rows == 1
        assert result.table == "public.companies"
        assert result.effect_observation.outcome is EffectOutcome.SUCCEEDED
        assert result.tool_data()["outcome"] == "committed"
        assert db.rows["new.test"]["name"] == "New Co"
        assert ("commit", False) in db.log
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("failure", "code"),
    (
        ("mutation_error", "write_constraint_violation"),
        ("mutation_permission", "write_permission_denied"),
        ("mutation_timeout", "write_statement_timeout"),
        ("mutation_conflict", "write_state_changed"),
        ("lock_timeout", "write_lock_timeout"),
    ),
)
async def test_a_failure_before_commit_is_known_not_applied(
    tmp_path, monkeypatch, failure, code
):
    agent, backend, db, source_id, write, preview = await _prepared(
        tmp_path, monkeypatch
    )
    errors = {
        "mutation_error": _ServerError("23505"),
        "mutation_permission": _ServerError("42501"),
        "mutation_timeout": _ServerError("57014"),
        "mutation_conflict": _ServerError("40001"),
    }
    db.mutation_error = errors.get(failure)
    if failure == "lock_timeout":
        db.lock_error = _ServerError("55P03")
    try:
        with pytest.raises(RelationalWriteError) as refused:
            await _execute(agent, backend, source_id, write, preview)
        assert refused.value.error_code == code
        assert "secret" not in str(refused.value)
        observation = refused.value.effect_observation
        assert observation is not None
        assert observation.outcome is EffectOutcome.NOT_APPLIED
        assert observation.payload is not None
        assert observation.payload["affected_rows"] == 0
        assert not db.rows
        assert ("rollback",) in db.log
    finally:
        await agent.close()


@pytest.mark.parametrize("difference", ("bad_count", "bad_key", "not_an_insert"))
async def test_rows_that_differ_from_the_card_roll_back(
    tmp_path, monkeypatch, difference
):
    agent, backend, db, source_id, write, preview = await _prepared(
        tmp_path, monkeypatch
    )
    setattr(db, difference, True)
    try:
        with pytest.raises(RelationalWriteError) as refused:
            await _execute(agent, backend, source_id, write, preview)
        assert refused.value.error_code == "write_affected_rows_mismatch"
        assert _outcome(refused.value) is EffectOutcome.NOT_APPLIED
        assert not db.rows and ("rollback",) in db.log
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("error", "code", "outcome"),
    (
        (ConnectionError("lost"), "write_outcome_unknown", EffectOutcome.UNCERTAIN),
        (
            _ServerError("23505"),
            "write_constraint_violation",
            EffectOutcome.NOT_APPLIED,
        ),
    ),
)
async def test_commit_certainty_is_classified_from_the_failure(
    tmp_path, monkeypatch, error, code, outcome
):
    agent, backend, db, source_id, write, preview = await _prepared(
        tmp_path, monkeypatch
    )
    db.commit_error = error
    try:
        with pytest.raises(RelationalWriteError) as refused:
            await _execute(agent, backend, source_id, write, preview)
        assert refused.value.error_code == code
        assert _outcome(refused.value) is outcome
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("planned", "code"),
    (
        (
            (("Insert", "public", "companies"), ("Delete", "public", "other")),
            "multiple_write_targets",
        ),
        ((("Update", "public", "other"),), "write_target_invalid"),
        ((("Update", "pg_temp_3", "companies"),), "write_target_invalid"),
        ((), "write_target_invalid"),
    ),
)
async def test_the_source_decides_the_target_before_preview_and_before_apply(
    tmp_path, monkeypatch, planned, code
):
    agent, backend, db, source_id, write, preview = await _prepared(
        tmp_path, monkeypatch
    )
    db.planned = list(planned)
    try:
        with pytest.raises(RelationalWriteError) as refused:
            await backend.preview(agent_id=agent.id, source_id=source_id, facts=write)
        assert refused.value.error_code == code
        with pytest.raises(RelationalWriteError) as refused:
            await _execute(agent, backend, source_id, write, preview)
        assert refused.value.error_code == code
        assert _outcome(refused.value) is EffectOutcome.NOT_APPLIED
        assert db.inserts() == 0 and not db.rows
    finally:
        await agent.close()


async def test_describing_a_write_releases_its_table_lock_before_the_scan(
    tmp_path, monkeypatch
):
    agent, backend, db, source_id, write, _preview = await _prepared(
        tmp_path, monkeypatch
    )
    try:
        await backend.preview(agent_id=agent.id, source_id=source_id, facts=write)
        names = [item[0] for item in db.log]
        assert ("describe", True) in db.log
        assert (
            names.index("savepoint")
            < names.index("describe")
            < names.index("savepoint_released")
            < names.index("scan")
        )
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("keys", "level"),
    (('[["id"], ["domain"]]', "rows"), ('[["id"]]', "count"), (None, "count")),
)
async def test_the_card_says_how_far_daita_can_verify(
    tmp_path, monkeypatch, keys, level
):
    agent, backend, db, source_id, write, _preview = await _prepared(
        tmp_path, monkeypatch
    )
    db.guardrails = {**db.guardrails, "keys": keys}
    try:
        preview = await backend.preview(
            agent_id=agent.id, source_id=source_id, facts=write
        )
        assert preview.verification == level
    finally:
        await agent.close()


async def test_a_failed_rollback_after_mutation_stays_uncertain(tmp_path, monkeypatch):
    agent, backend, db, source_id, write, preview = await _prepared(
        tmp_path, monkeypatch
    )
    db.mutation_error = _ServerError("23505")
    db.rollback_error = ConnectionError("lost")
    try:
        with pytest.raises(RelationalWriteError) as refused:
            await _execute(agent, backend, source_id, write, preview)
        assert refused.value.error_code == "write_outcome_unknown"
        assert _outcome(refused.value) is EffectOutcome.UNCERTAIN
    finally:
        await agent.close()


async def test_cancellation_during_commit_retains_uncertainty(tmp_path, monkeypatch):
    agent, backend, db, source_id, write, preview = await _prepared(
        tmp_path, monkeypatch
    )
    db.commit_error = asyncio.CancelledError()
    try:
        with pytest.raises(RelationalWriteCancelled) as cancelled:
            await _execute(agent, backend, source_id, write, preview)
        assert cancelled.value.effect_observation.outcome is EffectOutcome.UNCERTAIN
    finally:
        await agent.close()


async def test_cancellation_before_commit_is_known_not_applied(tmp_path, monkeypatch):
    agent, backend, db, source_id, write, preview = await _prepared(
        tmp_path, monkeypatch
    )
    db.mutation_error = asyncio.CancelledError()
    try:
        with pytest.raises(RelationalWriteCancelled) as cancelled:
            await _execute(agent, backend, source_id, write, preview)
        assert cancelled.value.effect_observation.outcome is EffectOutcome.NOT_APPLIED
    finally:
        await agent.close()


async def test_a_write_without_a_reserved_receipt_never_connects(tmp_path, monkeypatch):
    agent, backend, db, source_id, write, preview = await _prepared(
        tmp_path, monkeypatch
    )

    class Unreserved(_Execution):
        effect_receipt_id = None  # type: ignore[assignment]

    try:
        with pytest.raises(RelationalWriteError) as refused:
            await backend.apply(
                agent_id=agent.id,
                source_id=source_id,
                execution=Unreserved(),
                facts=write,
                preview_fingerprint=preview.preview_fingerprint,
                expected_affected_rows=preview.affected_rows,
            )
        assert refused.value.error_code == "write_receipt_unavailable"
        assert not db.log
    finally:
        await agent.close()


async def test_a_repeated_call_identity_returns_the_receipt_without_reconnecting(
    tmp_path, monkeypatch
):
    agent, provider, db, *_rest, batch, _clock, approvals = await create_fixture(
        tmp_path, monkeypatch
    )
    write = ToolCall(
        "write",
        "data_write",
        {
            "source": "Company research",
            "sql": batch["sql"],
            "parameters": batch["parameters"],
        },
    )
    provider.replace_script(
        (
            response(ToolCall("load", "toolbox_load", {"tool_names": ["data_write"]})),
            response(write),
            response(write),
            response(text="Done."),
        )
    )
    agent._embedded._capability_runtime._approval_handler = lambda request: _approve(
        approvals, request
    )
    try:
        await agent.run("Save it.")
        assert db.inserts() == 1
        assert len(approvals) == 1
    finally:
        await agent.close()


async def _approve(approvals: list[Any], request: Any) -> ApprovalDecision:
    approvals.append(request)
    return ApprovalDecision.APPROVE


@pytest.mark.parametrize(
    ("value", "type_name", "expected"),
    (
        ("1.50", "numeric", "1.50"),
        (7, "int8", 7),
        ("7", "int4", 7),
        (1.5, "float8", 1.5),
        (True, "bool", True),
        ("a", "text", "a"),
        (7, "text", "7"),
        (None, "date", None),
        ({"a": 1}, "jsonb", '{"a":1}'),
        ('{"a":1}', "json", '{"a":1}'),
    ),
)
def test_parameters_become_what_the_server_expects(
    value: object, type_name: str, expected: object
) -> None:
    assert str(native._bound_value(value, type_name)) == str(expected)


def test_a_parameter_that_cannot_fit_refuses_before_dispatch() -> None:
    class Parameter:
        name = "uuid"

    class Statement:
        def get_parameters(self) -> tuple[Parameter, ...]:
            return (Parameter(),)

    with pytest.raises(RelationalWriteError) as refused:
        native._bound_parameters(Statement(), ("not a uuid",))
    assert refused.value.error_code == "write_preview_failed"
