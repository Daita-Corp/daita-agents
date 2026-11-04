"""Offline checks of the release script and driver fault injection mechanics."""

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from daita import Agent, ApprovalDecision
from daita.capabilities import EffectOutcome
from tests.support.native_writes import create_fixture
from tests.support.postgresql_write_release import DriverProbe, WriteModel
from tests.support.workspace import workspace_for


@pytest.mark.parametrize(
    ("case", "code"),
    (
        ("valid", None),
        ("bytes", "write_too_large"),
        ("shape", "write_preview_failed"),
    ),
)
async def test_release_script_uses_current_public_contract_and_normalizes_invalid_input(
    tmp_path, monkeypatch, case, code
):
    # Reuse existing fake-I/O conformance support solely to verify that the new
    # script drives the current public contract. This is not PostgreSQL evidence.
    fixture = await create_fixture(tmp_path, monkeypatch)
    agent, _, db, _, _, _, _, batch, _, _ = fixture
    await agent.close()
    db.log.clear()
    model = WriteModel()
    approvals = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.open(
        "native-acceptance",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=model,
        model_profile=model.model_profile,
        approval_handler=approve,
    )
    try:
        arguments = dict(batch)
        if case == "bytes":
            arguments["parameters"] = ("new.test", "N" * 70_000, "https://e.test")
        elif case == "shape":
            arguments["sql"] = (
                "INSERT INTO public.companies (domain) VALUES ($1) "
                "ON CONFLICT ON CONSTRAINT companies_domain_key DO NOTHING"
            )
            arguments["parameters"] = ("new.test",)
        model.configure(arguments)
        result = await agent.run("Store the fixture company once.")
        assert result.reason == "completed"
        if code is not None:
            error = model.results["preview"].output["error"]
            assert isinstance(error, Mapping)
            assert error["code"] == code
            assert len(model.requests) == 3 and result.usage.total_tokens == 36
            assert not approvals and not db.rows and not db.log
            assert not await agent.list_effects()
            return
        assert model.results["preview"].is_error is False
        assert model.results["write"].is_error is False
        assert len(model.requests) == 4 and result.usage.total_tokens == 48
        receipts = await agent.list_effects()
        assert len(receipts) == 1 and receipts[0].outcome is EffectOutcome.SUCCEEDED
        assert len(approvals) == 1 and len(db.rows) == 1
    finally:
        await agent.close()


class Statement:
    def __init__(self, connection, sql):
        self.connection = connection
        self.sql = sql

    async def cursor(self, *args, **kwargs):
        self.connection.log.append(self.sql)
        yield {"__daita_key": "1", "__daita_inserted": False}


class Connection:
    def __init__(self):
        self.log = []
        self.terminated = False

    def get_server_pid(self):
        return 42

    def terminate(self):
        self.terminated = True
        self.log.append("terminate")

    def transaction(self, **kwargs):
        return SimpleNamespace(commit=self.commit)

    async def commit(self):
        self.log.append("real-commit")

    async def prepare(self, sql, **kwargs):
        return Statement(self, sql)

    async def fetch(self, sql, *args, **kwargs):
        self.log.append(sql)
        return [{"id": 123}]


async def _drain(rows):
    return [row async for row in rows]


@pytest.mark.parametrize(
    ("mode", "expected", "commits"),
    [
        (None, ["real-commit"], 1),
        ("disconnect_before_commit", ["terminate"], 0),
        ("disconnect_after_commit", ["real-commit", "terminate"], 1),
    ],
)
async def test_disconnect_injection_brackets_exactly_one_driver_commit(
    mode, expected, commits
):
    connection = Connection()
    probe = DriverProbe()
    probe.mode = mode
    transaction = probe.wrap(connection).transaction(isolation="read_committed")
    if mode is None:
        await transaction.commit()
    else:
        with pytest.raises(ConnectionError):
            await transaction.commit()
    assert connection.log == expected
    assert probe.commit_attempts == 1 and probe.server_commits == commits
    assert probe.write_pid == 42


async def test_readonly_preview_commit_is_not_a_write_fault_target():
    connection = Connection()
    probe = DriverProbe()
    probe.mode = "disconnect_before_commit"
    await probe.wrap(connection).transaction(readonly=True).commit()
    assert connection.log == ["real-commit"]
    assert probe.commit_attempts == 0 and probe.write_pid is None


@pytest.mark.parametrize("mode", ["pause_after_mutation", "pause_after_commit"])
async def test_cancellation_injection_waits_until_the_driver_operation_finishes(mode):
    connection = Connection()
    probe = DriverProbe()
    probe.mode = mode
    wrapped = probe.wrap(connection)
    if mode == "pause_after_mutation":
        statement = await wrapped.prepare("INSERT INTO canary VALUES ($1)")
        work = _drain(statement.cursor(1))
    else:
        work = wrapped.transaction(isolation="read_committed").commit()
    task = asyncio.create_task(work)
    try:
        await asyncio.wait_for(probe.reached.wait(), timeout=1)
        assert len(connection.log) == 1
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(connection.log) == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_count_mismatch_is_injected_after_real_update_completion():
    connection = Connection()
    probe = DriverProbe()
    probe.mode = "wrong_update_count"
    statement = await probe.wrap(connection).prepare("UPDATE canary SET name = $1")
    assert await _drain(statement.cursor("New")) == []
    assert connection.log == ["UPDATE canary SET name = $1"]
    assert probe.mutations == 1 and probe.server_commits == 0
