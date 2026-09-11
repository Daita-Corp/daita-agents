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


@pytest.mark.parametrize("case", ["valid", "bytes", "approval", "shape"])
async def test_release_script_uses_current_public_contract_and_normalizes_invalid_input(
    tmp_path, monkeypatch, case
):
    # Reuse existing fake-I/O conformance support solely to verify that the new
    # script drives the current public contract. This is not PostgreSQL evidence.
    fixture = await create_fixture(tmp_path, monkeypatch)
    agent, _, db, _, _, _, _, batch, _, _ = fixture
    await agent.close()
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
        arguments = {**batch, "evidence_call_ids": ()}
        if case in {"bytes", "approval"}:
            arguments["rows"] = [
                {
                    "domain": str(i),
                    "name": "N",
                    "evidence_url": (
                        "https://evidence.test/" if case == "bytes" else ""
                    )
                    + str(i),
                }
                for i in range(1000)
            ]
        elif case == "shape":
            arguments["rows"] = [{"domain": "new.test", "name": "New"}]
        model.configure("upsert", arguments)
        result = await agent.run("Store the fixture company once.")
        assert result.reason == "completed"
        if case != "valid":
            error = model.results["preview"].output["error"]
            assert isinstance(error, Mapping)
            assert error["code"] == "upsert_invalid_batch"
            assert "no write was dispatched" in str(error["message"])
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

    async def execute(self, sql, *args, **kwargs):
        self.log.append(sql)
        return "UPDATE 1"

    async def fetch(self, sql, *args, **kwargs):
        self.log.append(sql)
        return [{"id": 123}]


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
        work = wrapped.fetch("INSERT INTO canary VALUES ($1)", 1)
    else:
        work = wrapped.transaction().commit()
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
    status = await probe.wrap(connection).execute("UPDATE canary SET name = $1", "New")
    assert status == "UPDATE 0"
    assert connection.log == ["UPDATE canary SET name = $1"]
    assert probe.mutations == 1 and probe.server_commits == 0
