from _workspace_support import workspace_for
import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from daita import Agent
from daita.hosting.embedded import AgentHomeError, HostActiveError
from daita.llm.models import FinishReason, ModelProfile, ModelRequest, ModelResponse
from daita.storage.sqlite_records import SourceReadMode


class _BlockingProvider:
    provider_id = "mock:blocking-host-lifecycle"

    def __init__(self) -> None:
        self.model_profile = ModelProfile(
            id=self.provider_id,
            context_window_tokens=128_000,
            max_output_tokens=8_192,
            supports_tools=True,
            supports_parallel_tools=True,
        )
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def supports_request_policy(self, request: ModelRequest) -> bool:
        del request
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        del request
        self.started.set()
        await self.release.wait()
        return ModelResponse(finish_reason=FinishReason.STOP, text="done")


def _sqlite_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO records VALUES (1, 'one')")


async def test_close_retains_writer_lock_until_blocked_run_terminalizes(tmp_path):
    provider = _BlockingProvider()
    agent = await Agent.create(
        "close-drains-run",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        workspace=workspace_for(tmp_path),
    )
    run = asyncio.create_task(agent.run("answer without tools"))
    started = asyncio.create_task(provider.started.wait())
    done, _pending = await asyncio.wait(
        (run, started), timeout=1, return_when=asyncio.FIRST_COMPLETED
    )
    assert started in done, (
        repr(run.exception() or run.result())
        if run.done()
        else repr(
            [
                (frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno)
                for frame in run.get_stack()
            ]
        )
    )
    closing = asyncio.create_task(agent.close())
    await asyncio.sleep(0)

    with pytest.raises(HostActiveError):
        await asyncio.wait_for(
            Agent.open(
                "close-drains-run",
                root=tmp_path,
                model=_BlockingProvider(),
                model_profile=provider.model_profile,
                workspace=workspace_for(tmp_path),
            ),
            timeout=1,
        )
    assert not closing.done()

    provider.release.set()
    result = await asyncio.wait_for(run, timeout=1)
    await asyncio.wait_for(closing, timeout=1)

    reopened_provider = _BlockingProvider()
    reopened = await asyncio.wait_for(
        Agent.open(
            "close-drains-run",
            root=tmp_path,
            model=reopened_provider,
            model_profile=reopened_provider.model_profile,
            workspace=workspace_for(tmp_path),
        ),
        timeout=1,
    )
    try:
        persisted = await reopened.conversation_runs(result.conversation_id)
        assert persisted[-1].result == result
    finally:
        await reopened.close()


async def test_close_rejects_a_queued_run_before_releasing_writer_ownership(tmp_path):
    provider = _BlockingProvider()
    agent = await Agent.create(
        "close-rejects-queued-run",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        workspace=workspace_for(tmp_path),
    )
    active = asyncio.create_task(agent.run("active run"))
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    queued = asyncio.create_task(agent.run("queued run"))
    await asyncio.sleep(0)
    closing = asyncio.create_task(agent.close())
    await asyncio.sleep(0)

    provider.release.set()
    await asyncio.wait_for(active, timeout=1)
    with pytest.raises(AgentHomeError, match="closed"):
        await asyncio.wait_for(queued, timeout=1)
    await asyncio.wait_for(closing, timeout=1)

    reopened_provider = _BlockingProvider()
    reopened = await Agent.open(
        "close-rejects-queued-run",
        root=tmp_path,
        model=reopened_provider,
        model_profile=reopened_provider.model_profile,
        workspace=workspace_for(tmp_path),
    )
    await reopened.close()


@pytest.mark.parametrize(
    "operation",
    ("detach", "refresh", "apply_permissions", "candidate_review"),
)
async def test_foreground_run_serializes_owned_host_mutations_but_not_inspection(
    tmp_path,
    operation: str,
):
    database = tmp_path / f"{operation}.sqlite"
    _sqlite_database(database)
    provider = _BlockingProvider()
    agent = await Agent.create(
        f"run-mutation-{operation}",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        workspace=workspace_for(tmp_path),
    )
    source = await agent.attach_sqlite(database)
    permission_preview = None
    if operation == "apply_permissions":
        permission_preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode=SourceReadMode.NONE,
            read_resource_ids=(),
            postgresql_update_scopes={},
        )

    run = asyncio.create_task(agent.run("answer without tools", source_id=source.id))
    await asyncio.sleep(0)
    assert not run.done(), repr(run.exception()) if run.done() else ""
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    mutation: asyncio.Task[Any]
    if operation == "detach":
        mutation = asyncio.create_task(agent.detach(source.id))
    elif operation == "refresh":
        mutation = asyncio.create_task(agent.refresh_source(source.id))
    elif operation == "apply_permissions":
        assert permission_preview is not None
        mutation = asyncio.create_task(
            agent.apply_source_permissions(
                source_id=source.id,
                confirmation_fingerprint=(permission_preview.confirmation_fingerprint),
            )
        )
    else:
        mutation = asyncio.create_task(agent.review_learning_candidates())
    await asyncio.sleep(0)

    assert not mutation.done()
    assert await asyncio.wait_for(agent.list_sources(), timeout=1) == (source,)
    assert await asyncio.wait_for(
        agent.list_catalog_resources(source_id=source.id), timeout=1
    )

    provider.release.set()
    await asyncio.wait_for(run, timeout=1)
    await asyncio.wait_for(mutation, timeout=1)
    await asyncio.wait_for(agent.close(), timeout=1)
