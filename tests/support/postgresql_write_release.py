"""Test-only model script and fault injection around real asyncpg operations.

Catalog truth, SQL execution and receipt persistence remain production-owned.
Only explicitly selected driver responses and cancellation points are altered.
The disconnect modes bracket a real COMMIT at the driver boundary; they are not
a network proxy or evidence of actual packet loss.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate


class WriteModel:
    """One bounded load → preview → exact write → observation script per run."""

    provider_id = "mock:postgresql-write-release"
    model_profile = ModelProfile(
        id=provider_id,
        context_window_tokens=256_000,
        max_output_tokens=2_048,
        supports_tools=True,
    )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.configure("upsert", {})

    def configure(self, operation: str, arguments: Mapping[str, object]) -> None:
        assert operation in {"update", "upsert"}
        self.operation = operation
        self.arguments = dict(arguments)
        self.phase = 0
        self.results: dict[str, ToolResultBlock] = {}
        self.preview_only = False

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        for message in request.messages:
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    self.results[block.call_id] = block
        calls: tuple[ToolCall, ...] = ()
        preview_name = f"data_preview_{self.operation}_rows"
        write_name = f"data_{self.operation}_rows"
        if self.phase == 0:
            calls = (
                ToolCall(
                    id="load",
                    name="toolbox_load",
                    arguments={"tool_names": (preview_name, write_name)},
                ),
            )
        elif self.phase == 1:
            calls = (
                ToolCall(id="preview", name=preview_name, arguments=self.arguments),
            )
        elif self.phase == 2:
            preview = self.results["preview"]
            if not preview.is_error and not self.preview_only:
                data = preview.output["data"]
                assert isinstance(data, Mapping)
                arguments = {
                    **self.arguments,
                    "preview_fingerprint": data["preview_fingerprint"],
                }
                if self.operation == "update":
                    arguments["expected_affected_rows"] = data["matched_rows"]
                calls = (ToolCall(id="write", name=write_name, arguments=arguments),)
        elif self.phase != 3:
            raise AssertionError("unexpected extra model request")
        self.phase += 1
        return ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
            tool_calls=calls,
            text=None if calls else "Inspect the recorded tool result and receipt.",
            # Fictional deterministic model usage, never represented as live usage.
            usage=ModelUsage(
                input_tokens=10,
                output_tokens=2,
                cost_estimate=CostEstimate.complete(Decimal("0")),
            ),
        )


class DriverProbe:
    """Observe real writes; inject one explicitly selected failure per test."""

    def __init__(self) -> None:
        self.mode: str | None = None
        self.mutations = 0
        self.commit_attempts = 0
        self.server_commits = 0
        self.write_pid: int | None = None
        self.worker: asyncio.Task[Any] | None = None
        self.reached = asyncio.Event()
        self.release = asyncio.Event()
        self.connections: list[Any] = []

    def wrap(self, connection: Any) -> Any:
        self.connections.append(connection)
        return _Connection(connection, self)


class _Connection:
    def __init__(self, connection: Any, probe: DriverProbe) -> None:
        self.connection = connection
        self.probe = probe

    def __getattr__(self, name: str) -> Any:
        return getattr(self.connection, name)

    def transaction(self, **kwargs: Any) -> Any:
        transaction = self.connection.transaction(**kwargs)
        if kwargs.get("readonly", False):
            return transaction
        self.probe.write_pid = self.connection.get_server_pid()
        return _Transaction(transaction, self.connection, self.probe)

    async def _after_mutation(self) -> None:
        self.probe.mutations += 1
        if self.probe.mode == "pause_after_mutation" and self.probe.mutations == 1:
            self.probe.worker = asyncio.current_task()
            self.probe.reached.set()
            await self.probe.release.wait()

    async def fetch(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        result = await self.connection.fetch(sql, *args, **kwargs)
        if sql.startswith("INSERT INTO"):
            await self._after_mutation()
        return result

    async def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        result = await self.connection.execute(sql, *args, **kwargs)
        if sql.startswith("UPDATE "):
            await self._after_mutation()
            if self.probe.mode == "wrong_update_count":
                return "UPDATE 0"
        return result


class _Transaction:
    def __init__(self, transaction: Any, connection: Any, probe: DriverProbe) -> None:
        self.transaction = transaction
        self.connection = connection
        self.probe = probe

    def __getattr__(self, name: str) -> Any:
        return getattr(self.transaction, name)

    async def commit(self) -> None:
        self.probe.commit_attempts += 1
        if self.probe.mode == "disconnect_before_commit":
            self.connection.terminate()
            raise ConnectionError("test-only disconnect before COMMIT dispatch")
        await self.transaction.commit()
        self.probe.server_commits += 1
        if self.probe.mode == "disconnect_after_commit":
            self.connection.terminate()
            raise ConnectionError("test-only loss of a real COMMIT confirmation")
        if self.probe.mode == "pause_after_commit":
            self.probe.worker = asyncio.current_task()
            self.probe.reached.set()
            await self.probe.release.wait()
