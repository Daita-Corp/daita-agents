from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time

from daita import Agent
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.loop.models import LoopExitKind, Readiness, Turn
from daita.operations.checkpoints import ModelCallStatus, OperationSnapshot
from daita.operations.models import (
    ActionProposal,
    Evidence,
    Observation,
    OperationStatus,
)


class _TextContext:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert tools == ()
        message = operation.trigger.payload["message"]
        assert isinstance(message, str)
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock(message),),
                ),
            ),
        )


class _TextDomain:
    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        return ()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        raise AssertionError("text-only recovery has no actions")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("text-only recovery has no observations")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        assert text == "Recovered after the process was killed."
        return Readiness(
            allowed=True,
            code="ready.process_kill_recovery",
            message="The resumed response is complete.",
            evaluated_at=operation.operation.updated_at,
        )


class _RecoveryProvider:
    provider_id = "mock:real-process-kill"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text="Recovered after the process was killed.",
            finish_reason=FinishReason.STOP,
            usage=ModelUsage(input_tokens=9, output_tokens=6),
        )


_CHILD = textwrap.dedent("""
    import asyncio
    import os
    from pathlib import Path
    import sys

    from daita import Agent
    from daita.llm.models import (
        CanonicalMessage,
        MessageRole,
        ModelRequest,
        TextBlock,
    )
    from daita.loop.models import Readiness


    class Context:
        async def build(self, operation, turn, tools):
            assert tools == ()
            return ModelRequest(
                operation_id=operation.operation.id,
                turn_id=turn.id,
                messages=(
                    CanonicalMessage(
                        agent_id=operation.operation.agent_id,
                        operation_id=operation.operation.id,
                        turn_id=turn.id,
                        role=MessageRole.USER,
                        content=(TextBlock("Persist before blocking."),),
                    ),
                ),
            )


    class Domain:
        def tool_views(self, operation):
            return ()

        async def validate_action(self, call, operation):
            raise AssertionError("text-only child has no actions")

        async def project_observation(self, evidence):
            raise AssertionError("text-only child has no observations")

        async def evaluate_final_answer(self, text, operation):
            return Readiness(
                allowed=True,
                code="ready",
                message="ready",
                evaluated_at=operation.operation.updated_at,
            )


    class BlockingProvider:
        provider_id = "mock:real-process-kill"

        def __init__(self, marker):
            self.marker = marker

        async def generate(self, request):
            descriptor = os.open(
                self.marker,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                os.write(descriptor, request.operation_id.encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            await asyncio.Event().wait()
            raise AssertionError("the child must be killed while blocked")


    async def main():
        root = Path(sys.argv[1])
        marker = sys.argv[2]
        agent = await Agent.create(
            "atlas",
            root=root,
            model=BlockingProvider(marker),
            context_builder=Context(),
            domain=Domain(),
        )
        await agent.run("Start durable work, then block in provider I/O.")


    asyncio.run(main())
    """)


async def _wait_for_durable_marker(
    process: subprocess.Popen[bytes],
    marker: Path,
    *,
    timeout: float,
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists():
            operation_id = marker.read_text(encoding="utf-8")
            if operation_id:
                return operation_id
        return_code = process.poll()
        if return_code is not None:
            stderr = b"" if process.stderr is None else process.stderr.read()
            raise AssertionError(
                "recovery child exited before its durable checkpoint: "
                f"{return_code}: {stderr.decode('utf-8', errors='replace')}"
            )
        await asyncio.sleep(0.02)
    raise AssertionError("recovery child did not reach its durable checkpoint")


async def test_real_process_kill_reopens_and_resumes_the_committed_checkpoint(
    tmp_path: Path,
) -> None:
    next_root = Path(__file__).resolve().parents[2]
    marker = tmp_path / "provider-entered"
    state_root = tmp_path / "state"
    environment = os.environ.copy()
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(next_root / "src"), prior_pythonpath) if part
    )
    process = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(state_root), str(marker)],
        cwd=next_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        operation_id = await _wait_for_durable_marker(
            process,
            marker,
            timeout=10.0,
        )
        process.kill()
        return_code = process.wait(timeout=5.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)

    if os.name == "posix":
        assert return_code == -signal.SIGKILL
    else:
        assert return_code != 0

    provider = _RecoveryProvider()
    reopened = await Agent.open(
        "atlas",
        root=state_root,
        model=provider,
        context_builder=_TextContext(),
        domain=_TextDomain(),
    )
    try:
        interrupted = await reopened.inspect(operation_id)
        assert interrupted.operation.status is OperationStatus.RUNNING
        assert len(interrupted.model_calls) == 1
        assert interrupted.model_calls[0].status is ModelCallStatus.STARTED
        assert interrupted.events[-1].type == "model_call.started"

        result = await reopened.resume(operation_id)
        recovered = await reopened.inspect(operation_id)

        assert result.kind is LoopExitKind.COMPLETED
        assert result.operation_id == operation_id
        assert recovered.operation.status is OperationStatus.SUCCEEDED
        assert len(recovered.model_calls) == 1
        assert recovered.model_calls[0].status is ModelCallStatus.COMPLETED
        assert recovered.model_calls[0].response is not None
        assert recovered.model_calls[0].response.usage == ModelUsage(
            input_tokens=9,
            output_tokens=6,
        )
        assert len(provider.requests) == 1
        assert (
            sum(event.type == "model_call.started" for event in recovered.events) == 1
        )
        assert (
            sum(event.type == "model_response.recorded" for event in recovered.events)
            == 1
        )
        assert not any(
            event.type == "operation.interrupted" for event in recovered.events
        )
        state_path = reopened.home / "state.db"
    finally:
        await reopened.close()

    with sqlite3.connect(state_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
