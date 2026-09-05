"""Offline regression coverage for the live benchmark's discovery assertions."""

from datetime import UTC, datetime

import pytest

from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.loop.models import LoopExit, LoopExitKind, RunInput, Transcript

from ._support import RunCapture, assert_on_demand_invocation


@pytest.mark.parametrize(
    ("steps", "valid"),
    (
        (("load", "invoke"), True),
        (("search", "load", "invoke"), True),
        (("search", "invoke"), False),
        (("search", "failed_load", "invoke"), False),
        (("search", "wrong_load", "invoke"), False),
        (("search", "invoke", "load"), False),
        (("search", "load", "failed_invoke"), False),
        (("search", "load"), False),
    ),
    ids=(
        "direct-load",
        "search-then-load",
        "missing-load",
        "failed-load",
        "wrong-tool-load",
        "load-after-invocation",
        "failed-invocation",
        "missing-invocation",
    ),
)
def test_on_demand_assertion_requires_exact_successful_load_before_invocation(
    steps: tuple[str, ...], valid: bool
) -> None:
    now = datetime(2026, 9, 4, tzinfo=UTC)
    user = CanonicalMessage(MessageRole.USER, content=(TextBlock("Cancel job-1."),))
    calls = {
        "search": ToolCall("search", "toolbox_search", {"query": "cancel job"}),
        "load": ToolCall("load", "toolbox_load", {"tool_names": ["job_cancel"]}),
        "failed_load": ToolCall(
            "failed_load", "toolbox_load", {"tool_names": ["job_cancel"]}
        ),
        "wrong_load": ToolCall(
            "wrong_load", "toolbox_load", {"tool_names": ["job_list"]}
        ),
        "invoke": ToolCall("invoke", "job_cancel", {"job_id": "job-1"}),
        "failed_invoke": ToolCall("failed_invoke", "job_cancel", {"job_id": "job-1"}),
    }
    messages = [user]
    for step in steps:
        messages.extend(
            (
                CanonicalMessage(MessageRole.ASSISTANT, tool_calls=(calls[step],)),
                CanonicalMessage(
                    MessageRole.TOOL,
                    content=(
                        ToolResultBlock(step, is_error=step.startswith("failed")),
                    ),
                ),
            )
        )
    capture = RunCapture(
        result=LoopExit(
            "run-1",
            "conversation-1",
            LoopExitKind.COMPLETED,
            "completed",
            now,
            final_text="Done.",
        ),
        transcript=Transcript(
            RunInput("run-1", "agent-1", "Cancel job-1.", now), tuple(messages)
        ),
        requests=(
            ModelRequest(
                (user,),
                tools=tuple(
                    ToolDefinition(name, name, {"type": "object"})
                    for name in ("toolbox_search", "toolbox_load")
                ),
            ),
        ),
    )
    if valid:
        assert_on_demand_invocation(capture, "job_cancel")
    else:
        with pytest.raises(AssertionError):
            assert_on_demand_invocation(capture, "job_cancel")
