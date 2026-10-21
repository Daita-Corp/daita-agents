"""A raw HTTP client drives one question to its answer through an approval."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from daita import __version__
from daita.host.codec import decode
from daita.loop.models import LoopExit
from tests.host.conftest import StartHost
from tests.support.approval_learning import (
    MockModelProvider,
    _call,
    _memory_call,
    _stop,
)

pytestmark = pytest.mark.acceptance

NOTE = "Revenue means paid invoice subtotal."


async def test_raw_httpx_question_approval_and_answer(start_host: StartHost) -> None:
    model = MockModelProvider((_call(_memory_call(content=NOTE)), _stop("Saved.")))
    harness = await start_host(model=model)
    transport = httpx.AsyncHTTPTransport(uds=str(harness.paths.socket))
    headers = {"X-Daita-Protocol": "1", "X-Daita-Version": __version__}
    async with httpx.AsyncClient(
        transport=transport, base_url="http://daita-host", headers=headers, timeout=10
    ) as http:
        created = await http.post(
            "/v1/agents", json={"name": "sales", "cwd": str(harness.work())}
        )
        assert created.status_code == 201
        answer = run_ref = None
        # after=0 is older than any host sequence: the stream opens with a reset,
        # which tells this client it is attached before it asks the question.
        async with (
            asyncio.timeout(30),
            http.stream(
                "GET", "/v1/agents/sales/events", params={"after": "0"}, timeout=None
            ) as stream,
        ):
            async for line in stream.aiter_lines():
                if not line.startswith("data:"):
                    continue
                record = json.loads(line[len("data:") :])
                data = record["data"]
                if record["event"] == "reset" and run_ref is None:
                    submitted = await http.post(
                        "/v1/agents/sales/runs",
                        json={"message": "Remember what revenue means."},
                    )
                    assert submitted.status_code == 202
                    run_ref = submitted.json()["v"]["run_ref"]
                if record["event"] == "interaction" and data["v"]["op"] == "opened":
                    interaction = data["v"]["interaction"]["v"]
                    request = interaction["request"]["v"]
                    assert NOTE in request["review_document"]
                    answered = await http.post(
                        f"/v1/agents/sales/interactions/{interaction['id']}",
                        json={"decision": "approve"},
                    )
                    assert answered.status_code == 200
                if record["event"] == "run_state" and data["v"]["state"] == "completed":
                    answer = decode(data["v"]["exit"], LoopExit).final_text
                    break

        state = (await http.get(f"/v1/agents/sales/runs/{run_ref}")).json()["v"]
        memory = (await http.get("/v1/agents/sales/memory")).json()

    assert answer == "Saved."
    assert state["state"] == "completed"
    assert memory == NOTE
