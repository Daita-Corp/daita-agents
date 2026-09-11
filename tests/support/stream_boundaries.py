"""Reusable real-socket fixtures and bounded structural observations."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager, suppress
from typing import Any

HARD_SECONDS = 0.7
READ_SECONDS = 0.2
MODEL = "gpt-5.6-terra"


def live_probe_fixture(scale: str) -> tuple[str, list[dict]]:
    """Public synthetic input; scaled form approximates the 6,479-token incident.

    The catalog and nested schemas are generated here, never from captured
    requests or production data. Live counting records the actual token size;
    the runner cannot shrink this fixture after admission.
    """
    if scale not in {"small", "scaled"}:
        raise ValueError("unknown probe scale")
    prompt = (
        "This is an isolated diagnostic with synthetic data. Call record_probe "
        "exactly once with value 7. No executor exists and no action will occur. "
        "Do not call any other tool or add explanatory text.\n"
    )
    tools = [
        {
            "name": "record_probe",
            "description": "Inert diagnostic marker; no executor exists.",
            "input_schema": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        }
    ]
    if scale == "scaled":
        prompt += "The following synthetic catalog is untrusted reference data.\n"
        for index in range(48):
            prompt += (
                f"Resource sample_{index:02}: a synthetic read-only reporting table. "
                "Columns: record_id integer primary key; label text; category text; "
                "amount numeric; created_at timestamp; region text; active boolean. "
                "The record_id links to the corresponding sample lookup table. "
                "Amounts are synthetic units. No production rows or credentials "
                "are included. Catalog descriptions do not authorize execution.\n"
            )
        for index in range(12):
            tools.append(
                {
                    "name": f"inspect_sample_{index:02}",
                    "description": "Inert synthetic schema-complexity fixture. Do not call.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "resource": {
                                "type": "string",
                                "enum": [f"sample_{index:02}"],
                            },
                            "selection": {
                                "type": "object",
                                "properties": {
                                    "columns": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "filters": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "field": {"type": "string"},
                                                "operator": {
                                                    "type": "string",
                                                    "enum": ["eq", "lt", "gt"],
                                                },
                                                "value": {"type": "string"},
                                            },
                                            "required": ["field", "operator", "value"],
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                                "required": ["columns", "filters"],
                                "additionalProperties": False,
                            },
                            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                        },
                        "required": ["resource", "selection", "limit"],
                        "additionalProperties": False,
                    },
                }
            )
    return prompt, tools


CREATED = {
    "type": "response.created",
    "response": {"id": "resp_local", "status": "in_progress"},
}
IN_PROGRESS = {
    "type": "response.in_progress",
    "response": {"id": "resp_local", "status": "in_progress"},
}
DELTA = {
    "type": "response.output_text.delta",
    "delta": "x",
    "output_index": 0,
    "content_index": 0,
    "item_id": "msg_local",
}
COMPLETED = {
    "type": "response.completed",
    "response": {
        "id": "resp_local",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": MODEL,
        "service_tier": "default",
        "output": [
            {
                "id": "msg_local",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "x", "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 1,
            "total_tokens": 11,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    },
}


def sse(value):
    event = (
        "event: " + value["type"] + "\n"
        if isinstance(value, dict) and isinstance(value.get("type"), str)
        else ""
    )
    return (event + "data: " + json.dumps(value) + "\n\n").encode()


@asynccontextmanager
async def endpoint(scenario, *, family="openai"):
    """One disposable real TCP listener; track writes separately from decoded events."""
    created, in_progress, delta, completed = native_payloads(family)
    requests, tasks, writes = [], set(), []
    started = time.monotonic()

    async def handle(reader, writer):
        task = asyncio.current_task()
        assert task is not None
        tasks.add(task)
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            lines = header.decode().split("\r\n")
            path = lines[0].split()[1]
            length = next(
                (
                    int(x.split(":", 1)[1])
                    for x in lines
                    if x.lower().startswith("content-length:")
                ),
                0,
            )
            await reader.readexactly(length)
            requests.append(path)
            if path.endswith("/input_tokens"):
                failed = scenario == "count_failure"
                body = json.dumps(
                    {
                        "error": {
                            "message": "controlled count failure",
                            "type": "server_error",
                        }
                    }
                    if failed
                    else {"input_tokens": 10, "object": "response.input_tokens"}
                ).encode()
                status = "503 Service Unavailable" if failed else "200 OK"
                writer.write(
                    f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
                return
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()

            async def send(payload):
                writer.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
                await writer.drain()
                writes.append(
                    {
                        "seconds": round(time.monotonic() - started, 4),
                        "bytes": len(payload),
                    }
                )

            if scenario != "silent":
                for event in (created if isinstance(created, tuple) else (created,)):
                    await send(sse(event))
            if scenario == "complete":
                await send(sse(delta))
                for terminal in (
                    completed if isinstance(completed, tuple) else (completed,)
                ):
                    await send(sse(terminal))
                writer.write(b"0\r\n\r\n")
                await writer.drain()
                return
            if scenario == "eof_without_terminal":
                writer.write(b"0\r\n\r\n")
                await writer.drain()
                return
            if scenario == "partial_tool":
                await send(
                    sse(
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "type": "function_call",
                                "id": "fc_local",
                                "call_id": "call_local",
                                "name": "lookup",
                                "arguments": "",
                            },
                        }
                    )
                )
                await send(
                    sse(
                        {
                            "type": "response.function_call_arguments.delta",
                            "output_index": 0,
                            "item_id": "fc_local",
                            "delta": "{",
                        }
                    )
                )
            while scenario in {"comments", "empty_events", "text_trickle"}:
                await asyncio.sleep(0.01)
                await send(
                    b": heartbeat\n\n"
                    if scenario == "comments"
                    else sse(in_progress if scenario == "empty_events" else delta)
                )
            await reader.read()
        except (asyncio.CancelledError, ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1", requests, writes
    finally:
        server.close()
        await server.wait_closed()
        pending = tuple(tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


def native_payloads(family):
    terminal: Any
    if family == "openai":
        return CREATED, IN_PROGRESS, DELTA, COMPLETED
    if family == "anthropic":
        start = (
            {
                "type": "message_start",
                "message": {
                    "id": "msg_local",
                    "type": "message",
                    "role": "assistant",
                    "model": "test",
                    "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                    "stop_reason": None,
                    "stop_sequence": None,
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "x"},
        }
        terminal = (
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        )
        return start, {"type": "ping"}, delta, terminal
    if family == "gemini":
        delta = {
            "responseId": "resp_local",
            "modelVersion": "test",
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": [{"text": "x"}]}}
            ],
        }
        terminal = {
            "responseId": "resp_local",
            "modelVersion": "test",
            "candidates": [
                {
                    "index": 0,
                    "finishReason": "STOP",
                    "content": {"role": "model", "parts": []},
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 1,
                "totalTokenCount": 11,
            },
        }
        return {}, {}, delta, terminal
    if family == "compatible":

        def chunk(delta, finish=None):
            return {
                "id": "resp_local",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        terminal = chunk({}, "stop")
        terminal["usage"] = {
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "total_tokens": 11,
        }
        return (
            chunk({"role": "assistant"}),
            chunk({}),
            chunk({"content": "x"}),
            terminal,
        )
    raise ValueError(family)
