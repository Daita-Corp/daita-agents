"""Shared helpers extracted from ``test_provider_lifecycle.py``."""

from __future__ import annotations

import asyncio
import json

import httpx


class _ResponseBody(httpx.AsyncByteStream):
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.closed = False
        self.reading = asyncio.Event()

    async def __aiter__(self):
        yield b'data: {"type":"response.output_text.delta","delta":"done"}\n\n'
        if self.mode == "cancel":
            self.reading.set()
            await asyncio.Event().wait()
        if self.mode == "malformed":
            yield b'data: {"type":"response.output_text.delta","delta":123}\n\n'
            return
        response = {
            "id": "offline",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.6-terra",
            "output": [
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "done", "annotations": []}
                    ],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        yield (
            "data: "
            + json.dumps({"type": "response.completed", "response": response})
            + "\n\n"
        ).encode()

    async def aclose(self) -> None:
        self.closed = True
