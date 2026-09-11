"""Shared helpers extracted from ``test_streaming.py``."""

from __future__ import annotations


def _openai_text_response(text: str) -> dict[str, object]:
    return {
        "id": "resp-1",
        "status": "completed",
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": ""},
                    {"type": "output_text", "text": text},
                ],
            }
        ],
        "usage": None,
    }
