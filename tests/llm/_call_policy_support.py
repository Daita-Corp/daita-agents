"""Shared helpers extracted from ``test_call_policy.py``."""

from __future__ import annotations

from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    TextBlock,
)


def request(**kwargs):
    return ModelRequest(
        (CanonicalMessage(MessageRole.USER, content=(TextBlock("public"),)),), **kwargs
    )
