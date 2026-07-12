"""Normalization shared by DB model-response JSON boundaries."""

from __future__ import annotations

import re


def strip_json_fence(content: str) -> str:
    """Strip the exact optional JSON Markdown fence accepted by DB models."""
    stripped = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped
