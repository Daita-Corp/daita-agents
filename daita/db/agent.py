"""
User-facing facade for the new database runtime.
"""

from __future__ import annotations

from typing import Any

from .models import DbOperationResult, DbRequest, DbRuntimeInspection
from .runtime import DbRuntime


class DbAgent:
    """Facade returned by the future `Agent.from_db()` implementation."""

    def __init__(self, *, runtime: DbRuntime, name: str | None = None) -> None:
        self.runtime = runtime
        self.name = name

    @property
    def operations(self) -> tuple[DbOperationResult, ...]:
        """Typed operation results retained by the runtime."""
        return self.runtime.operation_results

    @property
    def audit_log(self) -> tuple[dict[str, Any], ...]:
        """Redacted operation audit summaries retained by the runtime."""
        return self.runtime.audit_log

    async def run(self, prompt: str, **kwargs) -> str:
        """Run a DB request and return the synthesized answer string."""
        result = await self.run_detailed(prompt, **kwargs)
        return result.answer or ""

    async def run_detailed(self, prompt: str, **kwargs) -> DbOperationResult:
        """Run a DB request and return the typed operation result."""
        request = _request_from_kwargs(prompt, kwargs)
        return await self.runtime.run(request)

    async def describe(self) -> DbRuntimeInspection:
        """Return runtime diagnostics for inspection."""
        return await self.runtime.inspect()

    async def stop(self) -> None:
        """Release runtime resources."""
        await self.runtime.teardown()

    async def teardown(self) -> None:
        """Alias for framework code that manages runtimes directly."""
        await self.stop()

    async def stream(self, prompt: str, **kwargs):
        """Streaming will be implemented once DB synthesis exists."""
        result = await self.run_detailed(prompt, **kwargs)
        yield result


def _request_from_kwargs(prompt: str, kwargs: dict[str, Any]) -> DbRequest:
    values = dict(kwargs)
    metadata = dict(values.pop("metadata", {}) or {})
    metadata.update(values)
    return DbRequest(
        prompt=prompt,
        user_id=metadata.pop("user_id", None),
        session_id=metadata.pop("session_id", None),
        source_scope=tuple(metadata.pop("source_scope", ()) or ()),
        mode=metadata.pop("mode", None),
        requested_capabilities=tuple(metadata.pop("requested_capabilities", ()) or ()),
        constraints=dict(metadata.pop("constraints", {}) or {}),
        metadata=metadata,
    )
