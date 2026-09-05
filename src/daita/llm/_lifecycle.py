"""Private cleanup mechanics shared by provider owners and stream consumers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypeVar

_T = TypeVar("_T")


async def await_cleanup(cleanup: asyncio.Future[None]) -> None:
    """Join one cleanup to completion, then propagate cancellation or its error.

    Owners retain the same task so concurrent and later callers observe the
    same outcome. Cleanup is never retried or left running after this returns.
    """

    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
    cleanup.result()
    if cancelled:
        raise asyncio.CancelledError


@asynccontextmanager
async def closing_stream(
    stream: AsyncIterator[_T],
    *,
    close: Callable[[], Awaitable[None]] | None = None,
) -> AsyncIterator[AsyncIterator[_T]]:
    """Release a request stream on completion, failure, cancellation, or exit.

    SDK adapters supply their native transport close operation. Canonical
    consumers use aclose in the iteration task/context: moving generator
    finalization into a new task can break ContextVar tokens across yields.
    Plain caller-supplied AsyncIterators remain supported.
    A cleanup failure must not replace the original model failure/cancellation.
    """

    cleanup = close if close is not None else getattr(stream, "aclose", None)
    failed = False
    try:
        yield stream
    except BaseException:
        failed = True
        raise
    finally:
        if cleanup is not None:
            try:
                if close is None:
                    await cleanup()
                else:
                    await await_cleanup(asyncio.ensure_future(cleanup()))
            except BaseException:
                if not failed:
                    raise
