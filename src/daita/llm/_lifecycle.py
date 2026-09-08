"""Private cleanup mechanics shared by provider owners and stream consumers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypeVar

from .models import ModelRequest

_T = TypeVar("_T")


def input_count_deadline(request: ModelRequest) -> float:
    """Bound one count without renewing the enclosing logical-request deadline."""
    deadline = asyncio.get_running_loop().time() + request.input_count_timeout_seconds
    return deadline if request.deadline is None else min(deadline, request.deadline)


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
    except asyncio.CancelledError as error:
        failed = True
        # A consumer can be cancelled between yielded events, while no anext is
        # active. Forward that cancellation through canonical generators so the
        # router can retain earlier attempt usage before their cleanup runs.
        throw = getattr(stream, "athrow", None) if close is None else None
        if callable(throw):
            try:
                await throw(error)
            except asyncio.CancelledError as cancelled:
                if cancelled is not error:
                    from .errors import (
                        interrupted_model_usage,
                        with_cancelled_model_usage,
                    )

                    with_cancelled_model_usage(
                        error, interrupted_model_usage(cancelled)
                    )
            except BaseException:
                pass  # Cleanup must not replace the consumer's cancellation.
        raise
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
