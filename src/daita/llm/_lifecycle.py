"""Private cleanup mechanics shared by provider owners and stream consumers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, TypeVar, cast

from .models import ModelRequest, ModelResponse, ModelStreamEvent

_T = TypeVar("_T")
_NATIVE_EOF = object()


async def await_cleanup(
    cleanup: asyncio.Future[Any], *, deadline: float, owner: NativeOwner
) -> None:
    """Observe one retained retirement with an explicit, never-renewed deadline."""
    shutdown = getattr(cleanup, "_daita_shutdown", None)
    if shutdown is None:
        owner.retain(cleanup)
        shutdown = NativeShutdown(owner, cleanup, deadline)
        setattr(cleanup, "_daita_shutdown", shutdown)  # noqa: B010
    await shutdown.wait(deadline=deadline)


def shutdown_deadline(deadline: float | None = None, *, seconds: float = 5.0) -> float:
    proposed = asyncio.get_running_loop().time() + seconds
    return proposed if deadline is None else deadline


@asynccontextmanager
async def closing_stream(
    stream: AsyncIterator[_T],
) -> AsyncIterator[AsyncIterator[_T]]:
    """Release a request stream on completion, failure, cancellation, or exit.

    Native transport release is bounded by AttemptLifecycle. Canonical
    consumers use aclose in the iteration task/context: moving generator
    finalization into a new task can break ContextVar tokens across yields.
    Plain caller-supplied AsyncIterators remain supported.
    A cleanup failure must not replace the original model failure/cancellation.
    """

    cleanup = getattr(stream, "aclose", None)
    failed = False
    try:
        yield stream
    except asyncio.CancelledError as error:
        failed = True
        # A consumer can be cancelled between yielded events, while no anext is
        # active. Forward that cancellation through canonical generators so the
        # router can retain earlier attempt usage before their cleanup runs.
        throw = getattr(stream, "athrow", None)
        if callable(throw):
            try:
                await cast(Callable[[BaseException], Awaitable[object]], throw)(error)
            except asyncio.CancelledError as cancelled:
                if cancelled is not error:
                    from .errors import (
                        interrupted_attempt_diagnostic,
                        interrupted_model_usage,
                        with_cancelled_model_usage,
                    )

                    with_cancelled_model_usage(
                        error, interrupted_model_usage(cancelled)
                    )
                    diagnostic = interrupted_attempt_diagnostic(cancelled)
                    if diagnostic is not None:
                        setattr(  # noqa: B010
                            error, "_daita_attempt_diagnostic", diagnostic
                        )
            except BaseException:  # noqa: BLE001
                pass  # Cleanup must not replace the consumer's cancellation.
        raise
    except BaseException:
        failed = True
        raise
    finally:
        if cleanup is not None:
            try:
                await cleanup()
            except BaseException:
                if not failed:
                    raise


class NativeOwner:
    """Retain unresolved native work and permanently close further admission.

    Only SDK/transport tasks may enter this owner, never tool execution or
    persistence. A late task is drained for exceptions but cannot heal the owner.
    """

    def __init__(self) -> None:
        self.tasks: set[asyncio.Future[Any]] = set()
        self.poisoned = False

    def require_available(self) -> None:
        if self.poisoned:
            from .errors import ModelProviderError, ProviderErrorCode, before_generation

            raise before_generation(
                ModelProviderError(ProviderErrorCode.OWNER_UNAVAILABLE),
                code="native_owner_unavailable",
            )

    def start(self, operation: Awaitable[_T]) -> asyncio.Future[_T]:
        self.require_available()
        task = asyncio.ensure_future(operation)
        return self.retain(task)

    def retain(self, task: asyncio.Future[_T]) -> asyncio.Future[_T]:
        """Account for release work belonging to an already admitted native call."""
        self.tasks.add(task)

        def drained(done: asyncio.Future[Any]) -> None:
            self.tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(drained)
        return task


class CloseCoordinator:
    """Own one retained provider close operation and its never-renewed outcome."""

    def __init__(self, owner: NativeOwner) -> None:
        self._owner = owner
        self._task: asyncio.Future[None] | None = None

    @property
    def started(self) -> bool:
        return self._task is not None

    async def close(
        self,
        operation: Callable[[], Awaitable[None]],
        *,
        deadline: float | None = None,
    ) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(operation())
        await await_cleanup(
            self._task,
            deadline=shutdown_deadline(deadline),
            owner=self._owner,
        )

    async def drain(self, *, deadline: float | None = None) -> None:
        """Close admission after the native work already owned by this provider."""
        limit = shutdown_deadline(deadline)
        if self._task is None:
            active = tuple(self._owner.tasks)

            async def finish() -> None:
                results = await asyncio.gather(
                    *(
                        await_cleanup(task, deadline=limit, owner=self._owner)
                        for task in active
                    ),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result

            self._task = asyncio.create_task(finish())
        await await_cleanup(self._task, deadline=limit, owner=self._owner)


async def join_until(task: asyncio.Future[_T], deadline: float) -> _T:
    """Wait without transferring cancellation to the task or waiting for its stop."""
    remaining = max(0.0, deadline - asyncio.get_running_loop().time())
    done, _ = await asyncio.wait((task,), timeout=remaining)
    if not done:
        raise TimeoutError
    return task.result()


class NativeStream:
    """One task owns the entire native scope; at most one item crosses to decoding.

    The consumer's generator/context never moves tasks. Demand and result futures
    have one slot and no prefetch: no next read starts until decoding asks for it.
    Native context-manager entry and exit both occur inside ``source`` in _run.
    """

    def __init__(
        self,
        owner: NativeOwner,
        source: AsyncIterator[_T],
        lifecycle: AttemptLifecycle | None = None,
    ) -> None:
        self._source = source
        self.lifecycle = lifecycle
        self._demand = asyncio.Event()
        self._result: asyncio.Future | None = None
        self._stopping = False
        self.task = owner.start(self._run())

    async def _run(self) -> None:
        try:
            while True:
                await self._demand.wait()
                self._demand.clear()
                if self._stopping:
                    return
                try:
                    item = await anext(self._source)
                except StopAsyncIteration:
                    if self._result is not None and not self._result.done():
                        self._result.set_result((False, None))
                    return
                if self._stopping:
                    return
                if self._result is not None and not self._result.done():
                    self._result.set_result(
                        (item is not _NATIVE_EOF, None if item is _NATIVE_EOF else item)
                    )
        finally:
            close = getattr(self._source, "aclose", None)
            if close is not None:
                await close()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._stopping:
            raise StopAsyncIteration
        if self.task.done():
            self.task.result()
            raise StopAsyncIteration
        self._result = asyncio.get_running_loop().create_future()
        self._demand.set()
        timeout = None
        lifecycle = self.lifecycle
        if lifecycle is not None:
            lifecycle.check_execution()
            timeout = max(0, lifecycle._expiry()[0] - asyncio.get_running_loop().time())
        done, _ = await asyncio.wait(
            (self._result, self.task),
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            assert lifecycle is not None
            lifecycle.timeout_reason = lifecycle._expiry()[1]
            raise lifecycle._timeout_error()
        if self._result in done:
            present, value = self._result.result()
            if present:
                return value
        else:
            self.task.result()
        raise StopAsyncIteration

    def stop(self, *, cancel: bool) -> None:
        if not self._stopping:
            self._stopping = True
            self._demand.set()
            if cancel and not self.task.done():
                self.task.cancel()


class NativeShutdown:
    """A once-only bounded retirement, with the native task retained by its owner."""

    def __init__(
        self,
        owner: NativeOwner,
        task: asyncio.Future[Any],
        deadline: float,
        *,
        cancel_confirms_release: bool = False,
    ):
        self.owner = owner
        self.native_task = task
        self.cancel_confirms_release = cancel_confirms_release
        self.deadline = deadline
        self.task = asyncio.create_task(self._finish())
        # An earlier waiter may leave before the retained supervisor finishes.
        self.task.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )

    async def _finish(self) -> None:
        from .errors import ModelProviderError, ProviderErrorCode

        try:
            await join_until(self.native_task, self.deadline)
        except TimeoutError:
            self.owner.poisoned = True
            raise ModelProviderError(ProviderErrorCode.CLEANUP_TIMEOUT) from None
        except asyncio.CancelledError:
            if not self.cancel_confirms_release:
                self.owner.poisoned = True
                raise ModelProviderError(ProviderErrorCode.CLEANUP_FAILED) from None
        except ModelProviderError:
            self.owner.poisoned = True
            raise
        except Exception:  # noqa: BLE001 - normalize native shutdown failures
            self.owner.poisoned = True
            raise ModelProviderError(ProviderErrorCode.CLEANUP_FAILED) from None

    async def wait(self, *, deadline: float | None = None) -> None:
        cancellation: asyncio.CancelledError | None = None
        limit = self.deadline if deadline is None else min(deadline, self.deadline)
        while not self.task.done():
            try:
                if limit < self.deadline:
                    await join_until(self.task, limit)
                else:
                    # Only this small bounded supervisor is shielded. Its one
                    # native join always expires at the retained deadline.
                    await asyncio.shield(self.task)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except TimeoutError:
                from .errors import ModelProviderError, ProviderErrorCode

                if cancellation is not None:
                    raise cancellation
                raise ModelProviderError(ProviderErrorCode.CLEANUP_TIMEOUT) from None
            except Exception:  # noqa: BLE001
                # Propagate the retained supervisor outcome below.
                break
        if cancellation is not None:
            raise cancellation
        self.task.result()


def materialize_request(
    request: ModelRequest, *, now: float | None = None, attempt: bool = True
) -> ModelRequest:
    """Intersect immutable caller ceilings before setup or native I/O."""
    if now is None:
        now = asyncio.get_running_loop().time()
    logical = min(
        request.deadline if request.deadline is not None else float("inf"),
        now + request.call_policy.max_request_seconds,
    )
    child = request.attempt_deadline
    if attempt:
        child = min(
            child if child is not None else float("inf"),
            logical,
            now + request.call_policy.max_attempt_seconds,
        )
    elif child is not None:
        child = min(child, logical)
    return replace(request, deadline=logical, attempt_deadline=child)


class AttemptLifecycle:
    """One authoritative clock/state for a native model attempt.

    Canonical decoding stays in the calling task. Only native awaitables and
    native scope iterators cross to NativeOwner; diagnostic projection is passive.
    """

    def __init__(
        self,
        owner: NativeOwner,
        request: ModelRequest,
        *,
        headers_supported: bool = False,
        observable: bool = False,
    ):
        from .errors import ProviderAttempt

        self.owner = owner
        self.request = materialize_request(request)
        self.policy = request.call_policy
        self.observation = ProviderAttempt(
            self.request, headers_supported=headers_supported
        )
        self.values = self.observation.values
        self.observable = observable
        self.phase = "admitted"
        self.first_progress: float | None = None
        self.last_progress: float | None = None
        self.dispatched: float | None = None
        self.terminal_response = None
        self.canonical_emitted = False
        self.cleanup_deadline: float | None = None
        self.timeout_reason: str | None = None
        self._phase_deadline: float | None = None
        self._native: set[asyncio.Future[Any]] = set()
        self._shutdowns: dict[asyncio.Future[Any], NativeShutdown] = {}

    def _expiry(self) -> tuple[float, str]:
        assert self.request.deadline is not None
        assert self.request.attempt_deadline is not None
        limits = [
            (self.request.deadline, "logical_deadline"),
            (self.request.attempt_deadline, "attempt_deadline"),
        ]
        if self._phase_deadline is not None:
            limits.append((self._phase_deadline, "input_count"))
        if (
            self.observable
            and self.dispatched is not None
            and self.phase == "generating"
        ):
            limits.append(
                (
                    (
                        (self.dispatched + self.policy.first_progress_timeout_seconds)
                        if self.last_progress is None
                        else (
                            self.last_progress
                            + self.policy.progress_idle_timeout_seconds
                        )
                    ),
                    "first_progress" if self.last_progress is None else "idle_progress",
                )
            )
        return min(limits, key=lambda item: item[0])

    @property
    def execution_deadline(self) -> float:
        return self._expiry()[0]

    def check_execution(self) -> None:
        if asyncio.get_running_loop().time() >= self._expiry()[0]:
            self.timeout_reason = self._expiry()[1]
            raise self._timeout_error()

    def _require_phase(self, *phases: str) -> None:
        if self.phase not in phases:
            raise RuntimeError(f"model attempt cannot advance from {self.phase}")

    async def __aenter__(self):
        self._require_phase("admitted")
        self.owner.require_available()
        self.check_execution()
        self.phase = "setup"
        return self

    async def __aexit__(self, kind, error, tb):
        from .errors import ModelProviderError, with_cancelled_model_usage

        try:
            for native in tuple(self._native):
                await self.retire(native, cancel=True, original=error)
        finally:
            if self.phase != "unresolved":
                self.phase = "closed"
        if isinstance(error, asyncio.CancelledError):
            with_cancelled_model_usage(error, self.usage())
        elif isinstance(error, ModelProviderError) and (
            self.terminal_response is not None or self.dispatched is None
        ):
            error.usage = self.usage()
        return False

    def usage(self):
        from decimal import Decimal

        from .models import ModelUsage
        from .pricing import CostEstimate

        if self.terminal_response is not None:
            return self.terminal_response.usage
        return ModelUsage(
            cost_estimate=(
                CostEstimate.complete(Decimal(0))
                if self.dispatched is None
                else CostEstimate.unavailable("model_attempt_interrupted")
            )
        )

    def track_response_release(self, response: object) -> None:
        """Retain close failures even when the SDK closes while reading the body.

        HTTPX marks a response closed before awaiting its stream's release.
        A failed release can therefore disappear on context exit. Wrap only
        this request's stream; the borrowed HTTP/SDK client stays untouched.
        """
        import httpx

        if not isinstance(response, httpx.Response):
            return
        if not isinstance(response.stream, httpx.AsyncByteStream):
            return
        source: httpx.AsyncByteStream = response.stream
        attempt = self

        class TrackedResponseStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                async for chunk in source:
                    yield chunk

            async def aclose(self) -> None:
                try:
                    await source.aclose()
                except Exception:
                    from .errors import ModelProviderError, ProviderErrorCode

                    attempt.owner.poisoned = True
                    attempt.phase = "unresolved"
                    attempt.values["cleanup_failure"] = "cleanup_failed"
                    raise ModelProviderError(
                        ProviderErrorCode.CLEANUP_FAILED, usage=attempt.usage()
                    ) from None

        response.stream = TrackedResponseStream()

    def _timeout_error(self):
        from .errors import (
            ModelProviderError,
            ProviderErrorCode,
            ProviderFailureDiagnostic,
            ProviderFailurePhase,
        )

        self.values["execution_expiry_phase"] = self.phase
        return ModelProviderError(
            ProviderErrorCode.TIMEOUT,
            usage=self.usage(),
            diagnostic=ProviderFailureDiagnostic(
                phase=ProviderFailurePhase.PROVIDER_BOUNDARY,
                code=self.timeout_reason or self._expiry()[1],
            ),
        )

    def start_count(self) -> None:
        self._require_phase("admitted", "setup")
        self.check_execution()
        assert self.request.attempt_deadline is not None
        self.phase = "count"
        self._phase_deadline = min(
            self.request.attempt_deadline,
            asyncio.get_running_loop().time() + self.policy.input_count_timeout_seconds,
        )
        self.observation.start_count()

    def counted(self, tokens: int) -> None:
        self._require_phase("count")
        self.observation.counted(tokens)
        self._phase_deadline = None
        self.phase = "setup"

    def dispatch(self) -> None:
        self._require_phase("admitted", "setup")
        self.check_execution()
        self.dispatched = asyncio.get_running_loop().time()
        self.phase = "generating"
        self.observation.mark("generation_submitted_seconds")

    def progress(self, fragment: str) -> None:
        if not isinstance(fragment, str):
            raise TypeError("native progress must be validated text")
        if not fragment or self.phase != "generating":
            return
        now = asyncio.get_running_loop().time()
        if self.first_progress is None:
            self.first_progress = now
        self.last_progress = now

    def canonical(self, event) -> None:
        self._require_phase("generating", "terminal_received")
        self.canonical_emitted = True

    def response(self, response):
        # Native nonstreaming scopes retain terminal usage before releasing
        # their response; the decoder can revisit that same result afterward.
        if self.phase == "closing" and self.terminal_response is not None:
            return self.observation.response(self.terminal_response)
        self._require_phase("generating", "terminal_received")
        self.terminal_response = response
        self.phase = "terminal_received"
        self.values["terminal_observed"] = True
        return self.observation.response(response)

    def mark(self, field: str) -> None:
        self.observation.mark(field)

    def native(self, *args) -> None:
        self.observation.native(*args)

    def headers(self, *args, **kwargs) -> None:
        if self.phase in {"closing", "closed", "unresolved"}:
            return
        self.observation.headers(*args, **kwargs)

    def transport_failure(self, *args, **kwargs) -> None:
        self.observation.transport_failure(*args, **kwargs)

    def finish(self, error) -> None:
        from dataclasses import asdict

        from .errors import ModelProviderError

        if isinstance(error, ModelProviderError):
            error.terminal_observed = self.terminal_response is not None
            error.canonical_emitted = self.canonical_emitted
            error.cleanup_unresolved = self.owner.poisoned
        start = self.observation.started
        self.values.update(
            {
                "policy": asdict(self.policy),
                "phase": self.phase,
                "timeout_reason": self.timeout_reason,
                "progress_mode": "observable" if self.observable else "unobservable",
                "first_substantive_progress_seconds": (
                    None
                    if self.first_progress is None
                    else round(self.first_progress - start, 6)
                ),
                "last_substantive_progress_seconds": (
                    None
                    if self.last_progress is None
                    else round(self.last_progress - start, 6)
                ),
                "canonical_emitted": self.canonical_emitted,
                "owner_poisoned": self.owner.poisoned,
                "cleanup_deadline_remaining_seconds": (
                    None
                    if self.cleanup_deadline is None
                    else max(
                        0, self.cleanup_deadline - asyncio.get_running_loop().time()
                    )
                ),
                "usage_complete": self.usage().cost_estimate.status.value,
            }
        )
        self.observation.finish(error)

    async def run_native(self, operation: Awaitable[_T]) -> _T:
        try:
            self._require_phase("admitted", "setup", "count", "generating")
            self.check_execution()
            self.owner.require_available()
        except BaseException:
            close = getattr(operation, "close", None)
            if close is not None:
                close()
            raise

        async def native() -> _T:
            try:
                return await operation
            except Exception as error:
                # Opaque SDK calls can close an HTTP response before handing it
                # to the adapter. Preserve explicit release-failure evidence
                # before SDK errors or TimeoutError are normalized below.
                self._raise_for_failed_response_release(error)
                raise

        task = self.owner.start(native())
        self._native.add(task)
        try:
            try:
                return await join_until(task, self._expiry()[0])
            except TimeoutError:
                self.timeout_reason = self._expiry()[1]
                raise self._timeout_error() from None
        except BaseException as error:
            await self.retire(task, cancel=True, original=error)
            raise
        finally:
            if task.done():
                self._native.discard(task)

    def _raise_for_failed_response_release(self, error: Exception) -> None:
        import httpx

        from .errors import ModelProviderError, ProviderErrorCode

        # Gemini's public nonstreaming/count APIs hide the HTTPX response.
        # A traceback through Response.aclose is positive evidence of failed
        # release, even if the SDK wraps that exception. Inspect code identity
        # only: never error text, frame locals, or SDK-private transport handles.
        close_code = httpx.Response.aclose.__code__
        current: BaseException | None = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            traceback = current.__traceback__
            while traceback is not None:
                if traceback.tb_frame.f_code is close_code:
                    self.owner.poisoned = True
                    self.phase = "unresolved"
                    self.values["cleanup_failure"] = "cleanup_failed"
                    raise ModelProviderError(
                        ProviderErrorCode.CLEANUP_FAILED, usage=self.usage()
                    ) from None
                traceback = traceback.tb_next
            current = current.__cause__ or current.__context__

    async def retire(
        self, task: asyncio.Future[Any], *, cancel: bool, original=None
    ) -> None:
        from .errors import ModelProviderError

        if task.done():
            self._native.discard(task)
            return
        self.begin_cleanup()
        assert self.cleanup_deadline is not None
        self.phase = "closing"
        if task not in self._shutdowns:
            if cancel:
                task.cancel()
            self._shutdowns[task] = NativeShutdown(
                self.owner,
                task,
                self.cleanup_deadline,
                cancel_confirms_release=cancel
                or getattr(task, "cancelling", lambda: 0)() > 0,
            )
        try:
            await self._shutdowns[task].wait()
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError) and not self.owner.poisoned:
                # The bounded supervisor confirmed native release before
                # preserving its waiter's cancellation; this owner is healthy.
                if original is None:
                    raise
                return
            self.owner.poisoned = True
            self.phase = "unresolved"
            self.values["cleanup_failure"] = "cleanup_failed"
            if isinstance(error, ModelProviderError):
                error.usage = self.usage()
            if original is None:
                from .errors import ProviderErrorCode

                if isinstance(error, (ModelProviderError, asyncio.CancelledError)):
                    raise
                raise ModelProviderError(
                    ProviderErrorCode.CLEANUP_FAILED, usage=self.usage()
                ) from None
        finally:
            self.observation.mark("cleanup_finished_seconds")
            if task.done():
                self._native.discard(task)

    def begin_cleanup(self) -> float:
        """Share one retirement deadline with nested native/process owners."""
        if self.cleanup_deadline is None:
            self.cleanup_deadline = (
                asyncio.get_running_loop().time() + self.policy.cleanup_timeout_seconds
            )
            self.observation.mark("cleanup_started_seconds")
        return self.cleanup_deadline

    @asynccontextmanager
    async def stream(self, source: AsyncIterator[_T]):
        self._require_phase("generating")
        self.check_execution()
        stream = NativeStream(self.owner, source, self)
        self._native.add(stream.task)
        failure = None
        try:
            yield stream
        except BaseException as error:
            failure = (
                None
                if isinstance(error, GeneratorExit)
                and self.terminal_response is not None
                else error
            )
            raise
        finally:
            stream.stop(cancel=failure is not None)
            await self.retire(stream.task, cancel=False, original=failure)
            if failure is None:
                try:
                    stream.task.result()
                except BaseException:
                    from .errors import ModelProviderError, ProviderErrorCode

                    self.owner.poisoned = True
                    self.values["cleanup_failure"] = "cleanup_failed"
                    raise ModelProviderError(
                        ProviderErrorCode.CLEANUP_FAILED, usage=self.usage()
                    ) from None
                self.check_execution()


async def execute_generate_attempt(
    owner: NativeOwner,
    request: ModelRequest,
    *,
    provider_id: str,
    boundary_name: str,
    operation: Callable[[ModelRequest, AttemptLifecycle], Awaitable[ModelResponse]],
    headers_supported: bool,
) -> ModelResponse:
    """Run the common canonical boundary around one native generate attempt."""
    from decimal import Decimal

    from .errors import (
        ModelProviderError,
        ProviderErrorCode,
        ProviderFailureDiagnostic,
        ProviderFailurePhase,
        detached_provider_error,
        interrupted_model_usage,
    )
    from .models import ModelUsage
    from .pricing import CostEstimate

    if not isinstance(request, ModelRequest):
        raise TypeError("request must be a canonical ModelRequest")
    attempt = AttemptLifecycle(owner, request, headers_supported=headers_supported)
    request = attempt.request
    failure: ModelProviderError | None = None
    try:
        request.remaining_after(
            ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
        )
        async with attempt:
            response = await operation(request, attempt)
            attempt.response(response)
            attempt.check_execution()
        attempt.finish(None)
        return attempt.observation.response(response)
    except TimeoutError as error:
        failure = ModelProviderError(
            ProviderErrorCode.TIMEOUT,
            "The model request deadline expired.",
            usage=interrupted_model_usage(error),
        )
    except (asyncio.CancelledError, GeneratorExit) as error:
        attempt.finish(error)
        raise
    except ImportError as error:
        attempt.finish(error)
        raise
    except ModelProviderError as error:
        failure = error
    except Exception:
        failure = ModelProviderError(
            ProviderErrorCode.MALFORMED_RESPONSE,
            f"{boundary_name} provider boundary failed",
            diagnostic=ProviderFailureDiagnostic(
                phase=ProviderFailurePhase.PROVIDER_BOUNDARY,
                code="unexpected_provider_boundary_failure",
            ),
        )
    if attempt.terminal_response is not None:
        failure.usage = attempt.usage()
    attempt.finish(failure)
    raise detached_provider_error(failure, provider_id=provider_id)


async def execute_stream_attempt(
    owner: NativeOwner,
    request: ModelRequest,
    *,
    provider_id: str,
    boundary_name: str,
    operation: Callable[
        [ModelRequest, AttemptLifecycle], AsyncIterator[ModelStreamEvent]
    ],
    headers_supported: bool,
) -> AsyncIterator[ModelStreamEvent]:
    """Run the common canonical boundary around one native streaming attempt."""
    from decimal import Decimal

    from .errors import (
        ModelProviderError,
        ProviderErrorCode,
        ProviderFailureDiagnostic,
        ProviderFailurePhase,
        detached_provider_error,
        interrupted_model_usage,
        with_cancelled_model_usage,
    )
    from .models import ModelStreamCompleted, ModelUsage
    from .pricing import CostEstimate

    if not isinstance(request, ModelRequest):
        raise TypeError("request must be a canonical ModelRequest")
    attempt = AttemptLifecycle(
        owner, request, headers_supported=headers_supported, observable=True
    )
    terminal_usage = None
    request = attempt.request
    failure: ModelProviderError | None = None
    try:
        request.remaining_after(
            ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
        )
        async with attempt:
            async with closing_stream(operation(request, attempt)) as events:
                async for event in events:
                    if isinstance(event, ModelStreamCompleted):
                        terminal_usage = event.response.usage
                        terminal = attempt.response(event.response)
                        break
                    attempt.canonical(event)
                    yield event
            attempt.check_execution()
            if terminal_usage is not None:
                attempt.finish(None)
                yield ModelStreamCompleted(attempt.observation.response(terminal))
        attempt.finish(None)
        return
    except TimeoutError as error:
        failure = ModelProviderError(
            ProviderErrorCode.TIMEOUT,
            "The model request deadline expired.",
            usage=interrupted_model_usage(error),
        )
    except (asyncio.CancelledError, GeneratorExit) as error:
        if isinstance(error, asyncio.CancelledError) and terminal_usage is not None:
            with_cancelled_model_usage(error, terminal_usage)
        attempt.finish(error)
        raise
    except ImportError as error:
        attempt.finish(error)
        raise
    except ModelProviderError as error:
        failure = error
    except Exception:
        failure = ModelProviderError(
            ProviderErrorCode.MALFORMED_RESPONSE,
            f"{boundary_name} provider boundary failed",
            diagnostic=ProviderFailureDiagnostic(
                phase=ProviderFailurePhase.PROVIDER_BOUNDARY,
                code="unexpected_provider_boundary_failure",
            ),
        )
    if terminal_usage is not None:
        failure.usage = terminal_usage
    if attempt.terminal_response is not None:
        failure.usage = attempt.usage()
    attempt.finish(failure)
    raise detached_provider_error(failure, provider_id=provider_id)


async def native_events(create, *, manager: bool = False, observe=None):
    """SDK scope only. The NativeStream task enters, reads and exits this scope."""
    if manager:
        async with create() as stream:
            if observe is not None:
                observe(stream)
            async for event in stream:
                yield event
            yield _NATIVE_EOF
    else:
        stream = await create()
        try:
            if observe is not None:
                observe(stream)
            async for event in stream:
                yield event
            yield _NATIVE_EOF
        finally:
            close = getattr(stream, "close", None) or getattr(stream, "aclose", None)
            if close is not None:
                await close()


def transport_timeout(
    request: ModelRequest | None = None, *, deadline: float | None = None
):
    import httpx

    if request is None:
        from .models import ModelCallPolicy

        policy = ModelCallPolicy()
        return httpx.Timeout(
            connect=policy.connect_timeout_seconds,
            read=policy.read_timeout_seconds,
            write=policy.write_timeout_seconds,
            pool=policy.pool_timeout_seconds,
        )
    request = materialize_request(request)
    assert request.attempt_deadline is not None
    remaining = max(
        0.001,
        min(
            request.attempt_deadline,
            deadline if deadline is not None else request.attempt_deadline,
        )
        - asyncio.get_running_loop().time(),
    )
    policy = request.call_policy
    return httpx.Timeout(
        connect=min(remaining, policy.connect_timeout_seconds),
        read=min(remaining, policy.read_timeout_seconds),
        write=min(remaining, policy.write_timeout_seconds),
        pool=min(remaining, policy.pool_timeout_seconds),
    )
