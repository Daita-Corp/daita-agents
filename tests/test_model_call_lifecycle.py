"""Native ownership regressions; emergency guards are never production bounds."""

import asyncio
from contextvars import ContextVar

import httpx
import openai
import pytest

from daita.llm._lifecycle import NativeOwner, NativeStream, join_until
from daita.llm.errors import ModelProviderError, ProviderErrorCode


async def test_native_scope_uses_one_context_and_no_prefetch():
    scope = ContextVar[str | None]("native_scope", default=None)
    read = []
    closed = []

    async def source():
        token = scope.set("owned")
        try:
            for value in range(4):
                read.append(value)
                yield value
        finally:
            scope.reset(token)
            closed.append(True)

    owner = NativeOwner()
    stream = NativeStream(owner, source())
    assert await anext(stream) == 0
    await asyncio.sleep(0)
    assert read == [0]
    assert scope.get() is None
    stream.stop(cancel=False)
    await join_until(stream.task, asyncio.get_running_loop().time() + 0.5)
    assert closed == [True]


async def test_cancel_resistance_retains_owner_and_discards_late_output():
    release = asyncio.Event()
    entered = asyncio.Event()

    async def source():
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        yield "late result"

    owner = NativeOwner()
    stream = NativeStream(owner, source())
    pending = asyncio.create_task(anext(stream))
    await entered.wait()
    stream.stop(cancel=True)
    start = asyncio.get_running_loop().time()
    try:
        with pytest.raises(TimeoutError):
            await join_until(stream.task, start + 0.03)
        owner.poisoned = True
        assert asyncio.get_running_loop().time() - start < 0.3
        assert stream.task in owner.tasks
        with pytest.raises(ModelProviderError) as error:
            owner.require_available()
        assert error.value.code is ProviderErrorCode.OWNER_UNAVAILABLE
    finally:
        release.set()
        await join_until(stream.task, asyncio.get_running_loop().time() + 0.5)
        with pytest.raises(StopAsyncIteration):
            await pending
    assert owner.poisoned
    assert not owner.tasks


async def test_actual_sdk_stream_scope_releases_borrowed_transport():
    scope = ContextVar[str | None]("http_body_owner", default=None)
    closed = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            self.token = scope.set("http")
            yield b'data: {"type":"response.output_text.delta","delta":"x","output_index":0,"content_index":0,"item_id":"m","sequence_number":1}\n\n'
            await asyncio.Event().wait()

        async def aclose(self):
            scope.reset(self.token)
            closed.append(True)

    client = openai.AsyncOpenAI(
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=Body()
                )
            )
        ),
    )

    async def source():
        native = await client.responses.create(model="test", input="test", stream=True)
        try:
            async for event in native:
                yield event
        finally:
            await native.close()

    stream = NativeStream(NativeOwner(), source())
    try:
        assert (await anext(stream)).delta == "x"
        stream.stop(cancel=False)
        await join_until(stream.task, asyncio.get_running_loop().time() + 0.5)
        assert closed == [True]
        assert not client.is_closed()
    finally:
        await client.close()


async def test_shutdown_is_once_only_bounded_and_shared_by_waiters():
    from daita.llm._lifecycle import NativeShutdown

    release = asyncio.Event()
    calls = []

    async def close():
        calls.append(1)
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass

    owner = NativeOwner()
    task = owner.start(close())
    start = asyncio.get_running_loop().time()
    closing = NativeShutdown(owner, task, start + 0.05)
    try:
        waiters = [asyncio.create_task(closing.wait()) for _ in range(5)]
        results = await asyncio.gather(*waiters, return_exceptions=True)
        assert all(isinstance(result, ModelProviderError) for result in results)
        assert all(
            isinstance(result, ModelProviderError)
            and result.code is ProviderErrorCode.CLEANUP_TIMEOUT
            for result in results
        )
        assert asyncio.get_running_loop().time() - start < 0.3
        assert owner.poisoned and task in owner.tasks
        with pytest.raises(ModelProviderError):
            await closing.wait()
        assert calls == [1]
    finally:
        release.set()
        await task
        await asyncio.gather(closing.task, return_exceptions=True)


async def test_cancel_during_close_preserves_cancellation_and_native_outcome():
    from daita.llm._lifecycle import NativeShutdown

    owner = NativeOwner()
    release = asyncio.Event()
    native = owner.start(release.wait())
    closing = NativeShutdown(owner, native, asyncio.get_running_loop().time() + 0.3)
    waiter = asyncio.create_task(closing.wait())
    await asyncio.sleep(0)
    waiter.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await closing.wait()
    assert not owner.poisoned


async def test_cancelled_earlier_close_waiter_preserves_original_cancellation():
    from daita.llm._lifecycle import NativeShutdown

    owner = NativeOwner()
    release = asyncio.Event()
    native = owner.start(release.wait())
    start = asyncio.get_running_loop().time()
    closing = NativeShutdown(owner, native, start + 0.08)
    waiter = asyncio.create_task(closing.wait(deadline=start + 0.03))
    try:
        await asyncio.sleep(0)
        waiter.cancel("user cancelled")
        with pytest.raises(asyncio.CancelledError, match="user cancelled"):
            await waiter
        assert not closing.task.done()
        assert native in owner.tasks
        with pytest.raises(ModelProviderError) as caught:
            await closing.wait()
        assert caught.value.code is ProviderErrorCode.CLEANUP_TIMEOUT
        assert owner.poisoned
    finally:
        release.set()
        await asyncio.gather(native, closing.task, return_exceptions=True)


@pytest.mark.parametrize("phase", ["closing", "closed", "unresolved"])
async def test_retired_attempt_cannot_restart_or_accept_late_output(phase):
    from test_model_call_policy import request

    from daita.llm._lifecycle import AttemptLifecycle

    life = AttemptLifecycle(NativeOwner(), request(), observable=True)
    async with life:
        life.dispatch()
        life.progress("first")
    life.phase = phase
    prior_progress = life.last_progress
    for action in (life.__aenter__,):
        with pytest.raises(RuntimeError):
            await action()
    for action in (life.start_count, life.dispatch, lambda: life.response(None)):
        with pytest.raises(RuntimeError):
            action()
    with pytest.raises(RuntimeError):
        await life.run_native(asyncio.sleep(0))
    life.progress("late")
    assert life.phase == phase
    assert life.last_progress == prior_progress
    assert life.terminal_response is None
    assert not life.owner.tasks


def test_virtual_progress_changes_only_idle_deadline(monkeypatch):
    from types import SimpleNamespace

    from test_model_call_policy import request

    from daita.llm._lifecycle import AttemptLifecycle

    now = [100.0]
    monkeypatch.setattr(
        asyncio, "get_running_loop", lambda: SimpleNamespace(time=lambda: now[0])
    )
    life = AttemptLifecycle(NativeOwner(), request(), observable=True)
    assert (life.request.deadline, life.request.attempt_deadline) == (280, 220)
    life.dispatch()
    assert life._expiry() == (160, "first_progress")
    now[0] = 110
    life.native("metadata")
    life.progress("")
    assert life._expiry() == (160, "first_progress")
    life.progress("same")
    assert life._expiry() == (140, "idle_progress")
    now[0] = 120
    life.progress("same")  # Equal incremental fragments are still new output.
    assert life._expiry() == (150, "idle_progress")
    assert (life.request.deadline, life.request.attempt_deadline) == (280, 220)
    life.phase = "unresolved"
    life.progress("late")
    assert life.last_progress == 120


async def test_cancel_resistant_attempt_returns_poisoned_and_rejects_replacement():
    from test_model_call_policy import request

    from daita.llm import ModelCallPolicy
    from daita.llm._lifecycle import AttemptLifecycle

    release = asyncio.Event()
    entered = asyncio.Event()
    owner = NativeOwner()
    policy = ModelCallPolicy(
        max_attempt_seconds=0.03,
        first_progress_timeout_seconds=0.03,
        progress_idle_timeout_seconds=0.03,
        cleanup_timeout_seconds=0.03,
    )
    life = AttemptLifecycle(owner, request(call_policy=policy))

    async def resistant():
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        return "late"

    start = asyncio.get_running_loop().time()
    try:
        with pytest.raises(ModelProviderError) as caught:
            async with life:
                await life.run_native(resistant())
        life.finish(caught.value)
        assert caught.value.code is ProviderErrorCode.TIMEOUT
        assert caught.value.cleanup_unresolved
        assert life.phase == "unresolved" and owner.poisoned
        assert asyncio.get_running_loop().time() - start < 0.3
        assert len(owner.tasks) == 1
        with pytest.raises(ModelProviderError) as unavailable:
            async with AttemptLifecycle(owner, request(call_policy=policy)):
                pytest.fail("replacement admitted")
        assert unavailable.value.code is ProviderErrorCode.OWNER_UNAVAILABLE
    finally:
        tasks = tuple(owner.tasks)
        release.set()
        await asyncio.gather(*tasks)
    assert life.phase == "unresolved" and owner.poisoned


@pytest.mark.parametrize("mode", ["success", "failure", "timeout", "cancel"])
async def test_terminal_usage_is_held_until_bounded_close(mode):
    from test_model_call_policy import request
    from test_provider_streaming import _openai_text_response

    from daita.llm import ModelCallPolicy
    from daita.llm.models import ModelStreamCompleted
    from daita.llm.providers.openai import OpenAIResponsesProvider

    release = asyncio.Event()
    closing = asyncio.Event()
    closed = []
    terminal = _openai_text_response("done")
    terminal["usage"] = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}

    class Stream:
        async def __aiter__(self):
            yield {"type": "response.completed", "response": terminal}

        async def close(self):
            closed.append("started")
            closing.set()
            if mode in {"timeout", "cancel"}:
                await release.wait()
            if mode == "failure":
                raise RuntimeError("private failure")
            closed.append("finished")

    class Responses:
        async def create(self, **kwargs: object) -> object:
            return Stream()

    class Client:
        responses = Responses()

        async def close(self) -> None:
            pass

    provider = OpenAIResponsesProvider("test", client=Client())
    policy = ModelCallPolicy(cleanup_timeout_seconds=0.05)
    events = []

    async def consume():
        async for event in provider.stream(request(call_policy=policy)):
            events.append(event)

    task = asyncio.create_task(consume())
    try:
        await closing.wait()
        assert events == []
        if mode == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError) as cancelled:
                await task
            from daita.llm.errors import interrupted_model_usage

            assert interrupted_model_usage(cancelled.value).total_tokens == 5
            assert not provider._native_owner.poisoned
        elif mode == "success":
            await task
            assert len(events) == 1 and isinstance(events[0], ModelStreamCompleted)
            assert events[0].response.usage.total_tokens == 5
        else:
            with pytest.raises(ModelProviderError) as caught:
                await task
            assert caught.value.terminal_observed
            assert caught.value.usage.total_tokens == 5
            assert caught.value.code in {
                ProviderErrorCode.CLEANUP_TIMEOUT,
                ProviderErrorCode.CLEANUP_FAILED,
            }
            assert events == []
            assert provider._native_owner.poisoned
        assert closed.count("started") == 1
    finally:
        native_tasks = tuple(provider._native_owner.tasks)
        release.set()
        await asyncio.gather(task, *native_tasks, return_exceptions=True)
        await provider.close()


async def test_five_nested_owners_share_one_cleanup_deadline():
    from daita.llm._lifecycle import await_cleanup, shutdown_deadline

    release = asyncio.Event()
    owners = [NativeOwner() for _ in range(5)]
    observed = []
    tasks = []
    start = asyncio.get_running_loop().time()
    limit = start + 0.06

    async def close(index, deadline):
        observed.append(deadline)
        if index == 4:
            await release.wait()
        else:
            native = asyncio.create_task(close(index + 1, deadline))
            tasks.append(native)
            await await_cleanup(
                native, deadline=shutdown_deadline(deadline), owner=owners[index]
            )

    root = asyncio.create_task(close(0, limit))
    try:
        with pytest.raises(ModelProviderError) as caught:
            await await_cleanup(root, deadline=limit, owner=owners[-1])
        assert caught.value.code is ProviderErrorCode.CLEANUP_TIMEOUT
        assert asyncio.get_running_loop().time() - start < 0.2
        assert observed == [limit] * 5
        assert all(owner.poisoned for owner in owners)
    finally:
        release.set()
        await asyncio.gather(root, *tasks, return_exceptions=True)


def test_explicit_shutdown_deadline_is_not_replaced_by_default(monkeypatch):
    from types import SimpleNamespace

    from daita.llm._lifecycle import shutdown_deadline

    monkeypatch.setattr(
        asyncio, "get_running_loop", lambda: SimpleNamespace(time=lambda: 100)
    )
    assert shutdown_deadline() == 105
    assert shutdown_deadline(120) == 120


@pytest.mark.parametrize("stage", ["read", "close"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, httpx.ReadTimeout, TimeoutError])
async def test_opaque_native_call_distinguishes_read_and_release_failure(
    stage, wrapped, error_type
):
    from test_model_call_policy import request

    from daita.llm._lifecycle import AttemptLifecycle

    owner = NativeOwner()
    attempt = AttemptLifecycle(owner, request())
    failure = error_type("private native failure")
    closes = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            if stage == "read":
                raise failure
            yield b"{}"

        async def aclose(self):
            closes.append(1)
            if stage == "close":
                raise failure

    async def sdk_call():
        response = httpx.Response(200, stream=Body())
        try:
            try:
                return await response.aread()
            finally:
                await response.aclose()
        except Exception as error:
            if wrapped:
                raise RuntimeError("SDK wrapper") from error
            raise

    with pytest.raises(Exception) as caught:
        async with attempt:
            attempt.dispatch()
            await attempt.run_native(sdk_call())

    assert closes == [1]
    if stage == "close":
        assert isinstance(caught.value, ModelProviderError)
        assert caught.value.code is ProviderErrorCode.CLEANUP_FAILED
        assert caught.value.usage.cost_estimate.status.value != "complete"
        assert owner.poisoned
    else:
        assert not owner.poisoned
        if error_type is TimeoutError and not wrapped:
            assert isinstance(caught.value, ModelProviderError)
            assert caught.value.code is ProviderErrorCode.TIMEOUT
        elif wrapped:
            assert caught.value.__cause__ is failure
        else:
            assert caught.value is failure


@pytest.mark.parametrize("family", ["openai", "anthropic", "gemini", "compatible"])
@pytest.mark.parametrize("mode", ["release", "failure", "timeout", "cancel"])
async def test_actual_sdk_nonstreaming_response_cleanup(family, mode):
    """Exercise real SDK/HTTPX response ownership without external HTTP.

    SDKs may close the HTTP body before returning a parsed model response.
    Until that handoff, usage is unknown; raw body bytes are not authenticated
    terminal usage. Failed release must still block replacement work.
    """
    import json
    from contextlib import AsyncExitStack
    from typing import Any, cast

    from test_model_call_policy import request
    from test_provider_streaming import _openai_text_response
    from test_provider_token_counting import provider_at

    from daita.llm import ModelCallPolicy
    from daita.llm.errors import interrupted_model_usage
    from daita.llm.providers.openai_compatible import OpenAICompatibleProvider

    if family == "openai":
        payload = _openai_text_response("done")
        payload["usage"] = {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11}
    elif family == "anthropic":
        payload = {
            "id": "msg_fixture",
            "type": "message",
            "role": "assistant",
            "model": "fixture-model",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 7, "output_tokens": 4},
        }
    elif family == "gemini":
        payload = {
            "responseId": "fixture",
            "modelVersion": "fixture-model",
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "role": "model",
                        "parts": [{"text": "done"}],
                    },
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 7,
                "candidatesTokenCount": 4,
                "totalTokenCount": 11,
            },
        }
    else:
        payload = {
            "id": "chatcmpl_fixture",
            "object": "chat.completion",
            "created": 1,
            "model": "fixture-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
        }
    closing = asyncio.Event()
    release = asyncio.Event()
    close_calls = []
    paths = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield json.dumps(payload).encode()

        async def aclose(self):
            close_calls.append(1)
            closing.set()
            if mode == "failure":
                raise RuntimeError("synthetic native close failure")
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    if mode != "timeout":
                        raise

    def respond(native_request):
        paths.append(native_request.url.path)
        return httpx.Response(
            200, headers={"content-type": "application/json"}, stream=Body()
        )

    policy = ModelCallPolicy(
        max_request_seconds=2,
        max_attempt_seconds=1,
        first_progress_timeout_seconds=1,
        progress_idle_timeout_seconds=1,
        cleanup_timeout_seconds=0.1,
    )
    provider: Any
    async with AsyncExitStack() as stack:
        if family == "compatible":
            http = await stack.enter_async_context(
                httpx.AsyncClient(transport=httpx.MockTransport(respond))
            )
            sdk = await stack.enter_async_context(
                openai.AsyncOpenAI(api_key="offline", http_client=http, max_retries=7)
            )
            provider = OpenAICompatibleProvider(
                "fixture-model",
                provider="custom",
                base_url="https://offline.test/v1",
                client=cast(Any, sdk),
            )
            stack.push_async_callback(provider.close)
        else:
            provider = await stack.enter_async_context(provider_at(family, respond))
        task = asyncio.create_task(provider.generate(request(call_policy=policy)))
        try:
            async with asyncio.timeout(
                4
            ):  # Later harness guard, never the production bound.
                await closing.wait()
                if mode == "release":
                    assert not task.done()
                    release.set()
                    response = await task
                    assert response.text == "done"
                    assert response.usage.total_tokens == 11
                    assert not provider._native_owner.poisoned
                elif mode == "cancel":
                    task.cancel("user cancelled")
                    with pytest.raises(asyncio.CancelledError) as cancelled:
                        await task
                    assert (
                        interrupted_model_usage(
                            cancelled.value
                        ).cost_estimate.status.value
                        != "complete"
                    )
                    assert not provider._native_owner.poisoned
                else:
                    with pytest.raises(ModelProviderError) as caught:
                        await task
                    assert caught.value.usage.cost_estimate.status.value != "complete"
                    if mode == "failure":
                        assert caught.value.code is ProviderErrorCode.CLEANUP_FAILED
                    assert caught.value.cleanup_unresolved
                    assert provider._native_owner.poisoned
                    with pytest.raises(ModelProviderError) as unavailable:
                        await provider.generate(request(call_policy=policy))
                    assert unavailable.value.code is ProviderErrorCode.OWNER_UNAVAILABLE
                assert len(paths) == 1
                assert close_calls == [1]
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(
                task, *tuple(provider._native_owner.tasks), return_exceptions=True
            )
