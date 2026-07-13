from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from daita.db import DbAgent, DbRuntime
from daita.db.llm_service import DbLLMConfig, DbLLMService


def _service_with_provider(*, close_error: Exception | None = None):
    service = DbLLMService(DbLLMConfig(provider="openai", model="test-model"))
    provider = AsyncMock()
    provider.aclose.side_effect = close_error
    service._provider = provider
    return service, provider


async def test_runtime_teardown_closes_owned_created_provider_once():
    service, provider = _service_with_provider()
    runtime = DbRuntime(db_llm_service=service, owns_db_llm_service=True)

    await runtime.teardown()
    await runtime.teardown()

    provider.aclose.assert_awaited_once_with()
    assert service._provider is None
    assert runtime.is_setup is False


async def test_runtime_teardown_before_first_llm_use_stays_lazy():
    service = DbLLMService(DbLLMConfig(provider="openai", model="test-model"))
    runtime = DbRuntime(db_llm_service=service, owns_db_llm_service=True)

    await runtime.teardown()

    assert service._provider is None


async def test_plugin_teardown_failure_still_closes_owned_llm_service():
    service, provider = _service_with_provider()
    runtime = DbRuntime(db_llm_service=service, owns_db_llm_service=True)
    await runtime.setup()
    runtime.registry.teardown_all = AsyncMock(
        side_effect=RuntimeError("plugin teardown failed")
    )

    with pytest.raises(RuntimeError, match="plugin teardown failed"):
        await runtime.teardown()

    provider.aclose.assert_awaited_once_with()
    assert runtime.setup_context is None
    assert runtime.is_setup is False


async def test_llm_teardown_failure_still_finalizes_runtime_state():
    service, provider = _service_with_provider(
        close_error=RuntimeError("llm teardown failed")
    )
    runtime = DbRuntime(db_llm_service=service, owns_db_llm_service=True)
    await runtime.setup()
    runtime.registry.teardown_all = AsyncMock()

    with pytest.raises(RuntimeError, match="llm teardown failed"):
        await runtime.teardown()

    runtime.registry.teardown_all.assert_awaited_once_with()
    provider.aclose.assert_awaited_once_with()
    assert service._provider is None
    assert runtime.setup_context is None
    assert runtime.is_setup is False


async def test_runtime_surfaces_all_teardown_failures():
    service, provider = _service_with_provider(
        close_error=RuntimeError("llm teardown failed")
    )
    runtime = DbRuntime(db_llm_service=service, owns_db_llm_service=True)
    await runtime.setup()
    runtime.registry.teardown_all = AsyncMock(
        side_effect=RuntimeError("plugin teardown failed")
    )

    with pytest.raises(ExceptionGroup) as caught:
        await runtime.teardown()

    assert [str(error) for error in caught.value.exceptions] == [
        "plugin teardown failed",
        "llm teardown failed",
    ]
    provider.aclose.assert_awaited_once_with()


async def test_explicitly_shared_llm_service_is_not_closed_by_runtime():
    service, provider = _service_with_provider()
    runtime = DbRuntime(db_llm_service=service, owns_db_llm_service=False)

    await runtime.teardown()

    provider.aclose.assert_not_awaited()
    assert service._provider is provider
    await service.aclose()


async def test_agent_stop_reaches_runtime_teardown():
    runtime = SimpleNamespace(teardown=AsyncMock())
    agent = DbAgent(runtime=runtime)

    await agent.stop()

    runtime.teardown.assert_awaited_once_with()
