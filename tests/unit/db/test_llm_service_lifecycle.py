from unittest.mock import AsyncMock

import pytest

from daita.db.llm_service import DbLLMConfig, DbLLMService


async def test_db_llm_service_closes_created_provider_once_and_clears_it():
    service = DbLLMService(DbLLMConfig(provider="openai", model="test-model"))
    provider = AsyncMock()
    service._provider = provider

    await service.aclose()
    await service.aclose()

    provider.aclose.assert_awaited_once_with()
    assert service._provider is None


async def test_db_llm_service_close_before_first_use_stays_lazy():
    service = DbLLMService(DbLLMConfig(provider="openai", model="test-model"))

    await service.aclose()

    assert service._provider is None


async def test_db_llm_service_surfaces_close_failure_and_clears_provider():
    service = DbLLMService(DbLLMConfig(provider="openai", model="test-model"))
    provider = AsyncMock()
    provider.aclose.side_effect = RuntimeError("provider close failed")
    service._provider = provider

    with pytest.raises(RuntimeError, match="provider close failed"):
        await service.aclose()

    assert service._provider is None
    await service.aclose()
    provider.aclose.assert_awaited_once_with()


async def test_db_llm_service_clears_compatibility_provider_without_close_hook():
    service = DbLLMService(DbLLMConfig(provider="openai", model="test-model"))
    provider = object()
    service._provider = provider

    await service.aclose()

    assert service._provider is None
