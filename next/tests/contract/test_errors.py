from __future__ import annotations

from typing import Any

import pytest

from daita import (
    AgentError,
    AuthenticationError,
    ConfigError,
    DaitaError,
    DataQualityError,
    ErrorRetryability,
    FocusDSLError,
    LLMError,
    PermanentError,
    PluginError,
    RateLimitError,
    RetryableError,
    SkillError,
    TransientError,
    ValidationError,
)
from daita.hosting.embedded import AgentNameError
from daita.adapters.protocols import ResourceAdapterError, ResourceNotFoundError
from daita.llm import ModelProviderError, ProviderErrorCode
from daita.security import SecretResolutionError
from daita.skills import SkillError as ServiceSkillError


def test_public_error_taxonomy_has_stable_subsystem_and_retry_categories() -> None:
    assert issubclass(AgentError, DaitaError)
    assert issubclass(LLMError, DaitaError)
    assert issubclass(ConfigError, DaitaError)
    assert issubclass(PluginError, DaitaError)
    assert issubclass(SkillError, PluginError)
    assert issubclass(TransientError, DaitaError)
    assert issubclass(RetryableError, DaitaError)
    assert issubclass(PermanentError, DaitaError)

    transient = TransientError()
    retryable = RetryableError()
    permanent = PermanentError()
    assert transient.retryability is ErrorRetryability.TRANSIENT
    assert transient.is_transient() and transient.is_retryable()
    assert retryable.retryability is ErrorRetryability.RETRYABLE
    assert retryable.is_retryable() and not retryable.is_transient()
    assert permanent.retryability is ErrorRetryability.PERMANENT
    assert permanent.is_permanent() and not permanent.is_retryable()


def test_existing_agent_skill_model_and_secret_errors_join_public_taxonomy() -> None:
    assert isinstance(AgentNameError("invalid agent name"), AgentError)
    assert isinstance(
        ResourceAdapterError("source-1", "source_unavailable", "Unavailable."),
        PluginError,
    )
    assert isinstance(ServiceSkillError("invalid skill"), SkillError)
    assert isinstance(
        SecretResolutionError(
            "secret_not_found",
            "The configured secret is unavailable.",
        ),
        ConfigError,
    )

    transient = ModelProviderError(ProviderErrorCode.TIMEOUT)
    authentication = ModelProviderError(
        ProviderErrorCode.AUTHENTICATION_ERROR,
        provider_id="openai:primary",
    )
    rate_limited = ModelProviderError(
        ProviderErrorCode.RATE_LIMIT_ERROR,
        provider_id="anthropic:primary",
        retry_after_seconds=1.5,
    )
    for error in (transient, authentication, rate_limited):
        assert isinstance(error, ModelProviderError)
        assert isinstance(error, LLMError)
    assert isinstance(authentication, AuthenticationError)
    assert isinstance(rate_limited, RateLimitError)
    assert transient.error_code == "timeout"
    assert transient.is_transient()
    assert authentication.error_code == "authentication_error"
    assert authentication.provider_id == "openai:primary"
    assert authentication.is_permanent()
    assert rate_limited.provider_id == "anthropic:primary"
    assert rate_limited.retry_after_seconds == 1.5
    assert rate_limited.is_transient()


def test_broad_subsystem_errors_never_claim_an_unsafe_retry() -> None:
    agent = AgentNameError("invalid agent name")
    plugin = ResourceNotFoundError("source-1", "resource-1")
    skill = ServiceSkillError("invalid skill")

    for error in (agent, plugin, skill):
        assert error.retryability is ErrorRetryability.UNKNOWN
        assert not error.is_transient()
        assert not error.is_retryable()
        assert not error.is_permanent()


def test_specific_public_errors_expose_only_bounded_safe_facts() -> None:
    rate_limit = RateLimitError(retry_after_seconds=1.5)
    authentication = AuthenticationError(provider_id="openai:primary")
    validation = ValidationError(field="source_id")
    data_quality = DataQualityError(violation_count=3)
    focus = FocusDSLError()

    assert rate_limit.retry_after_seconds == 1.5
    assert rate_limit.error_code == "rate_limit_error"
    assert authentication.provider_id == "openai:primary"
    assert validation.field == "source_id"
    assert data_quality.violation_count == 3
    assert focus.error_code == "focus_dsl_error"
    rendered = " ".join(
        str(error)
        for error in (
            rate_limit,
            authentication,
            validation,
            data_quality,
            focus,
        )
    )
    assert "credential" not in rendered.lower()
    assert "password" not in rendered.lower()


def test_validation_and_data_quality_errors_never_accept_rejected_values() -> None:
    unsafe_validation: Any = ValidationError
    unsafe_data_quality: Any = DataQualityError
    with pytest.raises(TypeError):
        unsafe_validation(value="private-password")
    with pytest.raises(TypeError):
        unsafe_data_quality(rows=[{"password": "private-password"}])
    with pytest.raises(ValueError, match="non-negative"):
        DataQualityError(violation_count=-1)


def test_error_codes_are_bounded_machine_identifiers() -> None:
    unsafe_error: Any = DaitaError
    with pytest.raises(ValueError, match="error_code"):
        DaitaError(error_code="Not Safe")
    with pytest.raises(TypeError, match="retryability"):
        unsafe_error(retryability="retryable")
