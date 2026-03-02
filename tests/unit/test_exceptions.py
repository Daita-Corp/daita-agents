"""
Unit tests for daita/core/exceptions.py

Covers:
  - Exception hierarchy and retry_hint correctness
  - is_transient / is_retryable / is_permanent predicates
  - classify_exception() for Daita and standard Python exceptions
  - create_contextual_error() wrapping and context preservation
"""

import pytest

from daita.core.exceptions import (
    AgentError,
    AuthenticationError,
    BackpressureError,
    CircuitBreakerOpenError,
    ConfigError,
    ConnectionError,
    DaitaError,
    LLMError,
    PermanentError,
    PermissionError,
    PluginError,
    RateLimitError,
    RetryableError,
    TaskTimeoutError,
    TransientError,
    ValidationError,
    classify_exception,
    create_contextual_error,
)


# ===========================================================================
# Exception retry_hint correctness
# ===========================================================================

class TestRetryHints:
    def test_daita_error_default_hint(self):
        err = DaitaError("msg")
        assert err.retry_hint == "unknown"

    def test_transient_error_hint(self):
        assert TransientError("msg").retry_hint == "transient"

    def test_retryable_error_hint(self):
        assert RetryableError("msg").retry_hint == "retryable"

    def test_permanent_error_hint(self):
        assert PermanentError("msg").retry_hint == "permanent"

    def test_rate_limit_error_is_transient(self):
        assert RateLimitError().retry_hint == "transient"

    def test_authentication_error_is_permanent(self):
        assert AuthenticationError().retry_hint == "permanent"

    def test_agent_error_default_is_retryable(self):
        assert AgentError("msg").retry_hint == "retryable"

    def test_config_error_is_permanent(self):
        assert ConfigError("bad config").retry_hint == "permanent"

    def test_llm_error_stores_provider_and_model(self):
        err = LLMError("fail", provider="openai", model="gpt-4")
        assert err.provider == "openai"
        assert err.model == "gpt-4"

    def test_plugin_error_stores_plugin_name(self):
        err = PluginError("fail", plugin_name="s3")
        assert err.plugin_name == "s3"

    def test_rate_limit_error_includes_retry_after_in_message(self):
        err = RateLimitError(retry_after=10)
        assert "10" in str(err)

    def test_task_timeout_error_includes_duration_in_message(self):
        err = TaskTimeoutError(timeout_duration=5.0)
        assert "5.0" in str(err)

    def test_connection_error_includes_host(self):
        err = ConnectionError(host="example.com", port=5432)
        assert "example.com" in str(err)

    def test_backpressure_error_includes_queue_size(self):
        err = BackpressureError(queue_size=50)
        assert "50" in str(err)

    def test_circuit_breaker_error_is_permanent(self):
        assert CircuitBreakerOpenError().retry_hint == "permanent"

    def test_validation_error_is_permanent(self):
        assert ValidationError("bad value").retry_hint == "permanent"


# ===========================================================================
# is_* predicate methods
# ===========================================================================

class TestPredicateMethods:
    def test_is_transient_true_for_transient(self):
        assert TransientError("msg").is_transient() is True

    def test_is_transient_false_for_permanent(self):
        assert PermanentError("msg").is_transient() is False

    def test_is_transient_false_for_retryable(self):
        assert RetryableError("msg").is_transient() is False

    def test_is_retryable_true_for_transient(self):
        # Transient implies retryable
        assert TransientError("msg").is_retryable() is True

    def test_is_retryable_true_for_retryable(self):
        assert RetryableError("msg").is_retryable() is True

    def test_is_retryable_false_for_permanent(self):
        assert PermanentError("msg").is_retryable() is False

    def test_is_permanent_true_for_permanent(self):
        assert PermanentError("msg").is_permanent() is True

    def test_is_permanent_false_for_transient(self):
        assert TransientError("msg").is_permanent() is False

    def test_is_permanent_false_for_retryable(self):
        assert RetryableError("msg").is_permanent() is False


# ===========================================================================
# classify_exception()
# ===========================================================================

class TestClassifyException:
    def test_classify_daita_transient(self):
        assert classify_exception(TransientError("msg")) == "transient"

    def test_classify_daita_retryable(self):
        assert classify_exception(RetryableError("msg")) == "retryable"

    def test_classify_daita_permanent(self):
        assert classify_exception(PermanentError("msg")) == "permanent"

    def test_classify_value_error_is_permanent(self):
        assert classify_exception(ValueError("bad")) == "permanent"

    def test_classify_type_error_is_permanent(self):
        assert classify_exception(TypeError("bad type")) == "permanent"

    def test_classify_key_error_is_permanent(self):
        assert classify_exception(KeyError("missing")) == "permanent"

    def test_classify_attribute_error_is_permanent(self):
        assert classify_exception(AttributeError("no attr")) == "permanent"

    def test_classify_os_error_is_transient(self):
        assert classify_exception(OSError("disk full")) == "transient"

    def test_classify_io_error_is_transient(self):
        assert classify_exception(IOError("io error")) == "transient"

    def test_classify_unknown_exception_is_retryable(self):
        class WeirdError(Exception):
            pass

        assert classify_exception(WeirdError("?")) == "retryable"


# ===========================================================================
# create_contextual_error()
# ===========================================================================

class TestCreateContextualError:
    def test_wraps_value_error_as_permanent(self):
        wrapped = create_contextual_error(ValueError("bad"))
        assert isinstance(wrapped, PermanentError)

    def test_wraps_os_error_as_transient(self):
        wrapped = create_contextual_error(OSError("fail"))
        assert isinstance(wrapped, TransientError)

    def test_wraps_unknown_error_as_retryable(self):
        class Unknown(Exception):
            pass

        wrapped = create_contextual_error(Unknown("?"))
        assert isinstance(wrapped, RetryableError)

    def test_preserves_original_message(self):
        wrapped = create_contextual_error(ValueError("original message"))
        assert "original message" in str(wrapped)

    def test_override_hint_permanent(self):
        # An OSError (normally transient) forced to permanent via override
        wrapped = create_contextual_error(OSError("fail"), retry_hint="permanent")
        assert isinstance(wrapped, PermanentError)

    def test_context_stored(self):
        ctx = {"source": "database", "query": "SELECT 1"}
        wrapped = create_contextual_error(ValueError("bad"), context=ctx)
        assert wrapped.context == ctx

    def test_context_defaults_to_empty_dict_when_none(self):
        wrapped = create_contextual_error(ValueError("bad"))
        assert isinstance(wrapped.context, dict)
