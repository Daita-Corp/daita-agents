"""
Unit tests for daita/config/base.py

Covers:
  - RetryPolicy: defaults, delay calculations per strategy, jitter, max_delay cap
  - RetryPolicy.execute_with_retry(): success on first try, retries on failure
  - AgentConfig: defaults, retry_enabled property, auto policy creation
"""

import asyncio

import pytest
from pydantic import ValidationError as PydanticValidationError

from daita.config.base import AgentConfig, AgentType, RetryPolicy, RetryStrategy

# ===========================================================================
# RetryPolicy — defaults and validation
# ===========================================================================


class TestRetryPolicyDefaults:
    def test_default_max_retries(self):
        policy = RetryPolicy()
        assert policy.max_retries == 3

    def test_default_base_delay(self):
        policy = RetryPolicy()
        assert policy.base_delay == 1.0

    def test_default_strategy_is_exponential(self):
        policy = RetryPolicy()
        assert policy.strategy == RetryStrategy.EXPONENTIAL

    def test_default_jitter_enabled(self):
        policy = RetryPolicy()
        assert policy.jitter is True

    def test_max_retries_below_zero_rejected(self):
        with pytest.raises(PydanticValidationError):
            RetryPolicy(max_retries=-1)

    def test_max_retries_above_twenty_rejected(self):
        with pytest.raises(PydanticValidationError):
            RetryPolicy(max_retries=21)


# ===========================================================================
# RetryPolicy — delay calculations
# ===========================================================================


class TestRetryPolicyDelayCalculations:
    def test_fixed_strategy_returns_base_delay(self):
        policy = RetryPolicy(strategy=RetryStrategy.FIXED, base_delay=2.0, jitter=False)
        for attempt in range(1, 5):
            assert policy.calculate_delay(attempt) == pytest.approx(2.0)

    def test_linear_strategy_scales_with_attempt(self):
        policy = RetryPolicy(
            strategy=RetryStrategy.LINEAR, base_delay=1.0, jitter=False
        )
        assert policy.calculate_delay(1) == pytest.approx(1.0)
        assert policy.calculate_delay(2) == pytest.approx(2.0)
        assert policy.calculate_delay(3) == pytest.approx(3.0)

    def test_exponential_strategy_doubles_each_attempt(self):
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL, base_delay=1.0, jitter=False
        )
        assert policy.calculate_delay(1) == pytest.approx(1.0)
        assert policy.calculate_delay(2) == pytest.approx(2.0)
        assert policy.calculate_delay(3) == pytest.approx(4.0)

    def test_max_delay_cap_respected(self):
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            max_delay=5.0,
            jitter=False,
        )
        # attempt 10 → 2^9 = 512 → capped at 5.0
        assert policy.calculate_delay(10) == pytest.approx(5.0)

    def test_jitter_produces_values_within_half_to_full(self):
        policy = RetryPolicy(
            strategy=RetryStrategy.FIXED, base_delay=10.0, max_delay=60.0, jitter=True
        )
        delays = [policy.calculate_delay(1) for _ in range(50)]
        # All values should be in [5.0, 10.0]
        assert all(5.0 <= d <= 10.0 for d in delays)

    def test_jitter_disabled_is_deterministic(self):
        policy = RetryPolicy(strategy=RetryStrategy.FIXED, base_delay=2.0, jitter=False)
        delays = {policy.calculate_delay(1) for _ in range(20)}
        assert len(delays) == 1  # Always the same value


# ===========================================================================
# RetryPolicy — execute_with_retry
# ===========================================================================


class TestRetryPolicyExecuteWithRetry:
    async def test_success_on_first_call(self):
        policy = RetryPolicy(max_retries=3, base_delay=0.1, jitter=False)
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await policy.execute_with_retry(func)
        assert result == "ok"
        assert call_count == 1

    async def test_retries_on_exception(self):
        # Use FIXED strategy with minimum delay to keep test fast
        policy = RetryPolicy(
            max_retries=2, strategy=RetryStrategy.FIXED, base_delay=0.1, jitter=False
        )
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("temporary failure")
            return "ok"

        result = await policy.execute_with_retry(func)
        assert result == "ok"
        assert call_count == 3

    async def test_raises_after_max_retries_exhausted(self):
        policy = RetryPolicy(
            max_retries=1, strategy=RetryStrategy.FIXED, base_delay=0.1, jitter=False
        )

        async def always_fails():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            await policy.execute_with_retry(always_fails)

    async def test_works_with_sync_function(self):
        policy = RetryPolicy(max_retries=1, base_delay=0.1, jitter=False)

        def sync_fn():
            return 42

        result = await policy.execute_with_retry(sync_fn)
        assert result == 42


# ===========================================================================
# AgentConfig
# ===========================================================================


class TestAgentConfig:
    def test_name_required(self):
        config = AgentConfig(name="MyAgent")
        assert config.name == "MyAgent"

    def test_default_type_is_standard(self):
        config = AgentConfig(name="X")
        assert config.type == AgentType.STANDARD

    def test_default_enabled_is_true(self):
        config = AgentConfig(name="X")
        assert config.enabled is True

    def test_retry_disabled_by_default(self):
        config = AgentConfig(name="X")
        assert config.retry_enabled is False

    def test_retry_enabled_when_enable_retry_true(self):
        config = AgentConfig(name="X", enable_retry=True)
        assert config.retry_enabled is True

    def test_enable_retry_auto_creates_default_policy(self):
        config = AgentConfig(name="X", enable_retry=True)
        assert config.retry_policy is not None
        assert isinstance(config.retry_policy, RetryPolicy)

    def test_custom_retry_policy_stored(self):
        policy = RetryPolicy(max_retries=5)
        config = AgentConfig(name="X", enable_retry=True, retry_policy=policy)
        assert config.retry_policy.max_retries == 5

    def test_retry_enabled_false_despite_provided_policy(self):
        # Providing a policy without enable_retry=True → retry_enabled still False
        policy = RetryPolicy(max_retries=2)
        config = AgentConfig(name="X", enable_retry=False, retry_policy=policy)
        assert config.retry_enabled is False
