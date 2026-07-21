"""Typed versioned configuration assembled from existing runtime owners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

from ._json import canonical_json
from .context.session import SessionCompressionPolicy
from .errors import ConfigError
from .llm.models import ModelProfile
from .llm.protocols import ModelProvider
from .llm.routing import ModelRoute, ModelRouter, RetryPolicy
from .loop.models import LoopBudgets
from .operations.governance import DefaultPolicyEvaluator, DefaultPolicyProfile


@dataclass(frozen=True, slots=True)
class AgentRuntimeDefaults:
    """One immutable, restart-stable binding of future-operation defaults.

    Model profiles and retry routing remain owned by the model-profile
    repository and ``ModelRouter``.  This record owns only defaults that would
    otherwise silently change when a different process reopens the agent.
    """

    schema_version: int = 1
    revision: int = 1
    budgets: LoopBudgets = LoopBudgets()
    policy_profile: DefaultPolicyProfile = DefaultPolicyProfile()
    session_compression_policy: SessionCompressionPolicy = SessionCompressionPolicy()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("runtime defaults schema_version must be 1")
        if self.revision != 1:
            raise ValueError("runtime defaults revision must be 1")
        if not isinstance(self.budgets, LoopBudgets):
            raise TypeError("runtime defaults budgets must be LoopBudgets")
        if not isinstance(self.policy_profile, DefaultPolicyProfile):
            raise TypeError(
                "runtime defaults policy_profile must be DefaultPolicyProfile"
            )
        if not isinstance(self.session_compression_policy, SessionCompressionPolicy):
            raise TypeError(
                "runtime defaults session_compression_policy must be "
                "SessionCompressionPolicy"
            )

    @property
    def _legacy_fingerprint(self) -> str:
        """Return the exact migration-14 fingerprint for legacy SQLite rows."""

        budgets = self.budgets
        policy = self.policy_profile
        encoded = canonical_json(
            {
                "budgets": {
                    "max_actions": budgets.max_actions,
                    "max_estimated_cost_usd": (
                        None
                        if budgets.max_estimated_cost_usd is None
                        else str(budgets.max_estimated_cost_usd)
                    ),
                    "max_identical_failures": budgets.max_identical_failures,
                    "max_observation_characters": budgets.max_observation_characters,
                    "max_repairs": budgets.max_repairs,
                    "max_total_tokens": budgets.max_total_tokens,
                    "max_turns": budgets.max_turns,
                    "max_wall_time_seconds": budgets.max_wall_time_seconds,
                    "task_timeout_seconds": budgets.task_timeout_seconds,
                },
                "policy_profile": {
                    "allow_destructive": policy.allow_destructive,
                    "id": policy.id,
                    "version": policy.version,
                },
                "revision": self.revision,
                "schema_version": self.schema_version,
            }
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    @property
    def fingerprint(self) -> str:
        budgets = self.budgets
        policy = self.policy_profile
        encoded = canonical_json(
            {
                "budgets": {
                    "max_actions": budgets.max_actions,
                    "max_estimated_cost_usd": (
                        None
                        if budgets.max_estimated_cost_usd is None
                        else str(budgets.max_estimated_cost_usd)
                    ),
                    "max_identical_failures": budgets.max_identical_failures,
                    "max_observation_characters": (budgets.max_observation_characters),
                    "max_repairs": budgets.max_repairs,
                    "max_total_tokens": budgets.max_total_tokens,
                    "max_turns": budgets.max_turns,
                    "max_wall_time_seconds": budgets.max_wall_time_seconds,
                    "task_timeout_seconds": budgets.task_timeout_seconds,
                },
                "policy_profile": {
                    "allow_destructive": policy.allow_destructive,
                    "id": policy.id,
                    "version": policy.version,
                },
                "revision": self.revision,
                "schema_version": self.schema_version,
                "session_compression_policy": _session_compression_policy_data(
                    self.session_compression_policy
                ),
            }
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


class AgentRuntimeDefaultsConflictError(RuntimeError):
    """A durable agent is already bound to different runtime defaults."""


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Public configuration input without an untyped settings bag.

    Model capabilities, loop limits, retry behavior, and governance remain
    owned by their existing records and services.  This record only binds an
    exact versioned set for the thin public ``Agent`` facade.
    """

    schema_version: int = 1
    model_route: ModelRoute | None = None
    model_profile: ModelProfile | None = None
    budgets: LoopBudgets = LoopBudgets()
    retry_policy: RetryPolicy = RetryPolicy()
    policy_profile: DefaultPolicyProfile = DefaultPolicyProfile()
    session_compression_policy: SessionCompressionPolicy = SessionCompressionPolicy()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("agent configuration schema_version must be 1")
        if self.model_route is not None and not isinstance(
            self.model_route,
            ModelRoute,
        ):
            raise TypeError("model_route must be a ModelRoute or None")
        if self.model_profile is not None and not isinstance(
            self.model_profile,
            ModelProfile,
        ):
            raise TypeError("model_profile must be a ModelProfile or None")
        if not isinstance(self.budgets, LoopBudgets):
            raise TypeError("budgets must be a LoopBudgets record")
        if not isinstance(self.retry_policy, RetryPolicy):
            raise TypeError("retry_policy must be a RetryPolicy")
        if not isinstance(self.policy_profile, DefaultPolicyProfile):
            raise TypeError("policy_profile must be a DefaultPolicyProfile")
        if not isinstance(self.session_compression_policy, SessionCompressionPolicy):
            raise TypeError(
                "session_compression_policy must be a SessionCompressionPolicy"
            )
        if self.model_route is not None:
            if (
                self.model_profile is not None
                and self.model_profile != self.model_route.model_profile
            ):
                raise ValueError("model_profile must match model_route.model_profile")
            if self.retry_policy not in {
                RetryPolicy(),
                self.model_route.retry_policy,
            }:
                raise ValueError("retry_policy must match model_route.retry_policy")

    @property
    def runtime_defaults(self) -> AgentRuntimeDefaults:
        """Return the state-owned subset of this public facade input."""

        return AgentRuntimeDefaults(
            budgets=self.budgets,
            policy_profile=self.policy_profile,
            session_compression_policy=self.session_compression_policy,
        )


_SESSION_COMPRESSION_POLICY_FIELDS = frozenset(
    {
        "compression_threshold_tokens",
        "max_corrections",
        "max_excerpt_characters",
        "max_summary_characters",
        "retain_latest_operations",
        "schema_version",
    }
)


def _session_compression_policy_data(
    policy: SessionCompressionPolicy,
) -> dict[str, object]:
    if not isinstance(policy, SessionCompressionPolicy):
        raise TypeError("policy must be a SessionCompressionPolicy")
    return {
        "compression_threshold_tokens": policy.compression_threshold_tokens,
        "max_corrections": policy.max_corrections,
        "max_excerpt_characters": policy.max_excerpt_characters,
        "max_summary_characters": policy.max_summary_characters,
        "retain_latest_operations": policy.retain_latest_operations,
        "schema_version": policy.schema_version,
    }


def _session_compression_policy_from_data(
    value: Mapping[str, object],
) -> SessionCompressionPolicy:
    if (
        not isinstance(value, Mapping)
        or set(value) != _SESSION_COMPRESSION_POLICY_FIELDS
    ):
        raise ValueError("session compression policy has unknown or missing fields")
    threshold = value["compression_threshold_tokens"]
    if threshold is not None and (
        not isinstance(threshold, int) or isinstance(threshold, bool)
    ):
        raise ValueError("session compression threshold must be an integer or None")
    integer_values: dict[str, int] = {}
    for field_name in (
        "max_corrections",
        "max_excerpt_characters",
        "max_summary_characters",
        "retain_latest_operations",
        "schema_version",
    ):
        item = value[field_name]
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"session compression {field_name} must be an integer")
        integer_values[field_name] = item
    return SessionCompressionPolicy(
        schema_version=integer_values["schema_version"],
        compression_threshold_tokens=threshold,
        retain_latest_operations=integer_values["retain_latest_operations"],
        max_summary_characters=integer_values["max_summary_characters"],
        max_excerpt_characters=integer_values["max_excerpt_characters"],
        max_corrections=integer_values["max_corrections"],
    )


def resolve_agent_configuration(
    config: AgentConfig | None,
    *,
    model: ModelProvider | None,
    model_profile: ModelProfile | None,
    policy: DefaultPolicyEvaluator | None,
    budgets: LoopBudgets | None,
) -> tuple[
    ModelProfile | None,
    DefaultPolicyEvaluator | None,
    LoopBudgets | None,
    SessionCompressionPolicy | None,
]:
    """Resolve one invocation without deciding persisted reopen semantics."""

    if config is not None and not isinstance(config, AgentConfig):
        raise TypeError("config must be an AgentConfig or None")
    if model_profile is not None and not isinstance(model_profile, ModelProfile):
        raise TypeError("model_profile must be a ModelProfile or None")
    if policy is not None and not isinstance(policy, DefaultPolicyEvaluator):
        raise TypeError("policy must be a DefaultPolicyEvaluator or None")
    if budgets is not None and not isinstance(budgets, LoopBudgets):
        raise TypeError("budgets must be a LoopBudgets record or None")
    if config is None:
        return model_profile, policy, budgets, None
    configured_profile = (
        config.model_route.model_profile
        if config.model_route is not None
        else config.model_profile
    )
    if (
        model_profile is not None
        and configured_profile is not None
        and model_profile != configured_profile
    ):
        raise ConfigError(
            "AgentConfig model profile conflicts with model_profile.",
            section="model",
            error_code="config_conflict",
        )
    if budgets is not None and budgets != config.budgets:
        raise ConfigError(
            "AgentConfig budgets conflict with budgets.",
            section="budgets",
            error_code="config_conflict",
        )
    if policy is not None and policy.profile != config.policy_profile:
        raise ConfigError(
            "AgentConfig policy profile conflicts with policy.",
            section="policy",
            error_code="config_conflict",
        )
    if config.model_route is not None:
        if config.retry_policy not in {
            RetryPolicy(),
            config.model_route.retry_policy,
        }:
            raise ConfigError(
                "AgentConfig retry policy conflicts with the model route.",
                section="model.retry",
                error_code="config_conflict",
            )
    elif isinstance(model, ModelRouter):
        if model.retry_policy != config.retry_policy:
            raise ConfigError(
                "AgentConfig retry policy conflicts with the model router.",
                section="model.retry",
                error_code="config_conflict",
            )
    elif config.retry_policy != RetryPolicy():
        raise ConfigError(
            "A non-default retry policy requires a configured ModelRouter.",
            section="model.retry",
            error_code="config_owner_required",
        )
    return (
        model_profile if model_profile is not None else configured_profile,
        policy or DefaultPolicyEvaluator(config.policy_profile),
        config.budgets,
        config.session_compression_policy,
    )


__all__ = [
    "AgentConfig",
    "AgentRuntimeDefaults",
    "AgentRuntimeDefaultsConflictError",
]
