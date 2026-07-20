"""Typed versioned configuration assembled from existing runtime owners."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from ._json import canonical_json
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
        )


def resolve_agent_configuration(
    config: AgentConfig | None,
    *,
    model: ModelProvider | None,
    model_profile: ModelProfile | None,
    policy: DefaultPolicyEvaluator | None,
    budgets: LoopBudgets | None,
) -> tuple[ModelProfile | None, DefaultPolicyEvaluator | None, LoopBudgets | None]:
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
        return model_profile, policy, budgets
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
    )


__all__ = [
    "AgentConfig",
    "AgentRuntimeDefaults",
    "AgentRuntimeDefaultsConflictError",
]
