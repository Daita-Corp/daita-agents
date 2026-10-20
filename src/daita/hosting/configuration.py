"""Persisted model configuration, credential references and model admission for one agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import cast
from uuid import uuid4

from .._json import canonical_json
from ..adapters.registry import SourceTypeRegistry
from ..config import AgentConfig
from ..learning_candidates import LEARNING_REVIEW_MAX_TOTAL_TOKENS
from ..llm.errors import ModelProviderError, ProviderErrorCode
from ..llm.factory import create_model_route_provider
from ..llm.models import (
    _DEFAULT_CALL_POLICY,
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelCallPolicy,
    ModelProfile,
    ModelRequest,
    ModelSensitivity,
    TextBlock,
    ToolDefinition,
)
from ..llm.profiles import reviewed_model_profile
from ..llm.protocols import ManagedModelProvider, ModelProvider
from ..llm.provider_definitions import admit_model_selection, provider_definition
from ..llm.routing import (
    ModelProviderRegistration,
    ModelRoute,
    ModelRouteCandidate,
    ModelRouter,
    RetryPolicy,
)
from ..loop.driver import ContextBuilder, ToolRuntime
from ..loop.models import LoopLimits
from ..observation import PrivateDiagnosticSink
from ..security import (
    CredentialSession,
    KeychainSecretProvider,
    KeychainStore,
    SecretProvider,
    SecretReference,
)
from .errors import (
    AgentHomeError,
    AgentModelConfigurationError,
    AgentNotConfiguredError,
)
from .execution_governor import RunAdmissionCoordinator

_MODEL_CONFIG_NAME = "config.json"


_MAX_MODEL_CONFIG_BYTES = 64 * 1_024


# One-shot validation and candidate review never retry or fall back.
_ONE_ATTEMPT_RETRY_POLICY = RetryPolicy(
    max_attempts_per_candidate=1, max_total_attempts=1, backoff_seconds=0
)


_MODEL_VALIDATION_TOOL_NAME = "daita_validate_tool_support"


_MODEL_VALIDATION_MAX_OUTPUT_TOKENS = 16


_REASONING_MODEL_VALIDATION_MAX_OUTPUT_TOKENS = 25_000


_CANDIDATE_REVIEWER_MAX_OUTPUT_TOKENS = LEARNING_REVIEW_MAX_TOTAL_TOKENS // 4


def _stage_c_model_routes(
    model: ModelProvider | None,
    model_route: ModelRoute | None,
) -> tuple[str, ...]:
    if model_route is not None:
        return tuple(candidate.provider_id for candidate in model_route.candidates)
    if isinstance(model, ModelRouter):
        return tuple(
            registration.provider.provider_id for registration in model.candidates
        )
    if model is None:
        return ()
    return (model.provider_id,)


def _model_execution_contracts(
    model: ModelProvider | None,
    profile: ModelProfile | None,
    route: ModelRoute | None,
    call_policy: ModelCallPolicy = _DEFAULT_CALL_POLICY,
) -> dict[str, str]:
    """Digest declared non-secret execution configuration, never SDK state."""

    if route is not None:
        material = _encode_model_route_contract(route)
        route_ids = tuple(candidate.provider_id for candidate in route.candidates)
    elif isinstance(model, ModelRouter):
        material = {
            "candidates": [
                {
                    "provider_id": candidate.provider.provider_id,
                    "profile": _encode_model_profile(candidate.profile),
                    "allowed_sensitivities": sorted(
                        item.value for item in candidate.allowed_sensitivities
                    ),
                }
                for candidate in model.candidates
            ],
            "retry_policy": {
                "max_attempts_per_candidate": model.retry_policy.max_attempts_per_candidate,
                "max_total_attempts": model.retry_policy.max_total_attempts,
                "backoff_seconds": model.retry_policy.backoff_seconds,
            },
        }
        route_ids = tuple(
            candidate.provider.provider_id for candidate in model.candidates
        )
    elif model is not None and profile is not None:
        material = {
            "provider_id": model.provider_id,
            "profile": _encode_model_profile(profile),
        }
        route_ids = (model.provider_id,)
    else:
        return {}
    return {
        route_id: "sha256:"
        + hashlib.sha256(
            canonical_json(
                {
                    "route_id": route_id,
                    "execution_configuration": material,
                    "model_call_policy": _encode_call_policy(call_policy),
                }
            ).encode("utf-8")
        ).hexdigest()
        for route_id in route_ids
    }


def _encode_model_route_contract(route: ModelRoute) -> dict[str, object]:
    """Encode exact runtime facts used to authorize frozen model execution."""

    return {
        "candidates": [
            {
                "allowed_sensitivities": sorted(
                    sensitivity.value for sensitivity in candidate.allowed_sensitivities
                ),
                "base_url": candidate.base_url,
                "profile": _encode_model_profile(candidate.profile),
                "provider_id": candidate.provider_id,
                "secret_reference": (
                    None
                    if candidate.secret_reference is None
                    else candidate.secret_reference.to_uri()
                ),
            }
            for candidate in route.candidates
        ],
        "retry_policy": {
            "max_attempts_per_candidate": route.retry_policy.max_attempts_per_candidate,
            "max_total_attempts": route.retry_policy.max_total_attempts,
            "backoff_seconds": route.retry_policy.backoff_seconds,
        },
    }


def _admit_host_model(
    model: ModelProvider | None,
    coordinator: RunAdmissionCoordinator,
    *,
    identity: str,
    home: Path | None,
    profile: ModelProfile | None = None,
    retry_policy: RetryPolicy = RetryPolicy(),
    owned: bool = False,
) -> ModelProvider | None:
    """Route one host model through the shared router retry and permit boundary.

    A caller-supplied router keeps its own candidates and policy. Any other
    provider becomes the sole candidate of a host router so injected and
    configured routes share the same retry, admission and attempt diagnostics.
    A caller-injected provider stays caller-owned and the router never closes
    it; a host-owned provider closes with its router. Either keeps serving every
    request sensitivity the host sends it.
    """

    if model is None:
        return None

    async def admit(key: str, deadline: float | None):
        return await coordinator.provider_permit(key, deadline=deadline)

    if isinstance(model, ModelRouter):
        if not model.provider_admission_bound:
            model.bind_provider_admission(admit)
        return model
    if profile is None:
        raise AgentNotConfiguredError("a configured model requires its ModelProfile")
    if model.provider_id != profile.id:
        raise AgentNotConfiguredError("model and profile identities differ")
    router = ModelRouter(
        (
            ModelProviderRegistration(
                provider=model,
                profile=profile,
                allowed_sensitivities=frozenset(ModelSensitivity),
                close_with_router=owned,
                concurrency_key=_injected_concurrency_key(
                    model.provider_id, identity=identity, home=home
                ),
            ),
        ),
        retry_policy=retry_policy,
    )
    router.bind_provider_admission(admit)
    return router


def _injected_concurrency_key(
    provider_id: str, *, identity: str, home: Path | None
) -> str:
    return f"injected:{provider_id}:{home.resolve() if home is not None else identity}"


def _resolve_candidate_reviewer_profile(
    model: ModelProvider | None,
    profile: ModelProfile | None,
) -> ModelProfile | None:
    """Admit one direct provider profile without retry or fallback."""

    if model is None:
        if profile is not None:
            raise ValueError("reviewer_profile requires reviewer_model")
        return None
    if isinstance(model, ModelRouter) and (
        len(model.candidates) != 1
        or model.retry_policy.max_attempts_per_candidate != 1
        or model.retry_policy.max_total_attempts != 1
    ):
        raise ValueError("candidate reviewer requires one attempt without fallback")
    resolved = profile
    if resolved is None:
        candidate = getattr(model, "model_profile", None)
        if isinstance(candidate, ModelProfile):
            resolved = candidate
    if not isinstance(resolved, ModelProfile):
        raise ValueError("candidate reviewer requires an explicit bounded ModelProfile")
    if resolved.max_output_tokens >= LEARNING_REVIEW_MAX_TOTAL_TOKENS:
        raise ValueError(
            "candidate reviewer profile output limit exceeds review token budget"
        )
    return resolved


def _candidate_reviewer_from_route(
    route: ModelRoute,
    *,
    secret_provider: SecretProvider | None,
) -> tuple[ManagedModelProvider, ModelProfile]:
    """Derive one bounded, direct reviewer from the persisted primary route."""

    primary = route.candidates[0]
    profile = replace(
        primary.profile,
        max_output_tokens=min(
            primary.profile.max_output_tokens,
            _CANDIDATE_REVIEWER_MAX_OUTPUT_TOKENS,
        ),
    )
    direct_route = ModelRoute(
        (
            ModelRouteCandidate(
                provider_id=primary.provider_id,
                profile=profile,
                base_url=primary.base_url,
                secret_reference=primary.secret_reference,
                allowed_sensitivities=primary.allowed_sensitivities,
            ),
        ),
        retry_policy=_ONE_ATTEMPT_RETRY_POLICY,
    )
    return (
        create_model_route_provider(
            direct_route,
            secret_provider=secret_provider,
        ),
        profile,
    )


def _resolve_configuration(
    config: AgentConfig | None,
    *,
    model: ModelProvider | None,
    model_profile: ModelProfile | None,
    limits: LoopLimits | None,
    secret_provider: SecretProvider | None,
) -> tuple[
    ModelProvider | None,
    ModelProfile | None,
    ModelRoute | None,
    LoopLimits,
    ModelCallPolicy,
]:
    if config is not None and not isinstance(config, AgentConfig):
        raise TypeError("config must be AgentConfig or None")
    route = None if config is None else config.model_route
    if route is not None:
        if model is not None:
            raise AgentNotConfiguredError(
                "model_route and injected model cannot be combined"
            )
        model = create_model_route_provider(route, secret_provider=secret_provider)
        model_profile = route.model_profile
    resolved_limits = limits or (LoopLimits() if config is None else config.limits)
    if model is not None and model_profile is None:
        raise AgentNotConfiguredError("a configured model requires its ModelProfile")
    if model is not None:
        assert model_profile is not None
        if model.provider_id != model_profile.id:
            raise AgentNotConfiguredError("model and profile identities differ")
    return (
        model,
        model_profile,
        route,
        resolved_limits,
        (ModelCallPolicy() if config is None else config.model_call_policy),
    )


def _resolve_credential_session(
    keychain: KeychainStore | None,
) -> tuple[CredentialSession, bool]:
    if isinstance(keychain, CredentialSession):
        return keychain, False
    return CredentialSession(keychain or KeychainSecretProvider()), True


def _diagnostic_sink(path: Path | None) -> PrivateDiagnosticSink | None:
    if path is None:
        return None
    if not isinstance(path, Path):
        raise TypeError("diagnostics_path must be pathlib.Path or None")
    return PrivateDiagnosticSink(path)


def _configuration_was_injected(
    config: AgentConfig | None,
    *,
    model: ModelProvider | None,
    model_profile: ModelProfile | None,
    context_builder: ContextBuilder | None,
    tools: ToolRuntime | None,
) -> bool:
    return any(
        value is not None
        for value in (
            config,
            model,
            model_profile,
            context_builder,
            tools,
        )
    )


def _model_profile(
    provider: str,
    model: str,
    *,
    context_window_tokens: int | None,
    max_output_tokens: int | None,
) -> ModelProfile:
    provider_id = f"{provider}:{model}"
    reviewed = reviewed_model_profile(provider_id)
    if reviewed is not None:
        if context_window_tokens is not None or max_output_tokens is not None:
            raise ValueError("reviewed models use their canonical profile limits")
        return reviewed
    if context_window_tokens is None or max_output_tokens is None:
        raise ValueError(
            "unreviewed models require explicit context and output token limits"
        )
    definition = provider_definition(provider)
    return ModelProfile(
        id=provider_id,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        supports_tools=True,
        supports_parallel_tools=(
            False if definition is None else definition.profile_supports_parallel_tools
        ),
        supports_structured_output=(
            False
            if definition is None
            else definition.profile_supports_structured_output
        ),
        supports_streaming=(
            False if definition is None else definition.profile_supports_streaming
        ),
        supports_reasoning=(
            False if definition is None else definition.profile_supports_reasoning
        ),
    )


def _model_route(
    provider: str,
    model: str,
    *,
    base_url: str | None,
    secret_reference: SecretReference | None,
    context_window_tokens: int | None,
    max_output_tokens: int | None,
) -> ModelRoute:
    profile = _model_profile(
        provider,
        model,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
    )
    return ModelRoute(
        (
            ModelRouteCandidate(
                provider_id=profile.id,
                profile=profile,
                base_url=base_url,
                secret_reference=secret_reference,
            ),
        )
    )


def _credential_account(agent_id: str, provider: str, nonce: str) -> str:
    digest = hashlib.sha256(
        f"{agent_id}\x00{provider}\x00{nonce}".encode("utf-8")
    ).hexdigest()[:24]
    return _credential_account_prefix(agent_id, provider) + digest


def _credential_account_prefix(agent_id: str, provider: str) -> str:
    prefix = f"{agent_id}:{provider}:"
    if len(prefix) + 24 > 256:
        agent_digest = hashlib.sha256(agent_id.encode("utf-8")).hexdigest()[:24]
        prefix = f"agent-{agent_digest}:{provider}:"
    return prefix


def _credential_reference_is_owned(
    reference: SecretReference,
    *,
    agent_id: str,
    provider: str,
) -> bool:
    if reference.scheme != "keychain":
        return False
    prefix = _credential_account_prefix(agent_id, provider)
    if not reference.name.startswith(prefix):
        return False
    suffix = reference.name[len(prefix) :]
    return len(suffix) == 24 and all(
        character in "0123456789abcdef" for character in suffix
    )


def _owned_source_credential_references(
    adapter_id: str,
    configuration: Mapping[str, object],
    *,
    agent_id: str,
    source_types: SourceTypeRegistry,
) -> tuple[SecretReference, ...]:
    """Return credentials this agent created for one source's declared fields."""

    source_type = source_types.get(adapter_id)
    references: list[SecretReference] = []
    for name in () if source_type is None else source_type.credential_fields:
        value = configuration.get(name)
        if not isinstance(value, str):
            continue
        try:
            reference = SecretReference.parse(value)
        except ValueError:
            continue
        if _credential_reference_is_owned(
            reference,
            agent_id=agent_id,
            provider=adapter_id,
        ):
            references.append(reference)
    return tuple(references)


def _owned_agent_credential_references(
    agent_id: str,
    *,
    model_document: object | None,
    source_configurations: tuple[tuple[str, Mapping[str, object]], ...],
    source_types: SourceTypeRegistry,
    mcp_reference_values: tuple[str, ...] = (),
) -> tuple[SecretReference, ...]:
    references: dict[str, SecretReference] = {}
    if model_document is not None:
        if not isinstance(model_document, dict):
            raise AgentHomeError(
                "model configuration is invalid; the agent home was preserved"
            )
        route = model_document.get("model_route")
        candidates = route.get("candidates") if isinstance(route, dict) else None
        if not isinstance(candidates, list):
            raise AgentHomeError(
                "model configuration is invalid; the agent home was preserved"
            )
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise AgentHomeError(
                    "model configuration is invalid; the agent home was preserved"
                )
            provider_id = candidate.get("provider_id")
            reference_value = candidate.get("secret_reference")
            if not isinstance(provider_id, str) or (
                reference_value is not None and not isinstance(reference_value, str)
            ):
                raise AgentHomeError(
                    "model configuration is invalid; the agent home was preserved"
                )
            provider, separator, _model = provider_id.partition(":")
            if not separator or reference_value is None:
                continue
            try:
                reference = SecretReference.parse(reference_value)
            except ValueError:
                continue
            if _credential_reference_is_owned(
                reference,
                agent_id=agent_id,
                provider=provider,
            ):
                references[reference.to_uri()] = reference
    for adapter_id, configuration in source_configurations:
        for reference in _owned_source_credential_references(
            adapter_id,
            configuration,
            agent_id=agent_id,
            source_types=source_types,
        ):
            references[reference.to_uri()] = reference
    for reference_value in mcp_reference_values:
        reference = SecretReference.parse(reference_value)
        if _credential_reference_is_owned(reference, agent_id=agent_id, provider="mcp"):
            references[reference.to_uri()] = reference
    return tuple(references[key] for key in sorted(references))


async def _validate_model_route(
    route: ModelRoute,
    *,
    secret_provider: SecretProvider,
    injected_provider: ModelProvider | None,
    injected_concurrency_key: str | None = None,
    call_policy: ModelCallPolicy = _DEFAULT_CALL_POLICY,
    provider_admission: (
        Callable[
            [str, float | None],
            Awaitable[AbstractAsyncContextManager[object]],
        ]
        | None
    ) = None,
) -> None:
    validation_candidates = tuple(
        replace(
            candidate,
            profile=replace(
                candidate.profile,
                max_output_tokens=min(
                    (
                        _REASONING_MODEL_VALIDATION_MAX_OUTPUT_TOKENS
                        if candidate.profile.supports_reasoning
                        else _MODEL_VALIDATION_MAX_OUTPUT_TOKENS
                    ),
                    candidate.profile.max_output_tokens,
                    candidate.profile.context_window_tokens - 1,
                ),
            ),
        )
        for candidate in route.candidates
    )
    validation_route = ModelRoute(validation_candidates, _ONE_ATTEMPT_RETRY_POLICY)
    owns_provider = injected_provider is None
    if owns_provider:
        provider = create_model_route_provider(
            validation_route,
            secret_provider=secret_provider,
        )
        if provider_admission is not None:
            assert isinstance(provider, ModelRouter)
            provider.bind_provider_admission(provider_admission)
    else:
        assert injected_provider is not None
        if len(validation_candidates) != 1:
            raise AgentNotConfiguredError(
                "an injected validation provider requires one route candidate"
            )
        candidate = validation_candidates[0]
        if injected_provider.provider_id != candidate.profile.id:
            raise AgentNotConfiguredError(
                "validation provider and configured model identities differ"
            )
        provider = ModelRouter(
            (
                ModelProviderRegistration(
                    provider=injected_provider,
                    profile=candidate.profile,
                    allowed_sensitivities=candidate.allowed_sensitivities,
                    concurrency_key=injected_concurrency_key,
                ),
            ),
            retry_policy=validation_route.retry_policy,
        )
        if provider_admission is not None:
            provider.bind_provider_admission(provider_admission)
    try:
        await _require_validated_model_provider(
            provider,
            call_policy=call_policy,
            deadline=asyncio.get_running_loop().time()
            + call_policy.max_request_seconds,
        )
    except BaseException:
        if owns_provider:
            assert isinstance(provider, ManagedModelProvider)
            try:
                await provider.close(
                    deadline=asyncio.get_running_loop().time()
                    + call_policy.cleanup_timeout_seconds
                )
            except BaseException:  # noqa: BLE001
                # Retain the original validation failure.
                pass  # Cleanup's once-only outcome stays on the provider owner.
        raise
    else:
        if owns_provider:
            assert isinstance(provider, ManagedModelProvider)
            await provider.close(
                deadline=asyncio.get_running_loop().time()
                + call_policy.cleanup_timeout_seconds
            )


async def _require_validated_model_provider(
    provider: ModelProvider,
    *,
    call_policy: ModelCallPolicy = _DEFAULT_CALL_POLICY,
    deadline: float | None = None,
) -> None:
    response = await provider.generate(
        ModelRequest(
            call_policy=call_policy,
            deadline=deadline,
            messages=(
                CanonicalMessage(
                    role=MessageRole.USER,
                    content=(TextBlock("Call the validation tool once."),),
                ),
            ),
            tools=(
                ToolDefinition(
                    name=_MODEL_VALIDATION_TOOL_NAME,
                    description="Prove provider tool-call compatibility.",
                    input_schema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                ),
            ),
            sensitivity=ModelSensitivity.PUBLIC,
            allow_parallel_tool_calls=False,
        )
    )
    if response.finish_reason is FinishReason.LENGTH:
        raise ModelProviderError(
            ProviderErrorCode.OUTPUT_LIMIT,
            "model exhausted the validation output budget",
        )
    if (
        response.finish_reason is not FinishReason.TOOL_CALLS
        or len(response.tool_calls) != 1
        or response.tool_calls[0].name != _MODEL_VALIDATION_TOOL_NAME
        or response.tool_calls[0].arguments
    ):
        raise ValueError("model validation did not prove native tool-call support")


def _keychain_references(
    config: AgentConfig | None,
) -> tuple[tuple[str, SecretReference], ...]:
    if config is None or config.model_route is None:
        return ()
    return tuple(
        (candidate.provider_id.partition(":")[0], reference)
        for candidate in config.model_route.candidates
        if (reference := candidate.secret_reference) is not None
        and reference.scheme == "keychain"
    )


def _read_model_configuration_document(home: Path) -> object | None:
    path = home / _MODEL_CONFIG_NAME
    if path.is_symlink():
        raise AgentHomeError("model configuration cannot be a symlink")
    if not path.exists():
        return None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("configuration is not a regular file")
            chunks: list[bytes] = []
            size = 0
            while size <= _MAX_MODEL_CONFIG_BYTES:
                chunk = os.read(
                    descriptor, min(8_192, _MAX_MODEL_CONFIG_BYTES + 1 - size)
                )
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            if size > _MAX_MODEL_CONFIG_BYTES:
                raise ValueError("configuration exceeds its byte bound")
        finally:
            os.close(descriptor)
        value = json.loads(
            b"".join(chunks).decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
        return value
    except AgentHomeError:
        raise
    except Exception:
        raise AgentHomeError("model configuration is invalid") from None


def _read_model_configuration(home: Path, agent_id: str) -> AgentConfig | None:
    value = _read_model_configuration_document(home)
    if value is None:
        return None
    try:
        return _decode_agent_config(value, agent_id=agent_id)
    except AgentHomeError:
        raise
    except Exception:
        raise AgentModelConfigurationError(
            "saved model configuration must be replaced"
        ) from None


def _write_model_configuration(home: Path, config: AgentConfig) -> None:
    if not isinstance(config, AgentConfig) or config.model_route is None:
        raise ValueError("persisted model configuration requires a model route")
    path = home / _MODEL_CONFIG_NAME
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise AgentHomeError("model configuration path is invalid")
    data = json.dumps(
        _encode_agent_config(config),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(data) > _MAX_MODEL_CONFIG_BYTES:
        raise AgentHomeError("model configuration exceeds its byte bound")
    temporary = home / f".{_MODEL_CONFIG_NAME}.{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        directory = os.open(home, os.O_RDONLY)
        try:
            try:
                os.fsync(directory)
            except OSError:
                pass
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _encode_call_policy(policy: ModelCallPolicy) -> dict[str, object]:
    from dataclasses import asdict

    return asdict(policy)


def _decode_call_policy(value: object) -> ModelCallPolicy:
    from dataclasses import fields

    values = _strict_mapping(value, {item.name for item in fields(ModelCallPolicy)})
    return ModelCallPolicy(**cast(Mapping[str, float], values))


def _encode_agent_config(config: AgentConfig) -> dict[str, object]:
    route = config.model_route
    assert route is not None
    return {
        "model_call_policy": _encode_call_policy(config.model_call_policy),
        "limits": {
            "max_estimated_cost_usd": (
                None
                if config.limits.max_estimated_cost_usd is None
                else str(config.limits.max_estimated_cost_usd)
            ),
            "max_steps": config.limits.max_steps,
            "max_total_tokens": config.limits.max_total_tokens,
            "max_wall_time_seconds": config.limits.max_wall_time_seconds,
        },
        "model_route": {
            "candidates": [
                {
                    "allowed_sensitivities": sorted(
                        sensitivity.value
                        for sensitivity in candidate.allowed_sensitivities
                    ),
                    "base_url": candidate.base_url,
                    "profile_limits": _persisted_profile_limits(candidate),
                    "provider_id": candidate.provider_id,
                    "secret_reference": (
                        None
                        if candidate.secret_reference is None
                        else candidate.secret_reference.to_uri()
                    ),
                }
                for candidate in route.candidates
            ],
            "retry_policy": {
                "max_attempts_per_candidate": route.retry_policy.max_attempts_per_candidate,
                "max_total_attempts": route.retry_policy.max_total_attempts,
                "backoff_seconds": route.retry_policy.backoff_seconds,
            },
        },
    }


def _persisted_profile_limits(candidate: ModelRouteCandidate) -> dict[str, int] | None:
    reviewed = reviewed_model_profile(candidate.provider_id)
    if reviewed is not None:
        if candidate.profile != reviewed:
            raise ValueError(
                "reviewed model route contains non-canonical profile facts"
            )
        return None
    provider_name, separator, model_name = candidate.provider_id.partition(":")
    if not separator:
        raise ValueError("provider ID is incomplete")
    expected = _model_profile(
        provider_name,
        model_name,
        context_window_tokens=candidate.profile.context_window_tokens,
        max_output_tokens=candidate.profile.max_output_tokens,
    )
    if candidate.profile != expected:
        raise ValueError("custom model route contains non-canonical profile facts")
    return {
        "context_window_tokens": candidate.profile.context_window_tokens,
        "max_output_tokens": candidate.profile.max_output_tokens,
    }


def _encode_model_profile(profile: ModelProfile) -> dict[str, object]:
    return {
        "available": profile.available,
        "context_window_tokens": profile.context_window_tokens,
        "data_routing_classification": profile.data_routing_classification,
        "healthy": profile.healthy,
        "id": profile.id,
        "input_cost_per_million_usd": None,
        "max_output_tokens": profile.max_output_tokens,
        "output_cost_per_million_usd": None,
        "supports_documents": profile.supports_documents,
        "supports_native_continuation": profile.supports_native_continuation,
        "supports_parallel_tools": profile.supports_parallel_tools,
        "supports_prompt_caching": profile.supports_prompt_caching,
        "supports_reasoning": profile.supports_reasoning,
        "supports_streaming": profile.supports_streaming,
        "supports_structured_output": profile.supports_structured_output,
        "supports_tools": profile.supports_tools,
        "supports_vision": profile.supports_vision,
    }


def _decode_agent_config(value: object, *, agent_id: str) -> AgentConfig:
    document = _strict_mapping(value, {"limits", "model_route", "model_call_policy"})
    route_value = _strict_mapping(
        document["model_route"],
        {"candidates", "retry_policy"},
    )
    raw_candidates = route_value["candidates"]
    if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= 5:
        raise ValueError("model route candidates are invalid")
    candidates: list[ModelRouteCandidate] = []
    for raw_candidate in raw_candidates:
        candidate = _strict_mapping(
            raw_candidate,
            {
                "allowed_sensitivities",
                "base_url",
                "profile_limits",
                "provider_id",
                "secret_reference",
            },
        )
        raw_allowed = candidate["allowed_sensitivities"]
        if not isinstance(raw_allowed, list) or not raw_allowed:
            raise ValueError("allowed sensitivities are invalid")
        reference_value = candidate["secret_reference"]
        if reference_value is not None and not isinstance(reference_value, str):
            raise TypeError("secret reference must be text or null")
        base_url = candidate["base_url"]
        if base_url is not None and not isinstance(base_url, str):
            raise TypeError("base URL must be text or null")
        provider_id = candidate["provider_id"]
        if not isinstance(provider_id, str):
            raise TypeError("provider ID must be text")
        provider_name, separator, model_name = provider_id.partition(":")
        if not separator:
            raise ValueError("provider ID is incomplete")
        _, _, admitted_url, requires_credential = admit_model_selection(
            provider_name,
            model_name,
            base_url,
        )
        reviewed_profile = reviewed_model_profile(provider_id)
        limits_value = candidate["profile_limits"]
        if limits_value is None:
            if reviewed_profile is None:
                raise AgentModelConfigurationError(
                    "saved model configuration must be replaced"
                )
            profile = reviewed_profile
        else:
            profile_limits = _strict_mapping(
                limits_value,
                {"context_window_tokens", "max_output_tokens"},
            )
            context_window_tokens = profile_limits["context_window_tokens"]
            max_output_tokens = profile_limits["max_output_tokens"]
            if (
                not isinstance(context_window_tokens, int)
                or isinstance(context_window_tokens, bool)
                or not isinstance(max_output_tokens, int)
                or isinstance(max_output_tokens, bool)
            ):
                raise ValueError("saved custom model limits are invalid")
            profile = _model_profile(
                provider_name,
                model_name,
                context_window_tokens=context_window_tokens,
                max_output_tokens=max_output_tokens,
            )
        reference = (
            None if reference_value is None else SecretReference.parse(reference_value)
        )
        if requires_credential and (
            reference is None or reference.scheme != "keychain"
        ):
            raise ValueError("persisted provider credential reference is incomplete")
        if requires_credential and (
            reference is None
            or not _credential_reference_is_owned(
                reference,
                agent_id=agent_id,
                provider=provider_name,
            )
        ):
            raise ValueError(
                "persisted provider credential reference belongs to another agent"
            )
        if not requires_credential and reference is not None:
            raise ValueError(
                "local or client-owned subscription configuration cannot contain "
                "a credential"
            )
        allowed = tuple(ModelSensitivity(cast(str, item)) for item in raw_allowed)
        if len(allowed) != len(set(allowed)):
            raise ValueError("allowed sensitivities cannot repeat")
        candidates.append(
            ModelRouteCandidate(
                provider_id=provider_id,
                profile=profile,
                base_url=admitted_url,
                secret_reference=reference,
                allowed_sensitivities=frozenset(allowed),
            )
        )
    raw_retry = _strict_mapping(
        route_value["retry_policy"],
        {"max_attempts_per_candidate", "max_total_attempts", "backoff_seconds"},
    )
    route = ModelRoute(
        tuple(candidates),
        RetryPolicy(
            max_attempts_per_candidate=cast(
                int, raw_retry["max_attempts_per_candidate"]
            ),
            max_total_attempts=cast(int, raw_retry["max_total_attempts"]),
            backoff_seconds=cast(float, raw_retry["backoff_seconds"]),
        ),
    )
    raw_limits = _strict_mapping(
        document["limits"],
        {
            "max_estimated_cost_usd",
            "max_steps",
            "max_total_tokens",
            "max_wall_time_seconds",
        },
    )
    cost = raw_limits["max_estimated_cost_usd"]
    if cost is not None and not isinstance(cost, str):
        raise TypeError("estimated cost must be text or null")
    return AgentConfig(
        model_route=route,
        model_call_policy=_decode_call_policy(document["model_call_policy"]),
        limits=LoopLimits(
            max_steps=cast(int, raw_limits["max_steps"]),
            max_total_tokens=cast(int, raw_limits["max_total_tokens"]),
            max_wall_time_seconds=cast(float, raw_limits["max_wall_time_seconds"]),
            max_estimated_cost_usd=None if cost is None else Decimal(cost),
        ),
    )


def _strict_mapping(value: object, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("configuration object fields are invalid")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("configuration object keys must be text")
    return cast(Mapping[str, object], value)


def _validate_custom_loop(
    model: ModelProvider | None,
    profile: ModelProfile | None,
    context: ContextBuilder | None,
    tools: ToolRuntime | None,
) -> None:
    del profile
    if (context is None) != (tools is None):
        raise AgentNotConfiguredError(
            "custom context_builder and tools must be configured together"
        )
    if model is None and context is not None:
        raise AgentNotConfiguredError("custom loop components require a model")
