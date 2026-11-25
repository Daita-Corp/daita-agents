"""Call a capability domain's optional session-aware hooks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from ._json import FrozenJsonObject
from .capabilities import CallProvenance, Capability, ToolExecution, ToolOutput
from .llm.models import ModelSensitivity, ToolCall
from .loop.models import RunInput
from .loop.session import RunSession

if TYPE_CHECKING:
    from .capability_runtime import CapabilityDomain, SideEffectPlan


async def _domain_project(
    domain: CapabilityDomain,
    run: RunInput,
    session: RunSession | None,
    *,
    call: ToolCall | None = None,
) -> tuple[str, ...]:
    project_call = getattr(domain, "project_call", None)
    if call is not None and callable(project_call):
        project = cast(Callable[..., Awaitable[tuple[str, ...]]], project_call)
        return await project(run, call, session)
    project_session = getattr(domain, "project_session", None)
    if session is not None and callable(project_session):
        project = cast(
            Callable[..., Awaitable[tuple[str, ...]]],
            project_session,
        )
        return await project(run, session)
    return await domain.project(run)


async def _domain_prepare_call(
    domain: CapabilityDomain,
    run: RunInput,
    call: ToolCall,
    capability: Capability,
    arguments: FrozenJsonObject,
    *,
    request_sensitivity: ModelSensitivity,
    session: RunSession | None,
) -> FrozenJsonObject:
    prepare_session_call = getattr(domain, "prepare_session_call", None)
    if session is not None and callable(prepare_session_call):
        prepare = cast(
            Callable[..., Awaitable[FrozenJsonObject]],
            prepare_session_call,
        )
        return await prepare(
            run,
            call,
            capability,
            arguments,
            request_sensitivity=request_sensitivity,
            session=session,
        )
    return await domain.prepare_call(
        run,
        call,
        capability,
        arguments,
        request_sensitivity=request_sensitivity,
    )


async def _domain_call_provenance(
    domain: CapabilityDomain,
    run: RunInput,
    call: ToolCall,
    capability: Capability,
    arguments: FrozenJsonObject,
    *,
    session: RunSession | None,
) -> CallProvenance:
    """Ask the owning domain for evidence its capability declares it derives."""

    if not capability.derives_call_provenance:
        return CallProvenance()
    derive = cast(
        Callable[..., Awaitable[Any]] | None,
        getattr(domain, "derive_call_provenance", None),
    )
    if not callable(derive):
        raise ValueError("capability derives provenance its domain cannot supply")
    provenance = await derive(run, call, capability, arguments, session=session)
    if not isinstance(provenance, CallProvenance):
        raise TypeError("domain call provenance must be CallProvenance")
    return provenance


async def _domain_side_effect_plan(
    domain: CapabilityDomain,
    run: RunInput,
    call: ToolCall,
    capability: Capability,
    execution: ToolExecution,
    fingerprint: FrozenJsonObject,
    *,
    session: RunSession | None,
) -> SideEffectPlan:
    session_plan = getattr(domain, "session_side_effect_plan", None)
    if session is not None and callable(session_plan):
        plan = cast(
            "Callable[..., Awaitable[SideEffectPlan]]",
            session_plan,
        )
        return await plan(
            run,
            call,
            capability,
            execution,
            fingerprint,
            session=session,
        )
    return await domain.side_effect_plan(
        run,
        call,
        capability,
        execution,
        fingerprint,
    )


async def _domain_finalize_output(
    domain: CapabilityDomain,
    run: RunInput,
    call: ToolCall,
    capability: Capability,
    arguments: FrozenJsonObject,
    output: ToolOutput,
    *,
    request_sensitivity: ModelSensitivity,
    session: RunSession | None,
) -> ToolOutput:
    finalize_session_output = getattr(domain, "finalize_session_output", None)
    if session is not None and callable(finalize_session_output):
        finalize = cast(
            Callable[..., Awaitable[ToolOutput]],
            finalize_session_output,
        )
        return await finalize(
            run,
            call,
            capability,
            arguments,
            output,
            request_sensitivity=request_sensitivity,
            session=session,
        )
    return await domain.finalize_output(
        run,
        call,
        capability,
        arguments,
        output,
        request_sensitivity=request_sensitivity,
    )
