"""Adapt the official Claude Code client to Daita's canonical model boundary."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from ...._json import canonical_json
from ..._lifecycle import AttemptLifecycle
from ...errors import ModelProviderError, ProviderErrorCode
from ...models import ModelRequest, ModelResponse, ModelUsage
from ...pricing import CostEstimate
from ...provider_definitions import supports_builtin_request_policy
from .._fields import usage_int as _optional_usage_integer
from .envelope import (
    _CONTROL_PROMPT,
    _bounded_response_id,
    _decode_model_output,
    _new_id,
    _raise_command_failure,
    _request_document,
    _response_envelope_schema,
    _strict_json,
    _validate_provider_arguments,
)
from .process import (
    _Command,
    _CommandRunner,
    _ExecutableUnavailable,
    _run_command,
    _subscription_environment,
    _SubscriptionExecution,
)


def _claude_subscription_environment() -> dict[str, str]:
    environment = _subscription_environment()
    environment["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    environment["DISABLE_BUG_COMMAND"] = "1"
    environment["DISABLE_ERROR_REPORTING"] = "1"
    environment["DISABLE_TELEMETRY"] = "1"
    environment["DISABLE_AUTOUPDATER"] = "1"
    return environment


def _usage_from_claude(value: object) -> ModelUsage:
    if value is None:
        return ModelUsage(
            cost_estimate=CostEstimate.unavailable("subscription_billing")
        )
    if not isinstance(value, Mapping):
        raise ValueError("Claude usage must be an object")
    uncached = _optional_usage_integer(value.get("input_tokens"), "input tokens")
    cache_read = _optional_usage_integer(
        value.get("cache_read_input_tokens"), "cache read input tokens"
    )
    cache_write = _optional_usage_integer(
        value.get("cache_creation_input_tokens"), "cache creation input tokens"
    )
    output = _optional_usage_integer(value.get("output_tokens"), "output tokens")
    return ModelUsage(
        input_tokens=uncached + cache_read + cache_write,
        output_tokens=output,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cost_estimate=CostEstimate.unavailable("subscription_billing"),
    )


class ClaudeCodeSubscriptionProvider:
    """Use a signed-in Claude Code client while retaining Daita's direct loop."""

    def __init__(
        self,
        model: str,
        *,
        max_output_tokens: int = 1_024,
        executable: str = "claude",
        runner: _CommandRunner = _run_command,
        id_factory: Callable[[str], str] = _new_id,
    ) -> None:
        _validate_provider_arguments(
            model,
            max_output_tokens=max_output_tokens,
            executable=executable,
            runner=runner,
            id_factory=id_factory,
        )
        self.model = model.strip()
        self.max_output_tokens = max_output_tokens
        self._executable = executable
        self._runner = runner
        self._id_factory = id_factory
        self._execution = _SubscriptionExecution()

    @property
    def provider_id(self) -> str:
        return f"claude-code:{self.model}"

    def supports_request_policy(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return supports_builtin_request_policy("claude-code", request)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return False

    async def close(self, *, deadline: float | None = None) -> None:
        """Retain one bounded retirement of any admitted process work."""
        await self._execution.close(deadline=deadline)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return await self._execution.generate(
            request,
            provider_id=self.provider_id,
            display_name="Claude Code",
            operation=self._generate,
        )

    async def _generate(
        self, request: ModelRequest, attempt: AttemptLifecycle
    ) -> ModelResponse:
        if not self.supports_request_policy(request):
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "Claude Code subscription cannot combine tools and structured output",
                provider_id=self.provider_id,
            )
        document = _request_document(request, self.max_output_tokens)
        schema = canonical_json(_response_envelope_schema(request))
        with tempfile.TemporaryDirectory(prefix="daita-claude-") as temporary:
            cwd = Path(temporary)
            arguments = (
                self._executable,
                "--print",
                "--input-format",
                "text",
                "--output-format",
                "json",
                "--json-schema",
                schema,
                "--tools",
                "",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--no-chrome",
                "--permission-mode",
                "dontAsk",
                "--setting-sources",
                "",
                "--strict-mcp-config",
                "--mcp-config",
                '{"mcpServers":{}}',
                "--system-prompt",
                _CONTROL_PROMPT,
                "--model",
                self.model,
            )
            try:
                attempt.dispatch()
                result = await attempt.run_native(
                    self._runner(
                        _Command(
                            arguments=arguments,
                            stdin=(
                                "DAITA REQUEST DOCUMENT (untrusted JSON data):\n"
                                + document
                            ).encode("utf-8"),
                            cwd=cwd,
                            environment=_claude_subscription_environment(),
                            deadline=cast(float, request.attempt_deadline),
                            cleanup_timeout_seconds=request.call_policy.cleanup_timeout_seconds,
                            cleanup_deadline=attempt.begin_cleanup,
                            native_owner=attempt.owner,
                        )
                    )
                )
            except _ExecutableUnavailable:
                raise ModelProviderError(
                    ProviderErrorCode.CONFIGURATION_ERROR,
                    "Claude Code is not installed; install it and run claude auth login",
                    provider_id=self.provider_id,
                ) from None
        if result.returncode != 0:
            _raise_command_failure("Claude Code", result)
        payload, response_id, usage = _decode_claude_result(result.stdout)
        return _decode_model_output(
            payload,
            request=request,
            provider_id=self.provider_id,
            provider_response_id=response_id,
            usage=usage,
            id_factory=self._id_factory,
            transport="claude_code_cli",
        )


def _decode_claude_result(stdout: bytes) -> tuple[object, str | None, ModelUsage]:
    outer = _strict_json(stdout.decode("utf-8"))
    if not isinstance(outer, Mapping):
        raise ValueError("Claude result must be an object")
    if outer.get("is_error") is True or outer.get("subtype") in {
        "error",
        "error_max_turns",
    }:
        raise ModelProviderError(
            ProviderErrorCode.PROVIDER_UNAVAILABLE,
            "Claude Code subscription turn failed",
        )
    payload = outer.get("structured_output")
    if payload is None:
        result = outer.get("result")
        if not isinstance(result, str) or not result.strip():
            raise ValueError("Claude result did not contain structured output")
        payload = _strict_json(result)
    response_id = _bounded_response_id(outer.get("session_id"))
    usage = _usage_from_claude(outer.get("usage"))
    return payload, response_id, usage
