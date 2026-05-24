"""Generic guardrails for the Agent model/tool loop."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


def make_error_fingerprint(tool_call: Dict[str, Any]) -> str:
    """Stable hash of (tool_name, arguments) used for loop detection."""
    args_hash = hashlib.md5(
        json.dumps(tool_call.get("arguments", {}), sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return f"{tool_call['name']}:{args_hash}"


def make_result_fingerprint(tool_call: Dict[str, Any], result: Dict[str, Any]) -> str:
    """Stable hash of (tool_name, arguments, result) for no-progress detection."""
    tool_name = str(tool_call.get("name") or "")
    payload = {
        "tool": tool_name,
        "arguments": _result_fingerprint_arguments(tool_name, tool_call),
        "result": _result_fingerprint_result(tool_name, result.get("result")),
    }
    result_hash = hashlib.md5(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    return f"{tool_call.get('name')}:{result_hash}"


def _result_fingerprint_arguments(
    tool_name: str, tool_call: Dict[str, Any]
) -> Dict[str, Any]:
    arguments = tool_call.get("arguments", {})
    if tool_name == "db_plan_query" and isinstance(arguments, dict):
        return {
            key: value
            for key, value in arguments.items()
            if key not in {"include_diagnostics", "debug"}
        }
    return arguments


def _result_fingerprint_result(tool_name: str, raw: Any) -> Any:
    if tool_name == "db_plan_query" and isinstance(raw, dict):
        return {
            "ok": raw.get("ok"),
            "route": raw.get("route"),
            "compiled_sql": raw.get("compiled_sql"),
            "validation_ok": bool((raw.get("validation") or {}).get("ok")),
            "suggested_next_tool": raw.get("suggested_next_tool"),
            "resolved_tables": raw.get("resolved_tables"),
            "unknown_tables": raw.get("unknown_tables"),
            "ambiguous_tables": raw.get("ambiguous_tables"),
        }
    return raw


def has_terminal_tool_result(
    results: list[Dict[str, Any]], terminal_tools: set[str]
) -> bool:
    """Return True when a configured terminal tool completed without an error."""
    if not terminal_tools:
        return False
    for result in results:
        if result.get("tool") not in terminal_tools:
            continue
        raw = result.get("result")
        if isinstance(raw, dict) and raw.get("error"):
            continue
        if isinstance(raw, dict) and (
            raw.get("repair_required")
            or raw.get("preflight_failed")
            or raw.get("blocked_repeat")
            or raw.get("guardrail")
        ):
            continue
        return True
    return False


def tool_loop_error_message(raw: Any) -> Optional[str]:
    """Return an error-like message for tool results that should count as loops."""
    if not isinstance(raw, dict):
        return None
    if raw.get("error"):
        return str(raw["error"])
    if raw.get("blocked_repeat"):
        return str(raw.get("message") or raw.get("status") or "blocked repeat")
    if raw.get("repair_required") or raw.get("preflight_failed"):
        return str(raw.get("message") or raw.get("guidance") or "repair required")
    return None


class ToolCallGuardrails:
    """Small generic guardrail policy for repeated no-progress tool calls."""

    def __init__(
        self,
        max_consecutive_identical_errors: int = 3,
        guidance_after_repeats: int = 2,
    ):
        self.max_consecutive_identical_errors = max_consecutive_identical_errors
        self.guidance_after_repeats = guidance_after_repeats

    def observe_tool_result(
        self, run_state, tool_call: Dict[str, Any], result: Dict[str, Any]
    ) -> "GuardrailDecision":
        """Record a tool result and return guidance or a hard stop when needed."""
        raw = result.get("result", {})
        fingerprint = _domain_retry_fingerprint(
            run_state, tool_call, raw, kind="error"
        ) or make_error_fingerprint(tool_call)
        loop_error = tool_loop_error_message(raw)
        if loop_error:
            count = run_state.failed_tool_fingerprints.get(fingerprint, 0) + 1
            run_state.failed_tool_fingerprints[fingerprint] = count
            if count >= self.max_consecutive_identical_errors:
                return GuardrailDecision(
                    hard_stop_message=(
                        f"Loop detected: '{tool_call['name']}' returned a repair/error result "
                        f"{count} consecutive times with identical arguments. Last result: {loop_error}"
                    )
                )
            if count >= self.guidance_after_repeats:
                return GuardrailDecision(
                    guidance_result=_guardrail_payload(
                        guardrail="repeated_tool_error",
                        message=(
                            "This tool returned the same repair/error result for the same "
                            "arguments. Change the arguments, inspect different context, "
                            "or synthesize from available information."
                        ),
                        suggested_next_step="change_arguments_or_synthesize",
                        tool_call=tool_call,
                        repeat_count=count,
                        last_result=raw,
                    )
                )
            return GuardrailDecision()
        run_state.failed_tool_fingerprints.pop(fingerprint, None)

        result_fingerprint = _domain_retry_fingerprint(
            run_state, tool_call, raw, kind="result"
        ) or make_result_fingerprint(tool_call, result)
        count = run_state.repeated_result_fingerprints.get(result_fingerprint, 0) + 1
        run_state.repeated_result_fingerprints[result_fingerprint] = count
        if count >= self.guidance_after_repeats:
            return GuardrailDecision(
                guidance_result=_guardrail_payload(
                    guardrail="repeated_no_progress",
                    message=(
                        "This call returned the same result for the same arguments. "
                        "Use different arguments, a different tool, or move to synthesis."
                    ),
                    suggested_next_step="synthesize_or_plan_differently",
                    tool_call=tool_call,
                    repeat_count=count,
                    last_result=raw,
                )
            )

        run_state.record_progress("tool_result", tool=tool_call["name"])
        return GuardrailDecision()


@dataclass
class GuardrailDecision:
    """Internal guardrail decision for one tool result."""

    guidance_result: Optional[Dict[str, Any]] = None
    hard_stop_message: Optional[str] = None


def _guardrail_payload(
    *,
    guardrail: str,
    message: str,
    suggested_next_step: str,
    tool_call: Dict[str, Any],
    repeat_count: int,
    last_result: Any,
) -> Dict[str, Any]:
    return {
        "guardrail": guardrail,
        "message": message,
        "suggested_next_step": suggested_next_step,
        "tool": tool_call.get("name"),
        "arguments": tool_call.get("arguments", {}),
        "repeat_count": repeat_count,
        "last_result": last_result,
    }


def _domain_retry_fingerprint(
    run_state: Any, tool_call: Dict[str, Any], raw_result: Any, *, kind: str
) -> Optional[str]:
    for domain_state in getattr(run_state, "domains", {}).values():
        callback = getattr(domain_state, "tool_retry_fingerprint", None)
        if not callable(callback):
            continue
        fingerprint = callback(tool_call, raw_result, kind=kind)
        if fingerprint:
            return str(fingerprint)
    return None
