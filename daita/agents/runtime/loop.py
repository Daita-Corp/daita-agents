"""Agent run controller: the owner of the autonomous model/tool loop."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ...core.exceptions import AgentError
from .contextvars import active_run_state
from .evidence import add_evidence
from .exit import RunExitPolicy
from .guardrails import ToolCallGuardrails, has_terminal_tool_result
from .llm_turn import nonstream_llm_turn, stream_llm_turn
from .retry import mark_whole_run_retry_suppressed, run_model_turn_with_retry
from .state import FinalAnswerReadiness, RunPhase, RunState
from .tools import append_tool_messages, execute_and_track_tool

if TYPE_CHECKING:
    from ...core.tools import AgentTool


class AgentRunController:
    """Internal controller for one Agent.run() autonomous loop."""

    def __init__(self, agent):
        self.agent = agent
        self.guardrails = ToolCallGuardrails()

    async def run(
        self,
        prompt: str,
        tools: Optional[List[Union[str, "AgentTool"]]],
        max_iterations: int,
        on_event: Optional[Any],
        initial_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        from ...core.streaming import EventType

        agent = self.agent
        if agent.llm is None:
            provider_name = agent._llm_provider_name or "openai"
            raise AgentError(
                f"Cannot execute: No API key found for '{provider_name}'. "
                f"Set {provider_name.upper()}_API_KEY environment variable "
                f"or pass api_key parameter to Agent."
            )

        run_state = RunState(agent_id=agent.agent_id)
        db_state = getattr(agent, "_daita_db_run_state", None)
        if db_state is not None:
            run_state.domains["db"] = db_state
        run_state.final_answer_readiness_hooks.extend(
            getattr(agent, "_daita_final_answer_readiness_hooks", []) or []
        )

        token = active_run_state.set(run_state)
        try:
            resolved_tools = await agent._prepare_tools_with_focus(tools)
            active_tools = resolved_tools
            final_synthesis_without_tools = bool(
                kwargs.pop("final_synthesis_without_tools", False)
            )
            terminal_tools = set(kwargs.pop("terminal_tools", []) or [])
            exit_policy = RunExitPolicy(
                allow_partial=bool(kwargs.pop("partial_exit", False))
            )

            agent._tool_call_history = []
            conversation = await agent._build_initial_conversation(
                prompt, initial_messages
            )
            tools_called = []
            terminal_evidence_seen = False
            latest_terminal_result = None

            for iteration in range(max_iterations):
                run_state.iteration_count = iteration + 1
                agent._emit_event(
                    on_event,
                    EventType.ITERATION,
                    iteration=iteration + 1,
                    max_iterations=max_iterations,
                )

                run_state.set_phase(RunPhase.MODEL_TURN)
                run_state.model_turn_count += 1
                if on_event:
                    llm_call = lambda: stream_llm_turn(
                        agent, conversation, active_tools, on_event, **kwargs
                    )
                else:
                    llm_call = lambda: nonstream_llm_turn(
                        agent, conversation, active_tools, **kwargs
                    )
                if agent.config.retry_enabled:
                    llm_result = await run_model_turn_with_retry(
                        llm_call,
                        policy=agent.config.retry_policy,
                        run_state=run_state,
                    )
                else:
                    llm_result = await llm_call()
                run_state.warnings.extend(llm_result.warnings)

                if llm_result.tool_calls:
                    if terminal_evidence_seen and not active_tools:
                        if (
                            "tool_calls_after_terminal_evidence_ignored"
                            not in run_state.warnings
                        ):
                            run_state.warnings.append(
                                "tool_calls_after_terminal_evidence_ignored"
                            )
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "The necessary tool evidence is already present. "
                                    "Do not call tools. Provide the final answer now."
                                ),
                            }
                        )
                        continue
                    run_state.set_phase(RunPhase.TOOL_EXECUTION)
                    results = []
                    for tool_call in llm_result.tool_calls:
                        result = await execute_and_track_tool(
                            agent, tool_call, active_tools, on_event
                        )
                        run_state.record_tool_call(result)
                        tools_called.append(result)
                        results.append(result)
                        self._record_tool_evidence(run_state, tool_call, result)

                        guardrail_decision = self.guardrails.observe_tool_result(
                            run_state, tool_call, result
                        )
                        if guardrail_decision.hard_stop_message:
                            run_state.set_phase(RunPhase.ERROR)
                            run_state.exit_reason = "repeated_tool_error"
                            raise AgentError(guardrail_decision.hard_stop_message)
                        if guardrail_decision.guidance_result is not None:
                            result["result"] = guardrail_decision.guidance_result
                            run_state.warnings.append(
                                guardrail_decision.guidance_result["guardrail"]
                            )

                    append_tool_messages(
                        agent, conversation, llm_result.tool_calls, results
                    )
                    terminal_result = has_terminal_tool_result(results, terminal_tools)
                    if final_synthesis_without_tools and terminal_result:
                        terminal_evidence_seen = True
                        latest_terminal_result = next(
                            (
                                result
                                for result in reversed(results)
                                if result.get("tool") in terminal_tools
                            ),
                            None,
                        )
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "Use the latest tool result as answer evidence. "
                                    "Do not call more tools. Provide the final answer now."
                                ),
                            }
                        )
                        active_tools = []
                    elif suggested_next_tool := self._suggested_next_tool(results):
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    f"The previous tool result suggested calling "
                                    f"{suggested_next_tool}. Call that tool next "
                                    "using the provided suggested_next_arguments."
                                ),
                            }
                        )
                        active_tools = [
                            tool
                            for tool in resolved_tools
                            if getattr(tool, "name", None) == suggested_next_tool
                        ]
                    else:
                        active_tools = resolved_tools
                    continue

                run_state.set_phase(RunPhase.SYNTHESIS)
                readiness = self._evaluate_final_answer_readiness(
                    run_state, resolved_tools, llm_result.text
                )
                if not readiness.allow_final:
                    conversation.append(
                        {"role": "user", "content": readiness.guidance or ""}
                    )
                    active_tools = [] if terminal_evidence_seen else resolved_tools
                    if iteration + 1 < max_iterations:
                        continue
                    break

                result = agent._build_final_result(
                    llm_result.text, tools_called, iteration + 1, on_event
                )
                run_state.set_phase(RunPhase.COMPLETE)
                run_state.exit_reason = "final_answer"
                result["diagnostics"] = run_state.diagnostic_summary()
                return result

            token_stats = agent.llm.get_token_stats() if agent.llm is not None else {}
            run_state.set_phase(RunPhase.ERROR)
            run_state.exit_reason = "max_iterations"
            if final_synthesis_without_tools and terminal_evidence_seen:
                run_state.set_phase(RunPhase.COMPLETE)
                run_state.exit_reason = "terminal_evidence_synthesis"
                result = agent._build_final_result(
                    self._terminal_result_text(latest_terminal_result),
                    tools_called,
                    max_iterations,
                    on_event,
                )
                result["diagnostics"] = run_state.diagnostic_summary()
                return result
            exit_decision = exit_policy.decide_max_iterations(run_state, prompt)
            if exit_decision.should_return:
                run_state.set_phase(RunPhase.COMPLETE)
                run_state.exit_reason = exit_decision.reason
                run_state.partial_result = exit_decision.result_text
                result = agent._build_final_result(
                    exit_decision.result_text or "",
                    tools_called,
                    max_iterations,
                    on_event,
                )
                result["partial"] = True
                self._evaluate_final_answer_readiness(run_state, resolved_tools, "")
                result["diagnostics"] = run_state.diagnostic_summary()
                return result
            self._evaluate_final_answer_readiness(run_state, resolved_tools, "")
            raise AgentError(
                f"Max iterations ({max_iterations}) reached without final answer",
                agent_id=agent.agent_id,
                task=prompt,
                context={
                    "max_iterations": max_iterations,
                    "iterations": max_iterations,
                    "tool_calls": tools_called,
                    "tokens": token_stats,
                    "diagnostics": run_state.diagnostic_summary(),
                },
            )
        except Exception as error:
            mark_whole_run_retry_suppressed(error, run_state)
            raise
        finally:
            active_run_state.reset(token)

    def _evaluate_final_answer_readiness(
        self, run_state: RunState, available_tools: List["AgentTool"], final_text: str
    ) -> FinalAnswerReadiness:
        combined = FinalAnswerReadiness()
        for hook in run_state.final_answer_readiness_hooks:
            try:
                decision = hook(run_state, final_text, available_tools)
            except Exception:
                continue
            if decision is None:
                continue
            run_state.diagnostics.update(decision.diagnostics)
            if decision.warning and decision.warning not in run_state.warnings:
                run_state.warnings.append(decision.warning)
            if not decision.allow_final:
                combined.allow_final = False
                combined.guidance = combined.guidance or decision.guidance
                combined.warning = combined.warning or decision.warning
        return combined

    def _record_tool_evidence(
        self, run_state: RunState, tool_call: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        tool_name = str(tool_call.get("name") or "")
        if tool_name.startswith(("db_", "catalog_")):
            return
        raw = result.get("result")
        if isinstance(raw, dict) and raw.get("error"):
            return
        add_evidence(
            run_state,
            domain="generic",
            kind="tool_result",
            source_tool=tool_name,
            payload={
                "arguments": tool_call.get("arguments", {}),
                "result": raw,
                "duration_ms": result.get("duration_ms"),
            },
        )

    def _terminal_result_text(self, result: Optional[Dict[str, Any]]) -> str:
        raw = (result or {}).get("result")
        if isinstance(raw, dict):
            rows = raw.get("rows")
            if isinstance(rows, list):
                if not rows:
                    return "No matching rows were returned."
                return "Query result: " + json.dumps(rows[:5], default=str)
            row_count = raw.get("row_count")
            if row_count == 0:
                return "No matching rows were returned."
            return "Query result: " + json.dumps(raw, default=str)
        if raw is None:
            return "The terminal tool completed, but returned no result payload."
        return str(raw)

    def _suggested_next_tool(self, results: List[Dict[str, Any]]) -> Optional[str]:
        for result in reversed(results):
            raw = result.get("result")
            if not isinstance(raw, dict):
                continue
            tool_name = raw.get("suggested_next_tool")
            if tool_name:
                return str(tool_name)
        return None
