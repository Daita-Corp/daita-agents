"""Request-scoped coordinator for ``Agent.from_db()`` runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ...runtime.state import FinalAnswerReadiness
from ..config.policies import TERMINAL_DB_TOOLS
from ..config.tool_profiles import DbToolProfile, select_db_tool_profile
from ..utils import string_list
from .state import DbRunState, set_db_run_state

GENERIC_CATALOG_TOOLS = {
    "search_catalog",
    "inspect_asset",
    "find_relationship_paths",
}
RELATIONAL_CATALOG_ALIASES = {
    "search_catalog": "catalog_search_schema",
    "inspect_asset": "catalog_inspect_table",
    "find_relationship_paths": "catalog_find_join_paths",
}
FROM_DB_SCHEMA_TOOLS = (
    "catalog_search_schema",
    "catalog_inspect_table",
    "catalog_find_join_paths",
)
DB_REPAIR_TOOLS = {
    "db_compile_and_query",
    "db_plan_query",
    "db_query",
    "db_validate_sql",
}


@dataclass(frozen=True)
class DbRunContract:
    """The DB-specific execution contract for one user request."""

    intent: str
    tools: tuple[str, ...]
    required_phases: tuple[str, ...]
    terminal_tools: tuple[str, ...]
    evidence_mode: str
    allow_catalog_final: bool
    require_executed_query: bool
    max_model_turns: int
    max_tool_calls: int
    max_repair_attempts: int
    final_synthesis_without_tools: bool = True


@dataclass(frozen=True)
class DbPreparedRun:
    """Prepared inputs for the generic ``AgentRunController``."""

    prompt: str
    context: str
    memory_snippets: tuple[str, ...]
    contract: DbRunContract
    state: DbRunState
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DbNextToolDecision:
    """Decision for a tool-returned ``suggested_next_tool``."""

    tool_name: Optional[str] = None
    allowed: bool = False
    guidance: Optional[str] = None
    warning: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DbToolCallDecision:
    """Decision before a model-requested tool call is executed."""

    allow: bool = True
    result: Optional[Dict[str, Any]] = None
    warning: Optional[str] = None
    guidance: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class DbRunOrchestrator:
    """Plain-Python owner of DB run policy for one ``from_db`` request."""

    def __init__(self, agent: Any, prompt: str, *, state: Optional[DbRunState] = None):
        self.agent = agent
        self.prompt = prompt
        self.state = state or DbRunState()
        self.contract: Optional[DbRunContract] = None
        self.memory_snippets: List[str] = []
        self._diagnostics: Dict[str, Any] = {}

    async def prepare(self) -> DbPreparedRun:
        """Create state, classify intent, build the contract, and render context."""
        from ..memory import recall_db_memory_context
        from .run_context import build_db_run_context
        from .tracing import db_trace_span

        self._install_state()
        async with db_trace_span(
            self.agent,
            "from_db.prepare_orchestrator",
            prompt=self.prompt[:200],
        ):
            async with db_trace_span(self.agent, "from_db.memory_recall") as (
                trace_manager,
                span_id,
            ):
                self.memory_snippets = await recall_db_memory_context(
                    self.agent, self.prompt
                )
                decision = (
                    getattr(self.agent, "_db_last_memory_recall_decision", None) or {}
                )
                trace_manager.record_output(
                    span_id,
                    {
                        "skipped": not bool(decision.get("recall", True)),
                        "reason": decision.get("reason"),
                        "matched_terms": decision.get("matched_terms", []),
                        "snippet_count": len(self.memory_snippets),
                    },
                )

            async with db_trace_span(self.agent, "from_db.build_contract") as (
                trace_manager,
                span_id,
            ):
                profile = select_db_tool_profile(self.agent, self.prompt)
                self.contract = self._contract_from_profile(profile)
                self.state.intent_kind = self.contract.intent
                self.state.run_contract = self.contract
                trace_manager.record_output(span_id, self.contract_summary())

            async with db_trace_span(self.agent, "from_db.build_runtime_context") as (
                trace_manager,
                span_id,
            ):
                context = build_db_run_context(
                    self.agent,
                    prompt=self.prompt,
                    memory_snippets=self.memory_snippets,
                    contract=self.contract,
                )
                trace_manager.record_output(
                    span_id,
                    {
                        "runtime_context_chars": len(context),
                        "memory_snippet_count": len(self.memory_snippets),
                        "intent_kind": self.contract.intent,
                    },
                )

        return DbPreparedRun(
            prompt=self.prompt,
            context=context,
            memory_snippets=tuple(self.memory_snippets),
            contract=self.contract,
            state=self.state,
            diagnostics=self.diagnostics(),
        )

    def attach_run_state(self, run_state: Any) -> None:
        """Attach this orchestrator to the generic run state."""
        run_state.domains["db"] = self.state
        run_state.domains["db_orchestrator"] = self
        if self.evaluate_final_readiness not in run_state.final_answer_readiness_hooks:
            run_state.final_answer_readiness_hooks.append(self.evaluate_final_readiness)

    def normalize_tool_call(self, run_state: Any, tool_call: Dict[str, Any]) -> None:
        """Normalize DB/catalog tool calls before execution."""
        tool_name = str(tool_call.get("name") or "")
        if tool_name in RELATIONAL_CATALOG_ALIASES:
            alias = RELATIONAL_CATALOG_ALIASES[tool_name]
            if alias in self._require_contract().tools:
                tool_call["name"] = alias
                tool_name = alias

        if (
            tool_name not in FROM_DB_SCHEMA_TOOLS
            and tool_name not in GENERIC_CATALOG_TOOLS
        ):
            return
        active_store_id = getattr(self.agent, "_db_catalog_store_id", None)
        active_catalog = getattr(self.agent, "_db_catalog", None)
        if not active_store_id or active_catalog is None:
            return
        arguments = tool_call.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
            tool_call["arguments"] = arguments
        provided = str(arguments.get("store_id") or "").strip()
        if provided == active_store_id:
            return
        if provided and _catalog_has_schema(active_catalog, provided):
            return
        arguments["store_id"] = active_store_id

    def before_tool_call(
        self, run_state: Any, tool_call: Dict[str, Any]
    ) -> DbToolCallDecision:
        """Enforce the active contract before executing a DB tool call."""
        contract = self._require_contract()
        tool_name = str(tool_call.get("name") or "")
        if tool_name not in contract.tools:
            return self._blocked_tool_call(
                tool_name,
                warning=f"db_tool_not_allowed:{tool_name}",
                message=(
                    f"{tool_name} is not allowed by this DB run contract. "
                    "Use only the selected tools for this request."
                ),
                diagnostics={"blocked_db_tool": tool_name},
            )

        if (
            int(getattr(run_state, "tool_call_count", 0) or 0)
            >= contract.max_tool_calls
        ):
            return self._blocked_tool_call(
                tool_name,
                warning="db_tool_budget_exhausted",
                message=(
                    "The DB tool-call budget for this request has been exhausted. "
                    "Synthesize from the available evidence instead of calling "
                    "more tools."
                ),
                diagnostics={
                    "db_tool_budget_exhausted": True,
                    "max_tool_calls": contract.max_tool_calls,
                },
            )

        if self._repair_budget_exhausted() and tool_name in DB_REPAIR_TOOLS:
            return self._blocked_tool_call(
                tool_name,
                warning="db_repair_budget_exhausted",
                message=(
                    "The SQL repair budget for this request has been exhausted. "
                    "Do not continue retrying SQL repair tools; synthesize from "
                    "available evidence or explain what blocked execution."
                ),
                diagnostics={
                    "db_repair_budget_exhausted": True,
                    "max_repair_attempts": contract.max_repair_attempts,
                },
            )

        return DbToolCallDecision()

    def observe_tool_result(
        self, run_state: Any, tool_call: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """Record DB facts and attach allowed handoff hints to tool results."""
        tool_name = str(tool_call.get("name") or "")
        raw = result.get("result")
        if tool_name in FROM_DB_SCHEMA_TOOLS:
            self.state.record_catalog_tool_result(
                tool_name, tool_call.get("arguments", {}) or {}, raw
            )
            handoff = self._catalog_handoff(tool_name, raw)
            if (
                isinstance(handoff, dict)
                and isinstance(raw, dict)
                and self._suggested_tool_allowed(handoff.get("suggested_next_tool"))
            ):
                raw.update(
                    {
                        key: value
                        for key, value in handoff.items()
                        if key not in raw or raw[key] in (None, "", [], {})
                    }
                )

    def terminal_result(
        self, results: Sequence[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Return the latest result that satisfies this contract's terminal policy."""
        contract = self._require_contract()
        for result in reversed(results):
            if result.get("tool") not in contract.terminal_tools:
                continue
            raw = result.get("result")
            if not self._result_is_successful_terminal(raw):
                continue
            if self._evidence_satisfies_contract_after_result(result):
                return result
        return None

    def synthesis_guidance(self, results: Sequence[Dict[str, Any]]) -> Optional[str]:
        """Return forced-synthesis guidance after budget or policy blocks."""
        for result in reversed(results):
            raw = result.get("result")
            if not isinstance(raw, dict):
                continue
            guardrail = str(raw.get("guardrail") or "")
            if guardrail.startswith(("db_tool_", "db_repair_", "db_tool_not_allowed")):
                return str(
                    raw.get("message")
                    or "Do not call more tools. Provide the final answer now."
                )
        return None

    def resolve_suggested_next_tool(
        self, results: Sequence[Dict[str, Any]], resolved_tools: Sequence[Any]
    ) -> Optional[DbNextToolDecision]:
        """Allow, block, or ignore tool-returned next-tool suggestions."""
        suggested = _suggested_next_tool(results)
        if not suggested:
            return None
        if self._suggested_tool_allowed(suggested):
            if self._repair_budget_exhausted() and suggested in DB_REPAIR_TOOLS:
                return DbNextToolDecision(
                    tool_name=suggested,
                    allowed=False,
                    guidance=(
                        "The previous tool result suggested another SQL repair step, "
                        "but the repair budget is exhausted. Synthesize from the "
                        "available evidence or explain what blocked execution."
                    ),
                    warning="db_repair_budget_exhausted",
                    diagnostics={
                        "db_repair_budget_exhausted": True,
                        "suppressed_suggested_next_tool": suggested,
                    },
                )
            return DbNextToolDecision(tool_name=suggested, allowed=True)

        available_names = _tool_names(resolved_tools)
        diagnostics = {"unavailable_suggested_next_tool": suggested}
        guidance = (
            f"The previous tool result suggested {suggested}, but that tool is not "
            "allowed by this DB run contract. Do not call unavailable tools."
        )
        if self._evidence_satisfies_contract():
            guidance += " Synthesize from the evidence already collected."
        elif available_names:
            guidance += " Choose one of the allowed tools: " + ", ".join(
                name
                for name in self._require_contract().tools
                if name in available_names
            )
        guidance += "."
        return DbNextToolDecision(
            tool_name=suggested,
            allowed=False,
            guidance=guidance,
            warning=f"suggested_next_tool_unavailable:{suggested}",
            diagnostics=diagnostics,
        )

    def evaluate_final_readiness(
        self, run_state: Any, final_text: str, available_tools: Iterable[Any]
    ) -> FinalAnswerReadiness | None:
        from .completeness import evaluate_db_final_answer_readiness

        return evaluate_db_final_answer_readiness(
            run_state,
            final_text,
            available_tools,
            contract=self._require_contract(),
        )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "db_run_contract": self.contract_summary(),
            "db_run_state": self.state.summary(),
            **self._diagnostics,
        }

    def contract_summary(self) -> Dict[str, Any]:
        contract = self.contract
        if contract is None:
            return {}
        return {
            "intent": contract.intent,
            "tools": list(contract.tools),
            "required_phases": list(contract.required_phases),
            "terminal_tools": list(contract.terminal_tools),
            "evidence_mode": contract.evidence_mode,
            "allow_catalog_final": contract.allow_catalog_final,
            "require_executed_query": contract.require_executed_query,
            "max_model_turns": contract.max_model_turns,
            "max_tool_calls": contract.max_tool_calls,
            "max_repair_attempts": contract.max_repair_attempts,
        }

    @staticmethod
    def start_state(agent: Any, state: Optional[DbRunState] = None) -> DbRunState:
        run_state = state or DbRunState()
        plugin = getattr(agent, "_db_plugin", None)
        set_db_run_state(agent, run_state, register_final_answer_readiness=False)
        if plugin is not None:
            set_db_run_state(plugin, run_state, register_final_answer_readiness=False)
            setattr(plugin, "_daita_sql_preflight_failures", {})
        return run_state

    def _install_state(self) -> None:
        self.state = self.start_state(self.agent, self.state)

    def _contract_from_profile(self, profile: DbToolProfile) -> DbRunContract:
        available = set(getattr(self.agent.tool_registry, "tool_names", []))
        tools = self._contract_tools(profile, available)
        evidence_mode = self._evidence_mode(profile.intent)
        allow_catalog_final = evidence_mode == "catalog"
        require_executed_query = evidence_mode == "query"
        return DbRunContract(
            intent=profile.intent,
            tools=tuple(tools),
            required_phases=tuple(profile.required_phases),
            terminal_tools=self._terminal_tools(profile.intent, tools),
            evidence_mode=evidence_mode,
            allow_catalog_final=allow_catalog_final,
            require_executed_query=require_executed_query,
            max_model_turns=self._max_model_turns(profile.intent),
            max_tool_calls=self._max_tool_calls(profile.intent),
            max_repair_attempts=self._max_repair_attempts(profile.intent),
        )

    def _contract_tools(self, profile: DbToolProfile, available: set[str]) -> List[str]:
        prompt_text = str(self.prompt or "").lower()
        selected: List[str] = []
        for name in profile.tools:
            if name in GENERIC_CATALOG_TOOLS and name not in prompt_text:
                alias = RELATIONAL_CATALOG_ALIASES[name]
                if alias in available:
                    selected.append(alias)
                self._diagnostics.setdefault("hidden_generic_catalog_tools", []).append(
                    name
                )
                continue
            selected.append(name)
        return _unique([name for name in selected if name in available])

    def _terminal_tools(self, intent: str, tools: Sequence[str]) -> tuple[str, ...]:
        tool_set = set(tools)
        if intent in {"schema_only", "schema_question", "schema_explain"}:
            return tuple(name for name in FROM_DB_SCHEMA_TOOLS if name in tool_set)
        if intent == "memory_only":
            return tuple(name for name in ("db_remember",) if name in tool_set)
        return tuple(name for name in TERMINAL_DB_TOOLS if name in tool_set)

    def _suggested_tool_allowed(self, tool_name: Any) -> bool:
        if not tool_name:
            return False
        return str(tool_name) in set(self._require_contract().tools)

    def _evidence_mode(self, intent: str) -> str:
        if intent in {"schema_only", "schema_question", "schema_explain"}:
            return "catalog"
        if intent == "memory_only":
            return "memory"
        if intent in {"conversational"}:
            return "none"
        return "query"

    def _max_model_turns(self, intent: str) -> int:
        if intent in {"schema_only", "schema_question", "schema_explain"}:
            return 2
        if intent in {"data_query_simple", "manual_sql", "memory_only"}:
            return 3
        if intent == "data_query_catalog_assisted":
            return 6
        return 4

    def _max_tool_calls(self, intent: str) -> int:
        if intent in {"schema_only", "schema_question", "schema_explain"}:
            return 5
        if intent in {"data_query_simple", "manual_sql", "memory_only"}:
            return 3
        if intent == "data_query_catalog_assisted":
            return 8
        return 5

    def _max_repair_attempts(self, intent: str) -> int:
        if intent in {"schema_only", "schema_question", "schema_explain"}:
            return 0
        if intent == "data_query_catalog_assisted":
            return 2
        return 1

    def _result_is_successful_terminal(self, raw: Any) -> bool:
        if isinstance(raw, dict) and raw.get("error"):
            return False
        if isinstance(raw, dict) and (
            raw.get("repair_required")
            or raw.get("preflight_failed")
            or raw.get("blocked_repeat")
            or raw.get("guardrail")
        ):
            return False
        return True

    def _blocked_tool_call(
        self,
        tool_name: str,
        *,
        warning: str,
        message: str,
        diagnostics: Dict[str, Any],
    ) -> DbToolCallDecision:
        payload = {
            "guardrail": warning,
            "message": message,
            "tool": tool_name,
            "suggested_next_step": "synthesize_from_available_evidence",
        }
        return DbToolCallDecision(
            allow=False,
            result=payload,
            warning=warning,
            guidance=message,
            diagnostics=diagnostics,
        )

    def _catalog_handoff(self, tool_name: str, result: Any) -> Optional[Dict[str, Any]]:
        contract = self._require_contract()
        if tool_name != "catalog_find_join_paths":
            return None
        if contract.intent not in {"data_query_catalog_assisted", "data_query"}:
            return None
        if "db_plan_query" not in contract.tools:
            return None
        if not isinstance(result, dict):
            return None
        if not result.get("reachable") or not result.get("paths"):
            return None
        if self.state.planned_queries:
            return None
        from_tables = string_list(
            result.get("from_tables") or result.get("from_assets")
        )
        to_tables = string_list(result.get("to_tables") or result.get("to_assets"))
        if not from_tables or not to_tables:
            return None
        return {
            "suggested_next_tool": "db_plan_query",
            "suggested_next_arguments": {
                "candidate_tables": _unique(from_tables + to_tables),
                "required_joins": [
                    {
                        "from_tables": from_tables,
                        "to_tables": to_tables,
                    }
                ],
            },
        }

    def _repair_budget_exhausted(self) -> bool:
        contract = self._require_contract()
        return (
            sum(self.state.failed_sql_fingerprints.values())
            > contract.max_repair_attempts
        )

    def _evidence_satisfies_contract_after_result(self, result: Dict[str, Any]) -> bool:
        if not self._evidence_satisfies_contract():
            if self._require_contract().evidence_mode == "memory":
                return True
            return False
        raw = result.get("result")
        contract = self._require_contract()
        if contract.evidence_mode == "catalog":
            return isinstance(raw, dict) and bool(
                raw.get("tables")
                or raw.get("columns")
                or raw.get("matched_fields")
                or raw.get("paths")
                or raw.get("table_name")
                or raw.get("asset")
            )
        return True

    def _evidence_satisfies_contract(self) -> bool:
        contract = self._require_contract()
        if contract.evidence_mode == "catalog":
            evidence = self.state.catalog_evidence or {}
            return bool(
                evidence.get("tables")
                or evidence.get("columns")
                or evidence.get("joins")
                or self.state.inspected_tables
            )
        if contract.evidence_mode == "query":
            return bool(self.state.executed_queries)
        if contract.evidence_mode == "memory":
            return False
        return True

    def _require_contract(self) -> DbRunContract:
        if self.contract is None:
            raise RuntimeError("DbRunOrchestrator.prepare() has not been called")
        return self.contract


def _catalog_has_schema(catalog: Any, store_id: str) -> bool:
    getter = getattr(catalog, "get_schema", None)
    if not callable(getter):
        return False
    try:
        return getter(store_id) is not None
    except Exception:
        return False


def _suggested_next_tool(results: Sequence[Dict[str, Any]]) -> Optional[str]:
    for result in reversed(results):
        raw = result.get("result")
        if not isinstance(raw, dict):
            continue
        tool_name = raw.get("suggested_next_tool")
        if tool_name:
            return str(tool_name)
    return None


def _tool_names(tools: Iterable[Any]) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        if isinstance(tool, str):
            names.add(tool)
        else:
            name = getattr(tool, "name", None)
            if name:
                names.add(str(name))
    return names


def _unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out
