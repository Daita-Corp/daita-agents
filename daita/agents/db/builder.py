"""
from_db() — build a fully configured Agent from a database source.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

from .audit import make_audited_run, make_audited_stream
from .cache import cache_key, detect_drift, load_cached_schema, save_schema_cache
from .context import attach_db_context
from .describe import attach_db_describe
from .memory import DBMemory, calibrate_db_memory
from .navigation import should_register_schema_navigation
from .policies import (
    BudgetPreset,
    SchemaPromptPolicy,
    ToolResultPolicy,
    schema_prompt_policy_for_budget,
)
from .prompt import build_prompt_result, infer_domain
from .presets import AUTO_TOOLKIT, resolve_mode_options
from .resolve import resolve_plugin
from .run_context import make_db_context_run, make_db_context_stream
from .sampling import sample_numeric_columns
from .schema import discover_schema
from .summary import build_db_summary

if TYPE_CHECKING:
    from ..agent import Agent
    from ...plugins.base_db import BaseDatabasePlugin

logger = logging.getLogger(__name__)

_GENERIC_DB_TOOLS = {
    "db_query",
    "db_count",
    "db_sample",
    "db_execute",
    "db_find",
    "db_aggregate",
}


async def from_db(
    source: Union[str, "BaseDatabasePlugin"],
    *,
    name: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    llm_provider: Optional[str] = None,
    prompt: Optional[str] = None,
    db_schema: Optional[str] = None,
    mode: str = "analyst",
    include_sample_values: Optional[bool] = None,
    redact_pii_columns: bool = True,
    lineage: Union[bool, Any, None] = None,
    memory: Union[bool, Any, None] = None,
    calibrate_memory: Optional[bool] = None,
    history: Union[bool, Any, None] = None,
    cache_ttl: Optional[int] = None,
    read_only: Optional[bool] = None,
    query_default_limit: Optional[int] = None,
    query_max_rows: Optional[int] = None,
    query_max_chars: Optional[int] = None,
    query_timeout: Optional[float] = None,
    allowed_tables: Optional[List[str]] = None,
    blocked_tables: Optional[List[str]] = None,
    blocked_columns: Optional[List[str]] = None,
    toolkit: Optional[str] = AUTO_TOOLKIT,
    quality: Optional[bool] = None,
    budget: BudgetPreset = "auto",
    schema_prompt_policy: Optional[SchemaPromptPolicy] = None,
    tool_result_policy: Optional[ToolResultPolicy] = None,
    **agent_kwargs: Any,
) -> "Agent":
    """
    Create a fully configured Agent from a database connection string or plugin.

    Connects to the database, discovers the schema, generates a system prompt,
    and returns an Agent with query tools ready to use.

    Args:
        source: Connection string (e.g. ``"postgresql://user:pass@host/db"``) or
                a :class:`BaseDatabasePlugin` instance.
        name: Agent name override. Defaults to ``"{domain} database agent"``.
        model: LLM model override.
        api_key: LLM API key override.
        llm_provider: LLM provider override.
        prompt: User-supplied context prepended to the auto-generated schema prompt.
        db_schema: DB schema name override (e.g. ``"public"`` for PostgreSQL).
        mode: Preset for DB-agent behavior. Supported values are ``"simple"``,
            ``"analyst"`` (default), ``"governed"``, and ``"data_team"``.
        include_sample_values: Include sample values for numeric columns in the schema
            prompt. Defaults to the selected mode's value.
        redact_pii_columns: When ``True`` (default), skip sampling columns whose names
            match common PII patterns.
        lineage: ``True`` to auto-create a :class:`LineagePlugin`, or pass an instance.
        memory: ``True`` to auto-create a :class:`MemoryPlugin`, or pass an instance.
        calibrate_memory: ``True`` to infer numeric unit conventions with an LLM call
            during construction. Defaults to ``False``.
        history: ``True`` to auto-create an in-memory :class:`ConversationHistory`, or pass
            an instance. Enables conversational drilldown across ``run()`` calls.
        cache_ttl: Schema cache TTL in seconds. ``None`` disables caching.
        read_only: When ``True`` (default), write tools are omitted.
        query_default_limit: LIMIT injected into DB query tools when omitted.
        query_max_rows: Maximum rows returned from DB query tools after execution.
        query_max_chars: Maximum serialized characters returned from DB query tools.
        query_timeout: Timeout in seconds for DB query tool execution.
        allowed_tables: Optional table allowlist enforced by DB query tools.
        blocked_tables: Optional table blocklist enforced by DB query tools.
        blocked_columns: Optional column blocklist enforced by DB query tools.
        toolkit: Which analyst toolkit to register. ``"analyst"`` registers
            6 analysis tools (pivot, correlate, anomalies, compare, similar, forecast).
            ``"all"`` is an alias for ``"analyst"``. ``None`` registers no toolkit.
            Defaults to the selected mode's toolkit.
        quality: Register data-quality tools. Defaults to the selected mode's value.
        budget: Prompt budget preset. Supported values are ``"auto"`` (default),
            ``"full"``, ``"compact"``, and ``"retrieval"``. Explicit
            ``schema_prompt_policy`` values override this preset.
        schema_prompt_policy: Advanced schema prompt budget controls.
        tool_result_policy: Advanced DB tool result compaction controls.
        **agent_kwargs: Forwarded to :class:`Agent.__init__`.

    Returns:
        Configured :class:`Agent` with DB tools registered.

    Raises:
        AgentError: If connection or schema discovery fails.
        ValueError: If the connection string scheme is unsupported or malformed.
    """
    resolved_options = _resolve_from_db_options(
        mode=mode,
        read_only=read_only,
        include_sample_values=include_sample_values,
        query_default_limit=query_default_limit,
        query_max_rows=query_max_rows,
        query_max_chars=query_max_chars,
        query_timeout=query_timeout,
        lineage=lineage,
        memory=memory,
        history=history,
        toolkit=toolkit,
        quality=quality,
    )
    mode = resolved_options["mode"]
    read_only = resolved_options["read_only"]
    include_sample_values = resolved_options["include_sample_values"]
    query_default_limit = resolved_options["query_default_limit"]
    query_max_rows = resolved_options["query_max_rows"]
    query_max_chars = resolved_options["query_max_chars"]
    query_timeout = resolved_options["query_timeout"]
    lineage = resolved_options["lineage"]
    memory = resolved_options["memory"]
    history = resolved_options["history"]
    toolkit = resolved_options["toolkit"]
    quality = resolved_options["quality"]

    plugin, was_created = resolve_plugin(source, read_only=read_only)
    _configure_db_plugin(
        plugin,
        read_only=read_only,
        query_default_limit=query_default_limit,
        query_max_rows=query_max_rows,
        query_max_chars=query_max_chars,
        query_timeout=query_timeout,
        allowed_tables=allowed_tables,
        blocked_tables=blocked_tables,
        blocked_columns=blocked_columns,
    )
    await _connect_db_plugin(plugin, was_created=was_created)

    schema, drift = await _discover_schema_with_cache(
        source,
        plugin,
        db_schema=db_schema,
        cache_ttl=cache_ttl,
        was_created=was_created,
    )

    if include_sample_values:
        await sample_numeric_columns(plugin, schema, redact_pii=redact_pii_columns)

    domain = infer_domain(schema)
    prompt_policy = schema_prompt_policy or schema_prompt_policy_for_budget(budget)
    initial_prompt_result = build_prompt_result(
        schema,
        domain,
        prompt,
        analyst_tools=_analyst_tool_names(schema, toolkit),
        schema_navigation_enabled=False,
        policy=prompt_policy,
    )
    schema_navigation_enabled = (
        should_register_schema_navigation(schema)
        or initial_prompt_result.strategy != "full"
        or initial_prompt_result.budget_exceeded
    )
    analyst_tools = _analyst_tool_names(schema, toolkit)
    prompt_result = build_prompt_result(
        schema,
        domain,
        prompt,
        analyst_tools=analyst_tools,
        schema_navigation_enabled=schema_navigation_enabled,
        policy=prompt_policy,
    )
    system_prompt = prompt_result.prompt

    agent = _create_db_agent(
        name=name or f"{domain} database agent",
        llm_provider=llm_provider,
        model=model,
        api_key=api_key,
        prompt=system_prompt,
        agent_kwargs=agent_kwargs,
    )
    _attach_db_state(agent, plugin=plugin, schema=schema, mode=mode, drift=drift)
    agent._db_prompt_metadata = {
        "strategy": prompt_result.strategy,
        "estimated_tokens": prompt_result.estimated_tokens,
        "table_count": prompt_result.table_count,
        "column_count": prompt_result.column_count,
        "omitted_table_count": len(prompt_result.omitted_tables),
        "budget_exceeded": prompt_result.budget_exceeded,
    }
    _remove_focus_from_db_tools(agent)

    _register_schema_navigation_tools(
        agent, plugin, schema, enabled=schema_navigation_enabled
    )
    _register_db_facade_tools(agent, plugin, schema)
    _remove_provider_db_tools(agent)
    _register_analyst_toolkit(agent, plugin, schema, toolkit=toolkit)
    _register_quality_tools(agent, plugin, enabled=quality)
    await _attach_lineage(agent, schema, lineage=lineage)
    _attach_memory(agent, domain=domain, name=name, memory=memory)
    await _calibrate_memory(
        agent,
        source=source,
        schema=schema,
        enabled=memory,
        calibrate=bool(calibrate_memory),
    )
    _attach_history(agent, history=history)
    _wrap_db_runtime(agent, tool_result_policy=tool_result_policy)

    return agent


def _resolve_from_db_options(mode: str, **overrides: Any) -> Dict[str, Any]:
    return resolve_mode_options(mode, overrides)


def _configure_db_plugin(
    plugin: Any,
    *,
    read_only: bool,
    query_default_limit: int,
    query_max_rows: int,
    query_max_chars: int,
    query_timeout: Optional[float],
    allowed_tables: Optional[List[str]],
    blocked_tables: Optional[List[str]],
    blocked_columns: Optional[List[str]],
) -> None:
    plugin.read_only = read_only
    plugin.query_default_limit = query_default_limit
    plugin.query_max_rows = query_max_rows
    plugin.query_max_chars = query_max_chars
    if query_timeout is not None:
        plugin.query_timeout = query_timeout
    plugin.allowed_tables = set(allowed_tables or [])
    plugin.blocked_tables = set(blocked_tables or [])
    plugin.blocked_columns = set(blocked_columns or [])


async def _connect_db_plugin(plugin: Any, *, was_created: bool) -> None:
    from ...core.exceptions import AgentError

    try:
        await plugin.connect()
    except Exception as exc:
        await _disconnect_if_owned(plugin, was_created=was_created)
        raise AgentError(f"Failed to connect to database: {exc}") from exc


async def _discover_schema_with_cache(
    source: Union[str, "BaseDatabasePlugin"],
    plugin: Any,
    *,
    db_schema: Optional[str],
    cache_ttl: Optional[int],
    was_created: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    from ...core.exceptions import AgentError

    schema = None
    cached_schema = None
    cache_key_val = None
    drift = None

    if cache_ttl is not None:
        cache_key_val = cache_key(source)
        cache_result = load_cached_schema(cache_key_val, cache_ttl)
        if cache_result is not None:
            cached_schema, is_expired = cache_result
            if not is_expired:
                schema = cached_schema

    if schema is not None:
        return schema, drift

    try:
        conn_string = source if isinstance(source, str) else None
        schema = await discover_schema(plugin, conn_string, db_schema)
    except Exception as exc:
        if cached_schema is not None:
            logger.warning(f"Schema discovery failed ({exc}), using expired cache")
            return cached_schema, drift
        await _disconnect_if_owned(plugin, was_created=was_created)
        raise AgentError(f"Schema discovery failed: {exc}") from exc

    if cached_schema is not None:
        drift = detect_drift(cached_schema, schema)
        if drift:
            logger.warning(f"Schema drift detected: {drift}")

    if cache_key_val is not None:
        save_schema_cache(cache_key_val, schema)

    return schema, drift


async def _disconnect_if_owned(plugin: Any, *, was_created: bool) -> None:
    if not was_created:
        return
    try:
        await plugin.disconnect()
    except Exception:
        pass


def _create_db_agent(
    *,
    name: str,
    llm_provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    prompt: str,
    agent_kwargs: Dict[str, Any],
) -> "Agent":
    # Lazy import Agent inside function body to avoid circular imports.
    from ..agent import Agent

    agent_kwargs.pop("focus", None)
    return Agent(
        name=name,
        llm_provider=llm_provider,
        model=model,
        api_key=api_key,
        prompt=prompt,
        **agent_kwargs,
    )


def _attach_db_state(
    agent: "Agent",
    *,
    plugin: Any,
    schema: Dict[str, Any],
    mode: str,
    drift: Optional[Dict[str, Any]],
) -> None:
    agent.add_plugin(plugin)
    agent._db_schema = schema
    agent._db_plugin = plugin
    agent._db_summary = build_db_summary(schema)
    agent._db_mode = mode
    agent._db_schema_drift = drift
    agent._db_monitor_events = []
    agent._db_findings = []
    agent._db_active_findings = {}


def _remove_focus_from_db_tools(agent: "Agent") -> None:
    """Hide and ignore Focus DSL for SQL query tools on from_db agents."""

    for tool in agent.tool_registry.tools:
        if not tool.name.endswith("_query"):
            continue
        properties = tool.parameters.get("properties")
        if isinstance(properties, dict):
            properties.pop("focus", None)
        tool.description = tool.description.replace(
            "Use the focus DSL or add LIMIT to avoid oversized responses "
            "(default LIMIT 50 applied if omitted).",
            "Add LIMIT to control result size; a default LIMIT is applied if omitted.",
        )
        original_handler = tool.handler

        async def _handler_without_focus(
            args: Dict[str, Any], _handler=original_handler
        ):
            if isinstance(args, dict):
                args.pop("focus", None)
            return await _handler(args)

        tool.handler = _handler_without_focus


def _register_analyst_toolkit(
    agent: "Agent",
    plugin: Any,
    schema: Dict[str, Any],
    *,
    toolkit: Optional[str],
) -> None:
    if toolkit not in ("analyst", "all"):
        return
    from .tools import register_analyst_tools

    register_analyst_tools(agent, plugin, schema)


def _register_schema_navigation_tools(
    agent: "Agent",
    plugin: Any,
    schema: Dict[str, Any],
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    from .tools import register_schema_navigation_tools

    register_schema_navigation_tools(agent, plugin, schema)


def _register_db_facade_tools(
    agent: "Agent", plugin: Any, schema: Dict[str, Any]
) -> None:
    from .tools.query_facade import create_db_query_tools

    for tool in create_db_query_tools(plugin, schema):
        agent.tool_registry.register(tool)


def _remove_provider_db_tools(agent: "Agent") -> None:
    """Keep provider-owned tools internal to from_db after generic facades exist."""

    for tool_name in list(agent.tool_registry.tool_names):
        if tool_name in _GENERIC_DB_TOOLS:
            continue
        if tool_name.endswith("_vector_search"):
            continue
        if tool_name.endswith("_vector_upsert"):
            continue
        tool = agent.tool_registry.get(tool_name)
        if tool is not None and tool.source == "plugin" and tool.category == "database":
            agent.tool_registry.remove(tool_name)


def _analyst_tool_names(schema: Dict[str, Any], toolkit: Optional[str]) -> List[str]:
    if toolkit not in ("analyst", "all"):
        return []
    names = ["correlate", "detect_anomalies"]
    if schema.get("database_type", "").lower() != "mongodb":
        names.extend(
            ["pivot_table", "compare_entities", "find_similar", "forecast_trend"]
        )
    return names


def _register_quality_tools(agent: "Agent", plugin: Any, *, enabled: bool) -> None:
    if not enabled:
        return
    from ...plugins.data_quality import DataQualityPlugin

    quality_plugin = DataQualityPlugin(db=plugin)
    agent.add_plugin(quality_plugin)
    agent._db_quality = quality_plugin


async def _attach_lineage(
    agent: "Agent",
    schema: Dict[str, Any],
    *,
    lineage: Union[bool, Any, None],
) -> None:
    if lineage is None or lineage is False:
        return
    if lineage is True:
        from ...plugins.lineage import LineagePlugin

        lineage_plugin = LineagePlugin()
    else:
        lineage_plugin = lineage

    agent.add_plugin(lineage_plugin)

    from ...core.graph.models import AgentGraphNode, NodeType
    from ...plugins.catalog.persistence import _derive_store

    store = _derive_store(schema)
    for fk in schema.get("foreign_keys", []):
        src_id = AgentGraphNode.make_id(NodeType.TABLE, f"{store}.{fk['source_table']}")
        tgt_id = AgentGraphNode.make_id(NodeType.TABLE, f"{store}.{fk['target_table']}")
        await lineage_plugin.register_flow(
            source_id=src_id,
            target_id=tgt_id,
            flow_type="references",
            transformation=f"{fk['source_column']} → {fk['target_column']}",
            metadata={
                "source": "schema_discovery",
                "store": store,
                "fk_source_column": fk["source_column"],
                "fk_target_column": fk["target_column"],
            },
        )

    agent._db_lineage = lineage_plugin


def _attach_memory(
    agent: "Agent",
    *,
    domain: str,
    name: Optional[str],
    memory: Union[bool, Any, None],
) -> None:
    if memory is None or memory is False:
        return
    if memory is True:
        from ...plugins.memory import MemoryPlugin

        workspace = name or f"{domain}_db_agent"
        memory_plugin = MemoryPlugin(workspace=workspace)
    else:
        memory_plugin = memory

    agent.add_plugin(memory_plugin)
    disabled_for = getattr(
        memory_plugin, "_daita_disable_lifecycle_context_for_agent_ids", None
    )
    if disabled_for is None:
        disabled_for = set()
        setattr(
            memory_plugin,
            "_daita_disable_lifecycle_context_for_agent_ids",
            disabled_for,
        )
    disabled_for.add(agent.agent_id)
    agent._db_memory = memory_plugin
    agent._db_memory_semantics = DBMemory(memory_plugin)


async def _calibrate_memory(
    agent: "Agent",
    *,
    source: Union[str, "BaseDatabasePlugin"],
    schema: Dict[str, Any],
    enabled: Union[bool, Any, None],
    calibrate: bool,
) -> None:
    if enabled is None or enabled is False:
        return
    if not calibrate:
        return
    if not hasattr(agent, "_db_memory_semantics"):
        return
    cache_key_calib = f"numeric_unit_calibration:{cache_key(source)}"
    await calibrate_db_memory(
        agent,
        schema,
        agent._db_memory_semantics,
        marker_key=cache_key_calib,
    )


def _attach_history(agent: "Agent", *, history: Union[bool, Any, None]) -> None:
    if history is None or history is False:
        return
    if history is True:
        from ..conversation import ConversationHistory

        history_obj = ConversationHistory()
    else:
        history_obj = history

    agent._db_history = history_obj


def _wrap_db_runtime(
    agent: "Agent", *, tool_result_policy: Optional[ToolResultPolicy]
) -> None:
    agent._db_audit_log: List[Dict[str, Any]] = []
    _attach_result_compaction(agent, policy=tool_result_policy)
    agent.run = make_audited_run(agent, make_db_context_run(agent, agent.run))
    agent.stream = make_audited_stream(
        agent, make_db_context_stream(agent, agent.stream)
    )
    attach_db_context(agent)
    attach_db_describe(agent)


def _attach_result_compaction(
    agent: "Agent", *, policy: Optional[ToolResultPolicy]
) -> None:
    from .result_compaction import compact_tool_result_for_context

    policy = policy or ToolResultPolicy()
    agent._db_tool_result_policy = policy

    def _compact(tool_name: str, result: Any) -> Any:
        return compact_tool_result_for_context(tool_name, result, policy=policy)

    agent._compact_tool_result_for_context = _compact
