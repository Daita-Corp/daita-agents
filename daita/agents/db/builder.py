"""
from_db() — build a fully configured Agent from a database source.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from .audit import make_audited_run, make_audited_stream
from .cache import cache_key, detect_drift, load_cached_schema, save_schema_cache
from .calibration import calibrate_numerics
from .prompt import build_prompt, infer_domain
from .resolve import resolve_plugin
from .sampling import sample_numeric_columns
from .schema import discover_schema

if TYPE_CHECKING:
    from ..agent import Agent
    from ...plugins.base_db import BaseDatabasePlugin

logger = logging.getLogger(__name__)

_DISCOVERY_TOOLS = {
    "postgres_list_tables", "postgres_get_schema", "postgres_inspect",
    "mysql_list_tables", "mysql_get_schema", "mysql_inspect",
    "sqlite_list_tables", "sqlite_get_schema", "sqlite_inspect",
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
    include_sample_values: bool = True,
    redact_pii_columns: bool = True,
    lineage: Union[bool, Any, None] = None,
    memory: Union[bool, Any, None] = None,
    history: Union[bool, Any, None] = None,
    cache_ttl: Optional[int] = None,
    read_only: bool = True,
    toolkit: Optional[str] = "analyst",
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
        include_sample_values: Include sample values for numeric columns in the schema
            prompt (default ``True``).
        redact_pii_columns: When ``True`` (default), skip sampling columns whose names
            match common PII patterns.
        lineage: ``True`` to auto-create a :class:`LineagePlugin`, or pass an instance.
        memory: ``True`` to auto-create a :class:`MemoryPlugin`, or pass an instance.
        history: ``True`` to auto-create an in-memory :class:`ConversationHistory`, or pass
            an instance. Enables conversational drilldown across ``run()`` calls.
        cache_ttl: Schema cache TTL in seconds. ``None`` disables caching.
        read_only: When ``True`` (default), write tools are omitted.
        toolkit: Which analyst toolkit to register. ``"analyst"`` (default) registers
            6 analysis tools (pivot, correlate, anomalies, compare, similar, forecast).
            ``"all"`` is an alias for ``"analyst"``. ``None`` registers no toolkit.
        **agent_kwargs: Forwarded to :class:`Agent.__init__`.

    Returns:
        Configured :class:`Agent` with DB tools registered.

    Raises:
        AgentError: If connection or schema discovery fails.
        ValueError: If the connection string scheme is unsupported or malformed.
    """
    from ...core.exceptions import AgentError

    plugin, was_created = resolve_plugin(source, read_only=read_only)

    try:
        await plugin.connect()
    except Exception as exc:
        if was_created:
            try:
                await plugin.disconnect()
            except Exception:
                pass
        raise AgentError(f"Failed to connect to database: {exc}") from exc

    # ------------------------------------------------------------------
    # Schema discovery with optional caching and drift detection
    # ------------------------------------------------------------------
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
                schema = cached_schema  # cache hit — skip discovery

    if schema is None:
        try:
            conn_string = source if isinstance(source, str) else None
            schema = await discover_schema(plugin, conn_string, db_schema)
        except Exception as exc:
            if cached_schema is not None:
                logger.warning(f"Schema discovery failed ({exc}), using expired cache")
                schema = cached_schema
            else:
                if was_created:
                    try:
                        await plugin.disconnect()
                    except Exception:
                        pass
                raise AgentError(f"Schema discovery failed: {exc}") from exc

        # Drift detection against previous cache
        if cached_schema is not None:
            drift = detect_drift(cached_schema, schema)
            if drift:
                logger.warning(f"Schema drift detected: {drift}")

        # Save new cache
        if cache_key_val is not None:
            save_schema_cache(cache_key_val, schema)

    if include_sample_values:
        await sample_numeric_columns(plugin, schema, redact_pii=redact_pii_columns)

    domain = infer_domain(schema)
    system_prompt = build_prompt(schema, domain, prompt)

    # Lazy import Agent inside function body to avoid circular imports.
    from ..agent import Agent

    agent = Agent(
        name=name or f"{domain} database agent",
        llm_provider=llm_provider,
        model=model,
        api_key=api_key,
        prompt=system_prompt,
        **agent_kwargs,
    )
    agent.add_plugin(plugin)
    agent._db_schema = schema
    agent._db_plugin = plugin

    # Schema discovery tools are redundant when from_db() already embeds the full
    # schema in the system prompt. Unregistering them saves ~250-300 tokens per
    # LLM call without any capability loss.
    for _tool_name in _DISCOVERY_TOOLS:
        agent.tool_registry.remove(_tool_name)

    # ------------------------------------------------------------------
    # Optional analyst toolkit — SQL + pandas/numpy in-process tools
    # ------------------------------------------------------------------
    if toolkit in ("analyst", "all"):
        from .tools import register_analyst_tools
        register_analyst_tools(agent, plugin, schema)

    if drift is not None:
        agent._db_schema_drift = drift
        agent.prompt = build_prompt(schema, domain, prompt)

    # ------------------------------------------------------------------
    # Optional LineagePlugin integration
    # ------------------------------------------------------------------
    if lineage is not None and lineage is not False:
        if lineage is True:
            from ...plugins.lineage import LineagePlugin
            lineage_plugin = LineagePlugin()
        else:
            lineage_plugin = lineage

        agent.add_plugin(lineage_plugin)

        for fk in schema.get("foreign_keys", []):
            await lineage_plugin.register_flow(
                source_id=f"table:{fk['source_table']}",
                target_id=f"table:{fk['target_table']}",
                flow_type="FLOWS_TO",
                transformation=f"{fk['source_column']} → {fk['target_column']}",
                metadata={
                    "source": "schema_discovery",
                    "fk_source_column": fk["source_column"],
                    "fk_target_column": fk["target_column"],
                },
            )

        agent._db_lineage = lineage_plugin

    # ------------------------------------------------------------------
    # Optional MemoryPlugin integration
    # ------------------------------------------------------------------
    if memory is not None and memory is not False:
        if memory is True:
            from ...plugins.memory import MemoryPlugin
            workspace = name or f"{domain}_db_agent"
            memory_plugin = MemoryPlugin(workspace=workspace)
        else:
            memory_plugin = memory

        agent.add_plugin(memory_plugin)
        agent._db_memory = memory_plugin

    # ------------------------------------------------------------------
    # Optional auto calibration — infer numeric column units on cold start
    # ------------------------------------------------------------------
    if memory is not None and memory is not False and hasattr(agent, "_db_memory"):
        memory_plugin = agent._db_memory
        cache_key_calib = f"_calib_{cache_key(source)}"
        already_calibrated = None
        if hasattr(memory_plugin, "recall"):
            try:
                already_calibrated = await memory_plugin.recall(cache_key_calib)
            except (TypeError, AttributeError):
                pass
        if not already_calibrated:
            await calibrate_numerics(agent, schema, memory_plugin)
            if hasattr(memory_plugin, "remember"):
                try:
                    await memory_plugin.remember(cache_key_calib, "done")
                except (TypeError, AttributeError):
                    pass

    # ------------------------------------------------------------------
    # Optional ConversationHistory — auto-inject into every run() call
    # ------------------------------------------------------------------
    if history is not None and history is not False:
        if history is True:
            from ..conversation import ConversationHistory
            history_obj = ConversationHistory()
        else:
            history_obj = history

        agent._db_history = history_obj

    # ------------------------------------------------------------------
    # Audit log — accumulates tool calls across all run() invocations.
    # Each entry: {timestamp, prompt, tool_calls: [{tool, arguments, result}]}
    # Access via agent._db_audit_log after any number of run() calls.
    # ------------------------------------------------------------------
    agent._db_audit_log: List[Dict[str, Any]] = []
    agent.run = make_audited_run(agent, agent.run)
    agent.stream = make_audited_stream(agent, agent.stream)

    return agent
