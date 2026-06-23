"""
Public factory for `Agent.from_db()` on the new DB runtime path.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, is_dataclass, replace
import re
from typing import Any, Literal
from urllib.parse import urlparse

from daita.agents.conversation import ConversationHistory
from daita.core.exceptions import AgentError
from daita.plugins.base_db import BaseDatabasePlugin
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.catalog.persistence import catalog_profile_key
from daita.plugins.postgresql import PostgreSQLPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import RuntimeStore, SQLiteRuntimeStore, current_host_runtime_context

from .agent import DbAgent
from .memory import calibrate_db_memory
from .llm_service import db_llm_service_from_config
from .models import DbLimits, DbMemoryConfig, DbRuntimeConfig, DbRuntimeOptions
from .runtime import DbRuntime

_DbMemoryOption = DbMemoryConfig | dict[str, Any] | Literal[False] | None

_MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "simple": {
        "read_only": True,
        "query_default_limit": 50,
        "query_max_rows": 100,
        "query_max_chars": 25000,
        "quality": False,
        "lineage": False,
        "memory": None,
    },
    "analyst": {
        "read_only": True,
        "query_default_limit": 50,
        "query_max_rows": 200,
        "query_max_chars": 50000,
        "quality": False,
        "lineage": False,
        "memory": None,
    },
    "governed": {
        "read_only": True,
        "query_default_limit": 25,
        "query_max_rows": 100,
        "query_max_chars": 25000,
        "query_timeout": 30,
        "quality": False,
        "lineage": True,
        "memory": None,
    },
    "data_team": {
        "read_only": True,
        "query_default_limit": 50,
        "query_max_rows": 200,
        "query_max_chars": 50000,
        "query_timeout": 60,
        "quality": True,
        "lineage": True,
        "memory": None,
    },
}


async def from_db(
    source: str | BaseDatabasePlugin,
    *,
    name: str | None = None,
    mode: str | None = None,
    config: DbRuntimeConfig | None = None,
    catalog: CatalogPlugin | None | bool = None,
    lineage: Any | bool | None = None,
    memory: _DbMemoryOption = None,
    quality: Any | bool | None = None,
    model: str | None = None,
    api_key: str | None = None,
    llm_provider: str | None = None,
    temperature: float | None = None,
    prompt: str | None = None,
    db_schema: str | None = None,
    include_sample_values: bool | None = None,
    redact_pii_columns: bool = True,
    history: Any | bool | None = None,
    cache_ttl: int | None = None,
    toolkit: str | None = None,
    calibrate_memory: bool | None = None,
    budget: Any | None = None,
    schema_prompt_policy: Any | None = None,
    tool_result_policy: Any | None = None,
    stateful: bool = False,
    runtime: DbRuntimeOptions | None = None,
    plugins: tuple[Any, ...] | list[Any] = (),
    skills: tuple[Any, ...] | list[Any] = (),
    read_only: bool | None = None,
    query_default_limit: int | None = None,
    query_max_rows: int | None = None,
    query_max_chars: int | None = None,
    query_timeout: float | None = None,
    allowed_tables: list[str] | tuple[str, ...] | None = None,
    blocked_tables: list[str] | tuple[str, ...] | None = None,
    blocked_columns: list[str] | tuple[str, ...] | None = None,
    **kwargs: Any,
) -> DbAgent:
    """Create a `DbAgent` backed by `DbRuntime`.

    Phase 10 intentionally moved the public `Agent.from_db()` classmethod to
    the new operation runtime. Phase 13 keeps moving DB-adjacent services onto
    this path by resolving Memory, Lineage, and DataQuality options into
    extension-first plugins.
    """
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            "Agent.from_db() on the DB runtime path received unsupported "
            f"argument(s): {unexpected}"
        )

    db_config = config or DbRuntimeConfig()
    mode_name, resolved_options = _resolve_factory_options(
        db_config,
        mode=mode,
        read_only=read_only,
        query_default_limit=query_default_limit,
        query_max_rows=query_max_rows,
        query_max_chars=query_max_chars,
        query_timeout=query_timeout,
        lineage=lineage,
        memory=memory,
        quality=quality,
    )
    source_plugin = _resolve_runtime_source(
        source,
        read_only=resolved_options["read_only"],
        query_default_limit=resolved_options["query_default_limit"],
        query_max_rows=resolved_options["query_max_rows"],
        query_max_chars=resolved_options["query_max_chars"],
        query_timeout=resolved_options.get("query_timeout"),
        allowed_tables=allowed_tables,
        blocked_tables=blocked_tables,
        blocked_columns=blocked_columns,
    )
    _ensure_converted_plugin(source_plugin)
    profile_key = catalog_profile_key(source, db_schema=db_schema)
    source_identity = _source_identity(source_plugin, profile_key)
    memory_config = _resolve_memory_config(
        resolved_options["memory"],
        source_identity=source_identity,
    )
    host_context = current_host_runtime_context()
    runtime_metadata = _runtime_metadata(
        db_config,
        mode=mode_name,
        prompt=prompt,
        db_schema=db_schema,
        model=model,
        api_key=api_key,
        llm_provider=llm_provider,
        temperature=temperature,
        include_sample_values=include_sample_values,
        redact_pii_columns=redact_pii_columns,
        history=history if not isinstance(history, ConversationHistory) else None,
        stateful=stateful,
        cache_ttl=cache_ttl,
        toolkit=toolkit,
        calibrate_memory=calibrate_memory,
        budget=budget,
        schema_prompt_policy=schema_prompt_policy,
        tool_result_policy=tool_result_policy,
        catalog_profile_key=profile_key,
        catalog_store_id=f"from_db:{profile_key}",
        catalog_keys=[f"from_db:{profile_key}"],
        memory=memory_config.to_dict(),
    )
    if host_context is not None:
        runtime_metadata["host_runtime"] = {
            "surface": host_context.surface,
            "delivery_defaults": list(host_context.delivery_defaults),
            "metadata": dict(host_context.metadata),
        }
    catalog_plugin = CatalogPlugin(auto_persist=False) if catalog is None else catalog
    service_plugins = _resolve_service_plugins(
        source_plugin,
        lineage=resolved_options["lineage"],
        memory=resolved_options["memory"],
        memory_config=memory_config,
        quality=resolved_options["quality"],
    )
    host_plugins = tuple(host_context.runtime_extensions) if host_context else ()
    host_services = dict(host_context.services) if host_context else {}
    runtime_store = _resolve_runtime_store(runtime)
    developer_plugins = (*tuple(db_config.plugins), *tuple(plugins))
    _ensure_host_extensions_are_unique(host_plugins, developer_plugins)
    runtime_plugins = tuple(
        plugin
        for plugin in (
            *((catalog_plugin,) if catalog_plugin is not False else ()),
            source_plugin,
            *service_plugins,
            *host_plugins,
            *developer_plugins,
            *tuple(skills),
        )
        if plugin is not None
    )
    runtime_config = DbRuntimeConfig(
        profile=mode_name,
        limits=_limits_for_options(db_config.limits, resolved_options),
        plugins=runtime_plugins,
        policies=db_config.policies,
        metadata=runtime_metadata,
    )
    db_llm_service = db_llm_service_from_config(
        model=model,
        llm_provider=llm_provider,
        api_key=api_key,
        temperature=temperature,
        agent_id=name,
    )
    runtime = DbRuntime(
        source=source,
        config=runtime_config,
        store=runtime_store,
        db_llm_service=db_llm_service if db_llm_service.available else None,
        host_services=host_services,
    )
    try:
        await runtime.setup(agent_id=name)
        if calibrate_memory:
            await calibrate_db_memory(
                runtime,
                source_owner=source_plugin.manifest.id,
                marker_key=(
                    "numeric_unit_calibration:"
                    f"{_calibration_source_key(source, source_plugin)}"
                ),
            )
    except Exception as exc:
        with suppress(Exception):
            await runtime.teardown()
        raise AgentError(
            f"Failed to initialize database runtime: {exc}",
            retry_hint=getattr(exc, "retry_hint", "retryable"),
            context={"source_type": type(source_plugin).__name__},
        ) from exc
    default_history = history if isinstance(history, ConversationHistory) else None
    if default_history is None and stateful:
        default_history = ConversationHistory(workspace=name)
    return DbAgent(runtime=runtime, name=name, default_history=default_history)


def _resolve_runtime_source(
    source: str | BaseDatabasePlugin,
    **options: Any,
) -> BaseDatabasePlugin:
    if isinstance(source, BaseDatabasePlugin):
        return source
    if not isinstance(source, str):
        raise ValueError(
            "source must be a converted database connection string, SQLite "
            "file path, or converted BaseDatabasePlugin instance"
        )
    if "://" not in source:
        if (
            source == ":memory:"
            or source.endswith(".db")
            or source.endswith(".sqlite")
            or source.endswith(".sqlite3")
        ):
            return SQLitePlugin(path=source, **_plugin_options(options))
        raise ValueError(
            f"Unsupported source: {source!r}. Phase 10 supports sqlite:// "
            "sources, SQLite file paths, and converted database plugins."
        )

    parsed = urlparse(source)
    scheme = parsed.scheme.lower()
    if scheme == "sqlite":
        path = parsed.path or parsed.netloc or ":memory:"
        return SQLitePlugin(path=path, **_plugin_options(options))
    if scheme in {"postgresql", "postgres"}:
        return PostgreSQLPlugin(connection_string=source, **_plugin_options(options))
    raise ValueError(
        f"Unsupported scheme for new DbRuntime path: {scheme!r}. "
        "Additional DB connectors are converted in later roadmap phases."
    )


def _plugin_options(options: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in options.items() if value is not None}


def _resolve_runtime_store(options: DbRuntimeOptions | None) -> RuntimeStore | None:
    if options is None:
        return None
    if not isinstance(options, DbRuntimeOptions):
        raise TypeError("runtime must be a DbRuntimeOptions instance")
    if options.store is None:
        return None
    if options.store == "sqlite":
        return SQLiteRuntimeStore(options.store_path)
    return options.store


def _resolve_factory_options(
    config: DbRuntimeConfig,
    *,
    mode: str | None,
    read_only: bool | None,
    query_default_limit: int | None,
    query_max_rows: int | None,
    query_max_chars: int | None,
    query_timeout: float | None,
    lineage: Any | bool | None,
    memory: _DbMemoryOption,
    quality: Any | bool | None,
) -> tuple[str, dict[str, Any]]:
    mode_name = (mode or config.profile or "analyst").lower()
    if mode_name not in _MODE_DEFAULTS:
        valid = ", ".join(sorted(_MODE_DEFAULTS))
        raise ValueError(f"Unknown from_db mode {mode!r}. Valid modes: {valid}")

    defaults = dict(_MODE_DEFAULTS[mode_name])
    explicit = {
        "read_only": read_only,
        "query_default_limit": query_default_limit,
        "query_max_rows": query_max_rows,
        "query_max_chars": query_max_chars,
        "query_timeout": query_timeout,
        "lineage": lineage,
        "memory": memory,
        "quality": quality,
    }
    for key, value in explicit.items():
        if value is not None:
            defaults[key] = value
    return mode_name, defaults


def _limits_for_options(limits: DbLimits, options: dict[str, Any]) -> DbLimits:
    return replace(
        limits,
        max_rows=options.get("query_max_rows") or limits.max_rows,
        timeout_seconds=options.get("query_timeout") or limits.timeout_seconds,
    )


def _runtime_metadata(config: DbRuntimeConfig, **values: Any) -> dict[str, Any]:
    metadata = dict(config.metadata)
    construction = {
        key: _metadata_value(value)
        for key, value in values.items()
        if value is not None and key not in {"api_key"}
    }
    if construction:
        metadata["from_db_options"] = construction
    return metadata


def _metadata_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def _ensure_converted_plugin(plugin: BaseDatabasePlugin) -> None:
    if getattr(plugin, "manifest", None) is None:
        raise ValueError(
            f"{type(plugin).__name__} is not converted to the extension-first "
            "DB runtime yet"
        )


def _resolve_service_plugins(
    source_plugin: BaseDatabasePlugin,
    *,
    lineage: Any | bool | None,
    memory: _DbMemoryOption,
    memory_config: DbMemoryConfig,
    quality: Any | bool | None,
) -> tuple[Any, ...]:
    resolved: list[Any] = []
    quality_plugin = _resolve_quality_plugin(quality, source_plugin)
    lineage_plugin = _resolve_lineage_plugin(lineage)
    memory_plugin = _resolve_memory_plugin(memory, memory_config)
    for plugin in (quality_plugin, lineage_plugin, memory_plugin):
        if plugin is not None:
            resolved.append(plugin)
    return tuple(resolved)


def _resolve_quality_plugin(
    quality: Any | bool | None,
    source_plugin: BaseDatabasePlugin,
) -> Any | None:
    if quality in (None, False):
        return None
    if quality is True:
        from daita.plugins.data_quality import DataQualityPlugin

        return DataQualityPlugin(db=source_plugin)
    if getattr(quality, "manifest", None) is None:
        raise ValueError("quality must be True, False, None, or a converted plugin")
    if getattr(quality, "_db", None) is None:
        try:
            quality._db = source_plugin
        except Exception:
            pass
    return quality


def _resolve_lineage_plugin(lineage: Any | bool | None) -> Any | None:
    if lineage in (None, False):
        return None
    if lineage is True:
        from daita.plugins.lineage import LineagePlugin

        return LineagePlugin()
    if getattr(lineage, "manifest", None) is None:
        raise ValueError("lineage must be True, False, None, or a converted plugin")
    return lineage


def _resolve_memory_plugin(
    memory: _DbMemoryOption,
    memory_config: DbMemoryConfig,
) -> Any | None:
    if not memory_config.enabled:
        return None

    _ensure_supported_memory_option(memory)
    from daita.plugins.memory import MemoryPlugin

    plugin = MemoryPlugin(
        workspace=_memory_workspace(memory_config),
        scope="project",
        auto_curate="manual",
        db_memory_mode=True,
        db_memory_retrieval_mode=memory_config.retrieval_mode,
        **_memory_plugin_kwargs(memory),
    )
    backend = _memory_backend(memory)
    if backend is not None:
        plugin.backend = backend
    return plugin


def _resolve_memory_config(
    memory: _DbMemoryOption,
    *,
    source_identity: str,
) -> DbMemoryConfig:
    if memory is False:
        return DbMemoryConfig(
            enabled=False,
            recall="off",
            learning="off",
            source_identity=source_identity,
        )
    if isinstance(memory, DbMemoryConfig):
        config = replace(
            memory, source_identity=memory.source_identity or source_identity
        )
        if not config.enabled:
            return replace(config, recall="off", learning="off")
        _validate_embedding_mode_config(config, memory)
        return config
    if isinstance(memory, dict):
        values = dict(memory)
        embedding_available = _memory_embedding_explicit(memory)
        backend = values.get("backend")
        if values.get("enabled") is False:
            values.setdefault("recall", "off")
            values.setdefault("learning", "off")
        for key in (
            "backend",
            "embedding_provider",
            "embedding_model",
            "embedder",
            "max_chunks",
            "default_ttl_days",
        ):
            values.pop(key, None)
        if backend is not None:
            values.setdefault("backend", _memory_backend_name(backend))
            values.setdefault(
                "structured_index",
                str(getattr(backend, "structured_index", "custom") or "custom"),
            )
            embedding_available = bool(
                getattr(backend, "embedding_available", embedding_available)
            )
        values["embedding_available"] = embedding_available
        values["source_identity"] = values.get("source_identity") or source_identity
        config = DbMemoryConfig(**values)
        _validate_embedding_mode_config(config, memory)
        return config
    _ensure_supported_memory_option(memory)
    return DbMemoryConfig(source_identity=source_identity)


def _ensure_supported_memory_option(memory: Any) -> None:
    if memory is None or memory is False:
        return
    if isinstance(memory, (DbMemoryConfig, dict)):
        return
    raise ValueError(
        "memory must be False, a DbMemoryConfig, a config mapping, or None"
    )


def _memory_plugin_kwargs(memory: _DbMemoryOption) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    kwargs = {}
    for key in (
        "embedding_provider",
        "embedding_model",
        "embedder",
        "max_chunks",
        "default_ttl_days",
    ):
        if key in memory:
            kwargs[key] = memory[key]
    return kwargs


def _memory_embedding_explicit(memory: _DbMemoryOption) -> bool:
    if not isinstance(memory, dict):
        return False
    return any(
        memory.get(key) is not None
        for key in ("embedding_provider", "embedding_model", "embedder")
    )


def _memory_backend_name(backend: Any) -> str:
    name = type(backend).__name__.lower()
    if "supabase" in name:
        return "supabase"
    if "local" in name:
        return "local"
    return name or "custom"


def _validate_embedding_mode_config(
    memory_config: DbMemoryConfig,
    memory: _DbMemoryOption,
) -> None:
    if not memory_config.enabled:
        return
    if memory_config.retrieval_mode == "structured":
        return
    if not _memory_embedding_explicit(memory):
        raise ValueError(
            "memory.retrieval_mode='hybrid' or 'embedding' requires an explicit "
            "embedding_provider, embedding_model, or embedder"
        )


def _memory_backend(memory: _DbMemoryOption) -> Any | None:
    if not isinstance(memory, dict):
        return None
    return memory.get("backend")


def _memory_workspace(memory_config: DbMemoryConfig) -> str:
    source_identity = memory_config.source_identity or "unknown-source"
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "_", source_identity).strip("_") or "source"


def _source_identity(source_plugin: BaseDatabasePlugin, profile_key: str) -> str:
    source_kind = getattr(getattr(source_plugin, "manifest", None), "id", None)
    return (
        f"{source_kind or type(source_plugin).__name__.lower()}:from_db:{profile_key}"
    )


def _ensure_host_extensions_are_unique(
    host_plugins: tuple[Any, ...],
    developer_plugins: tuple[Any, ...],
) -> None:
    host_ids = tuple(
        plugin_id
        for plugin in host_plugins
        if (plugin_id := _plugin_manifest_id(plugin)) is not None
    )
    duplicate_host_ids = {
        plugin_id for plugin_id in host_ids if host_ids.count(plugin_id) > 1
    }
    developer_ids = {
        plugin_id
        for plugin in developer_plugins
        if (plugin_id := _plugin_manifest_id(plugin)) is not None
    }
    duplicates = tuple(
        plugin_id
        for plugin_id in dict.fromkeys((*duplicate_host_ids, *host_ids))
        if plugin_id in duplicate_host_ids or plugin_id in developer_ids
    )
    if duplicates:
        raise ValueError(
            "duplicate hosted runtime extension id(s): " + ", ".join(duplicates)
        )


def _plugin_manifest_id(plugin: Any) -> str | None:
    manifest = getattr(plugin, "manifest", None)
    plugin_id = getattr(manifest, "id", None)
    return plugin_id if isinstance(plugin_id, str) and plugin_id else None


def _calibration_source_key(
    source: str | BaseDatabasePlugin,
    source_plugin: BaseDatabasePlugin,
) -> str:
    if isinstance(source, str):
        return source
    for attribute in ("path", "connection_string"):
        value = getattr(source_plugin, attribute, None)
        if value:
            return str(value)
    return source_plugin.manifest.id
