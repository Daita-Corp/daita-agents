"""
Public factory for `Agent.from_db()` on the new DB runtime path.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
from pathlib import Path
import re
from typing import Any
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
from .llm_service import DbLLMConfig, db_llm_service_from_config
from .models import (
    DbLimits,
    DbMemoryConfig,
    DbRuntimeConfig,
    DbRuntimeOptions,
    DbSourceOptions,
)
from .runtime import DbRuntime

_MODE_DEFAULTS: dict[str, tuple[DbSourceOptions, bool, bool]] = {
    "simple": (
        DbSourceOptions(
            include_sample_values=True,
            redact_pii_columns=True,
            read_only=True,
            query_default_limit=50,
            query_max_rows=100,
            query_max_chars=25000,
        ),
        False,
        False,
    ),
    "analyst": (
        DbSourceOptions(
            include_sample_values=True,
            redact_pii_columns=True,
            read_only=True,
            query_default_limit=50,
            query_max_rows=200,
            query_max_chars=50000,
        ),
        False,
        False,
    ),
    "governed": (
        DbSourceOptions(
            include_sample_values=True,
            redact_pii_columns=True,
            read_only=True,
            query_default_limit=25,
            query_max_rows=100,
            query_max_chars=25000,
            query_timeout=30,
        ),
        True,
        False,
    ),
    "data_team": (
        DbSourceOptions(
            include_sample_values=True,
            redact_pii_columns=True,
            read_only=True,
            query_default_limit=50,
            query_max_rows=200,
            query_max_chars=50000,
            query_timeout=60,
        ),
        True,
        True,
    ),
}


async def from_db(
    source: str | BaseDatabasePlugin,
    *,
    name: str | None = None,
    mode: str | None = None,
    config: DbRuntimeConfig | None = None,
    source_options: DbSourceOptions | None = None,
    llm: DbLLMConfig | None = None,
    runtime: DbRuntimeOptions | None = None,
    memory: DbMemoryConfig | None = None,
    catalog: CatalogPlugin | None | bool = None,
    lineage: Any | bool | None = None,
    quality: Any | bool | None = None,
    history: ConversationHistory | bool | None = None,
    stateful: bool = False,
    plugins: tuple[Any, ...] | list[Any] = (),
    skills: tuple[Any, ...] | list[Any] = (),
) -> DbAgent:
    """Create a `DbAgent` from typed binding records."""

    db_config = config or DbRuntimeConfig()
    if not isinstance(db_config, DbRuntimeConfig):
        raise TypeError("config must be a DbRuntimeConfig instance")
    (
        mode_name,
        effective_source_options,
        resolved_lineage,
        resolved_quality,
    ) = _resolve_factory_options(
        db_config,
        mode=mode,
        source_options=source_options,
        lineage=lineage,
        quality=quality,
    )
    effective_limits = _limits_for_options(db_config.limits, effective_source_options)
    effective_source_options = replace(
        effective_source_options,
        query_max_rows=effective_limits.max_rows,
        query_timeout=effective_limits.timeout_seconds,
    )
    source_plugin = _resolve_runtime_source(
        source,
        options=effective_source_options,
    )
    _ensure_converted_plugin(source_plugin)
    profile_key = catalog_profile_key(source, db_schema=effective_source_options.schema)
    source_identity = _source_identity(source_plugin, profile_key)
    memory_config = _resolve_memory_config(
        memory,
        source_identity=source_identity,
    )
    host_context = current_host_runtime_context()
    runtime_metadata = _runtime_metadata(
        db_config,
        mode=mode_name,
        source_options=effective_source_options.to_dict(),
        llm=llm.safe_metadata() if llm is not None else None,
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
        lineage=resolved_lineage,
        memory_config=memory_config,
        quality=resolved_quality,
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
        limits=effective_limits,
        execution=db_config.execution,
        source_options=effective_source_options,
        plugins=runtime_plugins,
        policies=db_config.policies,
        metadata=runtime_metadata,
    )
    db_llm_service = db_llm_service_from_config(llm, agent_id=name)
    db_runtime = DbRuntime(
        source=source,
        config=runtime_config,
        store=runtime_store,
        db_llm_service=db_llm_service if db_llm_service.available else None,
        host_services=host_services,
    )
    try:
        await db_runtime.setup(agent_id=name)
        if memory_config.enabled and memory_config.calibrate:
            await calibrate_db_memory(
                db_runtime,
                source_owner=source_plugin.manifest.id,
                marker_key=(
                    "numeric_unit_calibration:"
                    f"{_calibration_source_key(source, source_plugin)}"
                ),
            )
    except Exception as exc:
        with suppress(Exception):
            await db_runtime.teardown()
        raise AgentError(
            f"Failed to initialize database runtime: {exc}",
            retry_hint=getattr(exc, "retry_hint", "retryable"),
            context={"source_type": type(source_plugin).__name__},
        ) from exc
    default_history = history if isinstance(history, ConversationHistory) else None
    if default_history is None and stateful:
        default_history = ConversationHistory(workspace=name)
    return DbAgent(runtime=db_runtime, name=name, default_history=default_history)


def _resolve_runtime_source(
    source: str | BaseDatabasePlugin,
    *,
    options: DbSourceOptions,
) -> BaseDatabasePlugin:
    if isinstance(source, BaseDatabasePlugin):
        _apply_source_options(source, options)
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
            f"Unsupported source: {source!r}. Supported bindings are sqlite:// "
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
        "Use a converted BaseDatabasePlugin for another connector."
    )


def _plugin_options(options: DbSourceOptions) -> dict[str, Any]:
    values = options.to_dict()
    return {
        key: value
        for key, value in values.items()
        if key
        in {
            "schema",
            "include_sample_values",
            "redact_pii_columns",
            "read_only",
            "query_default_limit",
            "query_max_rows",
            "query_max_chars",
            "query_timeout",
            "allowed_tables",
            "blocked_tables",
            "blocked_columns",
        }
    }


def _apply_source_options(plugin: BaseDatabasePlugin, options: DbSourceOptions) -> None:
    plugin.read_only = bool(getattr(plugin, "read_only", False) or options.read_only)
    plugin.config["read_only"] = plugin.read_only
    if options.schema is not None:
        plugin.schema = options.schema
        plugin.config["schema"] = options.schema
    if options.include_sample_values is not None:
        plugin.include_sample_values = options.include_sample_values
        plugin.config["include_sample_values"] = options.include_sample_values
    if options.redact_pii_columns is not None:
        plugin.redact_pii_columns = options.redact_pii_columns
        plugin.config["redact_pii_columns"] = options.redact_pii_columns
    if options.query_default_limit is not None:
        plugin.query_default_limit = options.query_default_limit
        plugin.config["query_default_limit"] = options.query_default_limit
    if options.query_max_rows is not None:
        plugin.query_max_rows = min(
            int(getattr(plugin, "query_max_rows", options.query_max_rows)),
            options.query_max_rows,
        )
        plugin.config["query_max_rows"] = plugin.query_max_rows
    if options.query_max_chars is not None:
        plugin.query_max_chars = options.query_max_chars
        plugin.config["query_max_chars"] = options.query_max_chars
    if options.query_timeout is not None:
        plugin.query_timeout = min(
            float(getattr(plugin, "query_timeout", options.query_timeout)),
            options.query_timeout,
        )
        plugin.config["query_timeout"] = plugin.query_timeout
    current_allowed = set(getattr(plugin, "allowed_tables", set()) or set())
    requested_allowed = set(options.allowed_tables or ())
    if options.allowed_tables is not None:
        current_restricted = bool(
            getattr(plugin, "_allowed_tables_restricted", bool(current_allowed))
        )
        plugin.allowed_tables = (
            current_allowed & requested_allowed
            if current_restricted
            else requested_allowed
        )
        plugin._allowed_tables_restricted = True
        plugin.config["allowed_tables"] = tuple(sorted(plugin.allowed_tables))
    plugin.blocked_tables = set(
        getattr(plugin, "blocked_tables", set()) or set()
    ) | set(options.blocked_tables or ())
    plugin.blocked_columns = set(
        getattr(plugin, "blocked_columns", set()) or set()
    ) | set(options.blocked_columns or ())
    plugin.config["blocked_tables"] = tuple(sorted(plugin.blocked_tables))
    plugin.config["blocked_columns"] = tuple(sorted(plugin.blocked_columns))


def _resolve_runtime_store(options: DbRuntimeOptions | None) -> RuntimeStore | None:
    if options is None:
        return None
    if not isinstance(options, DbRuntimeOptions):
        raise TypeError("runtime must be a DbRuntimeOptions instance")
    if options.store is None:
        return None
    if options.store == "sqlite":
        if options.store_path is None:
            raise ValueError("store_path is required when store='sqlite'")
        return SQLiteRuntimeStore(Path(options.store_path))
    return options.store


def _resolve_factory_options(
    config: DbRuntimeConfig,
    *,
    mode: str | None,
    source_options: DbSourceOptions | None,
    lineage: Any | bool | None,
    quality: Any | bool | None,
) -> tuple[str, DbSourceOptions, Any | bool | None, Any | bool | None]:
    mode_name = (mode or config.profile or "analyst").lower()
    if mode_name not in _MODE_DEFAULTS:
        valid = ", ".join(sorted(_MODE_DEFAULTS))
        raise ValueError(f"Unknown from_db mode {mode!r}. Valid modes: {valid}")

    if source_options is not None and not isinstance(source_options, DbSourceOptions):
        raise TypeError("source_options must be a DbSourceOptions instance")
    mode_source_options, default_lineage, default_quality = _MODE_DEFAULTS[mode_name]
    effective_source_options = _merge_source_options(
        mode_source_options,
        config.source_options,
        source_options,
    )
    resolved_lineage = default_lineage if lineage is None else lineage
    resolved_quality = default_quality if quality is None else quality
    return mode_name, effective_source_options, resolved_lineage, resolved_quality


def _merge_source_options(*owners: DbSourceOptions | None) -> DbSourceOptions:
    field_names = (
        "schema",
        "include_sample_values",
        "redact_pii_columns",
        "query_default_limit",
        "query_max_rows",
        "query_max_chars",
        "query_timeout",
        "cache_ttl",
    )
    values: dict[str, Any] = {}
    applicable = tuple(owner for owner in owners if owner is not None)
    for name in field_names:
        populated = [getattr(owner, name) for owner in applicable]
        values[name] = next(
            (value for value in reversed(populated) if value is not None), None
        )
    read_only_values = [
        owner.read_only for owner in applicable if owner.read_only is not None
    ]
    values["read_only"] = any(read_only_values) if read_only_values else None
    allowed_sets = [
        set(owner.allowed_tables)
        for owner in applicable
        if owner.allowed_tables is not None
    ]
    values["allowed_tables"] = (
        tuple(sorted(set.intersection(*allowed_sets))) if allowed_sets else None
    )
    for name in ("blocked_tables", "blocked_columns"):
        sets = [
            set(getattr(owner, name))
            for owner in applicable
            if getattr(owner, name) is not None
        ]
        values[name] = tuple(sorted(set().union(*sets))) if sets else None
    return DbSourceOptions(**values)


def _limits_for_options(limits: DbLimits, options: DbSourceOptions) -> DbLimits:
    max_rows = (
        min(limits.max_rows, options.query_max_rows)
        if options.query_max_rows is not None
        else limits.max_rows
    )
    timeout_seconds = (
        min(limits.timeout_seconds, options.query_timeout)
        if options.query_timeout is not None
        else limits.timeout_seconds
    )
    return replace(
        limits,
        max_rows=max_rows,
        timeout_seconds=timeout_seconds,
    )


def _runtime_metadata(config: DbRuntimeConfig, **values: Any) -> dict[str, Any]:
    metadata = dict(config.metadata)
    construction = {key: value for key, value in values.items() if value is not None}
    if construction:
        metadata["from_db_options"] = construction
    return metadata


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
    memory_config: DbMemoryConfig,
    quality: Any | bool | None,
) -> tuple[Any, ...]:
    resolved: list[Any] = []
    quality_plugin = _resolve_quality_plugin(quality, source_plugin)
    lineage_plugin = _resolve_lineage_plugin(lineage)
    memory_plugin = _resolve_memory_plugin(memory_config)
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
            setattr(quality, "_db", source_plugin)
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
    memory_config: DbMemoryConfig,
) -> Any | None:
    if not memory_config.enabled:
        return None

    from daita.plugins.memory import MemoryPlugin

    plugin = MemoryPlugin(
        workspace=_memory_workspace(memory_config),
        scope="project",
        auto_curate="manual",
        db_memory_mode=True,
        db_memory_retrieval_mode=memory_config.retrieval_mode,
        **_memory_plugin_kwargs(memory_config),
    )
    if not isinstance(memory_config.backend, str):
        plugin.backend = memory_config.backend
    return plugin


def _resolve_memory_config(
    memory: DbMemoryConfig | None,
    *,
    source_identity: str,
) -> DbMemoryConfig:
    if memory is not None and not isinstance(memory, DbMemoryConfig):
        raise TypeError("memory must be a DbMemoryConfig instance")
    config = memory or DbMemoryConfig()
    backend = config.backend
    embedding_available = bool(
        config.embedding_available
        or config.embedding_provider
        or config.embedding_model
        or config.embedder
        or getattr(backend, "embedding_available", False)
    )
    structured_index = str(
        getattr(backend, "structured_index", config.structured_index)
        or config.structured_index
    )
    config = replace(
        config,
        embedding_available=embedding_available,
        structured_index=structured_index,
        source_identity=config.source_identity or source_identity,
    )
    _validate_embedding_mode_config(config)
    return config


def _memory_plugin_kwargs(memory: DbMemoryConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key in (
        "embedding_provider",
        "embedding_model",
        "embedder",
        "max_chunks",
        "default_ttl_days",
    ):
        value = getattr(memory, key)
        if value is not None:
            kwargs[key] = value
    return kwargs


def _validate_embedding_mode_config(
    memory_config: DbMemoryConfig,
) -> None:
    if not memory_config.enabled:
        return
    if memory_config.retrieval_mode == "structured":
        return
    if not any(
        value is not None
        for value in (
            memory_config.embedding_provider,
            memory_config.embedding_model,
            memory_config.embedder,
        )
    ):
        raise ValueError(
            "memory.retrieval_mode='hybrid' or 'embedding' requires an explicit "
            "embedding_provider, embedding_model, or embedder"
        )


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
