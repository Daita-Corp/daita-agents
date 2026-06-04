"""
Public factory for `Agent.from_db()` on the new DB runtime path.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, is_dataclass, replace
from typing import Any
from urllib.parse import urlparse

from daita.core.exceptions import AgentError
from daita.plugins.base_db import BaseDatabasePlugin
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.catalog.persistence import catalog_profile_key
from daita.plugins.postgresql import PostgreSQLPlugin
from daita.plugins.sqlite import SQLitePlugin

from .agent import DbAgent
from .memory import calibrate_db_memory
from .models import DbLimits, DbRuntimeConfig
from .runtime import DbRuntime

_MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "simple": {
        "read_only": True,
        "query_default_limit": 50,
        "query_max_rows": 100,
        "query_max_chars": 25000,
        "quality": False,
        "lineage": False,
        "memory": False,
    },
    "analyst": {
        "read_only": True,
        "query_default_limit": 50,
        "query_max_rows": 200,
        "query_max_chars": 50000,
        "quality": False,
        "lineage": False,
        "memory": False,
    },
    "governed": {
        "read_only": True,
        "query_default_limit": 25,
        "query_max_rows": 100,
        "query_max_chars": 25000,
        "query_timeout": 30,
        "quality": False,
        "lineage": True,
        "memory": False,
    },
    "data_team": {
        "read_only": True,
        "query_default_limit": 50,
        "query_max_rows": 200,
        "query_max_chars": 50000,
        "query_timeout": 60,
        "quality": True,
        "lineage": True,
        "memory": False,
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
    memory: Any | bool | None = None,
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
    plugins: tuple[Any, ...] | list[Any] = (),
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
        history=history,
        cache_ttl=cache_ttl,
        toolkit=toolkit,
        calibrate_memory=calibrate_memory,
        budget=budget,
        schema_prompt_policy=schema_prompt_policy,
        tool_result_policy=tool_result_policy,
        catalog_profile_key=profile_key,
        catalog_store_id=f"from_db:{profile_key}",
        catalog_keys=[f"from_db:{profile_key}"],
    )
    catalog_plugin = CatalogPlugin(auto_persist=False) if catalog is None else catalog
    service_plugins = _resolve_service_plugins(
        source_plugin,
        lineage=resolved_options["lineage"],
        memory=resolved_options["memory"],
        quality=resolved_options["quality"],
    )
    runtime_plugins = tuple(
        plugin
        for plugin in (
            *((catalog_plugin,) if catalog_plugin is not False else ()),
            source_plugin,
            *service_plugins,
            *db_config.plugins,
            *tuple(plugins),
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
    runtime = DbRuntime(source=source, config=runtime_config)
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
    return DbAgent(runtime=runtime, name=name)


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
    memory: Any | bool | None,
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
    memory: Any | bool | None,
    quality: Any | bool | None,
) -> tuple[Any, ...]:
    resolved: list[Any] = []
    quality_plugin = _resolve_quality_plugin(quality, source_plugin)
    lineage_plugin = _resolve_lineage_plugin(lineage)
    memory_plugin = _resolve_memory_plugin(memory)
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
    memory: Any | bool | None,
) -> Any | None:
    if memory in (None, False):
        return None
    if memory is True:
        from daita.plugins.memory import MemoryPlugin

        plugin = MemoryPlugin()
    else:
        plugin = memory
        if getattr(plugin, "manifest", None) is None:
            raise ValueError("memory must be True, False, None, or a converted plugin")
    return plugin


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
