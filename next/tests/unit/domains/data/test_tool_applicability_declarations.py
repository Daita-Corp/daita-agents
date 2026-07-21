from __future__ import annotations

from dataclasses import replace
from typing import cast

from daita.capabilities import ToolApplicability
from daita.catalog import (
    CATALOG_INSPECT_CAPABILITY_ID,
    CATALOG_SEARCH_CAPABILITY_ID,
    CATALOG_TRAVERSE_CAPABILITY_ID,
    CatalogService,
    catalog_declarations,
)
from daita.domains.data import (
    LOCAL_FILE_READ_CAPABILITY_ID,
    POSTGRESQL_QUERY_CAPABILITY_ID,
    SQLITE_QUERY_CAPABILITY_ID,
    SQLITE_UPDATE_CAPABILITY_ID,
    SQLITE_UPDATE_IMPACT_CAPABILITY_ID,
    TABULAR_COMPARE_CAPABILITY_ID,
    local_file_read_extension_declarations,
    postgresql_query_extension_declarations,
    sqlite_query_extension_declarations,
    sqlite_update_extension_declarations,
    tabular_comparison_extension_declarations,
)
from daita.extensions import ExtensionKind, ExtensionManifest


def test_builtin_tool_views_declare_exact_source_applicability() -> None:
    catalog = catalog_declarations(
        "agent-declarations",
        cast(CatalogService, object()),
    )
    extensions = (
        sqlite_query_extension_declarations(),
        postgresql_query_extension_declarations(),
        local_file_read_extension_declarations(),
        sqlite_update_extension_declarations(),
        tabular_comparison_extension_declarations(),
    )
    views = {
        view.capability_id: view.applicability
        for view in (
            *catalog.tool_views,
            *(view for extension in extensions for view in extension.tool_views),
        )
    }

    assert views == {
        CATALOG_SEARCH_CAPABILITY_ID: ToolApplicability(minimum_active_sources=1),
        CATALOG_INSPECT_CAPABILITY_ID: ToolApplicability(minimum_active_sources=1),
        CATALOG_TRAVERSE_CAPABILITY_ID: ToolApplicability(minimum_active_sources=1),
        SQLITE_QUERY_CAPABILITY_ID: ToolApplicability(
            source_adapter_ids=("sqlite",),
            minimum_active_sources=1,
        ),
        POSTGRESQL_QUERY_CAPABILITY_ID: ToolApplicability(
            source_adapter_ids=("postgresql",),
            minimum_active_sources=1,
        ),
        LOCAL_FILE_READ_CAPABILITY_ID: ToolApplicability(
            source_adapter_ids=("local-directory",),
            minimum_active_sources=1,
        ),
        SQLITE_UPDATE_IMPACT_CAPABILITY_ID: ToolApplicability(
            source_adapter_ids=("sqlite",),
            minimum_active_sources=1,
            required_configuration_flags=("write_access",),
        ),
        SQLITE_UPDATE_CAPABILITY_ID: ToolApplicability(
            source_adapter_ids=("sqlite",),
            minimum_active_sources=1,
            required_configuration_flags=("write_access",),
        ),
        TABULAR_COMPARE_CAPABILITY_ID: ToolApplicability(
            minimum_active_sources=2,
        ),
    }


def test_builtin_nondefault_applicability_participates_in_manifest_fingerprint() -> (
    None
):
    declarations = sqlite_query_extension_declarations()
    declared = ExtensionManifest(
        id="data.sqlite.query",
        version="2.0.0",
        kind=ExtensionKind.CAPABILITY_PROVIDER,
        declarations=declarations,
    )
    global_view = replace(
        declared,
        declarations=replace(
            declarations,
            tool_views=tuple(
                replace(view, applicability=ToolApplicability())
                for view in declarations.tool_views
            ),
        ),
    )

    assert declared.declaration_fingerprint != global_view.declaration_fingerprint
    assert declared.fingerprint != global_view.fingerprint
