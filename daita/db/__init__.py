"""
Database runtime facade.

This package is the operation-centric `from_db` architecture path. The legacy
`daita.agents.db` package now keeps only a package-level compatibility import
for `from_db`.
"""

from .agent import DbAgent
from .context import DbContextRenderer
from .evidence import DbEvidenceStore, InMemoryDbEvidenceStore
from .execution import DbExecutionOutcome, DbOperationExecutor
from .factory import from_db
from .catalog_prompt import (
    DBPromptReadModel,
    build_db_prompt_read_model,
    estimate_tokens,
)
from .models import (
    DbIntent,
    DbIntentKind,
    DbLimits,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
    DbRuntimeConfig,
    DbRuntimeInspection,
    DbRuntimeOptions,
    DbMemoryConfig,
)
from .monitor_commands import (
    DbCommandRouter,
    DbMonitorCommand,
    DbMonitorCommandService,
    DbMonitorPlanner,
    DbMonitorResolution,
    DbMonitorResolver,
    DbMonitorValidation,
)
from .monitor_scheduler import (
    DbMonitorActionRunner,
    DbMonitorDeliveryRunner,
    DbMonitorRunner,
    DbMonitorScheduler,
    DbMonitorSchedulerResult,
)
from .monitors import (
    DbMonitorInspection,
    DbMonitorMutation,
    DbMonitor,
    DbMonitorRun,
    DbMonitorState,
    DbMonitorStore,
    InMemoryDbMonitorStore,
    SQLiteDbMonitorStore,
)
from .memory import (
    DBMemory,
    DBMemoryRecord,
    calibrate_db_memory,
    db_memory_pii_error,
    db_memory_recall_decision,
    db_memory_record_from_payload,
    normalize_db_memory_record,
    recall_db_memory_context,
    recall_db_memory_records,
    write_db_memory_record,
    write_db_memory_records,
)
from .planning import DbContractBuilder, DbIntentClassifier
from .query_plan import (
    DbAggregationSpec,
    DbFilterSpec,
    DbJoinSpec,
    DbQueryPlan,
    DbQueryPlanCandidate,
    DbQueryPlanValidation,
)
from .query_planning import DbQueryPlanner
from .query_metadata import (
    IdentityMode,
    column_name,
    field_phrase_tokens,
    field_ref_matches_required,
    identity_column,
    is_numeric_type,
    matching_tables,
    metric_matches_required,
    normalize_field_phrase,
    normalize_identifier,
    required_field_matches_output,
    schema_table_columns,
    short_table_name,
    split_identifier,
    table_name,
)
from .query_catalog import (
    catalog_schema_snapshot,
    find_relationship_paths,
    has_likely_catalog_match,
    search_tables,
)
from .query_sql_validation import (
    sql_fingerprint,
    validate_sql_against_schema,
)
from .runtime import DbRuntime
from .runtime.extensions import HostedInAppMonitorDeliveryPlugin
from .synthesis import (
    DbAnswerCitation,
    DbAnswerSynthesisPayload,
    DbSynthesizer,
    DbSynthesisResult,
)
from .sql_analysis import (
    SqlAnalysis,
    SqlAnalysisError,
    SqlColumnRef,
    SqlSelectItem,
    SqlTableRef,
    analyze_sql,
    normalize_sqlglot_dialect,
)
from .verification import DbVerifier, DbVerificationResult

__all__ = [
    "DbAgent",
    "DBMemory",
    "DBMemoryRecord",
    "calibrate_db_memory",
    "db_memory_pii_error",
    "db_memory_recall_decision",
    "db_memory_record_from_payload",
    "normalize_db_memory_record",
    "recall_db_memory_context",
    "recall_db_memory_records",
    "write_db_memory_record",
    "write_db_memory_records",
    "DbIntent",
    "DbIntentKind",
    "DbLimits",
    "DbOperationContract",
    "DbOperationResult",
    "DbRequest",
    "DbRuntime",
    "HostedInAppMonitorDeliveryPlugin",
    "DbRuntimeConfig",
    "DbRuntimeInspection",
    "DbRuntimeOptions",
    "DbMemoryConfig",
    "DbCommandRouter",
    "DbMonitorCommand",
    "DbMonitorCommandService",
    "DbMonitorPlanner",
    "DbMonitorResolution",
    "DbMonitorResolver",
    "DbMonitorValidation",
    "DbMonitorActionRunner",
    "DbMonitorDeliveryRunner",
    "DbMonitorRunner",
    "DbMonitorScheduler",
    "DbMonitorSchedulerResult",
    "DbMonitor",
    "DbMonitorInspection",
    "DbMonitorMutation",
    "DbMonitorRun",
    "DbMonitorState",
    "DbMonitorStore",
    "InMemoryDbMonitorStore",
    "SQLiteDbMonitorStore",
    "DbContractBuilder",
    "DbIntentClassifier",
    "DbQueryPlan",
    "DbQueryPlanCandidate",
    "DbQueryPlanValidation",
    "DbJoinSpec",
    "DbFilterSpec",
    "DbAggregationSpec",
    "DbQueryPlanner",
    "validate_sql_against_schema",
    "catalog_schema_snapshot",
    "find_relationship_paths",
    "has_likely_catalog_match",
    "IdentityMode",
    "column_name",
    "field_phrase_tokens",
    "field_ref_matches_required",
    "identity_column",
    "is_numeric_type",
    "matching_tables",
    "metric_matches_required",
    "normalize_field_phrase",
    "normalize_identifier",
    "required_field_matches_output",
    "schema_table_columns",
    "short_table_name",
    "split_identifier",
    "table_name",
    "search_tables",
    "sql_fingerprint",
    "DbExecutionOutcome",
    "DbOperationExecutor",
    "DbEvidenceStore",
    "InMemoryDbEvidenceStore",
    "DbVerifier",
    "DbVerificationResult",
    "DbSynthesizer",
    "DbSynthesisResult",
    "DbAnswerCitation",
    "DbAnswerSynthesisPayload",
    "DbContextRenderer",
    "DBPromptReadModel",
    "build_db_prompt_read_model",
    "estimate_tokens",
    "SqlAnalysis",
    "SqlAnalysisError",
    "SqlColumnRef",
    "SqlSelectItem",
    "SqlTableRef",
    "analyze_sql",
    "normalize_sqlglot_dialect",
    "from_db",
]
