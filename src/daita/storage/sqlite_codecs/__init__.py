"""Collect explicit codec-v1 serializers used by the SQLite state store."""

from .autonomy import (
    decode_autonomous_followup,
    encode_autonomous_followup,
)
from .catalog import (
    decode_catalog_snapshot,
    decode_catalog_sync,
    encode_catalog_snapshot,
    encode_catalog_sync,
)
from .identity import (
    decode_identifier,
    decode_identity,
    encode_identifier,
    encode_identity,
)
from .distribution import (
    decode_conversation_inbox_target,
    decode_delivery,
    decode_distribution_plan,
    decode_outcome_contract,
    decode_outcome_reference,
    encode_conversation_inbox_target,
    encode_delivery,
    encode_distribution_plan,
    encode_outcome_contract,
    encode_outcome_reference,
)
from .jobs import decode_job_run, encode_job_run
from .learning import (
    decode_learning_candidate,
    decode_review_stamps,
    encode_learning_candidate,
    encode_review_stamps,
)
from .mcp_bindings import decode_mcp_binding, encode_mcp_binding
from .receipts import decode_receipt, encode_receipt
from .routines import (
    decode_routine_occurrence,
    decode_scheduled_routine,
    encode_routine_occurrence,
    encode_scheduled_routine,
)
from .semantics import decode_semantic_annotation, encode_semantic_annotation
from .source_permissions import (
    decode_postgresql_update_scope,
    decode_source_read_scope,
    encode_postgresql_update_scope,
    encode_source_read_scope,
)
from .sources import CurrentSourceAdapterError, decode_source, encode_source
from .transcripts import (
    decode_loop_exit,
    decode_message,
    decode_run_input,
    encode_loop_exit,
    encode_message,
    encode_run_input,
)

__all__ = [
    "CurrentSourceAdapterError",
    "decode_autonomous_followup",
    "decode_catalog_snapshot",
    "decode_catalog_sync",
    "decode_identifier",
    "decode_identity",
    "decode_conversation_inbox_target",
    "decode_delivery",
    "decode_distribution_plan",
    "decode_learning_candidate",
    "decode_job_run",
    "decode_loop_exit",
    "decode_mcp_binding",
    "decode_message",
    "decode_postgresql_update_scope",
    "decode_outcome_contract",
    "decode_outcome_reference",
    "decode_receipt",
    "decode_review_stamps",
    "decode_routine_occurrence",
    "decode_run_input",
    "decode_semantic_annotation",
    "decode_source",
    "decode_source_read_scope",
    "decode_scheduled_routine",
    "encode_autonomous_followup",
    "encode_catalog_snapshot",
    "encode_catalog_sync",
    "encode_identifier",
    "encode_identity",
    "encode_conversation_inbox_target",
    "encode_delivery",
    "encode_distribution_plan",
    "encode_learning_candidate",
    "encode_job_run",
    "encode_loop_exit",
    "encode_mcp_binding",
    "encode_message",
    "encode_postgresql_update_scope",
    "encode_outcome_contract",
    "encode_outcome_reference",
    "encode_receipt",
    "encode_review_stamps",
    "encode_routine_occurrence",
    "encode_run_input",
    "encode_semantic_annotation",
    "encode_source",
    "encode_source_read_scope",
    "encode_scheduled_routine",
]
