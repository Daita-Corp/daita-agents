"""
Schema normalization functions.

Converts raw ``discover_*`` output into a uniform normalized shape consumed by
agents and the schema dispatch layer in ``daita.agents.db.schema``.

Each store type has its own module (``_postgresql.py``, ``_s3.py``, …) holding
a single ``normalize_<store>()`` function. This module re-exports them and
provides the ``normalize_discovery()`` dispatcher keyed on ``database_type``.

Adding a new store:
  1. Create ``_mystore.py`` with ``normalize_mystore()``.
  2. Add it to the import list and to :data:`_NORMALIZERS` below.
"""

from typing import Any, Callable, Dict

# Shared helpers and dedup utilities — kept at package level for stable import paths.
from ._common import (
    _normalize_relational,
    deduplicate_stores,
    infer_environment,
    merge_store_sources,
)

# Per-store normalizers
from ._apigateway import normalize_apigateway
from ._bigquery import normalize_bigquery
from ._bigtable import normalize_bigtable
from ._documentdb import normalize_documentdb
from ._dynamodb import normalize_dynamodb
from ._firestore import normalize_firestore
from ._gcp_apigateway import normalize_gcp_apigateway
from ._gcs import normalize_gcs
from ._kinesis import normalize_kinesis
from ._memorystore import normalize_memorystore
from ._mongodb import normalize_mongodb
from ._mysql import normalize_mysql
from ._opensearch import normalize_opensearch
from ._postgresql import normalize_postgresql
from ._pubsub import normalize_pubsub_subscription, normalize_pubsub_topic
from ._s3 import normalize_s3
from ._sns import normalize_sns
from ._sqs import normalize_sqs

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# database_type -> normalizer. Unknown types pass through unchanged via
# normalize_discovery(). Add new stores by registering them here.
_NORMALIZERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    # Relational / document / API
    "postgresql": normalize_postgresql,
    "mysql": normalize_mysql,
    "mongodb": normalize_mongodb,
    # AWS
    "dynamodb": normalize_dynamodb,
    "s3": normalize_s3,
    "apigateway": normalize_apigateway,
    "sqs": normalize_sqs,
    "sns": normalize_sns,
    "opensearch": normalize_opensearch,
    "documentdb": normalize_documentdb,
    "kinesis": normalize_kinesis,
    # GCP
    "gcs": normalize_gcs,
    "bigquery": normalize_bigquery,
    "firestore": normalize_firestore,
    "bigtable": normalize_bigtable,
    "pubsub_topic": normalize_pubsub_topic,
    "pubsub_subscription": normalize_pubsub_subscription,
    "memorystore": normalize_memorystore,
    "gcp_apigateway": normalize_gcp_apigateway,
}


def normalize_discovery(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ``discover_*`` output to the normalized schema shape.

    Dispatches by ``raw["database_type"]``; returns *raw* unchanged for
    unrecognized types.

    Normalized shape::

        {
            "database_type": str,
            "database_name": str,
            "tables": [
                {
                    "name": str,
                    "row_count": int | None,
                    "columns": [
                        {"name": str, "type": str, "nullable": bool,
                         "is_primary_key": bool}
                    ],
                }
            ],
            "foreign_keys": [
                {"source_table": str, "source_column": str,
                 "target_table": str, "target_column": str}
            ],
            "table_count": int,
        }
    """
    handler = _NORMALIZERS.get(raw.get("database_type", "unknown"))
    return handler(raw) if handler else raw


__all__ = [
    # Dispatcher
    "normalize_discovery",
    # Relational / document / API
    "normalize_postgresql",
    "normalize_mysql",
    "normalize_mongodb",
    # AWS
    "normalize_dynamodb",
    "normalize_s3",
    "normalize_apigateway",
    "normalize_sqs",
    "normalize_sns",
    "normalize_opensearch",
    "normalize_documentdb",
    "normalize_kinesis",
    # GCP
    "normalize_gcs",
    "normalize_bigquery",
    "normalize_firestore",
    "normalize_bigtable",
    "normalize_pubsub_topic",
    "normalize_pubsub_subscription",
    "normalize_memorystore",
    "normalize_gcp_apigateway",
    # Store-level utilities
    "deduplicate_stores",
    "merge_store_sources",
    "infer_environment",
    # Shared internal helper (used by tests and future relational normalizers)
    "_normalize_relational",
]
