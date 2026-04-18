"""Firestore schema discovery by document sampling."""

import logging
from collections import defaultdict
from typing import Any, Optional

from ._gcp_common import gcp_credentials

logger = logging.getLogger(__name__)


def _infer_firestore_type(value: Any) -> str:
    """Return a coarse Firestore field type name."""
    match value:
        case bool():
            return "boolean"
        case int() | float():
            return "number"
        case str():
            return "string"
        case list():
            return "array"
        case dict():
            return "map"
        case None:
            return "null"
        case _:
            return type(value).__name__


async def discover_firestore(
    project: str,
    database: str = "(default)",
    credentials_path: Optional[str] = None,
    impersonate_service_account: Optional[str] = None,
    sample_size: int = 50,
) -> dict[str, Any]:
    """Enumerate Firestore collections and sample documents to infer fields.

    Returns a raw result dict mirroring ``discover_mongodb``'s shape so the
    same normalizer path can be reused.
    """
    try:
        from google.cloud import firestore, firestore_admin_v1
    except ImportError:
        raise ImportError(
            "google-cloud-firestore is required. "
            "Install with: pip install 'daita-agents[gcp]'"
        )

    logger.debug("discover_firestore: sampling %s/%s", project, database)

    creds, _ = gcp_credentials(
        credentials_path=credentials_path,
        impersonate_service_account=impersonate_service_account,
    )
    client = firestore.Client(project=project, credentials=creds, database=database)
    admin = firestore_admin_v1.FirestoreAdminClient(credentials=creds)

    collections_out: list[dict[str, Any]] = []
    try:
        collections = list(client.collections())
    except Exception as exc:
        logger.warning(
            "Firestore collections() failed for %s/%s: %s", project, database, exc
        )
        collections = []

    for coll in collections:
        fields_by_name: dict[str, set[str]] = defaultdict(set)
        doc_count = 0
        try:
            for doc in coll.limit(sample_size).stream():
                doc_count += 1
                for field_name, value in (doc.to_dict() or {}).items():
                    fields_by_name[field_name].add(_infer_firestore_type(value))
        except Exception as exc:
            logger.debug("Firestore sample failed for %s: %s", coll.id, exc)

        collections_out.append(
            {
                "collection_name": coll.id,
                "document_count": doc_count,
                "fields": [
                    {"field_name": name, "types": sorted(types)}
                    for name, types in sorted(fields_by_name.items())
                ],
                "indexes": _list_firestore_indexes(admin, project, database, coll.id),
            }
        )

    return {
        "database_type": "firestore",
        "project": project,
        "database": database,
        "collections": collections_out,
        "sample_size": sample_size,
    }


def _list_firestore_indexes(
    admin: Any, project: str, database: str, collection_id: str
) -> list[dict[str, Any]]:
    """List composite indexes for a Firestore collection group.

    Each index becomes ``{name, query_scope, state, fields: [{field_path,
    order, array_config}]}``. Errors (permission / legacy DB) degrade to ``[]``.
    """
    parent = (
        f"projects/{project}/databases/{database}" f"/collectionGroups/{collection_id}"
    )
    out: list[dict[str, Any]] = []
    try:
        for idx in admin.list_indexes(parent=parent):
            out.append(
                {
                    "name": idx.name.rsplit("/", 1)[-1] if idx.name else "",
                    "query_scope": idx.query_scope.name if idx.query_scope else "",
                    "state": idx.state.name if idx.state else "",
                    "fields": [
                        {
                            "field_path": f.field_path,
                            "order": f.order.name if f.order else "",
                            "array_config": (
                                f.array_config.name if f.array_config else ""
                            ),
                        }
                        for f in idx.fields
                    ],
                }
            )
    except Exception as exc:
        logger.debug("Firestore list_indexes failed for %s: %s", collection_id, exc)
    return out
