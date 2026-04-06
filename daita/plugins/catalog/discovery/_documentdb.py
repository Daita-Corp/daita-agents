"""DocumentDB schema discovery via document sampling.

DocumentDB is wire-protocol compatible with MongoDB, so we use the
motor (pymongo) driver to connect and sample documents.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def discover_documentdb(
    host: str,
    port: int = 27017,
    database: str = "admin",
    username: Optional[str] = None,
    password: Optional[str] = None,
    tls: bool = True,
    sample_size: int = 100,
) -> Dict[str, Any]:
    """
    Connect to a DocumentDB cluster and infer schema by sampling documents.

    Returns a raw result dict with collections and inferred field schemas.
    """
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
    except ImportError:
        raise ImportError(
            "motor is required. Install with: pip install 'daita-agents[mongodb]'"
        )

    # Build connection string
    auth = ""
    if username and password:
        auth = f"{username}:{password}@"
    tls_param = "tls=true&tlsAllowInvalidCertificates=true" if tls else ""
    conn_str = f"mongodb://{auth}{host}:{port}/{database}"
    if tls_param:
        conn_str += f"?{tls_param}&retryWrites=false"
    else:
        conn_str += "?retryWrites=false"

    logger.debug("discover_documentdb: connecting to %s:%d/%s", host, port, database)

    client = AsyncIOMotorClient(conn_str)
    db = client[database]

    try:
        collection_names = await db.list_collection_names()
        collections_schema = []

        for coll_name in collection_names:
            if coll_name.startswith("system."):
                continue

            collection = db[coll_name]

            # Sample documents
            cursor = collection.find().limit(sample_size)
            docs = await cursor.to_list(length=sample_size)

            # Infer schema from samples
            fields = {}
            for doc in docs:
                for key, value in doc.items():
                    if key not in fields:
                        fields[key] = {
                            "field_name": key,
                            "types": set(),
                            "sample_count": 0,
                        }
                    fields[key]["types"].add(type(value).__name__)
                    fields[key]["sample_count"] += 1

            # Convert sets to lists for JSON serialization
            for field in fields.values():
                field["types"] = list(field["types"])

            doc_count = None
            try:
                doc_count = await collection.estimated_document_count()
            except Exception:
                pass

            collections_schema.append(
                {
                    "collection_name": coll_name,
                    "document_count": doc_count,
                    "sampled_count": len(docs),
                    "fields": list(fields.values()),
                }
            )

        return {
            "database_type": "documentdb",
            "database": database,
            "host": host,
            "port": port,
            "collections": collections_schema,
            "collection_count": len(collections_schema),
            "sample_size": sample_size,
        }

    finally:
        client.close()
