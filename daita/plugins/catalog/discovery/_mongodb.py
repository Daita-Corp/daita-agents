"""MongoDB schema discovery via document sampling."""

import logging
from typing import Any, Dict

from ._utils import redact_url

logger = logging.getLogger(__name__)


async def discover_mongodb(
    connection_string: str,
    database: str,
    sample_size: int = 100,
) -> Dict[str, Any]:
    """
    Connect to a MongoDB database and infer its schema by sampling documents.

    Returns a raw result dict with collections and inferred field schemas.
    """
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
    except ImportError:
        raise ImportError(
            "motor is required. Install with: pip install 'daita-agents[mongodb]'"
        )

    logger.debug("discover_mongodb: connecting to %s", redact_url(connection_string))
    client = AsyncIOMotorClient(connection_string)
    db = client[database]

    try:
        collection_names = await db.list_collection_names()
        collections_schema = []

        for coll_name in collection_names:
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

            collections_schema.append(
                {
                    "collection_name": coll_name,
                    "document_count": await collection.estimated_document_count(),
                    "sampled_count": len(docs),
                    "fields": list(fields.values()),
                }
            )

        return {
            "database_type": "mongodb",
            "database": database,
            "collections": collections_schema,
            "collection_count": len(collections_schema),
            "sample_size": sample_size,
        }

    finally:
        client.close()
