"""
Raw database and API discovery functions.

Each function connects to a data source, queries its schema, and returns a raw
result dict. No persistence logic — callers handle persist+wrap.

Also provides connection-string utilities used across discovery operations.
"""

from ._utils import redact_url, parse_conn_url, validate_openapi_url, ssl_context
from ._postgres import discover_postgres
from ._mysql import discover_mysql
from ._mongodb import discover_mongodb
from ._openapi import discover_openapi
from ._dynamodb import discover_dynamodb
from ._s3 import discover_s3
from ._apigateway import discover_apigateway
from ._sqs import discover_sqs
from ._sns import discover_sns
from ._opensearch import discover_opensearch
from ._documentdb import discover_documentdb
from ._kinesis import discover_kinesis

__all__ = [
    # Utilities
    "redact_url",
    "parse_conn_url",
    "validate_openapi_url",
    "ssl_context",
    # Discovery functions
    "discover_postgres",
    "discover_mysql",
    "discover_mongodb",
    "discover_openapi",
    "discover_dynamodb",
    "discover_s3",
    "discover_apigateway",
    "discover_sqs",
    "discover_sns",
    "discover_opensearch",
    "discover_documentdb",
    "discover_kinesis",
]
