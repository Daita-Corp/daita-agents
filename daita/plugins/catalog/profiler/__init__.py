"""
Concrete profiler implementations wrapping existing discover_* logic.

Each profiler extends BaseProfiler and calls the standalone discovery
functions in the discovery package, converting output to NormalizedSchema.
"""

from ._common import _dict_to_normalized_schema
from ._postgres import PostgresProfiler
from ._mysql import MySQLProfiler
from ._mongodb import MongoDBProfiler
from ._dynamodb import DynamoDBProfiler
from ._s3 import S3Profiler
from ._apigateway import APIGatewayProfiler
from ._sqs import SQSProfiler
from ._sns import SNSProfiler
from ._opensearch import OpenSearchProfiler
from ._documentdb import DocumentDBProfiler
from ._kinesis import KinesisProfiler

__all__ = [
    "_dict_to_normalized_schema",
    "PostgresProfiler",
    "MySQLProfiler",
    "MongoDBProfiler",
    "DynamoDBProfiler",
    "S3Profiler",
    "APIGatewayProfiler",
    "SQSProfiler",
    "SNSProfiler",
    "OpenSearchProfiler",
    "DocumentDBProfiler",
    "KinesisProfiler",
]
