"""
Execution module — thin re-export shim for the daita-client package.

Provides programmatic agent/workflow execution without the CLI.
"""
from daita_client import DaitaClient, ExecutionResult, ExecutionError

__all__ = ["DaitaClient", "ExecutionResult", "ExecutionError"]
