"""
Plugin Tracing Integration for Daita Agents - Fixed Complete Version

Simplified automatic tracing for all plugin operations:
- Database queries (PostgreSQL, MySQL, MongoDB)
- API calls (REST)
- Cloud storage operations (S3)
- File operations
- Custom plugin executions

FIXED ISSUES:
- Completed all missing function implementations
- Fixed async/sync method detection
- Improved error handling and logging
- Added proper metadata extraction
- Fixed circular import issues
"""

import asyncio
import logging
import time
import functools
from typing import Dict, Any, Optional, Callable, List
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class TracedPlugin:
    """
    Complete traced plugin wrapper that automatically captures all operations.
    
    Transparently adds tracing to any plugin without changing the interface.
    """
    
    def __init__(self, original_plugin: Any, plugin_name: str):
        """
        Initialize traced plugin wrapper.
        
        Args:
            original_plugin: The original plugin instance
            plugin_name: Name of plugin (postgresql, rest, etc.)
        """
        self._original = original_plugin
        self._plugin_name = plugin_name
        
        # Import here to avoid circular imports
        from .tracing import get_trace_manager
        self._trace_manager = get_trace_manager()
        
        # Copy over all attributes from original plugin
        for attr_name in dir(original_plugin):
            if not attr_name.startswith('_') and not hasattr(self, attr_name):
                attr = getattr(original_plugin, attr_name)
                if callable(attr):
                    # Wrap callable methods with tracing
                    setattr(self, attr_name, self._wrap_method(attr, attr_name))
                else:
                    # Copy non-callable attributes directly
                    setattr(self, attr_name, attr)
        
        logger.debug(f"Created traced {plugin_name} plugin")
    
    def _wrap_method(self, method: Callable, method_name: str) -> Callable:
        """Wrap a plugin method with automatic tracing."""
        
        if asyncio.iscoroutinefunction(method):
            @functools.wraps(method)
            async def async_traced_method(*args, **kwargs):
                return await self._trace_async_method(method, method_name, args, kwargs)
            return async_traced_method
        else:
            @functools.wraps(method)
            def sync_traced_method(*args, **kwargs):
                return self._trace_sync_method(method, method_name, args, kwargs)
            return sync_traced_method
    
    async def _trace_async_method(self, method: Callable, method_name: str, args: tuple, kwargs: dict):
        """Trace an async plugin method."""
        from .tracing import TraceType, TraceStatus
        
        async with self._trace_manager.span(
            operation_name=f"{self._plugin_name}_{method_name}",
            trace_type=TraceType.TOOL_EXECUTION,
            tool_name=self._plugin_name,
            tool_operation=method_name,
            input_data=self._prepare_input_data(args, kwargs)
        ) as span_id:
            
            start_time = time.time()
            try:
                result = await method(*args, **kwargs)
                
                # Extract metadata from result
                metadata = self._extract_result_metadata(result)
                
                # Update span with success metadata
                self._trace_manager.end_span(
                    span_id=span_id,
                    status=TraceStatus.SUCCESS,
                    output_data=result,
                    duration_ms=(time.time() - start_time) * 1000,
                    **metadata
                )
                
                return result
                
            except Exception as e:
                # Update span with error metadata
                self._trace_manager.end_span(
                    span_id=span_id,
                    status=TraceStatus.ERROR,
                    error_message=str(e),
                    duration_ms=(time.time() - start_time) * 1000
                )
                raise
    
    def _trace_sync_method(self, method: Callable, method_name: str, args: tuple, kwargs: dict):
        """Trace a sync plugin method."""
        from .tracing import TraceType, TraceStatus
        
        # For sync methods, we need to create a simple span without async context
        span_id = self._trace_manager.start_span(
            operation_name=f"{self._plugin_name}_{method_name}",
            trace_type=TraceType.TOOL_EXECUTION,
            tool_name=self._plugin_name,
            tool_operation=method_name,
            input_data=self._prepare_input_data(args, kwargs)
        )
        
        start_time = time.time()
        try:
            result = method(*args, **kwargs)
            
            # Extract metadata from result
            metadata = self._extract_result_metadata(result)
            
            # Update span with success metadata
            self._trace_manager.end_span(
                span_id=span_id,
                status=TraceStatus.SUCCESS,
                output_data=result,
                duration_ms=(time.time() - start_time) * 1000,
                **metadata
            )
            
            return result
            
        except Exception as e:
            # Update span with error metadata
            self._trace_manager.end_span(
                span_id=span_id,
                status=TraceStatus.ERROR,
                error_message=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            raise
    
    def _prepare_input_data(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Prepare input data for tracing (sanitized)."""
        try:
            # Sanitize sensitive data
            safe_kwargs = {}
            for key, value in kwargs.items():
                if key.lower() in ['password', 'token', 'secret', 'key', 'auth']:
                    safe_kwargs[key] = '[REDACTED]'
                elif isinstance(value, str) and len(value) > 200:
                    safe_kwargs[key] = value[:200] + '...'
                else:
                    safe_kwargs[key] = value
            
            return {
                "args_count": len(args),
                "kwargs": safe_kwargs,
                "method_type": "async" if asyncio.iscoroutinefunction(args[0] if args else None) else "sync"
            }
        except Exception as e:
            return {"input_error": str(e)}
    
    def _extract_result_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract metadata from operation result."""
        metadata = {
            "result_type": type(result).__name__,
            "success": True
        }
        
        try:
            # Count results for database/collections
            if hasattr(result, '__len__') and not isinstance(result, str):
                metadata["result_count"] = len(result)
            elif isinstance(result, (list, tuple)):
                metadata["result_count"] = len(result)
            
            # API response codes
            if hasattr(result, 'status_code'):
                metadata["status_code"] = result.status_code
                metadata["success"] = 200 <= result.status_code < 300
            
            # Content size for string responses
            if isinstance(result, str):
                metadata["content_size"] = len(result)
            elif hasattr(result, 'text'):
                metadata["content_size"] = len(result.text)
            elif hasattr(result, 'content'):
                metadata["content_size"] = len(str(result.content))
            
            # Database-specific metadata
            if hasattr(result, 'rowcount'):
                metadata["rows_affected"] = result.rowcount
            elif hasattr(result, 'inserted_id'):
                metadata["inserted_id"] = str(result.inserted_id)
            
        except Exception as e:
            metadata["metadata_error"] = str(e)
        
        return metadata
    
    # Delegate all other attributes to the original plugin
    def __getattr__(self, name):
        return getattr(self._original, name)
    
    def __setattr__(self, name, value):
        if name.startswith('_') or name in ['_original', '_plugin_name', '_trace_manager']:
            super().__setattr__(name, value)
        else:
            setattr(self._original, name, value)


# Simple plugin wrapper function

def trace_plugin(plugin: Any, plugin_name: str = None) -> Any:
    """
    Automatically wrap any plugin with tracing capabilities.
    
    Args:
        plugin: Original plugin instance
        plugin_name: Plugin name (auto-detected if not provided)
        
    Returns:
        Traced plugin with automatic operation tracking
    """
    if plugin is None:
        return None
    
    # Auto-detect plugin name if not provided
    if not plugin_name:
        plugin_name = plugin.__class__.__name__.lower().replace('plugin', '')
        if not plugin_name:
            plugin_name = 'unknown_plugin'
    
    return TracedPlugin(plugin, plugin_name)


# Traced plugin factory functions

def traced_postgresql(**kwargs):
    """Create PostgreSQL plugin with automatic tracing."""
    try:
        from ..plugins.postgresql import postgresql
        plugin = postgresql(**kwargs)
        return trace_plugin(plugin, "postgresql")
    except ImportError as e:
        logger.error(f"PostgreSQL plugin not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL plugin: {e}")
        return None


def traced_mysql(**kwargs):
    """Create MySQL plugin with automatic tracing."""
    try:
        from ..plugins.mysql import mysql
        plugin = mysql(**kwargs)
        return trace_plugin(plugin, "mysql")
    except ImportError as e:
        logger.error(f"MySQL plugin not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create MySQL plugin: {e}")
        return None


def traced_mongodb(**kwargs):
    """Create MongoDB plugin with automatic tracing."""
    try:
        from ..plugins.mongodb import mongodb
        plugin = mongodb(**kwargs)
        return trace_plugin(plugin, "mongodb")
    except ImportError as e:
        logger.error(f"MongoDB plugin not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create MongoDB plugin: {e}")
        return None


def traced_rest(**kwargs):
    """Create REST API plugin with automatic tracing."""
    try:
        from ..plugins.rest import rest
        plugin = rest(**kwargs)
        return trace_plugin(plugin, "rest")
    except ImportError as e:
        logger.error(f"REST plugin not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create REST plugin: {e}")
        return None


def traced_s3(**kwargs):
    """Create S3 plugin with automatic tracing."""
    try:
        from ..plugins.s3 import s3
        plugin = s3(**kwargs)
        return trace_plugin(plugin, "s3")
    except ImportError as e:
        logger.error(f"S3 plugin not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create S3 plugin: {e}")
        return None


def traced_slack(**kwargs):
    """Create Slack plugin with automatic tracing."""
    try:
        from ..plugins.slack import slack
        plugin = slack(**kwargs)
        return trace_plugin(plugin, "slack")
    except ImportError as e:
        logger.error(f"Slack plugin not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create Slack plugin: {e}")
        return None


def traced_elasticsearch(**kwargs):
    """Create Elasticsearch plugin with automatic tracing."""
    try:
        from ..plugins.elasticsearch import elasticsearch
        plugin = elasticsearch(**kwargs)
        return trace_plugin(plugin, "elasticsearch")
    except ImportError as e:
        logger.error(f"Elasticsearch plugin not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create Elasticsearch plugin: {e}")
        return None


# Context managers for batch operations

@asynccontextmanager
async def traced_transaction(db_plugin, operation_name: str = "transaction"):
    """
    Context manager for tracing database transactions.
    
    Usage:
        async with traced_transaction(db, "user_registration"):
            await db.query("INSERT INTO users ...")
            await db.query("INSERT INTO profiles ...")
    """
    from .tracing import get_trace_manager, TraceType
    trace_manager = get_trace_manager()
    
    async with trace_manager.span(
        operation_name=f"transaction_{operation_name}",
        trace_type=TraceType.TOOL_EXECUTION,
        tool_name=getattr(db_plugin, '_plugin_name', 'database'),
        tool_operation=operation_name,
        transaction_type="database"
    ):
        try:
            yield db_plugin
        except Exception as e:
            logger.error(f"Transaction {operation_name} failed: {e}")
            raise


@asynccontextmanager  
async def traced_api_batch(api_plugin, batch_name: str = "api_batch"):
    """
    Context manager for tracing API batch operations.
    
    Usage:
        async with traced_api_batch(api, "user_sync"):
            user_data = await api.get("/users/123")
            await api.post("/sync", json=user_data)
    """
    from .tracing import get_trace_manager, TraceType
    trace_manager = get_trace_manager()
    
    async with trace_manager.span(
        operation_name=f"api_batch_{batch_name}",
        trace_type=TraceType.TOOL_EXECUTION,
        tool_name=getattr(api_plugin, '_plugin_name', 'api'),
        tool_operation=batch_name,
        batch_type="api"
    ):
        try:
            yield api_plugin
        except Exception as e:
            logger.error(f"API batch {batch_name} failed: {e}")
            raise


# Utility functions

def get_plugin_traces(plugin_name: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent plugin execution traces."""
    try:
        from .tracing import get_trace_manager
        trace_manager = get_trace_manager()
        operations = trace_manager.get_recent_operations(limit=limit * 2)
        
        # Filter for tool executions
        tool_ops = [
            op for op in operations 
            if op.get('type') == 'tool_execution'
        ]
        
        # Filter by plugin name if specified
        if plugin_name:
            tool_ops = [
                op for op in tool_ops
                if op.get('metadata', {}).get('tool_name') == plugin_name
            ]
        
        return tool_ops[:limit]
    except Exception as e:
        logger.error(f"Error getting plugin traces: {e}")
        return []


def get_plugin_stats(plugin_name: Optional[str] = None) -> Dict[str, Any]:
    """Get comprehensive plugin usage statistics."""
    try:
        traces = get_plugin_traces(plugin_name, limit=50)
        
        if not traces:
            return {"total_operations": 0, "success_rate": 0}
        
        total_ops = len(traces)
        successful_ops = len([t for t in traces if t.get('status') == 'success'])
        
        # Calculate average latency
        latencies = [t.get('duration_ms', 0) for t in traces if t.get('duration_ms')]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        # Operation distribution
        operations = {}
        for trace in traces:
            op_name = trace.get('metadata', {}).get('tool_operation', 'unknown')
            operations[op_name] = operations.get(op_name, 0) + 1
        
        # Error distribution
        errors = {}
        for trace in traces:
            if trace.get('status') == 'error':
                error_msg = trace.get('error', 'Unknown error')
                # Categorize errors
                if 'timeout' in error_msg.lower():
                    error_type = 'timeout'
                elif 'connection' in error_msg.lower():
                    error_type = 'connection'
                elif 'permission' in error_msg.lower() or 'auth' in error_msg.lower():
                    error_type = 'permission'
                else:
                    error_type = 'other'
                errors[error_type] = errors.get(error_type, 0) + 1
        
        return {
            "plugin_name": plugin_name,
            "total_operations": total_ops,
            "successful_operations": successful_ops,
            "failed_operations": total_ops - successful_ops,
            "success_rate": successful_ops / total_ops if total_ops > 0 else 0,
            "average_latency_ms": avg_latency,
            "operation_distribution": operations,
            "error_distribution": errors,
            "total_latency_ms": sum(latencies)
        }
    except Exception as e:
        logger.error(f"Error getting plugin stats: {e}")
        return {"total_operations": 0, "success_rate": 0}


# Advanced tracing helpers

def create_custom_traced_plugin(plugin_instance: Any, plugin_name: str, custom_metadata: Dict[str, Any] = None) -> TracedPlugin:
    """
    Create a custom traced plugin with additional metadata.
    
    Args:
        plugin_instance: The plugin to wrap
        plugin_name: Name for tracing
        custom_metadata: Additional metadata to include in traces
        
    Returns:
        TracedPlugin instance with custom metadata
    """
    traced_plugin = TracedPlugin(plugin_instance, plugin_name)
    
    if custom_metadata:
        # Add custom metadata to the traced plugin
        traced_plugin._custom_metadata = custom_metadata
        
        # Override the metadata extraction to include custom data
        original_extract = traced_plugin._extract_result_metadata
        
        def enhanced_extract(result):
            metadata = original_extract(result)
            metadata.update(custom_metadata)
            return metadata
        
        traced_plugin._extract_result_metadata = enhanced_extract
    
    return traced_plugin


# Export everything
__all__ = [
    # Main tracing functions
    "trace_plugin",
    
    # Traced plugin factories
    "traced_postgresql",
    "traced_mysql", 
    "traced_mongodb",
    "traced_rest",
    "traced_s3",
    "traced_slack",
    "traced_elasticsearch",
    
    # Context managers
    "traced_transaction",
    "traced_api_batch",
    
    # Utility functions
    "get_plugin_traces",
    "get_plugin_stats",
    
    # Advanced functions
    "create_custom_traced_plugin",
    
    # Plugin class (for advanced usage)
    "TracedPlugin"
]