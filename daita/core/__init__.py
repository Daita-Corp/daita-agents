"""
Core functionality for Daita Agents.

This module provides the essential building blocks for the Daita framework:
- Exception hierarchy with intelligent retry hints
- Core interfaces that define contracts for agents, LLM providers, and processors
- Focus system for data filtering and selection
- Relay system for agent-to-agent communication
- Workflow orchestration for multi-agent systems
- Unified tracing system for automatic observability
- Plugin tracing for tool execution monitoring
- Decision tracing for agent reasoning capture

The core module is designed to be imported by other framework components
and provides the foundational abstractions that everything else builds upon.
"""

# Exception hierarchy - Must be imported first to avoid circular imports
from .exceptions import (
    # Base exceptions
    DaitaError,
    AgentError,
    ConfigError,
    LLMError,
    PluginError,
    WorkflowError,
    # Retry-specific exceptions with built-in hints
    TransientError,
    RetryableError,
    PermanentError,
    # Specific transient errors
    RateLimitError,
    TimeoutError,
    ConnectionError,
    ServiceUnavailableError,
    TemporaryError,
    TooManyRequestsError,
    # Specific retryable errors
    ResourceBusyError,
    DataInconsistencyError,
    ProcessingQueueFullError,
    # Specific permanent errors
    AuthenticationError,
    PermissionError,
    ValidationError,
    InvalidDataError,
    NotFoundError,
    BadRequestError,
    # Circuit breaker errors
    CircuitBreakerOpenError,
    # Utility functions
    classify_exception,
    create_contextual_error,
)

# Core interfaces - Define contracts for framework components
from .interfaces import (
    LLMProvider,
    DatabaseBackend,
)

# Focus system - DSL-based data filtering before LLM sees results
from .focus import (
    apply_focus,
    FocusQuery,
    parse as parse_focus,
    register_backend,
    FocusDSLError,
)

# Unified Tracing System - OTel-backed automatic observability for all operations
from .tracing import (
    # Main tracing interface
    get_trace_manager,
    TraceManager,
    configure_tracing,
    # Trace types and status
    TraceType,
    TraceStatus,
    # Core tracing components
    TraceSpan,
    TraceContext,
)

# Plugin Tracing System - Automatic tool execution monitoring
from .plugin_tracing import (
    # Main plugin tracing function
    trace_plugin,
    # Traced plugin factories
    traced_postgresql,
    traced_mysql,
    traced_mongodb,
    traced_rest,
    # Context managers for batch operations
    traced_transaction,
    traced_api_batch,
    # Query and statistics functions
    get_plugin_traces,
    get_plugin_stats,
    # Advanced plugin tracing class
    TracedPlugin,
)

# Decision Tracing System - Agent reasoning and confidence capture
from .decision_tracing import (
    # Main decision tracing interfaces
    record_decision_point,
    trace_decision,
    # Decision types
    DecisionType,
    # Helper functions for common decision patterns
    record_classification_decision,
    record_analysis_decision,
    record_recommendation_decision,
    # Query and analysis functions
    get_recent_decisions,
    get_decision_stats,
    # Advanced decision tracing classes
    DecisionRecorder,
    DecisionContext,
)

# Export everything that other modules need
__all__ = [
    # Exception hierarchy
    "DaitaError",
    "AgentError",
    "ConfigError",
    "LLMError",
    "PluginError",
    "WorkflowError",
    # Retry-specific exceptions
    "TransientError",
    "RetryableError",
    "PermanentError",
    # Specific transient errors
    "RateLimitError",
    "TimeoutError",
    "ConnectionError",
    "ServiceUnavailableError",
    "TemporaryError",
    "TooManyRequestsError",
    # Specific retryable errors
    "ResourceBusyError",
    "DataInconsistencyError",
    "ProcessingQueueFullError",
    # Specific permanent errors
    "AuthenticationError",
    "PermissionError",
    "ValidationError",
    "InvalidDataError",
    "NotFoundError",
    "BadRequestError",
    # Circuit breaker errors
    "CircuitBreakerOpenError",
    # Exception utility functions
    "classify_exception",
    "create_contextual_error",
    # Core interfaces
    "Agent",
    "LLMProvider",
    "DatabaseBackend",
    # Focus system
    "apply_focus",
    "FocusQuery",
    "parse_focus",
    "register_backend",
    "FocusDSLError",
    # === Unified Tracing System (OTel-backed) ===
    # Main tracing interface
    "get_trace_manager",
    "TraceManager",
    "configure_tracing",
    # Trace types and status
    "TraceType",
    "TraceStatus",
    # Core tracing components
    "TraceSpan",
    "TraceContext",
    # === Plugin Tracing System ===
    # Main plugin tracing
    "trace_plugin",
    # Traced plugin factories
    "traced_postgresql",
    "traced_mysql",
    "traced_mongodb",
    "traced_rest",
    # Plugin batch operations
    "traced_transaction",
    "traced_api_batch",
    # Plugin analytics
    "get_plugin_traces",
    "get_plugin_stats",
    # Advanced plugin tracing
    "TracedPlugin",
    # === Decision Tracing System ===
    # Main decision tracing
    "record_decision_point",
    "trace_decision",
    # Decision types
    "DecisionType",
    # Decision helpers
    "record_classification_decision",
    "record_analysis_decision",
    "record_recommendation_decision",
    # Decision analytics
    "get_recent_decisions",
    "get_decision_stats",
    # Advanced decision tracing
    "DecisionRecorder",
    "DecisionContext",
]

# Core module metadata
CORE_INFO = {
    "components": [
        "exceptions",
        "interfaces",
        "focus",
        "tracing",
        "plugin_tracing",
        "decision_tracing",
    ],
    "description": "Core abstractions and utilities for the Daita framework",
    "stability": "stable",
}


def get_core_info() -> dict:
    """Get information about core module components."""
    return CORE_INFO.copy()


__all__.append("get_core_info")
