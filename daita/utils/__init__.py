"""
Utilities for Daita Agents.

This module provides utility functions and helpers used throughout the framework.

Note: Token tracking has been removed from this module. All token tracking
is now handled automatically by the unified tracing system in daita.core.tracing.
LLM providers automatically track tokens without any manual setup required.

For token and performance statistics, use:
- agent.get_trace_stats() - Get comprehensive stats for a specific agent
- trace_manager.get_agent_metrics(agent_id) - Get metrics via trace manager
- trace_manager.get_global_metrics() - Get system-wide metrics
"""

__all__ = [
    # No utilities currently exported - all functionality moved to core modules
]