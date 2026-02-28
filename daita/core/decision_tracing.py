"""
Agent Decision Tracing Utilities for Daita Agents - Fixed Complete Version

Provides simple tools for agents to trace their decision-making process,
reasoning chains, and confidence scores. Integrates seamlessly with the
TraceManager for automatic decision observability.

FIXED ISSUES:
- Completed the trace_decision decorator implementation
- Fixed async context manager issues
- Added missing helper functions
- Improved error handling
- Fixed circular import issues
"""

import asyncio
import logging
import time
import functools
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DecisionEventType(str, Enum):
    """Types of decision events that can be streamed in real-time."""
    DECISION_STARTED = "decision_started"
    CONFIDENCE_UPDATED = "confidence_updated"
    REASONING_ADDED = "reasoning_added"
    ALTERNATIVE_ADDED = "alternative_added"
    FACTOR_SET = "factor_set"
    DECISION_COMPLETED = "decision_completed"

@dataclass
class DecisionEvent:
    """Real-time decision event for streaming."""
    event_type: DecisionEventType
    decision_point: str
    span_id: Optional[str]
    timestamp: float
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "decision_point": self.decision_point,
            "span_id": self.span_id,
            "timestamp": self.timestamp,
            "data": self.data
        }

class DecisionType(str, Enum):
    """Common types of agent decisions."""
    CLASSIFICATION = "classification"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"
    ROUTING = "routing"
    VALIDATION = "validation"
    SELECTION = "selection"
    PRIORITIZATION = "prioritization"
    CUSTOM = "custom"

@dataclass
class DecisionContext:
    """Simple container for decision context and metadata."""
    decision_point: str
    decision_type: DecisionType
    confidence_score: float = 0.0
    reasoning_chain: List[str] = None
    alternatives: List[str] = None
    factors: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.reasoning_chain is None:
            self.reasoning_chain = []
        if self.alternatives is None:
            self.alternatives = []
        if self.factors is None:
            self.factors = {}

class DecisionRecorder:
    """
    Context manager for recording agent decisions.
    
    Provides a simple interface for agents to record their decision-making
    process without needing to understand the underlying tracing system.
    """
    
    def __init__(self, decision_point: str, decision_type: Union[str, DecisionType] = DecisionType.CUSTOM, agent_id: Optional[str] = None, stream_callback: Optional[Callable[[DecisionEvent], None]] = None):
        self.decision_point = decision_point
        self.decision_type = DecisionType(decision_type) if isinstance(decision_type, str) else decision_type
        self.agent_id = agent_id
        self.span_id = None
        self.context = DecisionContext(decision_point, self.decision_type)
        self.stream_callback = stream_callback
        
        # Import here to avoid circular imports
        from .tracing import get_trace_manager
        self.trace_manager = get_trace_manager()
    
    def _emit_event(self, event_type: DecisionEventType, data: Dict[str, Any]):
        """Emit a real-time decision event if callback is provided."""
        event = DecisionEvent(
            event_type=event_type,
            decision_point=self.decision_point,
            span_id=self.span_id,
            timestamp=time.time(),
            data=data
        )
        
        # Emit to local callback if provided
        if self.stream_callback:
            try:
                self.stream_callback(event)
            except Exception as e:
                logger.debug(f"Error calling local decision stream callback: {e}")
        
        # Also emit to TraceManager for centralized streaming
        if self.agent_id:
            try:
                self.trace_manager.emit_decision_event(self.agent_id, event)
            except Exception as e:
                logger.debug(f"Error emitting decision event to TraceManager: {e}")
    
    async def __aenter__(self):
        """Start decision tracing."""
        try:
            # Start the decision span
            self.span_id = self.trace_manager.start_span(
                operation_name=f"decision_{self.decision_point}",
                trace_type="decision_trace",  # Use string to avoid enum import issues
                agent_id=self.agent_id,
                decision_point=self.decision_point,
                decision_type=self.decision_type.value
            )
            
            logger.debug(f"Started decision trace: {self.decision_point}")
            
            # Emit decision started event
            self._emit_event(DecisionEventType.DECISION_STARTED, {
                "decision_type": self.decision_type.value,
                "agent_id": self.agent_id
            })
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to start decision trace: {e}")
            # Return self anyway so calling code doesn't break
            return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End decision tracing with recorded context."""
        try:
            if self.span_id:
                # Record the decision details
                self.trace_manager.record_decision(
                    span_id=self.span_id,
                    confidence=self.context.confidence_score,
                    reasoning=self.context.reasoning_chain,
                    alternatives=self.context.alternatives,
                    **self.context.factors
                )
                
                # End the span
                from .tracing import TraceStatus
                status = TraceStatus.ERROR if exc_type else TraceStatus.SUCCESS
                error_message = str(exc_val) if exc_val else None
                
                self.trace_manager.end_span(
                    span_id=self.span_id,
                    status=status,
                    error_message=error_message,
                    output_data=self.get_summary()
                )
                
                logger.debug(f"Ended decision trace: {self.decision_point} (confidence: {self.context.confidence_score:.2f})")
                
                # Emit decision completed event
                self._emit_event(DecisionEventType.DECISION_COMPLETED, {
                    "final_confidence": self.context.confidence_score,
                    "total_reasoning_steps": len(self.context.reasoning_chain),
                    "total_alternatives": len(self.context.alternatives),
                    "status": status.value if hasattr(status, 'value') else str(status),
                    "success": exc_type is None
                })
        
        except Exception as e:
            logger.error(f"Failed to end decision trace: {e}")
    
    def set_confidence(self, confidence: float):
        """Set confidence score (0.0 to 1.0)."""
        old_confidence = self.context.confidence_score
        self.context.confidence_score = max(0.0, min(1.0, confidence))
        
        # Emit confidence updated event
        self._emit_event(DecisionEventType.CONFIDENCE_UPDATED, {
            "old_confidence": old_confidence,
            "new_confidence": self.context.confidence_score,
            "confidence_change": self.context.confidence_score - old_confidence
        })
    
    def add_reasoning(self, reasoning: str):
        """Add a reasoning step to the decision chain."""
        self.context.reasoning_chain.append(reasoning)
        
        # Emit reasoning added event
        self._emit_event(DecisionEventType.REASONING_ADDED, {
            "reasoning": reasoning,
            "step_number": len(self.context.reasoning_chain),
            "total_steps": len(self.context.reasoning_chain)
        })
    
    def add_alternative(self, alternative: str):
        """Add an alternative option that was considered."""
        self.context.alternatives.append(alternative)
        
        # Emit alternative added event
        self._emit_event(DecisionEventType.ALTERNATIVE_ADDED, {
            "alternative": alternative,
            "total_alternatives": len(self.context.alternatives)
        })
    
    def set_factor(self, key: str, value: Any):
        """Set a decision factor (e.g., data_quality: 0.9)."""
        self.context.factors[key] = value
        
        # Emit factor set event
        self._emit_event(DecisionEventType.FACTOR_SET, {
            "factor_key": key,
            "factor_value": value,
            "total_factors": len(self.context.factors)
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the decision for logging or return."""
        return {
            "decision_point": self.decision_point,
            "decision_type": self.decision_type.value,
            "confidence": self.context.confidence_score,
            "reasoning_steps": len(self.context.reasoning_chain),
            "alternatives_considered": len(self.context.alternatives),
            "factors": list(self.context.factors.keys())
        }

# Context manager factory
@asynccontextmanager
async def record_decision_point(
    decision_point: str, 
    decision_type: Union[str, DecisionType] = DecisionType.CUSTOM,
    agent_id: Optional[str] = None,
    stream_callback: Optional[Callable[[DecisionEvent], None]] = None
):
    """
    Context manager for recording a decision point.
    
    Usage:
        async with record_decision_point("data_classification") as decision:
            result = classify(data)
            decision.set_confidence(0.85)
            decision.add_reasoning("Pattern match found")
            return result
    """
    recorder = DecisionRecorder(decision_point, decision_type, agent_id, stream_callback)
    async with recorder as decision:
        yield decision

# Complete decorator for automatic decision tracing
def trace_decision(
    decision_point: str, 
    decision_type: Union[str, DecisionType] = DecisionType.CUSTOM,
    extract_confidence: bool = True,
    extract_reasoning: bool = True
):
    """
    Decorator for automatic decision tracing.
    
    The decorated function can return:
    1. Just the result
    2. (result, confidence)  
    3. (result, confidence, reasoning_list)
    4. (result, {"confidence": X, "reasoning": [...], "alternatives": [...]})
    
    Usage:
        @trace_decision("classification", DecisionType.CLASSIFICATION)
        async def classify_data(self, data):
            # Your logic here
            result = {"class": "positive"}
            confidence = 0.85
            reasoning = ["Pattern A detected", "Threshold met"]
            return result, confidence, reasoning
    """
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Extract agent_id from self if available
                agent_id = None
                if args and hasattr(args[0], 'agent_id'):
                    agent_id = args[0].agent_id
                
                async with record_decision_point(decision_point, decision_type, agent_id) as decision:
                    try:
                        result = await func(*args, **kwargs)
                        
                        # Extract decision metadata from result
                        confidence, reasoning, alternatives = _extract_decision_metadata(
                            result, extract_confidence, extract_reasoning
                        )
                        
                        if confidence is not None:
                            decision.set_confidence(confidence)
                        
                        if reasoning:
                            for reason in reasoning:
                                decision.add_reasoning(reason)
                        
                        if alternatives:
                            for alt in alternatives:
                                decision.add_alternative(alt)
                        
                        # Return just the main result (strip metadata)
                        if isinstance(result, tuple) and len(result) > 1:
                            return result[0]
                        return result
                        
                    except Exception as e:
                        logger.error(f"Decision function {func.__name__} failed: {e}")
                        raise
            
            return async_wrapper
        else:
            @functools.wraps(func)  
            def sync_wrapper(*args, **kwargs):
                # For sync functions, use asyncio.run if needed
                async def async_exec():
                    agent_id = None
                    if args and hasattr(args[0], 'agent_id'):
                        agent_id = args[0].agent_id
                    
                    async with record_decision_point(decision_point, decision_type, agent_id) as decision:
                        try:
                            result = func(*args, **kwargs)
                            
                            confidence, reasoning, alternatives = _extract_decision_metadata(
                                result, extract_confidence, extract_reasoning
                            )
                            
                            if confidence is not None:
                                decision.set_confidence(confidence)
                            
                            if reasoning:
                                for reason in reasoning:
                                    decision.add_reasoning(reason)
                            
                            if alternatives:
                                for alt in alternatives:
                                    decision.add_alternative(alt)
                            
                            if isinstance(result, tuple) and len(result) > 1:
                                return result[0]
                            return result
                            
                        except Exception as e:
                            logger.error(f"Decision function {func.__name__} failed: {e}")
                            raise
                
                try:
                    # Try to get current event loop
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're already in an async context
                        task = asyncio.create_task(async_exec())
                        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                        return task
                    else:
                        return loop.run_until_complete(async_exec())
                except RuntimeError:
                    # No event loop, create one
                    return asyncio.run(async_exec())
            
            return sync_wrapper
    
    return decorator

def _extract_decision_metadata(result, extract_confidence: bool, extract_reasoning: bool) -> Tuple[Optional[float], List[str], List[str]]:
    """Extract confidence, reasoning, and alternatives from function result."""
    confidence = None
    reasoning = []
    alternatives = []
    
    try:
        if isinstance(result, tuple):
            if len(result) >= 2 and extract_confidence:
                # (result, confidence) or (result, confidence, reasoning)
                conf_value = result[1]
                if isinstance(conf_value, (int, float)):
                    confidence = float(conf_value)
                elif isinstance(conf_value, dict):
                    # (result, metadata_dict)
                    confidence = conf_value.get('confidence')
                    reasoning = conf_value.get('reasoning', [])
                    alternatives = conf_value.get('alternatives', [])
            
            if len(result) >= 3 and extract_reasoning:
                # (result, confidence, reasoning)
                reasoning_value = result[2]
                if isinstance(reasoning_value, list):
                    reasoning = reasoning_value
                elif isinstance(reasoning_value, str):
                    reasoning = [reasoning_value]
            
            if len(result) >= 4:
                # (result, confidence, reasoning, alternatives)
                alt_value = result[3]
                if isinstance(alt_value, list):
                    alternatives = alt_value
        
        elif isinstance(result, dict) and 'confidence' in result:
            # Result is a dict with metadata
            confidence = result.get('confidence')
            reasoning = result.get('reasoning', [])
            alternatives = result.get('alternatives', [])
    
    except Exception as e:
        logger.debug(f"Error extracting decision metadata: {e}")
    
    return confidence, reasoning, alternatives

# Helper functions for common decision patterns

async def record_classification_decision(
    decision_point: str,
    classification_result: str,
    confidence: float,
    features_used: List[str],
    alternatives_considered: Optional[List[str]] = None,
    feature_weights: Optional[Dict[str, float]] = None,
    agent_id: Optional[str] = None,
    stream_callback: Optional[Callable[[DecisionEvent], None]] = None
) -> Dict[str, Any]:
    """
    Helper for recording classification decisions.
    
    Usage:
        result = await record_classification_decision(
            "sentiment_analysis",
            classification_result="positive",
            confidence=0.87,
            features_used=["word_sentiment", "context_analysis"],
            alternatives_considered=["neutral", "negative"],
            feature_weights={"word_sentiment": 0.6, "context_analysis": 0.4}
        )
    """
    async with record_decision_point(decision_point, DecisionType.CLASSIFICATION, agent_id, stream_callback) as decision:
        decision.set_confidence(confidence)
        
        for feature in features_used:
            decision.add_reasoning(f"Feature used: {feature}")
        
        if alternatives_considered:
            for alt in alternatives_considered:
                decision.add_alternative(alt)
        
        if feature_weights:
            for feature, weight in feature_weights.items():
                decision.set_factor(f"weight_{feature}", weight)
        
        return {
            "classification": classification_result,
            "confidence": confidence,
            "decision_summary": decision.get_summary()
        }

async def record_analysis_decision(
    decision_point: str,
    analysis_result: Dict[str, Any],
    confidence: float,
    key_insights: List[str],
    data_quality_factors: Optional[Dict[str, float]] = None,
    agent_id: Optional[str] = None,
    stream_callback: Optional[Callable[[DecisionEvent], None]] = None
) -> Dict[str, Any]:
    """
    Helper for recording analysis decisions.
    
    Usage:
        result = await record_analysis_decision(
            "financial_analysis",
            analysis_result={"trend": "upward", "volatility": "low"},
            confidence=0.91,
            key_insights=["Strong growth pattern", "Low risk indicators"],
            data_quality_factors={"completeness": 0.95, "accuracy": 0.88}
        )
    """
    async with record_decision_point(decision_point, DecisionType.ANALYSIS, agent_id, stream_callback) as decision:
        decision.set_confidence(confidence)
        
        for insight in key_insights:
            decision.add_reasoning(insight)
        
        if data_quality_factors:
            for factor, value in data_quality_factors.items():
                decision.set_factor(factor, value)
        
        return {
            "analysis": analysis_result,
            "confidence": confidence,
            "decision_summary": decision.get_summary()
        }

async def record_recommendation_decision(
    decision_point: str,
    recommendation: str,
    confidence: float,
    rationale: List[str],
    alternatives_considered: Optional[List[str]] = None,
    risk_factors: Optional[Dict[str, str]] = None,
    agent_id: Optional[str] = None,
    stream_callback: Optional[Callable[[DecisionEvent], None]] = None
) -> Dict[str, Any]:
    """
    Helper for recording recommendation decisions.
    
    Usage:
        result = await record_recommendation_decision(
            "investment_advice",
            recommendation="buy",
            confidence=0.78,
            rationale=["Strong fundamentals", "Market conditions favorable"],
            alternatives_considered=["hold", "sell"],
            risk_factors={"market_volatility": "medium", "liquidity": "high"}
        )
    """
    async with record_decision_point(decision_point, DecisionType.RECOMMENDATION, agent_id, stream_callback) as decision:
        decision.set_confidence(confidence)
        
        for reason in rationale:
            decision.add_reasoning(reason)
        
        if alternatives_considered:
            for alt in alternatives_considered:
                decision.add_alternative(alt)
        
        if risk_factors:
            for risk, level in risk_factors.items():
                decision.set_factor(f"risk_{risk}", level)
        
        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "decision_summary": decision.get_summary()
        }

# Utility functions for decision analysis

def get_recent_decisions(agent_id: Optional[str] = None, decision_type: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent decision traces."""
    try:
        from .tracing import get_trace_manager
        trace_manager = get_trace_manager()
        operations = trace_manager.get_recent_operations(agent_id=agent_id, limit=limit * 2)
        
        # Filter for decision traces
        decisions = [
            op for op in operations 
            if op.get('type') == 'decision_trace'
        ]
        
        # Filter by decision type if specified
        if decision_type:
            decisions = [
                op for op in decisions
                if op.get('metadata', {}).get('decision_type') == decision_type
            ]
        
        return decisions[:limit]
    except Exception as e:
        logger.error(f"Error getting recent decisions: {e}")
        return []

def register_agent_decision_stream(agent_id: str, callback: Callable[[DecisionEvent], None]) -> None:
    """
    Register a callback to receive streaming decision events for a specific agent.
    
    This provides centralized decision event streaming through the TraceManager.
    
    Usage:
        def my_callback(event: DecisionEvent):
            print(f"Decision event: {event.event_type} - {event.data}")
        
        register_agent_decision_stream("my-agent-id", my_callback)
    """
    try:
        from .tracing import get_trace_manager
        trace_manager = get_trace_manager()
        trace_manager.register_decision_stream_callback(agent_id, callback)
        logger.info(f"Registered decision stream for agent {agent_id}")
    except Exception as e:
        logger.error(f"Failed to register agent decision stream: {e}")

def unregister_agent_decision_stream(agent_id: str, callback: Callable[[DecisionEvent], None]) -> None:
    """
    Unregister a decision stream callback for a specific agent.
    
    Usage:
        unregister_agent_decision_stream("my-agent-id", my_callback)
    """
    try:
        from .tracing import get_trace_manager
        trace_manager = get_trace_manager()
        trace_manager.unregister_decision_stream_callback(agent_id, callback)
        logger.info(f"Unregistered decision stream for agent {agent_id}")
    except Exception as e:
        logger.error(f"Failed to unregister agent decision stream: {e}")

def get_streaming_agents() -> List[str]:
    """Get list of agents that have decision streaming enabled."""
    try:
        from .tracing import get_trace_manager
        trace_manager = get_trace_manager()
        return trace_manager.get_streaming_agents()
    except Exception as e:
        logger.error(f"Failed to get streaming agents: {e}")
        return []

def get_decision_stats(agent_id: Optional[str] = None, decision_type: Optional[str] = None) -> Dict[str, Any]:
    """Get decision statistics for analysis."""
    try:
        decisions = get_recent_decisions(agent_id, decision_type, limit=50)
        
        if not decisions:
            return {"total_decisions": 0, "average_confidence": 0}
        
        # Calculate statistics
        total_decisions = len(decisions)
        successful_decisions = len([d for d in decisions if d.get('status') == 'success'])
        
        # Extract confidence scores
        confidences = []
        for decision in decisions:
            confidence = decision.get('metadata', {}).get('confidence_score')
            if confidence is not None:
                confidences.append(confidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Decision type distribution
        type_counts = {}
        for decision in decisions:
            dec_type = decision.get('metadata', {}).get('decision_type', 'unknown')
            type_counts[dec_type] = type_counts.get(dec_type, 0) + 1
        
        return {
            "total_decisions": total_decisions,
            "successful_decisions": successful_decisions,
            "success_rate": successful_decisions / total_decisions if total_decisions > 0 else 0,
            "average_confidence": avg_confidence,
            "confidence_count": len(confidences),
            "decision_types": type_counts,
            "agent_id": agent_id,
            "filter_type": decision_type
        }
    except Exception as e:
        logger.error(f"Error getting decision stats: {e}")
        return {"total_decisions": 0, "average_confidence": 0}

# Export everything
__all__ = [
    # Enums
    "DecisionType",
    "DecisionEventType",
    
    # Event classes
    "DecisionEvent",
    
    # Main interfaces
    "record_decision_point",
    "trace_decision",
    
    # Helper functions
    "record_classification_decision",
    "record_analysis_decision", 
    "record_recommendation_decision",
    
    # Analysis functions
    "get_recent_decisions",
    "get_decision_stats",
    
    # Streaming functions
    "register_agent_decision_stream",
    "unregister_agent_decision_stream", 
    "get_streaming_agents",
    
    # Classes (for advanced usage)
    "DecisionRecorder",
    "DecisionContext"
]