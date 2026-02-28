"""
Core exceptions for Daita Agents.

Provides a hierarchy of exceptions with built-in retry behavior hints
to help agents make intelligent retry decisions.
"""

class DaitaError(Exception):
    """Base exception for all Daita errors."""
    
    def __init__(self, message: str, retry_hint: str = "unknown", context: dict = None):
        """
        Initialize Daita error.
        
        Args:
            message: Error message
            retry_hint: Hint for retry behavior ("transient", "retryable", "permanent", "unknown")
            context: Additional error context
        """
        super().__init__(message)
        self.retry_hint = retry_hint
        self.context = context or {}
    
    def is_transient(self) -> bool:
        """Check if this error is likely transient."""
        return self.retry_hint == "transient"
    
    def is_retryable(self) -> bool:
        """Check if this error might be retryable."""
        return self.retry_hint in ["transient", "retryable"]
    
    def is_permanent(self) -> bool:
        """Check if this error is permanent."""
        return self.retry_hint == "permanent"

class AgentError(DaitaError):
    """Exception raised by agents during operation."""
    
    def __init__(self, message: str, agent_id: str = None, task: str = None, retry_hint: str = "retryable", context: dict = None):
        """
        Initialize agent error.
        
        Args:
            message: Error message
            agent_id: ID of the agent that failed
            task: Task that was being executed
            retry_hint: Hint for retry behavior
            context: Additional error context
        """
        super().__init__(message, retry_hint, context)
        self.agent_id = agent_id
        self.task = task

class ConfigError(DaitaError):
    """Exception raised for configuration issues."""
    
    def __init__(self, message: str, config_section: str = None, context: dict = None):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_section: Section of config that caused the error
            context: Additional error context
        """
        super().__init__(message, retry_hint="permanent", context=context)
        self.config_section = config_section

class LLMError(DaitaError):
    """Exception raised by LLM providers."""
    
    def __init__(self, message: str, provider: str = None, model: str = None, retry_hint: str = "retryable", context: dict = None):
        """
        Initialize LLM error.
        
        Args:
            message: Error message
            provider: LLM provider name
            model: Model name
            retry_hint: Hint for retry behavior
            context: Additional error context
        """
        super().__init__(message, retry_hint, context)
        self.provider = provider
        self.model = model

class PluginError(DaitaError):
    """Exception raised by plugins."""
    
    def __init__(self, message: str, plugin_name: str = None, retry_hint: str = "retryable", context: dict = None):
        """
        Initialize plugin error.
        
        Args:
            message: Error message
            plugin_name: Name of the plugin that failed
            retry_hint: Hint for retry behavior
            context: Additional error context
        """
        super().__init__(message, retry_hint, context)
        self.plugin_name = plugin_name

class WorkflowError(DaitaError):
    """Exception raised by workflow operations."""

    def __init__(self, message: str, workflow_name: str = None, retry_hint: str = "retryable", context: dict = None):
        """
        Initialize workflow error.

        Args:
            message: Error message
            workflow_name: Name of the workflow that failed
            retry_hint: Hint for retry behavior
            context: Additional error context
        """
        super().__init__(message, retry_hint, context)
        self.workflow_name = workflow_name

class RoutingError(DaitaError):
    """Exception raised during task routing operations."""

    def __init__(self, message: str, task: str = None, available_agents: list = None, retry_hint: str = "retryable", context: dict = None):
        """
        Initialize routing error.

        Args:
            message: Error message
            task: Task that failed to route
            available_agents: List of available agents
            retry_hint: Hint for retry behavior
            context: Additional error context
        """
        super().__init__(message, retry_hint, context)
        self.task = task
        self.available_agents = available_agents or []

# ======= Retry-Specific Exception Classes =======

class TransientError(DaitaError):
    """
    Exception for temporary issues that are likely to resolve quickly.
    
    Examples: Network timeouts, rate limits, temporary service unavailability
    These errors should be retried with minimal delay.
    """
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, retry_hint="transient", context=context)

class RetryableError(DaitaError):
    """
    Exception for issues that might be resolved with a different approach or after delay.
    
    Examples: Resource temporarily unavailable, processing queue full, 
    temporary data inconsistency. These errors should be retried with backoff.
    """
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, retry_hint="retryable", context=context)

class PermanentError(DaitaError):
    """
    Exception for issues that will not be resolved by retrying.
    
    Examples: Authentication failures, permission errors, invalid configuration,
    malformed data. These errors should not be retried.
    """
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, retry_hint="permanent", context=context)

# ======= Specific Transient Errors =======

class RateLimitError(TransientError):
    """Exception for API rate limiting."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None, context: dict = None):
        context = context or {}
        if retry_after:
            context['retry_after'] = retry_after
            message = f"{message} (retry after {retry_after}s)"
        super().__init__(message, context)
        self.retry_after = retry_after

class TimeoutError(TransientError):
    """Exception for timeout issues."""
    
    def __init__(self, message: str = "Operation timed out", timeout_duration: float = None, context: dict = None):
        context = context or {}
        if timeout_duration:
            context['timeout_duration'] = timeout_duration
            message = f"{message} (after {timeout_duration}s)"
        super().__init__(message, context)
        self.timeout_duration = timeout_duration

class ConnectionError(TransientError):
    """Exception for connection issues."""
    
    def __init__(self, message: str = "Connection failed", host: str = None, port: int = None, context: dict = None):
        context = context or {}
        if host:
            context['host'] = host
        if port:
            context['port'] = port
            message = f"{message} (to {host}:{port})" if host else f"{message} (port {port})"
        super().__init__(message, context)
        self.host = host
        self.port = port

class ServiceUnavailableError(TransientError):
    """Exception for service unavailability."""
    
    def __init__(self, message: str = "Service temporarily unavailable", service_name: str = None, context: dict = None):
        context = context or {}
        if service_name:
            context['service_name'] = service_name
            message = f"{message}: {service_name}"
        super().__init__(message, context)
        self.service_name = service_name

class TemporaryError(TransientError):
    """Generic temporary error that should retry quickly."""
    pass

class TooManyRequestsError(TransientError):
    """Exception for too many requests (429 HTTP status)."""
    
    def __init__(self, message: str = "Too many requests", retry_after: int = None, context: dict = None):
        context = context or {}
        if retry_after:
            context['retry_after'] = retry_after
        super().__init__(message, context)
        self.retry_after = retry_after

# ======= Specific Retryable Errors =======

class ResourceBusyError(RetryableError):
    """Exception for busy resources that might become available."""
    
    def __init__(self, message: str = "Resource is busy", resource_name: str = None, context: dict = None):
        context = context or {}
        if resource_name:
            context['resource_name'] = resource_name
            message = f"{message}: {resource_name}"
        super().__init__(message, context)
        self.resource_name = resource_name

class DataInconsistencyError(RetryableError):
    """Exception for temporary data inconsistency."""
    
    def __init__(self, message: str = "Data inconsistency detected", data_source: str = None, context: dict = None):
        context = context or {}
        if data_source:
            context['data_source'] = data_source
        super().__init__(message, context)
        self.data_source = data_source

class ProcessingQueueFullError(RetryableError):
    """Exception for full processing queues."""
    
    def __init__(self, message: str = "Processing queue is full", queue_name: str = None, context: dict = None):
        context = context or {}
        if queue_name:
            context['queue_name'] = queue_name
        super().__init__(message, context)
        self.queue_name = queue_name

# ======= Specific Permanent Errors =======

class AuthenticationError(PermanentError):
    """Exception for authentication failures."""
    
    def __init__(self, message: str = "Authentication failed", provider: str = None, context: dict = None):
        context = context or {}
        if provider:
            context['provider'] = provider
        super().__init__(message, context)
        self.provider = provider

class PermissionError(PermanentError):
    """Exception for permission/authorization failures."""
    
    def __init__(self, message: str = "Permission denied", resource: str = None, action: str = None, context: dict = None):
        context = context or {}
        if resource:
            context['resource'] = resource
        if action:
            context['action'] = action
        super().__init__(message, context)
        self.resource = resource
        self.action = action

class ValidationError(PermanentError):
    """Exception for data validation failures."""
    
    def __init__(self, message: str = "Validation failed", field: str = None, value: str = None, context: dict = None):
        context = context or {}
        if field:
            context['field'] = field
        if value:
            context['value'] = str(value)[:100]  # Truncate long values
        super().__init__(message, context)
        self.field = field
        self.value = value

class InvalidDataError(PermanentError):
    """Exception for invalid or malformed data."""
    
    def __init__(self, message: str = "Invalid data format", data_type: str = None, expected_format: str = None, context: dict = None):
        context = context or {}
        if data_type:
            context['data_type'] = data_type
        if expected_format:
            context['expected_format'] = expected_format
        super().__init__(message, context)
        self.data_type = data_type
        self.expected_format = expected_format

class NotFoundError(PermanentError):
    """Exception for missing resources."""
    
    def __init__(self, message: str = "Resource not found", resource_type: str = None, resource_id: str = None, context: dict = None):
        context = context or {}
        if resource_type:
            context['resource_type'] = resource_type
        if resource_id:
            context['resource_id'] = resource_id
        super().__init__(message, context)
        self.resource_type = resource_type
        self.resource_id = resource_id

class BadRequestError(PermanentError):
    """Exception for malformed requests."""
    
    def __init__(self, message: str = "Bad request", request_type: str = None, context: dict = None):
        context = context or {}
        if request_type:
            context['request_type'] = request_type
        super().__init__(message, context)
        self.request_type = request_type

# ======= Circuit Breaker Specific Errors =======

class CircuitBreakerOpenError(PermanentError):
    """Exception when circuit breaker is open."""
    
    def __init__(self, message: str = "Circuit breaker is open", agent_name: str = None, failure_count: int = None, context: dict = None):
        context = context or {}
        if agent_name:
            context['agent_name'] = agent_name
        if failure_count:
            context['failure_count'] = failure_count
        super().__init__(message, context)
        self.agent_name = agent_name
        self.failure_count = failure_count

# ======= Reliability Infrastructure Errors =======

class BackpressureError(RetryableError):
    """Exception when backpressure limits are exceeded."""
    
    def __init__(self, message: str = "Backpressure limit exceeded", agent_id: str = None, queue_size: int = None, context: dict = None):
        context = context or {}
        if agent_id:
            context['agent_id'] = agent_id
        if queue_size is not None:
            context['queue_size'] = queue_size
            message = f"{message} (queue size: {queue_size})"
        super().__init__(message, context)
        self.agent_id = agent_id
        self.queue_size = queue_size

class TaskTimeoutError(TransientError):
    """Exception when a task times out."""
    
    def __init__(self, message: str = "Task execution timed out", task_id: str = None, timeout_duration: float = None, context: dict = None):
        context = context or {}
        if task_id:
            context['task_id'] = task_id
        if timeout_duration:
            context['timeout_duration'] = timeout_duration
            message = f"{message} after {timeout_duration}s"
        super().__init__(message, context)
        self.task_id = task_id
        self.timeout_duration = timeout_duration

class AcknowledgmentTimeoutError(TransientError):
    """Exception when message acknowledgment times out."""
    
    def __init__(self, message: str = "Message acknowledgment timed out", message_id: str = None, timeout_duration: float = None, context: dict = None):
        context = context or {}
        if message_id:
            context['message_id'] = message_id
        if timeout_duration:
            context['timeout_duration'] = timeout_duration
        super().__init__(message, context)
        self.message_id = message_id
        self.timeout_duration = timeout_duration

class TaskNotFoundError(PermanentError):
    """Exception when a referenced task cannot be found."""
    
    def __init__(self, message: str = "Task not found", task_id: str = None, context: dict = None):
        context = context or {}
        if task_id:
            context['task_id'] = task_id
            message = f"{message}: {task_id}"
        super().__init__(message, context)
        self.task_id = task_id

class ReliabilityConfigurationError(PermanentError):
    """Exception for invalid reliability configuration."""
    
    def __init__(self, message: str = "Invalid reliability configuration", config_key: str = None, context: dict = None):
        context = context or {}
        if config_key:
            context['config_key'] = config_key
            message = f"{message}: {config_key}"
        super().__init__(message, context)
        self.config_key = config_key

class DeadLetterQueueError(RetryableError):
    """Exception related to dead letter queue operations."""
    
    def __init__(self, message: str = "Dead letter queue operation failed", operation: str = None, context: dict = None):
        context = context or {}
        if operation:
            context['operation'] = operation
            message = f"{message}: {operation}"
        super().__init__(message, context)
        self.operation = operation

# ======= Utility Functions =======

def classify_exception(exception: Exception) -> str:
    """
    Classify any exception to determine retry behavior.
    
    Args:
        exception: The exception to classify
        
    Returns:
        Retry hint: "transient", "retryable", "permanent", or "unknown"
    """
    # If it's already a Daita exception, use its hint
    if isinstance(exception, DaitaError):
        return exception.retry_hint
    
    # Classify standard Python exceptions
    exception_name = exception.__class__.__name__
    
    # Transient errors (standard library)
    transient_exceptions = {
        'TimeoutError', 'ConnectionError', 'ConnectionResetError', 
        'ConnectionAbortedError', 'ConnectionRefusedError',
        'OSError', 'IOError', 'socket.timeout'
    }
    
    # Permanent errors (standard library)  
    permanent_exceptions = {
        'ValueError', 'TypeError', 'AttributeError', 'KeyError',
        'IndexError', 'NameError', 'SyntaxError', 'ImportError',
        'FileNotFoundError', 'PermissionError'
    }
    
    if exception_name in transient_exceptions:
        return "transient"
    elif exception_name in permanent_exceptions:
        return "permanent"
    else:
        return "retryable"  # Default to retryable for unknown exceptions

def create_contextual_error(
    base_exception: Exception,
    context: dict = None,
    retry_hint: str = None
) -> DaitaError:
    """
    Wrap a standard exception in a Daita exception with context.
    
    Args:
        base_exception: The original exception
        context: Additional context information
        retry_hint: Override retry hint classification
        
    Returns:
        Wrapped DaitaError with context and retry hint
    """
    message = str(base_exception)
    hint = retry_hint or classify_exception(base_exception)
    
    # Choose appropriate Daita exception type
    if hint == "transient":
        return TransientError(message, context)
    elif hint == "permanent":
        return PermanentError(message, context)
    else:
        return RetryableError(message, context)