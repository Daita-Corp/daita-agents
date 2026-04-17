"""
Core exceptions for Daita Agents.

Provides a hierarchy of exceptions with built-in retry behavior hints
to help agents make intelligent retry decisions.
"""


class DaitaError(Exception):
    """Base exception for all Daita errors."""

    def __init__(self, message: str, retry_hint: str = "unknown", context: dict = None):
        super().__init__(message)
        self.retry_hint = retry_hint
        self.context = context or {}

    def _enrich(self, context: dict | None, **fields) -> dict:
        """Merge caller-supplied context with named fields, skipping None values."""
        ctx = context or {}
        ctx.update({k: v for k, v in fields.items() if v is not None})
        return ctx

    def is_transient(self) -> bool:
        return self.retry_hint == "transient"

    def is_retryable(self) -> bool:
        return self.retry_hint in ("transient", "retryable")

    def is_permanent(self) -> bool:
        return self.retry_hint == "permanent"


# ======= Domain Errors =======


class AgentError(DaitaError):
    """Exception raised by agents during operation."""

    def __init__(
        self,
        message: str,
        agent_id: str = None,
        task: str = None,
        retry_hint: str = "retryable",
        context: dict = None,
    ):
        super().__init__(
            message, retry_hint, self._enrich(context, agent_id=agent_id, task=task)
        )
        self.agent_id = agent_id
        self.task = task


class ConfigError(DaitaError):
    """Exception raised for configuration issues."""

    def __init__(self, message: str, config_section: str = None, context: dict = None):
        super().__init__(
            message, "permanent", self._enrich(context, config_section=config_section)
        )
        self.config_section = config_section


class LLMError(DaitaError):
    """Exception raised by LLM providers."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        model: str = None,
        retry_hint: str = "retryable",
        context: dict = None,
    ):
        super().__init__(
            message, retry_hint, self._enrich(context, provider=provider, model=model)
        )
        self.provider = provider
        self.model = model


class PluginError(DaitaError):
    """Exception raised by plugins."""

    def __init__(
        self,
        message: str,
        plugin_name: str = None,
        retry_hint: str = "retryable",
        context: dict = None,
    ):
        super().__init__(
            message, retry_hint, self._enrich(context, plugin_name=plugin_name)
        )
        self.plugin_name = plugin_name


class SkillError(PluginError):
    """Exception raised by skills (configuration, dependency resolution, etc.)."""

    pass


class WorkflowError(DaitaError):
    """Exception raised by workflow operations."""

    def __init__(
        self,
        message: str,
        workflow_name: str = None,
        retry_hint: str = "retryable",
        context: dict = None,
    ):
        super().__init__(
            message, retry_hint, self._enrich(context, workflow_name=workflow_name)
        )
        self.workflow_name = workflow_name


class RoutingError(DaitaError):
    """Exception raised during task routing operations."""

    def __init__(
        self,
        message: str,
        task: str = None,
        available_agents: list = None,
        retry_hint: str = "retryable",
        context: dict = None,
    ):
        super().__init__(message, retry_hint, self._enrich(context, task=task))
        self.task = task
        self.available_agents = available_agents or []


# ======= Retry-Specific Base Classes =======


class TransientError(DaitaError):
    """Temporary issues likely to resolve quickly (timeouts, rate limits)."""

    def __init__(self, message: str, context: dict = None):
        super().__init__(message, retry_hint="transient", context=context)


class RetryableError(DaitaError):
    """Issues that might resolve with a different approach or after delay."""

    def __init__(self, message: str, context: dict = None):
        super().__init__(message, retry_hint="retryable", context=context)


class PermanentError(DaitaError):
    """Issues that will not be resolved by retrying."""

    def __init__(self, message: str, context: dict = None):
        super().__init__(message, retry_hint="permanent", context=context)


# ======= Specific Transient Errors =======


class RateLimitError(TransientError):
    """Exception for API rate limiting."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = None,
        context: dict = None,
    ):
        if retry_after:
            message = f"{message} (retry after {retry_after}s)"
        super().__init__(message, self._enrich(context, retry_after=retry_after))
        self.retry_after = retry_after


class TimeoutError(TransientError):
    """Exception for timeout issues."""

    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_duration: float = None,
        context: dict = None,
    ):
        if timeout_duration:
            message = f"{message} (after {timeout_duration}s)"
        super().__init__(
            message, self._enrich(context, timeout_duration=timeout_duration)
        )
        self.timeout_duration = timeout_duration


class ConnectionError(TransientError):
    """Exception for connection issues."""

    def __init__(
        self,
        message: str = "Connection failed",
        host: str = None,
        port: int = None,
        context: dict = None,
    ):
        if port:
            message = (
                f"{message} (to {host}:{port})" if host else f"{message} (port {port})"
            )
        super().__init__(message, self._enrich(context, host=host, port=port))
        self.host = host
        self.port = port


class ServiceUnavailableError(TransientError):
    """Exception for service unavailability."""

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        service_name: str = None,
        context: dict = None,
    ):
        if service_name:
            message = f"{message}: {service_name}"
        super().__init__(message, self._enrich(context, service_name=service_name))
        self.service_name = service_name


class TemporaryError(TransientError):
    """Generic temporary error that should retry quickly."""

    pass


class TooManyRequestsError(TransientError):
    """Exception for too many requests (429 HTTP status)."""

    def __init__(
        self,
        message: str = "Too many requests",
        retry_after: int = None,
        context: dict = None,
    ):
        super().__init__(message, self._enrich(context, retry_after=retry_after))
        self.retry_after = retry_after


# ======= Specific Retryable Errors =======


class ResourceBusyError(RetryableError):
    """Exception for busy resources that might become available."""

    def __init__(
        self,
        message: str = "Resource is busy",
        resource_name: str = None,
        context: dict = None,
    ):
        if resource_name:
            message = f"{message}: {resource_name}"
        super().__init__(message, self._enrich(context, resource_name=resource_name))
        self.resource_name = resource_name


class DataInconsistencyError(RetryableError):
    """Exception for temporary data inconsistency."""

    def __init__(
        self,
        message: str = "Data inconsistency detected",
        data_source: str = None,
        context: dict = None,
    ):
        super().__init__(message, self._enrich(context, data_source=data_source))
        self.data_source = data_source


class ProcessingQueueFullError(RetryableError):
    """Exception for full processing queues."""

    def __init__(
        self,
        message: str = "Processing queue is full",
        queue_name: str = None,
        context: dict = None,
    ):
        super().__init__(message, self._enrich(context, queue_name=queue_name))
        self.queue_name = queue_name


# ======= Specific Permanent Errors =======


class AuthenticationError(PermanentError):
    """Exception for authentication failures."""

    def __init__(
        self,
        message: str = "Authentication failed",
        provider: str = None,
        context: dict = None,
    ):
        super().__init__(message, self._enrich(context, provider=provider))
        self.provider = provider


class PermissionError(PermanentError):
    """Exception for permission/authorization failures."""

    def __init__(
        self,
        message: str = "Permission denied",
        resource: str = None,
        action: str = None,
        context: dict = None,
    ):
        super().__init__(
            message, self._enrich(context, resource=resource, action=action)
        )
        self.resource = resource
        self.action = action


class ValidationError(PermanentError):
    """Exception for data validation failures."""

    def __init__(
        self,
        message: str = "Validation failed",
        field: str = None,
        value: str = None,
        context: dict = None,
    ):
        ctx = self._enrich(context, field=field)
        if value is not None:
            ctx["value"] = str(value)[:100]
        super().__init__(message, ctx)
        self.field = field
        self.value = value


class InvalidDataError(PermanentError):
    """Exception for invalid or malformed data."""

    def __init__(
        self,
        message: str = "Invalid data format",
        data_type: str = None,
        expected_format: str = None,
        context: dict = None,
    ):
        super().__init__(
            message,
            self._enrich(context, data_type=data_type, expected_format=expected_format),
        )
        self.data_type = data_type
        self.expected_format = expected_format


class FocusDSLError(PermanentError):
    """Exception for Focus DSL parse or execution errors."""

    def __init__(self, message: str = "Focus DSL error", context: dict = None):
        super().__init__(message, context)


class DataQualityError(PermanentError):
    """Exception raised when data fails ItemAssertion checks."""

    def __init__(
        self,
        message: str = "Data quality violation",
        violations: list = None,
        table: str = None,
        context: dict = None,
    ):
        super().__init__(
            message, self._enrich(context, table=table, violations=violations)
        )
        self.violations = violations or []
        self.table = table


class NotFoundError(PermanentError):
    """Exception for missing resources."""

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: str = None,
        resource_id: str = None,
        context: dict = None,
    ):
        super().__init__(
            message,
            self._enrich(context, resource_type=resource_type, resource_id=resource_id),
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class BadRequestError(PermanentError):
    """Exception for malformed requests."""

    def __init__(
        self,
        message: str = "Bad request",
        request_type: str = None,
        context: dict = None,
    ):
        super().__init__(message, self._enrich(context, request_type=request_type))
        self.request_type = request_type


# ======= Circuit Breaker Specific Errors =======


class CircuitBreakerOpenError(PermanentError):
    """Exception when circuit breaker is open."""

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        agent_name: str = None,
        failure_count: int = None,
        context: dict = None,
    ):
        super().__init__(
            message,
            self._enrich(context, agent_name=agent_name, failure_count=failure_count),
        )
        self.agent_name = agent_name
        self.failure_count = failure_count


# ======= Reliability Infrastructure Errors =======


class BackpressureError(RetryableError):
    """Exception when backpressure limits are exceeded."""

    def __init__(
        self,
        message: str = "Backpressure limit exceeded",
        agent_id: str = None,
        queue_size: int = None,
        context: dict = None,
    ):
        if queue_size is not None:
            message = f"{message} (queue size: {queue_size})"
        super().__init__(
            message, self._enrich(context, agent_id=agent_id, queue_size=queue_size)
        )
        self.agent_id = agent_id
        self.queue_size = queue_size


class TaskTimeoutError(TransientError):
    """Exception when a task times out."""

    def __init__(
        self,
        message: str = "Task execution timed out",
        task_id: str = None,
        timeout_duration: float = None,
        context: dict = None,
    ):
        if timeout_duration:
            message = f"{message} after {timeout_duration}s"
        super().__init__(
            message,
            self._enrich(context, task_id=task_id, timeout_duration=timeout_duration),
        )
        self.task_id = task_id
        self.timeout_duration = timeout_duration


class AcknowledgmentTimeoutError(TransientError):
    """Exception when message acknowledgment times out."""

    def __init__(
        self,
        message: str = "Message acknowledgment timed out",
        message_id: str = None,
        timeout_duration: float = None,
        context: dict = None,
    ):
        super().__init__(
            message,
            self._enrich(
                context, message_id=message_id, timeout_duration=timeout_duration
            ),
        )
        self.message_id = message_id
        self.timeout_duration = timeout_duration


class TaskNotFoundError(PermanentError):
    """Exception when a referenced task cannot be found."""

    def __init__(
        self, message: str = "Task not found", task_id: str = None, context: dict = None
    ):
        if task_id:
            message = f"{message}: {task_id}"
        super().__init__(message, self._enrich(context, task_id=task_id))
        self.task_id = task_id


class ReliabilityConfigurationError(PermanentError):
    """Exception for invalid reliability configuration."""

    def __init__(
        self,
        message: str = "Invalid reliability configuration",
        config_key: str = None,
        context: dict = None,
    ):
        if config_key:
            message = f"{message}: {config_key}"
        super().__init__(message, self._enrich(context, config_key=config_key))
        self.config_key = config_key


class DeadLetterQueueError(RetryableError):
    """Exception related to dead letter queue operations."""

    def __init__(
        self,
        message: str = "Dead letter queue operation failed",
        operation: str = None,
        context: dict = None,
    ):
        if operation:
            message = f"{message}: {operation}"
        super().__init__(message, self._enrich(context, operation=operation))
        self.operation = operation


# ======= Utility Functions =======


_TRANSIENT_EXCEPTIONS = frozenset(
    {
        "TimeoutError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "ConnectionRefusedError",
        "OSError",
        "IOError",
        "socket.timeout",
    }
)

_PERMANENT_EXCEPTIONS = frozenset(
    {
        "ValueError",
        "TypeError",
        "AttributeError",
        "KeyError",
        "IndexError",
        "NameError",
        "SyntaxError",
        "ImportError",
        "FileNotFoundError",
        "PermissionError",
    }
)


def classify_exception(exception: Exception) -> str:
    """Classify any exception to determine retry behavior."""
    if isinstance(exception, DaitaError):
        return exception.retry_hint

    name = exception.__class__.__name__
    if name in _TRANSIENT_EXCEPTIONS:
        return "transient"
    if name in _PERMANENT_EXCEPTIONS:
        return "permanent"
    return "retryable"


def create_contextual_error(
    base_exception: Exception, context: dict = None, retry_hint: str = None
) -> DaitaError:
    """Wrap a standard exception in a Daita exception with context."""
    message = str(base_exception)
    hint = retry_hint or classify_exception(base_exception)

    match hint:
        case "transient":
            return TransientError(message, context)
        case "permanent":
            return PermanentError(message, context)
        case _:
            return RetryableError(message, context)
