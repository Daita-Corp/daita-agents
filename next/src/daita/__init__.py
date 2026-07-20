"""Daita autonomous-agent v2 replacement package."""

from .agent import Agent
from .adapters import LocalDirectorySource, PostgreSQLSource, SQLiteSource
from .config import AgentConfig
from .errors import (
    AgentError,
    AuthenticationError,
    ConfigError,
    DaitaError,
    DataQualityError,
    ErrorRetryability,
    FocusDSLError,
    LLMError,
    PermanentError,
    PluginError,
    RateLimitError,
    RetryableError,
    SkillError,
    TransientError,
    ValidationError,
)
from .events import EventAudience, project_committed_event
from .extensions import (
    ConfiguredExtension,
    ExtensionBinding,
    ExtensionKind,
    ExtensionLoadError,
    ExtensionManifest,
    ExtensionRegistration,
    ExtensionRegistry,
    LocalCapability,
    RegistryDiagnostic,
    tool,
)
from .hosting import AgentHost
from .learning import LearningProposal, LearningProposalState
from .llm import (
    ModelRoute,
    ModelRouteCandidate,
    RetryPolicy,
    RetryStrategy,
    create_llm_provider,
)
from .security import (
    CompositeSecretProvider,
    EmptySecretProvider,
    EnvironmentSecretProvider,
    KeychainSecretProvider,
    SecretProvider,
    SecretReference,
    SecretResolutionError,
)
from .telemetry import (
    CommittedEventObserver,
    TelemetryExporter,
    TelemetryExportFailure,
)

__version__ = "2.0.0a0"

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentError",
    "AgentHost",
    "AuthenticationError",
    "CompositeSecretProvider",
    "CommittedEventObserver",
    "ConfigError",
    "ConfiguredExtension",
    "DaitaError",
    "DataQualityError",
    "EmptySecretProvider",
    "EnvironmentSecretProvider",
    "ErrorRetryability",
    "EventAudience",
    "ExtensionKind",
    "ExtensionBinding",
    "ExtensionLoadError",
    "ExtensionManifest",
    "ExtensionRegistration",
    "ExtensionRegistry",
    "FocusDSLError",
    "KeychainSecretProvider",
    "LLMError",
    "LearningProposal",
    "LearningProposalState",
    "LocalCapability",
    "LocalDirectorySource",
    "ModelRoute",
    "ModelRouteCandidate",
    "PermanentError",
    "PluginError",
    "PostgreSQLSource",
    "RateLimitError",
    "RegistryDiagnostic",
    "RetryPolicy",
    "RetryStrategy",
    "RetryableError",
    "SecretProvider",
    "SecretReference",
    "SecretResolutionError",
    "SkillError",
    "SQLiteSource",
    "TransientError",
    "TelemetryExporter",
    "TelemetryExportFailure",
    "ValidationError",
    "__version__",
    "create_llm_provider",
    "project_committed_event",
    "tool",
]
