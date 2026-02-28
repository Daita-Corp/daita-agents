"""
Base class for database plugins.

Provides common connection management, error handling, and context manager
support for all database plugins in the Daita framework.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from ..core.exceptions import PluginError, ConnectionError as DaitaConnectionError, ValidationError

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)

class BaseDatabasePlugin(ABC):
    """
    Base class for all database plugins with common connection management.
    
    This class provides:
    - Standardized connection/disconnection lifecycle
    - Context manager support for automatic cleanup
    - Common error handling patterns
    - Consistent configuration patterns
    
    Database-specific plugins should inherit from this class and implement
    the abstract methods for their specific database requirements.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize base database plugin.
        
        Args:
            **kwargs: Database-specific configuration parameters
        """
        # Common connection state
        self._connection = None
        self._pool = None
        self._client = None
        self._db = None
        
        # Connection configuration
        self.config = kwargs
        self.timeout = kwargs.get('timeout', 30)
        self.max_retries = kwargs.get('max_retries', 3)
        
        logger.debug(f"{self.__class__.__name__} initialized with config keys: {list(kwargs.keys())}")
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to the database.
        
        This method must be implemented by each database plugin to handle
        the specific connection logic for that database type.
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the database and clean up resources.
        
        This method must be implemented by each database plugin to handle
        the specific disconnection and cleanup logic for that database type.
        """
        pass
    
    @property
    def is_connected(self) -> bool:
        """
        Check if the database connection is active.
        
        Returns:
            True if connected, False otherwise
        """
        return (
            self._connection is not None or 
            self._pool is not None or 
            self._client is not None
        )
    
    async def __aenter__(self):
        """Async context manager entry - automatically connect."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - automatically disconnect."""
        await self.disconnect()
    
    def _validate_connection(self) -> None:
        """
        Validate that the database connection is available.
        
        Raises:
            ValidationError: If not connected to database
        """
        if not self.is_connected:
            raise ValidationError(
                f"{self.__class__.__name__} is not connected to database",
                field="connection_state"
            )
    
    def _handle_connection_error(self, error: Exception, operation: str) -> None:
        """
        Handle database connection errors with consistent logging.
        
        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            
        Raises:
            PluginError: Wrapped database error with context
        """
        error_msg = f"{self.__class__.__name__} {operation} failed: {str(error)}"
        logger.error(error_msg)
        
        # Choose appropriate exception type based on the original error
        if isinstance(error, ImportError):
            # Missing dependency - permanent error
            raise PluginError(
                error_msg,
                plugin_name=self.__class__.__name__,
                retry_hint="permanent",
                context={"operation": operation, "original_error": str(error)}
            ) from error
        else:
            # Connection issues - typically transient
            raise DaitaConnectionError(
                error_msg,
                context={"plugin": self.__class__.__name__, "operation": operation}
            ) from error
    
    def get_tools(self) -> List['AgentTool']:
        """
        Get agent-usable tools from this database plugin.

        Override in subclasses to expose database operations as LLM tools.

        Returns:
            List of AgentTool instances
        """
        return []

    @property
    def has_tools(self) -> bool:
        """Check if plugin exposes any tools"""
        return len(self.get_tools()) > 0

    @property
    def info(self) -> Dict[str, Any]:
        """
        Get information about the database plugin.

        Returns:
            Dictionary with plugin information
        """
        return {
            'plugin_type': self.__class__.__name__,
            'connected': self.is_connected,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'config_keys': list(self.config.keys())
        }