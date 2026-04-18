"""
Core interfaces for Daita Agents.

Defines the essential contracts that components must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AgentABC(ABC):
    """Abstract base interface for all agents. Use Agent or BaseAgent for concrete implementations."""

    @abstractmethod
    async def _process(
        self,
        task: str,
        data: Any = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """INTERNAL: Process a task with data and context."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the agent."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the agent."""
        pass


class LLMProvider(ABC):
    """Interface for language model providers."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass

    @property
    @abstractmethod
    def info(self) -> Dict[str, Any]:
        """Get information about the LLM provider."""
        pass

    def set_agent_id(self, agent_id: str) -> None:
        """Set agent ID for tracing context. Override in custom providers."""
        pass


class EmbeddingProvider(ABC):
    """Interface for embedding providers."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings in a batch."""
        pass
