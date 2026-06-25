"""
Core interfaces for Daita Agents.

Defines the essential contracts that components must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


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
