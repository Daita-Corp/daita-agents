"""
Base class for vector database plugins.

Provides minimal abstract interface for vector operations across different providers.
"""
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from .base_db import BaseDatabasePlugin


class BaseVectorPlugin(BaseDatabasePlugin):
    """
    Base class for vector database plugins with common interface.

    This class provides a minimal abstraction for vector operations while
    allowing each plugin to expose provider-native features and filter syntax.

    Vector-specific plugins should inherit from this class and implement
    the abstract methods for their specific vector database.
    """

    @abstractmethod
    async def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Insert or update vectors with associated metadata.

        Args:
            ids: List of unique identifiers for the vectors
            vectors: List of vector embeddings (each vector is a list of floats)
            metadata: Optional list of metadata dictionaries (one per vector)
            **kwargs: Provider-specific parameters

        Returns:
            Dictionary with operation results (format varies by provider)
        """
        pass

    @abstractmethod
    async def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Any] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            vector: Query vector (list of floats)
            top_k: Maximum number of results to return
            filter: Provider-specific filter syntax
            **kwargs: Provider-specific parameters

        Returns:
            List of results with scores and metadata
        """
        pass

    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Delete vectors by ID or filter.

        Args:
            ids: Optional list of IDs to delete
            filter: Optional provider-specific filter
            **kwargs: Provider-specific parameters

        Returns:
            Dictionary with deletion results
        """
        pass

    @abstractmethod
    async def fetch(
        self,
        ids: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch vectors by ID.

        Args:
            ids: List of IDs to fetch
            **kwargs: Provider-specific parameters

        Returns:
            List of vectors with metadata
        """
        pass
