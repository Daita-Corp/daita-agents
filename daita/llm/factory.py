"""
Factory for creating LLM provider instances.
"""
import logging
from typing import Optional

from ..core.exceptions import LLMError
from .base import BaseLLMProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .grok import GrokProvider
from .gemini import GeminiProvider
from .mock import MockLLMProvider

logger = logging.getLogger(__name__)

# Registry of available providers
PROVIDER_REGISTRY = {
    'openai': OpenAIProvider,
    'anthropic': AnthropicProvider,
    'grok': GrokProvider,
    'gemini': GeminiProvider,
    'mock': MockLLMProvider,
}

def create_llm_provider(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Factory function to create LLM provider instances.
    
    Args:
        provider: Provider name ('openai', 'anthropic', 'grok', 'gemini', 'mock')
        model: Model identifier
        api_key: API key for authentication
        agent_id: Agent ID for token tracking
        **kwargs: Additional provider-specific parameters
        
    Returns:
        LLM provider instance
        
    Raises:
        LLMError: If provider is not supported
        
    Examples:
        >>> # Create OpenAI provider with token tracking
        >>> llm = create_llm_provider('openai', 'gpt-4', api_key='sk-...', agent_id='my_agent')
        
        >>> # Create Anthropic provider with token tracking
        >>> llm = create_llm_provider('anthropic', 'claude-3-sonnet-20240229', agent_id='my_agent')
        
        >>> # Create Grok provider
        >>> llm = create_llm_provider('grok', 'grok-beta', api_key='xai-...', agent_id='my_agent')
        
        >>> # Create Gemini provider
        >>> llm = create_llm_provider('gemini', 'gemini-1.5-pro', api_key='AIza...', agent_id='my_agent')
        
        >>> # Create mock provider for testing
        >>> llm = create_llm_provider('mock', 'test-model', agent_id='test_agent')
    """
    provider_name = provider.lower()
    
    if provider_name not in PROVIDER_REGISTRY:
        available_providers = list(PROVIDER_REGISTRY.keys())
        raise LLMError(
            f"Unsupported LLM provider: {provider}. "
            f"Available providers: {available_providers}"
        )
    
    provider_class = PROVIDER_REGISTRY[provider_name]
    
    try:
        # Pass agent_id to provider for token tracking
        return provider_class(
            model=model, 
            api_key=api_key, 
            agent_id=agent_id,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Failed to create {provider} provider: {str(e)}")
        raise LLMError(f"Failed to create {provider} provider: {str(e)}")

def register_llm_provider(name: str, provider_class) -> None:
    """
    Register a custom LLM provider.
    
    Args:
        name: Provider name
        provider_class: Provider class that implements LLMProvider interface
    """
    PROVIDER_REGISTRY[name.lower()] = provider_class
    logger.info(f"Registered custom LLM provider: {name}")

def list_available_providers() -> list:
    """Get list of available LLM providers."""
    return list(PROVIDER_REGISTRY.keys())