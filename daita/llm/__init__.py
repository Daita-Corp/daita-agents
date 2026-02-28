"""
LLM provider integrations for Daita Agents.

This module provides a unified interface for different LLM providers:
- OpenAI (GPT-4, GPT-3.5-turbo, etc.)
- Anthropic (Claude models)
- Google Gemini
- xAI Grok
- Mock provider for testing

The factory pattern allows easy switching between providers while maintaining
a consistent interface for agents.

Usage:
    ```python
    from daita.llm import create_llm_provider
    
    # Create OpenAI provider
    llm = create_llm_provider("openai", "gpt-4", api_key="sk-...")
    response = await llm.generate("Hello, world!")
    
    # Create Anthropic provider  
    llm = create_llm_provider("anthropic", "claude-3-sonnet-20240229")
    response = await llm.generate("Analyze this data...")
    ```
"""

# Factory and registry functions
from .factory import (
    create_llm_provider,
    register_llm_provider,
    list_available_providers
)

# Base class for custom providers
from .base import BaseLLMProvider

# Concrete provider implementations
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .grok import GrokProvider
from .gemini import GeminiProvider
from .mock import MockLLMProvider

__all__ = [
    # Factory functions
    "create_llm_provider",
    "register_llm_provider", 
    "list_available_providers",
    
    # Base class
    "BaseLLMProvider",
    
    # Provider implementations
    "OpenAIProvider",
    "AnthropicProvider",
    "GrokProvider",
    "GeminiProvider",
    "MockLLMProvider",
]