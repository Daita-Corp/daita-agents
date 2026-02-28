"""
Runtime settings management.

Simplified settings system focused on essential configuration.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, ClassVar
from pydantic import BaseModel
from dotenv import load_dotenv


def _find_dotenv_file() -> Optional[Path]:
    """
    Search current and parent directories for .env file.

    Searches up to 10 levels up from current working directory.
    This improves developer experience when running from subdirectories.

    Returns:
        Path to .env file if found, None otherwise
    """
    current = Path.cwd()
    for _ in range(10):
        env_file = current / '.env'
        if env_file.exists():
            return env_file
        if current.parent == current:
            break
        current = current.parent
    return None


# Load environment variables from .env (search parent dirs)
_env_file = _find_dotenv_file()
if _env_file:
    load_dotenv(_env_file)
else:
    load_dotenv()  # Fallback to default behavior

class Settings(BaseModel):
    """Runtime settings for Daita framework."""
    
    # API settings (SECURE - from environment only)
    api_key: Optional[str] = None
    api_endpoint: str = ""  # Must be set via DAITA_API_ENDPOINT env var
    dashboard_url: str = ""  # Must be set via DAITA_DASHBOARD_URL env var
    
    # Deployment settings
    deployment_id: Optional[str] = None
    environment: str = "development"
    
    # Local mode settings
    local_mode: bool = True
    cache_dir: Path = Path.home() / ".daita" / "cache"
    log_level: str = "INFO"
    
    # LLM Provider API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None
    grok_api_key: Optional[str] = None
    
    # LLM settings
    default_model: str = "gpt-4"
    default_provider: str = "openai"
    default_temperature: float = 0.7
    
    def validate_endpoint(self, endpoint: str) -> str:
        """Validate API endpoint for security."""
        if not endpoint:
            raise ValueError("API endpoint cannot be empty")
        
        if not endpoint.startswith('https://'):
            raise ValueError("API endpoint must use HTTPS")
        
        return endpoint
    
    def __init__(self, **data):
        # Override with environment variables
        env_overrides = {}
        
        # Lambda environment detection - use /tmp for cache
        if os.getenv('AWS_LAMBDA_FUNCTION_NAME') or os.getenv('DAITA_RUNTIME') == 'lambda':
            env_overrides["cache_dir"] = Path("/tmp/.daita/cache")
        
        # Main Daita API key
        if os.getenv("DAITA_API_KEY"):
            env_overrides["api_key"] = os.getenv("DAITA_API_KEY")
        
        # API endpoints - REQUIRED from environment
        api_endpoint = os.getenv("DAITA_API_ENDPOINT")
        if api_endpoint:
            try:
                self.validate_endpoint(api_endpoint)
                env_overrides["api_endpoint"] = api_endpoint
            except ValueError as e:
                raise ValueError(f"Invalid DAITA_API_ENDPOINT environment variable: {e}")
        
        dashboard_url = os.getenv("DAITA_DASHBOARD_URL") 
        if dashboard_url:
            try:
                self.validate_endpoint(dashboard_url)
                env_overrides["dashboard_url"] = dashboard_url
            except ValueError as e:
                raise ValueError(f"Invalid DAITA_DASHBOARD_URL environment variable: {e}")
        
        # Deployment settings
        if os.getenv("DAITA_DEPLOYMENT_ID"):
            env_overrides["deployment_id"] = os.getenv("DAITA_DEPLOYMENT_ID")
        
        if os.getenv("DAITA_ENVIRONMENT"):
            env_overrides["environment"] = os.getenv("DAITA_ENVIRONMENT")
        
        # LLM Provider API Keys
        if os.getenv("OPENAI_API_KEY"):
            env_overrides["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("ANTHROPIC_API_KEY"):
            env_overrides["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
        
        if os.getenv("GOOGLE_API_KEY"):
            env_overrides["google_api_key"] = os.getenv("GOOGLE_API_KEY")
        
        if os.getenv("GEMINI_API_KEY"):
            env_overrides["gemini_api_key"] = os.getenv("GEMINI_API_KEY")
        
        if os.getenv("XAI_API_KEY"):
            env_overrides["xai_api_key"] = os.getenv("XAI_API_KEY")
        
        if os.getenv("GROK_API_KEY"):
            env_overrides["grok_api_key"] = os.getenv("GROK_API_KEY")
        
        # General settings
        if os.getenv("DAITA_LOCAL_MODE"):
            env_overrides["local_mode"] = os.getenv("DAITA_LOCAL_MODE").lower() == "true"
        
        if os.getenv("DAITA_LOG_LEVEL"):
            env_overrides["log_level"] = os.getenv("DAITA_LOG_LEVEL")
        
        if os.getenv("DAITA_DEFAULT_MODEL"):
            env_overrides["default_model"] = os.getenv("DAITA_DEFAULT_MODEL")
        
        if os.getenv("DAITA_DEFAULT_PROVIDER"):
            env_overrides["default_provider"] = os.getenv("DAITA_DEFAULT_PROVIDER")
        
        if os.getenv("DAITA_DEFAULT_TEMPERATURE"):
            try:
                env_overrides["default_temperature"] = float(os.getenv("DAITA_DEFAULT_TEMPERATURE"))
            except (ValueError, TypeError):
                pass  # Use default if invalid
        
        # Merge with provided data
        super().__init__(**{**data, **env_overrides})
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # SECURITY: Only validate endpoints in Lambda runtime
        # Customer deployments use hardcoded production endpoints
        if os.getenv('DAITA_RUNTIME') == 'lambda':
            if not self.api_endpoint:
                raise ValueError("DAITA_API_ENDPOINT environment variable required for Lambda runtime")
            if not self.dashboard_url:
                raise ValueError("DAITA_DASHBOARD_URL environment variable required for Lambda runtime")
    
    def get_llm_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a specific LLM provider.
        
        Args:
            provider: Provider name (openai, anthropic, google, gemini, xai, grok)
            
        Returns:
            API key for the provider or None if not set
        """
        provider = provider.lower()
        
        if provider == "openai":
            return self.openai_api_key
        elif provider == "anthropic":
            return self.anthropic_api_key
        elif provider in ["google", "gemini"]:
            return self.google_api_key or self.gemini_api_key
        elif provider in ["xai", "grok"]:
            return self.xai_api_key or self.grok_api_key
        
        return None
    
    def get_daita_api_key(self) -> Optional[str]:
        """
        Get Daita API key with fallback to provider keys.
        
        Returns:
            Daita API key or first available provider key
        """
        return (
            self.api_key or
            self.openai_api_key or
            self.anthropic_api_key or
            self.google_api_key or
            self.gemini_api_key or
            self.xai_api_key or
            self.grok_api_key
        )

# Global settings instance
settings = Settings()