"""
Flexible provider manager for embeddings and LLMs.
Handles dynamic provider discovery, lazy initialization, and graceful fallbacks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type
import os
import importlib
from dataclasses import dataclass
from .config_manager import config


@dataclass
class ProviderInfo:
    """Information about a provider."""
    name: str
    display_name: str
    requires_api_key: bool
    api_key_env: Optional[str]
    is_available: bool
    error_message: Optional[str] = None


class BaseProvider(ABC):
    """Base class for all providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name."""
        pass
    
    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Whether this provider requires an API key."""
        pass
    
    @abstractmethod
    def is_available(self) -> tuple[bool, Optional[str]]:
        """Check if provider is available. Returns (is_available, error_message)."""
        pass
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize the provider. Returns True if successful."""
        pass


class ProviderManager:
    """Manages dynamic provider discovery and initialization."""
    
    def __init__(self):
        self._providers: Dict[str, Type[BaseProvider]] = {}
        self._initialized_providers: Dict[str, BaseProvider] = {}
        self._register_builtin_providers()
    
    def _register_builtin_providers(self):
        """Register built-in providers."""
        # Import and register providers dynamically
        try:
            from .providers.sentence_transformers_provider import SentenceTransformersEmbeddingProvider
            self.register_provider("sentence_transformers", SentenceTransformersEmbeddingProvider)
        except ImportError:
            pass
        
        try:
            from .providers.openai_provider import OpenAIEmbeddingProvider, OpenAILLMProvider
            self.register_provider("openai_embedding", OpenAIEmbeddingProvider)
            self.register_provider("openai_llm", OpenAILLMProvider)
        except ImportError:
            pass
        
        try:
            from .providers.anthropic_provider import AnthropicLLMProvider
            self.register_provider("anthropic", AnthropicLLMProvider)
        except ImportError:
            pass
    
    def register_provider(self, name: str, provider_class: Type[BaseProvider]):
        """Register a new provider."""
        self._providers[name] = provider_class
    
    def get_available_providers(self, provider_type: str = None) -> List[ProviderInfo]:
        """Get list of available providers."""
        available = []
        
        for name, provider_class in self._providers.items():
            if provider_type and not name.startswith(provider_type):
                continue
                
            try:
                provider_instance = provider_class()
                is_available, error_msg = provider_instance.is_available()
                
                info = ProviderInfo(
                    name=name,
                    display_name=provider_instance.display_name,
                    requires_api_key=provider_instance.requires_api_key,
                    api_key_env=getattr(provider_instance, 'api_key_env', None),
                    is_available=is_available,
                    error_message=error_msg
                )
                available.append(info)
            except Exception as e:
                info = ProviderInfo(
                    name=name,
                    display_name=name.title(),
                    requires_api_key=True,
                    api_key_env=None,
                    is_available=False,
                    error_message=str(e)
                )
                available.append(info)
        
        return available
    
    def get_provider(self, name: str, **kwargs) -> Optional[BaseProvider]:
        """Get an initialized provider instance."""
        if name in self._initialized_providers:
            return self._initialized_providers[name]
        
        if name not in self._providers:
            return None
        
        try:
            provider_class = self._providers[name]
            provider_instance = provider_class()
            
            if provider_instance.initialize(**kwargs):
                self._initialized_providers[name] = provider_instance
                return provider_instance
        except Exception as e:
            print(f"Failed to initialize provider {name}: {e}")
        
        return None
    
    def switch_provider(self, old_name: str, new_name: str, **kwargs) -> bool:
        """Switch from one provider to another."""
        # Remove old provider
        if old_name in self._initialized_providers:
            del self._initialized_providers[old_name]
        
        # Initialize new provider
        new_provider = self.get_provider(new_name, **kwargs)
        return new_provider is not None
    
    def get_best_available_provider(self, provider_type: str, preferred: List[str] = None) -> Optional[str]:
        """Get the best available provider of a given type."""
        available = self.get_available_providers(provider_type)
        available_names = [p.name for p in available if p.is_available]
        
        if not available_names:
            return None
        
        # Try preferred providers first
        if preferred:
            for pref in preferred:
                if pref in available_names:
                    return pref
        
        # Return first available
        return available_names[0]


# Global provider manager instance
provider_manager = ProviderManager() 