"""
Flexible embedding manager with dynamic provider switching and graceful fallbacks.
"""

from typing import List, Optional, Dict, Any
from .provider_manager import provider_manager, ProviderInfo
from .config_manager import config


class FlexibleEmbeddingManager:
    """Flexible embedding manager with dynamic provider switching."""
    
    def __init__(self):
        self.current_provider = None
        self.provider_name = None
        self._initialize_best_provider()
    
    def _initialize_best_provider(self):
        """Initialize the best available provider."""
        # Try to get the configured provider first
        preferred_provider = None
        try:
            if config.embeddings.default_provider == "sentence_transformers":
                preferred_provider = "sentence_transformers"
            elif config.embeddings.default_provider == "openai":
                preferred_provider = "openai_embedding"
        except:
            pass
        
        # Get best available provider
        available = provider_manager.get_available_providers()
        embedding_providers = [p for p in available if "embedding" in p.name or p.name == "sentence_transformers"]
        
        # Try preferred first
        if preferred_provider:
            for provider_info in embedding_providers:
                if provider_info.name == preferred_provider and provider_info.is_available:
                    if self._switch_to_provider(provider_info.name):
                        return
        
        # Try any available provider
        for provider_info in embedding_providers:
            if provider_info.is_available:
                if self._switch_to_provider(provider_info.name):
                    return
        
        print("Warning: No embedding providers available")
    
    def _switch_to_provider(self, provider_name: str) -> bool:
        """Switch to a specific provider."""
        try:
            # Get configuration for the provider
            init_params = self._get_provider_config(provider_name)
            
            provider = provider_manager.get_provider(provider_name, **init_params)
            if provider:
                self.current_provider = provider
                self.provider_name = provider_name
                print(f"Initialized embedding provider: {provider.display_name}")
                return True
        except Exception as e:
            print(f"Failed to switch to provider {provider_name}: {e}")
        
        return False
    
    def _get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration parameters for a provider."""
        params = {}
        
        try:
            if provider_name == "sentence_transformers":
                params = {
                    "model_name": config.embeddings.sentence_transformers.get("model_name", "all-MiniLM-L6-v2")
                }
            elif provider_name == "openai_embedding":
                params = {
                    "model": config.embeddings.openai.get("model", "text-embedding-ada-002"),
                    "api_key_env": config.embeddings.openai.get("api_key_env", "OPENAI_API_KEY")
                }
        except:
            pass
        
        return params
    
    def get_available_providers(self) -> List[ProviderInfo]:
        """Get list of available embedding providers."""
        available = provider_manager.get_available_providers()
        return [p for p in available if "embedding" in p.name or p.name == "sentence_transformers"]
    
    def switch_provider(self, provider_name: str) -> bool:
        """Switch to a different embedding provider."""
        if provider_name == self.provider_name:
            return True
        
        return self._switch_to_provider(provider_name)
    
    @property
    def provider_type(self) -> Optional[str]:
        """Get current provider type."""
        return self.provider_name
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self.current_provider and hasattr(self.current_provider, 'dimension'):
            return self.current_provider.dimension
        return 384  # Default fallback
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        if not self.current_provider:
            return None
        
        if hasattr(self.current_provider, 'embed_text'):
            return self.current_provider.embed_text(text)
        return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        if not self.current_provider:
            return [None] * len(texts)
        
        if hasattr(self.current_provider, 'embed_batch'):
            return self.current_provider.embed_batch(texts, batch_size)
        
        # Fallback to individual embeddings
        return [self.embed_text(text) for text in texts]
    
    def is_available(self) -> bool:
        """Check if embedding manager is available."""
        return self.current_provider is not None 