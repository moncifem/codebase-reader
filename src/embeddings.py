"""
Embeddings module for generating vector representations of code chunks.
Supports multiple embedding providers: sentence transformers and OpenAI.
"""

import os
from typing import List, Optional, Union
from abc import ABC, abstractmethod
import numpy as np

from .config_manager import config


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the dimension of the embeddings."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using SentenceTransformers (local/offline)."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.embeddings.sentence_transformers['model_name']
        self._model = None
        self._dimension = None
    
    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                # Get dimension by encoding a dummy text
                dummy_embedding = self._model.encode("test")
                self._dimension = len(dummy_embedding)
            except ImportError:
                raise ImportError("sentence-transformers package is required for local embeddings")
            except Exception as e:
                raise RuntimeError(f"Failed to load SentenceTransformer model: {e}")
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if self._dimension is None:
            # Trigger model loading to get dimension
            _ = self.model
        return self._dimension
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            return [0.0] * self.dimension
        
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []
        
        # Filter out empty texts and keep track of indices
        non_empty_texts = []
        indices_map = []
        
        for i, text in enumerate(texts):
            if text.strip():
                non_empty_texts.append(text)
                indices_map.append(i)
        
        if not non_empty_texts:
            return [[0.0] * self.dimension] * len(texts)
        
        # Generate embeddings for non-empty texts
        embeddings = self.model.encode(non_empty_texts)
        
        # Map back to original order, filling empty texts with zero vectors
        result = []
        non_empty_idx = 0
        
        for i, text in enumerate(texts):
            if text.strip() and non_empty_idx < len(embeddings):
                result.append(embeddings[non_empty_idx].tolist())
                non_empty_idx += 1
            else:
                result.append([0.0] * self.dimension)
        
        return result


class OpenAIProvider(EmbeddingProvider):
    """Embedding provider using OpenAI's API."""
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or config.embeddings.openai['model']
        self.api_key = api_key or config.get_api_key(config.embeddings.openai['api_key_env'])
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required but not provided")
        
        self._client = None
        self._dimension = None
    
    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package is required for OpenAI embeddings")
        return self._client
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if self._dimension is None:
            # OpenAI ada-002 has 1536 dimensions
            if "ada-002" in self.model:
                self._dimension = 1536
            else:
                # Fallback: make a test request to get dimension
                test_embedding = self.embed_text("test")
                self._dimension = len(test_embedding)
        return self._dimension
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            return [0.0] * self.dimension
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embedding: {e}")
            return [0.0] * self.dimension
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []
        
        # Filter out empty texts
        non_empty_texts = [text for text in texts if text.strip()]
        
        if not non_empty_texts:
            return [[0.0] * self.dimension] * len(texts)
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=non_empty_texts
            )
            embeddings = [item.embedding for item in response.data]
            
            # Map back to original order
            result = []
            embedding_idx = 0
            
            for text in texts:
                if text.strip() and embedding_idx < len(embeddings):
                    result.append(embeddings[embedding_idx])
                    embedding_idx += 1
                else:
                    result.append([0.0] * self.dimension)
            
            return result
            
        except Exception as e:
            print(f"Error generating OpenAI embeddings: {e}")
            return [[0.0] * self.dimension] * len(texts)


class EmbeddingManager:
    """Manages embedding providers and provides a unified interface."""
    
    def __init__(self, provider_type: Optional[str] = None):
        self.provider_type = provider_type or config.embeddings.default_provider
        self._provider = None
    
    @property
    def provider(self) -> EmbeddingProvider:
        """Get the current embedding provider."""
        if self._provider is None:
            self._provider = self._create_provider()
        return self._provider
    
    def _create_provider(self) -> EmbeddingProvider:
        """Create an embedding provider based on configuration."""
        if self.provider_type == "sentence_transformers":
            return SentenceTransformerProvider()
        elif self.provider_type == "openai":
            return OpenAIProvider()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider_type}")
    
    def switch_provider(self, provider_type: str) -> None:
        """Switch to a different embedding provider."""
        if provider_type not in ["sentence_transformers", "openai"]:
            raise ValueError(f"Unsupported embedding provider: {provider_type}")
        
        self.provider_type = provider_type
        self._provider = None  # Reset to force recreation
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.provider.embed_text(text)
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for a batch of texts with optional batching."""
        if len(texts) <= batch_size:
            return self.provider.embed_batch(texts)
        
        # Process in smaller batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.provider.embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the current embedding provider."""
        return self.provider.dimension
    
    def get_available_providers(self) -> List[str]:
        """Get list of available embedding providers."""
        available = []
        
        # Check sentence transformers
        try:
            import sentence_transformers
            available.append("sentence_transformers")
        except ImportError:
            pass
        
        # Check OpenAI
        if config.get_api_key(config.embeddings.openai['api_key_env']):
            try:
                import openai
                available.append("openai")
            except ImportError:
                pass
        
        return available 