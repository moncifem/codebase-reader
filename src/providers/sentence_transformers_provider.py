"""
SentenceTransformers embedding provider with flexible initialization.
"""

from typing import List, Optional, Union
import numpy as np
from ..provider_manager import BaseProvider


class SentenceTransformersEmbeddingProvider(BaseProvider):
    """SentenceTransformers embedding provider."""
    
    def __init__(self):
        self.model = None
        self.model_name = "all-MiniLM-L6-v2"
        self._dimension = None
    
    @property
    def name(self) -> str:
        return "sentence_transformers"
    
    @property
    def display_name(self) -> str:
        return "SentenceTransformers (Local)"
    
    @property
    def requires_api_key(self) -> bool:
        return False
    
    def is_available(self) -> tuple[bool, Optional[str]]:
        """Check if SentenceTransformers is available."""
        try:
            import sentence_transformers
            return True, None
        except ImportError:
            return False, "sentence-transformers package not installed. Run: pip install sentence-transformers"
    
    def initialize(self, model_name: str = None, **kwargs) -> bool:
        """Initialize the SentenceTransformers model."""
        try:
            import sentence_transformers
            import torch
            
            self.model_name = model_name or self.model_name
            
            # Determine device (prefer GPU if available, else CPU)
            # Forcing CPU to debug 'meta tensor' issue
            device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"SentenceTransformers: Using device: {device}")

            self.model = sentence_transformers.SentenceTransformer(self.model_name, device=device)
            
            # Get model dimension
            self._dimension = self.model.get_sentence_embedding_dimension()
            return True
            
        except Exception as e:
            print(f"Failed to initialize SentenceTransformers: {e}")
            return False
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None and self.model:
            self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension or 384  # Default for all-MiniLM-L6-v2
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        if not self.model:
            return None
        
        try:
            text = self._preprocess_text(text)
            if not text:
                return None
            
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        if not self.model:
            return [None] * len(texts)
        
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Filter out empty texts but keep track of original indices
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(processed_texts):
                if text:
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if not valid_texts:
                return [None] * len(texts)
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=False
            )
            
            # Map back to original order
            result = [None] * len(texts)
            for i, embedding in enumerate(embeddings):
                original_index = valid_indices[i]
                result[original_index] = embedding.tolist()
            
            return result
            
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding."""
        if not text or not text.strip():
            return ""
        
        # Remove excessive whitespace and newlines
        text = " ".join(text.split())
        return text.strip() 