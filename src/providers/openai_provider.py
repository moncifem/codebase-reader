"""
OpenAI providers for embeddings and LLM with flexible initialization.
"""

from typing import List, Optional
import os
from ..provider_manager import BaseProvider


class OpenAIEmbeddingProvider(BaseProvider):
    """OpenAI embedding provider."""
    
    def __init__(self):
        self.client = None
        self.model = "text-embedding-ada-002"
        self.api_key_env = "OPENAI_API_KEY"
        self._dimension = 1536
    
    @property
    def name(self) -> str:
        return "openai_embedding"
    
    @property
    def display_name(self) -> str:
        return "OpenAI Embeddings"
    
    @property
    def requires_api_key(self) -> bool:
        return True
    
    def is_available(self) -> tuple[bool, Optional[str]]:
        """Check if OpenAI is available."""
        try:
            import openai
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                return False, f"API key not found in environment: {self.api_key_env}"
            return True, None
        except ImportError:
            return False, "openai package not installed. Run: pip install openai"
    
    def initialize(self, model: str = None, api_key_env: str = None, **kwargs) -> bool:
        """Initialize the OpenAI client."""
        try:
            import openai
            
            self.model = model or self.model
            self.api_key_env = api_key_env or self.api_key_env
            
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                return False
            
            self.client = openai.OpenAI(api_key=api_key)
            return True
            
        except Exception as e:
            print(f"Failed to initialize OpenAI embeddings: {e}")
            return False
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        if not self.client:
            return None
        
        try:
            text = self._preprocess_text(text)
            if not text:
                return None
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Error generating OpenAI embedding: {e}")
            return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        if not self.client:
            return [None] * len(texts)
        
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Process in batches
            results = [None] * len(texts)
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                valid_batch = []
                valid_indices = []
                
                for j, text in enumerate(batch):
                    if text:
                        valid_batch.append(text)
                        valid_indices.append(i + j)
                
                if not valid_batch:
                    continue
                
                response = self.client.embeddings.create(
                    model=self.model,
                    input=valid_batch
                )
                
                for k, embedding_data in enumerate(response.data):
                    original_index = valid_indices[k]
                    results[original_index] = embedding_data.embedding
            
            return results
            
        except Exception as e:
            print(f"Error generating OpenAI batch embeddings: {e}")
            return [None] * len(texts)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding."""
        if not text or not text.strip():
            return ""
        
        # OpenAI recommends removing newlines for better performance
        text = text.replace("\n", " ").replace("\r", " ")
        text = " ".join(text.split())
        return text.strip()


class OpenAILLMProvider(BaseProvider):
    """OpenAI LLM provider."""
    
    def __init__(self):
        self.client = None
        self.model = "gpt-3.5-turbo"
        self.api_key_env = "OPENAI_API_KEY"
        self.max_tokens = 1000
        self.temperature = 0.1
    
    @property
    def name(self) -> str:
        return "openai_llm"
    
    @property
    def display_name(self) -> str:
        return "OpenAI GPT"
    
    @property
    def requires_api_key(self) -> bool:
        return True
    
    def is_available(self) -> tuple[bool, Optional[str]]:
        """Check if OpenAI is available."""
        try:
            import openai
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                return False, f"API key not found in environment: {self.api_key_env}"
            return True, None
        except ImportError:
            return False, "openai package not installed. Run: pip install openai"
    
    def initialize(self, model: str = None, api_key_env: str = None, 
                   max_tokens: int = None, temperature: float = None, **kwargs) -> bool:
        """Initialize the OpenAI client."""
        try:
            import openai
            
            self.model = model or self.model
            self.api_key_env = api_key_env or self.api_key_env
            self.max_tokens = max_tokens or self.max_tokens
            self.temperature = temperature or self.temperature
            
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                return False
            
            self.client = openai.OpenAI(api_key=api_key)
            return True
            
        except Exception as e:
            print(f"Failed to initialize OpenAI LLM: {e}")
            return False
    
    def generate_response(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate a response from the LLM."""
        if not self.client:
            return None
        
        try:
            params = {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs
            }
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating OpenAI response: {str(e)}" 