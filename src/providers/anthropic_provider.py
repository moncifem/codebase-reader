"""
Anthropic LLM provider with flexible initialization.
"""

from typing import Optional
import os
from ..provider_manager import BaseProvider


class AnthropicLLMProvider(BaseProvider):
    """Anthropic LLM provider."""
    
    def __init__(self):
        self.client = None
        self.model = "claude-3-haiku-20240307"
        self.api_key_env = "ANTHROPIC_API_KEY"
        self.max_tokens = 1000
        self.temperature = 0.1
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    @property
    def display_name(self) -> str:
        return "Anthropic Claude"
    
    @property
    def requires_api_key(self) -> bool:
        return True
    
    def is_available(self) -> tuple[bool, Optional[str]]:
        """Check if Anthropic is available."""
        try:
            import anthropic
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                return False, f"API key not found in environment: {self.api_key_env}"
            return True, None
        except ImportError:
            return False, "anthropic package not installed. Run: pip install anthropic"
    
    def initialize(self, model: str = None, api_key_env: str = None, 
                   max_tokens: int = None, temperature: float = None, **kwargs) -> bool:
        """Initialize the Anthropic client."""
        try:
            import anthropic
            
            self.model = model or self.model
            self.api_key_env = api_key_env or self.api_key_env
            self.max_tokens = max_tokens or self.max_tokens
            self.temperature = temperature or self.temperature
            
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                return False
            
            self.client = anthropic.Anthropic(api_key=api_key)
            return True
            
        except Exception as e:
            print(f"Failed to initialize Anthropic: {e}")
            return False
    
    def generate_response(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate a response from Claude."""
        if not self.client:
            return None
        
        try:
            params = {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs
            }
            
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            return response.content[0].text
            
        except Exception as e:
            return f"Error generating Anthropic response: {str(e)}" 