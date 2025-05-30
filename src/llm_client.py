"""
LLM client for code analysis and Q&A.
Supports multiple LLM providers including OpenAI and Anthropic.
"""

from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import json
from .config_manager import config


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def analyze_code(self, code: str, question: str) -> str:
        """Analyze code and answer a specific question about it."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, model: str, api_key: str, **kwargs):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
            self.default_params = kwargs
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        params = {**self.default_params, **kwargs}
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider."""
    
    def __init__(self, model: str, api_key: str, **kwargs):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
            self.default_params = kwargs
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic API."""
        params = {**self.default_params, **kwargs}
        
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating response: {str(e)}"


class LLMClient:
    """Main LLM client that manages different providers."""
    
    def __init__(self):
        self.provider = None
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the LLM provider based on configuration."""
        llm_config = config.llm
        provider_type = llm_config.default_provider
        
        if provider_type == "openai":
            openai_config = llm_config.openai
            api_key = config.get_api_key(openai_config["api_key_env"])
            if not api_key:
                raise ValueError(f"API key not found in environment: {openai_config['api_key_env']}")
            
            self.provider = OpenAIProvider(
                model=openai_config["model"],
                api_key=api_key,
                max_tokens=openai_config.get("max_tokens", 1000),
                temperature=openai_config.get("temperature", 0.1)
            )
        
        elif provider_type == "anthropic":
            if not llm_config.anthropic:
                raise ValueError("Anthropic configuration not found in config file")
                
            anthropic_config = llm_config.anthropic
            api_key = config.get_api_key(anthropic_config["api_key_env"])
            if not api_key:
                raise ValueError(f"API key not found in environment: {anthropic_config['api_key_env']}")
            
            self.provider = AnthropicProvider(
                model=anthropic_config["model"],
                api_key=api_key,
                max_tokens=anthropic_config.get("max_tokens", 1000),
                temperature=anthropic_config.get("temperature", 0.1)
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")
    
    def query(self, prompt: str, **kwargs) -> str:
        """Send a query to the LLM."""
        if not self.provider:
            return "LLM provider not initialized"
        return self.provider.generate_response(prompt, **kwargs)
    
    def explain_code(self, code: str, language: str = "unknown") -> str:
        """Explain what a piece of code does."""
        prompt = f"""
Please explain the following {language} code in a clear and concise way:

```{language}
{code}
```

Provide a high-level explanation of what this code does, its purpose, and any important details.
"""
        return self.query(prompt)
    
    def suggest_improvements(self, code: str, language: str = "unknown") -> str:
        """Suggest improvements for a piece of code."""
        prompt = f"""
Please analyze the following {language} code and suggest improvements:

```{language}
{code}
```

Focus on:
1. Code quality and readability
2. Performance optimizations
3. Best practices
4. Potential bugs or issues
5. Security considerations
"""
        return self.query(prompt)
    
    def find_security_issues(self, code: str, language: str = "unknown") -> str:
        """Analyze code for potential security vulnerabilities."""
        prompt = f"""
Please analyze the following {language} code for potential security vulnerabilities:

```{language}
{code}
```

Look for:
1. Input validation issues
2. SQL injection vulnerabilities
3. XSS vulnerabilities
4. Authentication/authorization flaws
5. Data exposure risks
6. Other security concerns

Provide specific recommendations for fixing any issues found.
"""
        return self.query(prompt)
    
    def generate_documentation(self, code: str, language: str = "unknown") -> str:
        """Generate documentation for a piece of code."""
        prompt = f"""
Please generate comprehensive documentation for the following {language} code:

```{language}
{code}
```

Include:
1. Function/class descriptions
2. Parameter descriptions
3. Return value descriptions
4. Usage examples
5. Any important notes or warnings
"""
        return self.query(prompt)
    
    def compare_implementations(self, code1: str, code2: str, language: str = "unknown") -> str:
        """Compare two code implementations."""
        prompt = f"""
Please compare these two {language} implementations:

Implementation 1:
```{language}
{code1}
```

Implementation 2:
```{language}
{code2}
```

Analyze:
1. Functionality differences
2. Performance implications
3. Code quality and readability
4. Maintainability
5. Which approach is better and why
"""
        return self.query(prompt)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        available = []
        
        # Check OpenAI
        if config.get_api_key(config.llm.openai['api_key_env']):
            try:
                import openai
                available.append("openai")
            except ImportError:
                pass
        
        return available 