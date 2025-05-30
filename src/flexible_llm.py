"""
Flexible LLM client with dynamic provider switching and graceful fallbacks.
"""

from typing import List, Optional, Dict, Any
from .provider_manager import provider_manager, ProviderInfo
from .config_manager import config


class FlexibleLLMClient:
    """Flexible LLM client with dynamic provider switching."""
    
    def __init__(self):
        self.current_provider = None
        self.provider_name = None
        self._initialize_best_provider()
    
    def _initialize_best_provider(self):
        """Initialize the best available provider."""
        # Try to get the configured provider first
        preferred_provider = None
        try:
            if config.llm.default_provider == "anthropic":
                preferred_provider = "anthropic"
            elif config.llm.default_provider == "openai":
                preferred_provider = "openai_llm"
        except:
            pass
        
        # Get best available provider
        available = provider_manager.get_available_providers()
        llm_providers = [p for p in available if "llm" in p.name or p.name == "anthropic"]
        
        # Try preferred first
        if preferred_provider:
            for provider_info in llm_providers:
                if provider_info.name == preferred_provider and provider_info.is_available:
                    if self._switch_to_provider(provider_info.name):
                        return
        
        # Try any available provider
        for provider_info in llm_providers:
            if provider_info.is_available:
                if self._switch_to_provider(provider_info.name):
                    return
        
        print("Warning: No LLM providers available")
    
    def _switch_to_provider(self, provider_name: str) -> bool:
        """Switch to a specific provider."""
        try:
            # Get configuration for the provider
            init_params = self._get_provider_config(provider_name)
            
            provider = provider_manager.get_provider(provider_name, **init_params)
            if provider:
                self.current_provider = provider
                self.provider_name = provider_name
                print(f"Initialized LLM provider: {provider.display_name}")
                return True
        except Exception as e:
            print(f"Failed to switch to provider {provider_name}: {e}")
        
        return False
    
    def _get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration parameters for a provider."""
        params = {}
        
        try:
            if provider_name == "openai_llm":
                params = {
                    "model": config.llm.openai.get("model", "gpt-3.5-turbo"),
                    "api_key_env": config.llm.openai.get("api_key_env", "OPENAI_API_KEY"),
                    "max_tokens": config.llm.openai.get("max_tokens", 1000),
                    "temperature": config.llm.openai.get("temperature", 0.1)
                }
            elif provider_name == "anthropic":
                if config.llm.anthropic:
                    params = {
                        "model": config.llm.anthropic.get("model", "claude-3-haiku-20240307"),
                        "api_key_env": config.llm.anthropic.get("api_key_env", "ANTHROPIC_API_KEY"),
                        "max_tokens": config.llm.anthropic.get("max_tokens", 1000),
                        "temperature": config.llm.anthropic.get("temperature", 0.1)
                    }
        except:
            pass
        
        return params
    
    def get_available_providers(self) -> List[ProviderInfo]:
        """Get list of available LLM providers."""
        available = provider_manager.get_available_providers()
        return [p for p in available if "llm" in p.name or p.name == "anthropic"]
    
    def switch_provider(self, provider_name: str) -> bool:
        """Switch to a different LLM provider."""
        if provider_name == self.provider_name:
            return True
        
        return self._switch_to_provider(provider_name)
    
    @property
    def provider_type(self) -> Optional[str]:
        """Get current provider type."""
        return self.provider_name
    
    def generate_response(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate a response from the LLM."""
        if not self.current_provider:
            return "No LLM provider available"
        
        if hasattr(self.current_provider, 'generate_response'):
            return self.current_provider.generate_response(prompt, **kwargs)
        return "LLM provider does not support response generation"
    
    def query(self, prompt: str, **kwargs) -> str:
        """Send a query to the LLM."""
        result = self.generate_response(prompt, **kwargs)
        return result or "No response generated"
    
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
    
    def is_available(self) -> bool:
        """Check if LLM client is available."""
        return self.current_provider is not None 