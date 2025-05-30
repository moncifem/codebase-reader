"""
LLM client module for flexible language model interactions.
Supports multiple LLM providers for code analysis and question answering.
"""

from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

from .config_manager import config


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate a response to a prompt with optional context."""
        pass
    
    @abstractmethod
    def analyze_code(self, code: str, question: str) -> str:
        """Analyze code and answer a specific question about it."""
        pass


class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI's API."""
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or config.llm.openai['model']
        self.api_key = api_key or config.get_api_key(config.llm.openai['api_key_env'])
        self.max_tokens = config.llm.openai.get('max_tokens', 1000)
        self.temperature = config.llm.openai.get('temperature', 0.1)
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required but not provided")
        
        self._client = None
    
    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package is required for OpenAI LLM")
        return self._client
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate a response to a prompt with optional context."""
        messages = []
        
        if context:
            messages.append({
                "role": "system",
                "content": f"Use the following context to answer the user's question:\n\n{context}"
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def analyze_code(self, code: str, question: str) -> str:
        """Analyze code and answer a specific question about it."""
        prompt = f"""
Analyze the following code and answer the question:

Code:
```
{code}
```

Question: {question}

Please provide a detailed analysis and answer.
"""
        
        return self.generate_response(prompt)


class LLMClient:
    """Main LLM client that manages different providers."""
    
    def __init__(self, provider_type: Optional[str] = None):
        self.provider_type = provider_type or config.llm.default_provider
        self._provider = None
    
    @property
    def provider(self) -> LLMProvider:
        """Get the current LLM provider."""
        if self._provider is None:
            self._provider = self._create_provider()
        return self._provider
    
    def _create_provider(self) -> LLMProvider:
        """Create an LLM provider based on configuration."""
        if self.provider_type == "openai":
            return OpenAIProvider()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider_type}")
    
    def switch_provider(self, provider_type: str) -> None:
        """Switch to a different LLM provider."""
        if provider_type not in ["openai"]:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")
        
        self.provider_type = provider_type
        self._provider = None  # Reset to force recreation
    
    def ask_question(self, question: str, search_results: List[Dict[str, Any]] = None) -> str:
        """
        Ask a question with optional search results as context.
        
        Args:
            question: The user's question
            search_results: List of relevant code chunks from vector search
            
        Returns:
            LLM response
        """
        if not search_results:
            return self.provider.generate_response(question)
        
        # Build context from search results
        context_parts = []
        for i, result in enumerate(search_results[:5]):  # Limit to top 5 results
            metadata = result.get('metadata', {})
            file_path = metadata.get('file_path', 'Unknown')
            language = metadata.get('language', 'unknown')
            start_line = metadata.get('start_line', 0)
            end_line = metadata.get('end_line', 0)
            similarity = result.get('similarity', 0)
            
            context_parts.append(f"""
## Code Chunk {i+1} (Similarity: {similarity:.2f})
**File:** {file_path}
**Language:** {language}
**Lines:** {start_line}-{end_line}

```{language}
{result['content']}
```
""")
        
        context = "\n".join(context_parts)
        
        enhanced_prompt = f"""
Based on the following code chunks from the codebase, please answer this question:

{question}

Please provide a comprehensive answer that references specific parts of the code when relevant.
"""
        
        return self.provider.generate_response(enhanced_prompt, context)
    
    def explain_code(self, code_chunk: Dict[str, Any]) -> str:
        """
        Explain what a specific code chunk does.
        
        Args:
            code_chunk: Dictionary containing code content and metadata
            
        Returns:
            Explanation of the code
        """
        metadata = code_chunk.get('metadata', {})
        file_path = metadata.get('file_path', 'Unknown')
        language = metadata.get('language', 'unknown')
        
        question = f"""
Please explain what this {language} code does:

File: {file_path}
Language: {language}

Provide a clear explanation of:
1. What the code does
2. Key functions or classes
3. Important logic or algorithms
4. Any notable patterns or design decisions
"""
        
        return self.provider.analyze_code(code_chunk['content'], question)
    
    def suggest_improvements(self, code_chunk: Dict[str, Any]) -> str:
        """
        Suggest improvements for a code chunk.
        
        Args:
            code_chunk: Dictionary containing code content and metadata
            
        Returns:
            Suggestions for code improvements
        """
        metadata = code_chunk.get('metadata', {})
        language = metadata.get('language', 'unknown')
        
        question = f"""
Please analyze this {language} code and suggest improvements:

Focus on:
1. Code quality and readability
2. Performance optimizations
3. Security considerations
4. Best practices for {language}
5. Potential bugs or issues

Please provide specific, actionable suggestions with explanations.
"""
        
        return self.provider.analyze_code(code_chunk['content'], question)
    
    def find_security_issues(self, code_chunk: Dict[str, Any]) -> str:
        """
        Analyze code for potential security issues.
        
        Args:
            code_chunk: Dictionary containing code content and metadata
            
        Returns:
            Security analysis and recommendations
        """
        metadata = code_chunk.get('metadata', {})
        language = metadata.get('language', 'unknown')
        
        question = f"""
Please analyze this {language} code for security vulnerabilities:

Look for:
1. Input validation issues
2. SQL injection vulnerabilities
3. XSS vulnerabilities
4. Authentication/authorization issues
5. Data exposure risks
6. Cryptographic issues
7. Other common security patterns for {language}

Provide specific findings and remediation suggestions.
"""
        
        return self.provider.analyze_code(code_chunk['content'], question)
    
    def generate_documentation(self, code_chunk: Dict[str, Any]) -> str:
        """
        Generate documentation for a code chunk.
        
        Args:
            code_chunk: Dictionary containing code content and metadata
            
        Returns:
            Generated documentation
        """
        metadata = code_chunk.get('metadata', {})
        file_path = metadata.get('file_path', 'Unknown')
        language = metadata.get('language', 'unknown')
        
        question = f"""
Please generate comprehensive documentation for this {language} code:

File: {file_path}

Include:
1. Overview of what the code does
2. Function/class descriptions
3. Parameter descriptions
4. Return value descriptions
5. Usage examples
6. Any important notes or warnings

Format the documentation appropriately for {language} (e.g., docstrings for Python, JSDoc for JavaScript).
"""
        
        return self.provider.analyze_code(code_chunk['content'], question)
    
    def compare_implementations(self, code_chunks: List[Dict[str, Any]]) -> str:
        """
        Compare multiple code implementations.
        
        Args:
            code_chunks: List of code chunks to compare
            
        Returns:
            Comparison analysis
        """
        if len(code_chunks) < 2:
            return "At least 2 code chunks are required for comparison."
        
        # Build comparison prompt
        comparison_text = "Please compare the following code implementations:\n\n"
        
        for i, chunk in enumerate(code_chunks[:3]):  # Limit to 3 chunks
            metadata = chunk.get('metadata', {})
            file_path = metadata.get('file_path', f'Unknown {i+1}')
            language = metadata.get('language', 'unknown')
            
            comparison_text += f"""
## Implementation {i+1}
**File:** {file_path}
**Language:** {language}

```{language}
{chunk['content']}
```
"""
        
        comparison_text += """
Please analyze and compare these implementations focusing on:
1. Functionality differences
2. Performance characteristics
3. Code quality and maintainability
4. Design patterns used
5. Pros and cons of each approach
6. Recommendations for which to use when
"""
        
        return self.provider.generate_response(comparison_text)
    
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