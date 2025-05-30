"""
Configuration manager for the codebase reader application.
Handles loading and managing application settings from config.yaml.
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class VectorDBConfig:
    type: str
    persist_directory: str
    collection_name: str
    distance_metric: str = "cosine"  # Default to cosine distance


@dataclass
class EmbeddingConfig:
    default_provider: str
    sentence_transformers: Dict[str, Any]
    openai: Dict[str, Any]


@dataclass
class LLMConfig:
    default_provider: str
    openai: Dict[str, Any]
    anthropic: Optional[Dict[str, Any]] = None  # Optional for backward compatibility


@dataclass
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    max_file_size_mb: int


@dataclass
class UIConfig:
    page_title: str
    page_icon: str
    layout: str


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config: Optional[Dict[str, Any]] = None
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    @property
    def vector_db(self) -> VectorDBConfig:
        """Get vector database configuration."""
        config_data = self._config['vector_db']
        # Ensure backward compatibility - add default distance_metric if not present
        if 'distance_metric' not in config_data:
            config_data['distance_metric'] = 'cosine'
        return VectorDBConfig(**config_data)
    
    @property
    def embeddings(self) -> EmbeddingConfig:
        """Get embeddings configuration."""
        config = self._config['embeddings']
        return EmbeddingConfig(**config)
    
    @property
    def llm(self) -> LLMConfig:
        """Get LLM configuration."""
        config_data = self._config['llm']
        # Ensure backward compatibility - add default anthropic config if not present
        if 'anthropic' not in config_data:
            config_data['anthropic'] = None
        return LLMConfig(**config_data)
    
    @property
    def chunking(self) -> ChunkingConfig:
        """Get chunking configuration."""
        config = self._config['chunking']
        return ChunkingConfig(**config)
    
    @property
    def ui(self) -> UIConfig:
        """Get UI configuration."""
        config = self._config['ui']
        return UIConfig(**config)
    
    @property
    def supported_extensions(self) -> list:
        """Get list of supported file extensions."""
        return self._config['supported_extensions']
    
    @property
    def ignore_patterns(self) -> list:
        """Get list of patterns to ignore when scanning codebases."""
        return self._config['ignore_patterns']
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """Get API key from environment variables."""
        return os.getenv(key_name)
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update a configuration value."""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        with open(self.config_path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)


# Global configuration instance
config = ConfigManager() 