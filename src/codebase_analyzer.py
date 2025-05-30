"""
Main codebase analyzer that orchestrates all components.
Provides high-level API for codebase analysis operations.
"""

import os
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path

from .config_manager import config
from .codebase_reader import CodebaseReader, CodeChunk
from .vector_store import VectorStore
from .embeddings import EmbeddingManager
from .llm_client import LLMClient


class CodebaseAnalyzer:
    """Main analyzer that orchestrates all components."""
    
    def __init__(self):
        self.reader = CodebaseReader()
        self.vector_store = VectorStore()
        self.embedding_manager = EmbeddingManager()
        self.llm_client = LLMClient()
        
        self._processed_files: Dict[str, str] = {}  # file_path -> file_hash
    
    def index_codebase(self, root_path: str, update_existing: bool = True) -> Dict[str, Any]:
        """
        Index a codebase into the vector store.
        
        Args:
            root_path: Root directory of the codebase
            update_existing: Whether to update existing files
            
        Returns:
            Dictionary with indexing results
        """
        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Directory not found: {root_path}")
        
        # Get codebase statistics
        stats = self.reader.get_codebase_stats(root_path)
        
        # Track progress
        results = {
            'total_files': stats['total_files'],
            'processed_files': 0,
            'added_chunks': 0,
            'updated_files': 0,
            'skipped_files': 0,
            'errors': []
        }
        
        # Get existing files in vector store
        existing_files = {f['file_path']: f['file_hash'] 
                         for f in self.vector_store.get_files_in_store()}
        
        # Process each file
        for chunk in self.reader.process_codebase(root_path):
            file_path = chunk.file_path
            file_hash = chunk.file_hash
            
            try:
                # Check if file needs updating
                if file_path in existing_files:
                    if not update_existing or existing_files[file_path] == file_hash:
                        results['skipped_files'] += 1
                        continue
                    else:
                        # Delete old version
                        self.vector_store.delete_by_file_path(file_path)
                        results['updated_files'] += 1
                
                # Add chunk to vector store
                self.vector_store.add_chunks([chunk])
                results['added_chunks'] += 1
                
                # Track processed file
                if file_path not in self._processed_files:
                    self._processed_files[file_path] = file_hash
                    results['processed_files'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                results['errors'].append(error_msg)
                print(error_msg)
        
        return results
    
    def add_file(self, file_path: str) -> Dict[str, Any]:
        """
        Add a single file to the vector store.
        
        Args:
            file_path: Path to the file to add
            
        Returns:
            Dictionary with operation results
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        results = {
            'file_path': file_path,
            'added_chunks': 0,
            'success': False,
            'error': None
        }
        
        try:
            # Process the file
            chunks = self.reader.process_file(file_path)
            
            if chunks:
                # Remove existing chunks for this file
                self.vector_store.delete_by_file_path(file_path)
                
                # Add new chunks
                self.vector_store.add_chunks(chunks)
                results['added_chunks'] = len(chunks)
                results['success'] = True
                
                # Update tracking
                file_hash = self.reader.get_file_hash(file_path)
                self._processed_files[file_path] = file_hash
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Error adding file {file_path}: {e}")
        
        return results
    
    def remove_file(self, file_path: str) -> Dict[str, Any]:
        """
        Remove a file from the vector store.
        
        Args:
            file_path: Path of the file to remove
            
        Returns:
            Dictionary with operation results
        """
        results = {
            'file_path': file_path,
            'deleted_chunks': 0,
            'success': False
        }
        
        try:
            deleted_count = self.vector_store.delete_by_file_path(file_path)
            results['deleted_chunks'] = deleted_count
            results['success'] = deleted_count > 0
            
            # Remove from tracking
            if file_path in self._processed_files:
                del self._processed_files[file_path]
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Error removing file {file_path}: {e}")
        
        return results
    
    def search_code(self, query: str, n_results: int = 10, 
                   language_filter: Optional[str] = None,
                   file_path_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for code chunks matching a query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            language_filter: Filter by programming language
            file_path_filter: Filter by file path pattern
            
        Returns:
            List of matching code chunks
        """
        return self.vector_store.search(
            query=query,
            n_results=n_results,
            language_filter=language_filter,
            file_path_filter=file_path_filter
        )
    
    def ask_question(self, question: str, n_context_chunks: int = 5) -> str:
        """
        Ask a question about the codebase.
        
        Args:
            question: Question to ask
            n_context_chunks: Number of relevant chunks to include as context
            
        Returns:
            LLM response
        """
        # Search for relevant context
        search_results = self.search_code(question, n_results=n_context_chunks)
        
        # Get LLM response with context
        return self.llm_client.ask_question(question, search_results)
    
    def explain_code_chunk(self, chunk_id: str) -> Optional[str]:
        """
        Get an explanation of a specific code chunk.
        
        Args:
            chunk_id: ID of the chunk to explain
            
        Returns:
            LLM explanation or None if chunk not found
        """
        chunk = self.vector_store.get_chunk_by_id(chunk_id)
        if not chunk:
            return None
        
        return self.llm_client.explain_code(chunk)
    
    def suggest_improvements(self, chunk_id: str) -> Optional[str]:
        """
        Get improvement suggestions for a specific code chunk.
        
        Args:
            chunk_id: ID of the chunk to analyze
            
        Returns:
            LLM suggestions or None if chunk not found
        """
        chunk = self.vector_store.get_chunk_by_id(chunk_id)
        if not chunk:
            return None
        
        return self.llm_client.suggest_improvements(chunk)
    
    def find_security_issues(self, chunk_id: str) -> Optional[str]:
        """
        Analyze a code chunk for security issues.
        
        Args:
            chunk_id: ID of the chunk to analyze
            
        Returns:
            Security analysis or None if chunk not found
        """
        chunk = self.vector_store.get_chunk_by_id(chunk_id)
        if not chunk:
            return None
        
        return self.llm_client.find_security_issues(chunk)
    
    def generate_documentation(self, chunk_id: str) -> Optional[str]:
        """
        Generate documentation for a specific code chunk.
        
        Args:
            chunk_id: ID of the chunk to document
            
        Returns:
            Generated documentation or None if chunk not found
        """
        chunk = self.vector_store.get_chunk_by_id(chunk_id)
        if not chunk:
            return None
        
        return self.llm_client.generate_documentation(chunk)
    
    def find_similar_code(self, code_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find code chunks similar to the given code text.
        
        Args:
            code_text: Code to find similar chunks for
            n_results: Number of similar chunks to return
            
        Returns:
            List of similar code chunks
        """
        return self.vector_store.search(code_text, n_results=n_results)
    
    def get_codebase_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the indexed codebase.
        
        Returns:
            Dictionary with codebase statistics and summary
        """
        # Get vector store stats
        store_stats = self.vector_store.get_collection_stats()
        
        # Get file information
        files_info = self.vector_store.get_files_in_store()
        
        # Calculate additional metrics
        total_lines = sum(
            f.get('total_size', 0) for f in files_info
        ) // 50  # Rough estimate of lines (50 chars per line average)
        
        summary = {
            'total_chunks': store_stats['total_chunks'],
            'unique_files': store_stats['unique_files'],
            'languages': store_stats['languages'],
            'total_size_bytes': store_stats['total_size_bytes'],
            'estimated_lines': total_lines,
            'embedding_provider': self.embedding_manager.provider_type,
            'llm_provider': self.llm_client.provider_type,
            'files': files_info
        }
        
        return summary
    
    def check_for_updates(self, root_path: str) -> List[Dict[str, Any]]:
        """
        Check which files in the codebase have been modified since last indexing.
        
        Args:
            root_path: Root directory of the codebase
            
        Returns:
            List of files that need updating
        """
        updates_needed = []
        
        # Get current files
        current_files = self.reader.scan_codebase(root_path)
        
        # Get existing files in vector store
        existing_files = {f['file_path']: f['file_hash'] 
                         for f in self.vector_store.get_files_in_store()}
        
        for file_path in current_files:
            current_hash = self.reader.get_file_hash(file_path)
            
            if file_path not in existing_files:
                # New file
                updates_needed.append({
                    'file_path': file_path,
                    'status': 'new',
                    'language': self.reader.detect_language(file_path)
                })
            elif existing_files[file_path] != current_hash:
                # Modified file
                updates_needed.append({
                    'file_path': file_path,
                    'status': 'modified',
                    'language': self.reader.detect_language(file_path)
                })
        
        # Check for deleted files
        for file_path in existing_files:
            if file_path not in current_files:
                updates_needed.append({
                    'file_path': file_path,
                    'status': 'deleted',
                    'language': 'unknown'
                })
        
        return updates_needed
    
    def clear_index(self) -> bool:
        """
        Clear all indexed data.
        
        Returns:
            True if successful
        """
        try:
            self.vector_store.clear_collection()
            self._processed_files.clear()
            return True
        except Exception as e:
            print(f"Error clearing index: {e}")
            return False
    
    def switch_embedding_provider(self, provider: str) -> bool:
        """
        Switch to a different embedding provider.
        
        Args:
            provider: Name of the embedding provider
            
        Returns:
            True if successful
        """
        try:
            self.embedding_manager.switch_provider(provider)
            self.vector_store.embedding_manager = self.embedding_manager
            return True
        except Exception as e:
            print(f"Error switching embedding provider: {e}")
            return False
    
    def switch_llm_provider(self, provider: str) -> bool:
        """
        Switch to a different LLM provider.
        
        Args:
            provider: Name of the LLM provider
            
        Returns:
            True if successful
        """
        try:
            self.llm_client.switch_provider(provider)
            return True
        except Exception as e:
            print(f"Error switching LLM provider: {e}")
            return False 