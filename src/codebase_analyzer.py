"""
Main codebase analyzer with flexible provider system and graceful fallbacks.
Orchestrates code reading, embedding, vector storage, and LLM analysis.
"""

import os
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path

from .config_manager import config
from .codebase_reader import CodebaseReader, CodeChunk
from .vector_store import VectorStore
from .flexible_embeddings import FlexibleEmbeddingManager
from .flexible_llm import FlexibleLLMClient
from .provider_manager import provider_manager


class CodebaseAnalyzer:
    """
    Main analyzer that orchestrates all components with flexible provider management.
    """
    
    def __init__(self):
        """Initialize the analyzer with flexible providers."""
        print("Initializing Codebase Analyzer...")
        
        # Initialize core components
        self.reader = CodebaseReader()
        
        # Initialize flexible providers with graceful fallbacks
        self.embedding_manager = FlexibleEmbeddingManager()
        self.llm_client = FlexibleLLMClient()
        
        # Initialize vector store only if we have embeddings
        self.vector_store = None
        if self.embedding_manager.is_available():
            try:
                self.vector_store = VectorStore(
                    persist_directory=config.vector_db.persist_directory,
                    collection_name=config.vector_db.collection_name
                )
            except Exception as e:
                print(f"Warning: Failed to initialize vector store: {e}")
        
        self._processed_files: Dict[str, str] = {}  # file_path -> file_hash
        self._print_status()
    
    def _print_status(self):
        """Print initialization status."""
        print("\n=== Codebase Analyzer Status ===")
        print(f"📖 Code Reader: ✅ Ready")
        
        if self.embedding_manager.is_available():
            print(f"🔢 Embeddings: ✅ {self.embedding_manager.current_provider.display_name}")
        else:
            print(f"🔢 Embeddings: ❌ No providers available")
        
        if self.llm_client.is_available():
            print(f"🤖 LLM: ✅ {self.llm_client.current_provider.display_name}")
        else:
            print(f"🤖 LLM: ❌ No providers available")
        
        if self.vector_store:
            print(f"💾 Vector Store: ✅ ChromaDB")
        else:
            print(f"💾 Vector Store: ❌ Not available")
        print("================================\n")
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get detailed provider status information."""
        status = {
            "embedding_providers": [],
            "llm_providers": [],
            "current_embedding": None,
            "current_llm": None,
            "vector_store_available": self.vector_store is not None
        }
        
        # Get embedding provider status
        embedding_providers = self.embedding_manager.get_available_providers()
        for provider in embedding_providers:
            status["embedding_providers"].append({
                "name": provider.name,
                "display_name": provider.display_name,
                "available": provider.is_available,
                "requires_api_key": provider.requires_api_key,
                "error": provider.error_message
            })
        
        # Get LLM provider status
        llm_providers = self.llm_client.get_available_providers()
        for provider in llm_providers:
            status["llm_providers"].append({
                "name": provider.name,
                "display_name": provider.display_name,
                "available": provider.is_available,
                "requires_api_key": provider.requires_api_key,
                "error": provider.error_message
            })
        
        # Current providers
        if self.embedding_manager.current_provider:
            status["current_embedding"] = {
                "name": self.embedding_manager.provider_name,
                "display_name": self.embedding_manager.current_provider.display_name
            }
        
        if self.llm_client.current_provider:
            status["current_llm"] = {
                "name": self.llm_client.provider_name,
                "display_name": self.llm_client.current_provider.display_name
            }
        
        return status
    
    def switch_embedding_provider(self, provider_name: str) -> bool:
        """Switch to a different embedding provider."""
        success = self.embedding_manager.switch_provider(provider_name)
        
        if success and self.embedding_manager.is_available():
            # Reinitialize vector store
            try:
                self.vector_store = VectorStore(
                    persist_directory=config.vector_db.persist_directory,
                    collection_name=config.vector_db.collection_name
                )
                print(f"✅ Switched to embedding provider: {self.embedding_manager.current_provider.display_name}")
                return True
            except Exception as e:
                print(f"❌ Failed to reinitialize vector store: {e}")
                return False
        
        return success
    
    def switch_llm_provider(self, provider_name: str) -> bool:
        """Switch to a different LLM provider."""
        success = self.llm_client.switch_provider(provider_name)
        
        if success:
            print(f"✅ Switched to LLM provider: {self.llm_client.current_provider.display_name}")
        
        return success
    
    def index_codebase(self, codebase_path: str, update_existing: bool = True) -> Dict[str, Any]:
        """
        Index a codebase with flexible provider support.
        
        Args:
            codebase_path: Path to the codebase directory
            update_existing: Whether to update existing files
            
        Returns:
            Dictionary with indexing results
        """
        if not self.embedding_manager.is_available():
            return {
                "success": False,
                "error": "No embedding providers available",
                "processed_files": 0,
                "added_chunks": 0,
                "updated_files": 0,
                "skipped_files": 0,
                "errors": [],
                "indexed_files": []
            }
        
        if not self.vector_store:
            return {
                "success": False,
                "error": "Vector store not available",
                "processed_files": 0,
                "added_chunks": 0,
                "updated_files": 0,
                "skipped_files": 0,
                "errors": [],
                "indexed_files": []
            }
        
        print(f"Starting codebase indexing: {codebase_path}")
        
        # Read and chunk the codebase
        code_chunks = list(self.reader.process_codebase(codebase_path))
        
        if not code_chunks:
            return {
                "success": False,
                "error": "No code chunks found",
                "processed_files": 0,
                "added_chunks": 0,
                "updated_files": 0,
                "skipped_files": 0,
                "errors": [],
                "indexed_files": []
            }
        
        print(f"Found {len(code_chunks)} code chunks")
        
        # Group chunks by file for processing
        files_processed = set()
        added_chunks = 0
        updated_files = 0
        skipped_files = 0
        errors = []
        
        # Group chunks by file path to handle updates properly
        chunks_by_file = {}
        for chunk in code_chunks:
            file_path = chunk.file_path
            if file_path not in chunks_by_file:
                chunks_by_file[file_path] = []
            chunks_by_file[file_path].append(chunk)
        
        # Process each file
        for file_path, file_chunks in chunks_by_file.items():
            try:
                # Generate embeddings for all chunks in this file
                texts = [chunk.content for chunk in file_chunks]
                embeddings = self.embedding_manager.embed_batch(texts)
                
                # Filter out chunks with failed embeddings
                valid_chunks = []
                valid_embeddings = []
                for chunk, embedding in zip(file_chunks, embeddings):
                    if embedding is not None:
                        valid_chunks.append(chunk)
                        valid_embeddings.append(embedding)
                    else:
                        errors.append(f"Failed to generate embedding for chunk in {chunk.file_path}")
                
                if valid_chunks:
                    # If update_existing is True, delete existing chunks for this file first
                    if update_existing:
                        deleted_count = self.vector_store.delete_by_file_path(file_path)
                        if deleted_count > 0:
                            updated_files += 1
                    
                    # Add the chunks with embeddings to the vector store
                    # We need to add embeddings to the chunks - let's modify the VectorStore call
                    self._add_chunks_with_embeddings(valid_chunks, valid_embeddings)
                    
                    added_chunks += len(valid_chunks)
                    files_processed.add(file_path)
                
            except Exception as e:
                error_msg = f"Error processing file {file_path}: {str(e)}"
                errors.append(error_msg)
                print(f"❌ {error_msg}")

        result = {
            "success": True,
            "processed_files": len(files_processed),
            "added_chunks": added_chunks,
            "updated_files": updated_files,
            "skipped_files": skipped_files,
            "errors": errors,
            "indexed_files": list(files_processed)
        }
        
        print(f"✅ Indexing complete: {added_chunks} chunks from {len(files_processed)} files")
        return result
    
    def _add_chunks_with_embeddings(self, chunks: List, embeddings: List[List[float]]) -> None:
        """Add chunks with their embeddings to the vector store."""
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.content)
            
            # Prepare metadata (ChromaDB doesn't support nested dicts well)
            metadata = {
                'file_path': chunk.file_path,
                'start_line': chunk.start_line,
                'end_line': chunk.end_line,
                'language': chunk.language,
                'file_hash': chunk.file_hash,
                'chunk_index': chunk.chunk_index,
                'file_size': chunk.metadata.get('file_size', 0),
                'total_lines': chunk.metadata.get('total_lines', 0),
                'chunk_size': chunk.metadata.get('chunk_size', 0),
                'embedding_model': self.embedding_manager.provider_type
            }
            metadatas.append(metadata)
        
        # Add to ChromaDB directly
        try:
            self.vector_store.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        except Exception as e:
            print(f"Error adding chunks to vector store: {e}")
            # Fallback: try to add them one by one to handle duplicates
            for i, (chunk_id, doc, metadata, embedding) in enumerate(zip(ids, documents, metadatas, embeddings)):
                try:
                    self.vector_store.collection.add(
                        ids=[chunk_id],
                        documents=[doc],
                        metadatas=[metadata],
                        embeddings=[embedding]
                    )
                except Exception as e2:
                    print(f"Failed to add individual chunk {chunk_id}: {e2}")
    
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
                   language_filter: str = None, file_path_filter: str = None) -> List[Dict[str, Any]]:
        """
        Search for code using semantic similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            language_filter: Filter by programming language
            file_path_filter: Filter by file path pattern
            
        Returns:
            List of search results with similarity scores
        """
        if not self.embedding_manager.is_available() or not self.vector_store:
            return []
        
        try:
            # Use VectorStore's search method which handles embedding generation internally
            results = self.vector_store.search(
                query=query,
                embedding_manager=self.embedding_manager,
                n_results=n_results,
                language_filter=language_filter,
                file_path_filter=file_path_filter
            )
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def ask_question(self, question: str, n_context_chunks: int = 5) -> str:
        """
        Ask a question about the codebase with context.
        
        Args:
            question: The question to ask
            n_context_chunks: Number of relevant code chunks to include as context
            
        Returns:
            AI-generated response
        """
        if not self.llm_client.is_available():
            return "❌ No LLM providers available. Please configure an API key or check your configuration."
        
        # Get relevant code chunks for context
        context_chunks = []
        if self.embedding_manager.is_available() and self.vector_store:
            context_chunks = self.search_code(question, n_context_chunks)
        
        # Build context for the LLM
        if context_chunks:
            context_text = "\n\n".join([
                f"File: {chunk['metadata']['file_path']}\n"
                f"Language: {chunk['metadata']['language']}\n"
                f"Lines: {chunk['metadata']['start_line']}-{chunk['metadata']['end_line']}\n"
                f"Code:\n{chunk['content']}"
                for chunk in context_chunks
            ])
            
            prompt = f"""
Based on the following code from the codebase, please answer the question:

QUESTION: {question}

RELEVANT CODE:
{context_text}

Please provide a comprehensive answer that references the specific code when relevant.
"""
        else:
            prompt = f"""
Please answer the following question about a codebase:

QUESTION: {question}

Note: No specific code context is available, so please provide a general answer based on common programming practices.
"""
        
        return self.llm_client.query(prompt)
    
    def explain_code_chunk(self, chunk_id: str) -> Optional[str]:
        """Explain a specific code chunk."""
        if not self.llm_client.is_available():
            return "❌ No LLM providers available"
        
        if not self.vector_store:
            return "❌ Vector store not available"
        
        chunk_data = self.vector_store.get_chunk_by_id(chunk_id)
        if not chunk_data:
            return "❌ Chunk not found"
        
        metadata = chunk_data['metadata']
        return self.llm_client.explain_code(
            chunk_data['content'], 
            metadata.get('language', 'unknown')
        )
    
    def suggest_improvements(self, chunk_id: str) -> Optional[str]:
        """Suggest improvements for a specific code chunk."""
        if not self.llm_client.is_available():
            return "❌ No LLM providers available"
        
        if not self.vector_store:
            return "❌ Vector store not available"
        
        chunk_data = self.vector_store.get_chunk_by_id(chunk_id)
        if not chunk_data:
            return "❌ Chunk not found"
        
        metadata = chunk_data['metadata']
        return self.llm_client.suggest_improvements(
            chunk_data['content'], 
            metadata.get('language', 'unknown')
        )
    
    def find_security_issues(self, chunk_id: str) -> Optional[str]:
        """Find security issues in a specific code chunk."""
        if not self.llm_client.is_available():
            return "❌ No LLM providers available"
        
        if not self.vector_store:
            return "❌ Vector store not available"
        
        chunk_data = self.vector_store.get_chunk_by_id(chunk_id)
        if not chunk_data:
            return "❌ Chunk not found"
        
        metadata = chunk_data['metadata']
        return self.llm_client.find_security_issues(
            chunk_data['content'], 
            metadata.get('language', 'unknown')
        )
    
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
        if not self.embedding_manager.is_available() or not self.vector_store:
            return []
        
        return self.vector_store.search(
            query=code_text, 
            embedding_manager=self.embedding_manager,
            n_results=n_results
        )
    
    def get_codebase_summary(self) -> Dict[str, Any]:
        """Get a summary of the indexed codebase."""
        if not self.vector_store:
            return {
                "total_chunks": 0,
                "unique_files": 0,
                "languages": {},
                "total_size_bytes": 0,
                "files": []
            }
        
        return self.vector_store.get_collection_stats()
    
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
        """Clear the vector index."""
        if not self.vector_store:
            return False
        
        try:
            self.vector_store.clear_collection()
            self._processed_files.clear()
            print("✅ Index cleared successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to clear index: {e}")
            return False 