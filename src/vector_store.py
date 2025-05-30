"""
Vector store module for managing ChromaDB operations.
Handles storage and retrieval of code chunks with their embeddings.
Based on best practices from embeddings and vector database research.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import uuid

from .config_manager import config
from .codebase_reader import CodeChunk
from .embeddings import EmbeddingManager


class VectorStore:
    """Manages vector database operations using ChromaDB."""
    
    def __init__(self, collection_name: Optional[str] = None, persist_directory: Optional[str] = None):
        self.collection_name = collection_name or config.vector_db.collection_name
        self.persist_directory = persist_directory or config.vector_db.persist_directory
        self.distance_metric = getattr(config.vector_db, 'distance_metric', 'cosine')
        
        self._client = None
        self._collection = None
        self.embedding_manager = EmbeddingManager()
    
    @property
    def client(self):
        """Lazy load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                # Create persist directory if it doesn't exist
                os.makedirs(self.persist_directory, exist_ok=True)
                
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
            except ImportError:
                raise ImportError("chromadb package is required for vector storage")
        return self._client
    
    @property
    def collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            try:
                # Try to get existing collection
                self._collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                # Create new collection with optimized settings
                # Using cosine distance as recommended for text embeddings
                collection_metadata = {
                    "hnsw:space": self.distance_metric,
                    "description": "Codebase chunks with semantic embeddings"
                }
                
                self._collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata=collection_metadata
                )
        return self._collection
    
    def add_chunks(self, chunks: List[CodeChunk], batch_size: int = 100) -> None:
        """
        Add code chunks to the vector store.
        
        Args:
            chunks: List of code chunks to add
            batch_size: Number of chunks to process in each batch
        """
        if not chunks:
            return
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._add_batch(batch)
    
    def _add_batch(self, chunks: List[CodeChunk]) -> None:
        """Add a batch of chunks to the vector store."""
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        # Prepare text for embedding (clean up as recommended in research)
        chunk_texts = []
        for chunk in chunks:
            # Replace newlines with spaces as recommended by OpenAI
            clean_text = chunk.content.replace('\n', ' ')
            chunk_texts.append(clean_text)
        
        # Generate embeddings for all chunks in batch
        chunk_embeddings = self.embedding_manager.embed_batch(chunk_texts)
        
        for chunk, embedding in zip(chunks, chunk_embeddings):
            ids.append(chunk.id)
            documents.append(chunk.content)  # Keep original for display
            
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
            embeddings.append(embedding)
        
        # Add to ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        except Exception as e:
            # Handle potential duplicate IDs
            print(f"Error adding batch to vector store: {e}")
            self._handle_duplicate_ids(ids, documents, metadatas, embeddings)
    
    def _handle_duplicate_ids(self, ids: List[str], documents: List[str], 
                             metadatas: List[Dict], embeddings: List[List[float]]) -> None:
        """Handle duplicate IDs by updating existing entries."""
        for id_, doc, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            try:
                # Try to update existing entry
                self.collection.update(
                    ids=[id_],
                    documents=[doc],
                    metadatas=[metadata],
                    embeddings=[embedding]
                )
            except Exception:
                # If update fails, try to add with new ID
                new_id = f"{id_}_{uuid.uuid4().hex[:8]}"
                try:
                    self.collection.add(
                        ids=[new_id],
                        documents=[doc],
                        metadatas=[metadata],
                        embeddings=[embedding]
                    )
                except Exception as e:
                    print(f"Failed to add chunk {id_}: {e}")
    
    def search(self, query: str, embedding_manager, n_results: int = 10, 
              language_filter: Optional[str] = None,
              file_path_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar code chunks using semantic similarity.
        Uses cosine distance as recommended for text embeddings.
        
        Args:
            query: Search query text
            embedding_manager: Embedding manager to generate query embeddings
            n_results: Number of results to return
            language_filter: Filter by programming language
            file_path_filter: Filter by file path pattern
            
        Returns:
            List of search results with metadata
        """
        if not query.strip():
            return []
        
        # Clean query text as recommended
        clean_query = query.replace('\n', ' ')
        
        # Generate embedding for query
        query_embedding = embedding_manager.embed_text(clean_query)
        if not query_embedding:
            return []
        
        # Build where clause for filtering
        where_clause = {}
        if language_filter:
            where_clause['language'] = language_filter
        if file_path_filter:
            where_clause['file_path'] = {"$contains": file_path_filter}
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results with proper similarity scores
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    # Convert cosine distance to similarity score (closer to 1 = more similar)
                    similarity = 1 - distance
                    
                    result = {
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': distance,
                        'similarity': similarity
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk by its ID.
        
        Args:
            chunk_id: Unique identifier of the chunk
            
        Returns:
            Chunk data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas']
            )
            
            if results['ids'] and results['ids'][0]:
                return {
                    'id': results['ids'][0],
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            
        except Exception as e:
            print(f"Error getting chunk by ID: {e}")
        
        return None
    
    def delete_by_file_path(self, file_path: str) -> int:
        """
        Delete all chunks from a specific file.
        
        Args:
            file_path: Path of the file whose chunks to delete
            
        Returns:
            Number of chunks deleted
        """
        try:
            # Get all chunks for this file
            results = self.collection.get(
                where={'file_path': file_path},
                include=['documents', 'metadatas']
            )
            
            if results['ids']:
                chunk_ids = results['ids']
                self.collection.delete(ids=chunk_ids)
                return len(chunk_ids)
            
        except Exception as e:
            print(f"Error deleting chunks for file {file_path}: {e}")
        
        return 0
    
    def delete_by_file_hash(self, file_hash: str) -> int:
        """
        Delete all chunks with a specific file hash.
        
        Args:
            file_hash: Hash of the file whose chunks to delete
            
        Returns:
            Number of chunks deleted
        """
        try:
            results = self.collection.get(
                where={'file_hash': file_hash},
                include=['documents', 'metadatas']
            )
            
            if results['ids']:
                chunk_ids = results['ids']
                self.collection.delete(ids=chunk_ids)
                return len(chunk_ids)
            
        except Exception as e:
            print(f"Error deleting chunks for file hash {file_hash}: {e}")
        
        return 0
    
    def update_chunks(self, chunks: List[CodeChunk]) -> None:
        """
        Update existing chunks in the vector store.
        
        Args:
            chunks: List of code chunks to update
        """
        for chunk in chunks:
            # Delete old version if exists
            self.delete_by_file_path(chunk.file_path)
        
        # Add new versions
        self.add_chunks(chunks)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get all metadata to compute stats
            results = self.collection.get(include=['metadatas'])
            
            if not results['metadatas']:
                return {
                    'total_chunks': 0,
                    'unique_files': 0,
                    'languages': {},
                    'total_size_bytes': 0,
                    'embedding_providers': {}
                }
            
            stats = {
                'total_chunks': len(results['metadatas']),
                'unique_files': len(set(m['file_path'] for m in results['metadatas'])),
                'languages': {},
                'total_size_bytes': sum(m.get('chunk_size', 0) for m in results['metadatas']),
                'embedding_providers': {}
            }
            
            # Count languages and embedding providers
            for metadata in results['metadatas']:
                lang = metadata.get('language', 'unknown')
                stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
                
                provider = metadata.get('embedding_model', 'unknown')
                stats['embedding_providers'][provider] = stats['embedding_providers'].get(provider, 0) + 1
            
            return stats
            
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {
                'total_chunks': 0,
                'unique_files': 0,
                'languages': {},
                'total_size_bytes': 0,
                'embedding_providers': {}
            }
    
    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self._collection = None  # Reset to force recreation
            # Recreate collection
            _ = self.collection
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def get_files_in_store(self) -> List[Dict[str, Any]]:
        """
        Get list of all files currently in the vector store.
        
        Returns:
            List of file information
        """
        try:
            results = self.collection.get(include=['metadatas'])
            
            if not results['metadatas']:
                return []
            
            # Group by file path
            files_info = {}
            for metadata in results['metadatas']:
                file_path = metadata['file_path']
                if file_path not in files_info:
                    files_info[file_path] = {
                        'file_path': file_path,
                        'language': metadata.get('language', 'unknown'),
                        'file_hash': metadata.get('file_hash', ''),
                        'chunk_count': 0,
                        'total_size': 0,
                        'embedding_model': metadata.get('embedding_model', 'unknown')
                    }
                
                files_info[file_path]['chunk_count'] += 1
                files_info[file_path]['total_size'] += metadata.get('chunk_size', 0)
            
            return list(files_info.values())
            
        except Exception as e:
            print(f"Error getting files in store: {e}")
            return [] 