"""
Codebase reader module for scanning, reading, and chunking code files.
Supports multiple programming languages and intelligent file filtering.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Tuple
from dataclasses import dataclass, asdict
import fnmatch

from .config_manager import config


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    language: str
    file_hash: str
    chunk_index: int
    metadata: Dict[str, Any]


class CodebaseReader:
    """Reads and processes codebases into manageable chunks."""
    
    def __init__(self):
        self.supported_extensions = config.supported_extensions
        self.ignore_patterns = config.ignore_patterns
        self.chunk_size = config.chunking.chunk_size
        self.chunk_overlap = config.chunking.chunk_overlap
        self.max_file_size_mb = config.chunking.max_file_size_mb
    
    def scan_codebase(self, root_path: str) -> List[str]:
        """
        Scan a codebase directory and return list of relevant files.
        
        Args:
            root_path: Root directory to scan
            
        Returns:
            List of file paths to process
        """
        relevant_files = []
        root_path = Path(root_path)
        
        if not root_path.exists():
            raise FileNotFoundError(f"Directory not found: {root_path}")
        
        for file_path in root_path.rglob("*"):
            if self._should_include_file(file_path):
                relevant_files.append(str(file_path))
        
        return sorted(relevant_files)
    
    def _should_include_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be included in processing.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file should be included
        """
        # Skip directories
        if file_path.is_dir():
            return False
        
        # Check file size
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return False
        except OSError:
            return False
        
        # Check extension
        if file_path.suffix.lower() not in self.supported_extensions:
            return False
        
        # Check ignore patterns
        file_str = str(file_path)
        for pattern in self.ignore_patterns:
            if pattern in file_str or fnmatch.fnmatch(file_str, f"*{pattern}*"):
                return False
        
        return True
    
    def read_file_content(self, file_path: str) -> str:
        """
        Read content from a file with proper encoding handling.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                break
        
        return ""
    
    def get_file_hash(self, file_path: str) -> str:
        """
        Generate MD5 hash of file content for change detection.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 hash of file content
        """
        content = self.read_file_content(file_path)
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def detect_language(self, file_path: str) -> str:
        """
        Detect programming language from file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected language name
        """
        extension = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown',
            '.txt': 'text',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml'
        }
        
        return language_map.get(extension, 'unknown')
    
    def chunk_content(self, content: str, file_path: str) -> List[CodeChunk]:
        """
        Split file content into overlapping chunks.
        
        Args:
            content: File content to chunk
            file_path: Path to the source file
            
        Returns:
            List of code chunks
        """
        if not content.strip():
            return []
        
        lines = content.split('\n')
        chunks = []
        
        # Calculate lines per chunk
        avg_chars_per_line = len(content) / len(lines) if lines else 1
        lines_per_chunk = max(1, int(self.chunk_size / avg_chars_per_line))
        overlap_lines = max(1, int(self.chunk_overlap / avg_chars_per_line))
        
        file_hash = self.get_file_hash(file_path)
        language = self.detect_language(file_path)
        
        chunk_index = 0
        i = 0
        
        while i < len(lines):
            end_idx = min(i + lines_per_chunk, len(lines))
            chunk_lines = lines[i:end_idx]
            chunk_content = '\n'.join(chunk_lines)
            
            if chunk_content.strip():  # Only create non-empty chunks
                chunk_id = self._generate_chunk_id(file_path, chunk_index)
                
                chunk = CodeChunk(
                    id=chunk_id,
                    file_path=file_path,
                    content=chunk_content,
                    start_line=i + 1,  # 1-indexed
                    end_line=end_idx,   # 1-indexed
                    language=language,
                    file_hash=file_hash,
                    chunk_index=chunk_index,
                    metadata={
                        'file_size': len(content),
                        'total_lines': len(lines),
                        'chunk_size': len(chunk_content)
                    }
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Move to next chunk with overlap
            i += max(1, lines_per_chunk - overlap_lines)
        
        return chunks
    
    def _generate_chunk_id(self, file_path: str, chunk_index: int) -> str:
        """
        Generate unique ID for a code chunk.
        
        Args:
            file_path: Path to the source file
            chunk_index: Index of the chunk within the file
            
        Returns:
            Unique chunk ID
        """
        path_hash = hashlib.md5(file_path.encode('utf-8')).hexdigest()[:8]
        return f"{path_hash}_{chunk_index}"
    
    def process_file(self, file_path: str) -> List[CodeChunk]:
        """
        Process a single file into code chunks.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of code chunks
        """
        try:
            content = self.read_file_content(file_path)
            if not content:
                return []
            
            chunks = self.chunk_content(content, file_path)
            return chunks
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []
    
    def process_codebase(self, root_path: str) -> Iterator[CodeChunk]:
        """
        Process entire codebase and yield code chunks.
        
        Args:
            root_path: Root directory to process
            
        Yields:
            Code chunks from the codebase
        """
        files = self.scan_codebase(root_path)
        
        for file_path in files:
            chunks = self.process_file(file_path)
            for chunk in chunks:
                yield chunk
    
    def get_codebase_stats(self, root_path: str) -> Dict[str, Any]:
        """
        Get statistics about a codebase.
        
        Args:
            root_path: Root directory to analyze
            
        Returns:
            Dictionary with codebase statistics
        """
        files = self.scan_codebase(root_path)
        
        stats = {
            'total_files': len(files),
            'languages': {},
            'total_size_mb': 0,
            'total_lines': 0
        }
        
        for file_path in files:
            language = self.detect_language(file_path)
            if language not in stats['languages']:
                stats['languages'][language] = 0
            stats['languages'][language] += 1
            
            try:
                file_size = Path(file_path).stat().st_size
                stats['total_size_mb'] += file_size / (1024 * 1024)
                
                content = self.read_file_content(file_path)
                stats['total_lines'] += len(content.split('\n'))
            except Exception:
                pass
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        
        return stats 