#!/usr/bin/env python3
"""
Basic test script to verify the codebase reader application components.
Run this script to check if all components are properly installed and configured.
"""

import sys
import os
import tempfile
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.config_manager import config, ConfigManager
        print("✅ Config manager imported successfully")
    except Exception as e:
        print(f"❌ Failed to import config manager: {e}")
        return False
    
    try:
        from src.codebase_reader import CodebaseReader, CodeChunk
        print("✅ Codebase reader imported successfully")
    except Exception as e:
        print(f"❌ Failed to import codebase reader: {e}")
        return False
    
    try:
        from src.embeddings import EmbeddingManager
        print("✅ Embeddings module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import embeddings module: {e}")
        return False
    
    try:
        from src.vector_store import VectorStore
        print("✅ Vector store imported successfully")
    except Exception as e:
        print(f"❌ Failed to import vector store: {e}")
        return False
    
    try:
        from src.llm_client import LLMClient
        print("✅ LLM client imported successfully")
    except Exception as e:
        print(f"❌ Failed to import LLM client: {e}")
        return False
    
    try:
        from src.codebase_analyzer import CodebaseAnalyzer
        print("✅ Codebase analyzer imported successfully")
    except Exception as e:
        print(f"❌ Failed to import codebase analyzer: {e}")
        return False
    
    return True


def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.config_manager import config
        
        # Test basic config access
        vector_db_config = config.vector_db
        print(f"✅ Vector DB config: {vector_db_config.type}")
        
        embeddings_config = config.embeddings
        print(f"✅ Embeddings config: {embeddings_config.default_provider}")
        
        supported_extensions = config.supported_extensions
        print(f"✅ Supported extensions: {len(supported_extensions)} types")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_codebase_reader():
    """Test basic codebase reader functionality."""
    print("\nTesting codebase reader...")
    
    try:
        from src.codebase_reader import CodebaseReader
        
        reader = CodebaseReader()
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def hello_world():
    '''A simple hello world function.'''
    print("Hello, World!")
    return "success"

if __name__ == "__main__":
    result = hello_world()
    print(f"Result: {result}")
""")
            temp_file = f.name
        
        try:
            # Test file processing
            chunks = reader.process_file(temp_file)
            print(f"✅ Processed test file into {len(chunks)} chunks")
            
            if chunks:
                chunk = chunks[0]
                print(f"✅ First chunk: {len(chunk.content)} characters, language: {chunk.language}")
            
            # Test language detection
            language = reader.detect_language(temp_file)
            print(f"✅ Detected language: {language}")
            
            return True
            
        finally:
            # Clean up temp file
            os.unlink(temp_file)
        
    except Exception as e:
        print(f"❌ Codebase reader test failed: {e}")
        return False


def test_sentence_transformers():
    """Test if SentenceTransformers embedding provider works."""
    print("\nTesting SentenceTransformers embedding provider...")
    
    try:
        from src.embeddings import SentenceTransformerProvider
        
        provider = SentenceTransformerProvider()
        
        # Test embedding generation
        test_text = "def hello_world(): print('Hello, World!')"
        embedding = provider.embed_text(test_text)
        
        print(f"✅ Generated embedding of dimension: {len(embedding)}")
        print(f"✅ Provider dimension: {provider.dimension}")
        
        # Test batch embedding
        batch_texts = ["print('hello')", "def function():", "import os"]
        batch_embeddings = provider.embed_batch(batch_texts)
        print(f"✅ Generated {len(batch_embeddings)} batch embeddings")
        
        return True
        
    except Exception as e:
        print(f"❌ SentenceTransformers test failed: {e}")
        print("Note: This is expected if sentence-transformers is not installed")
        return False


def test_chromadb():
    """Test if ChromaDB works."""
    print("\nTesting ChromaDB...")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Create a temporary ChromaDB instance
        with tempfile.TemporaryDirectory() as temp_dir:
            client = chromadb.PersistentClient(
                path=temp_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            collection = client.create_collection(name="test_collection")
            print("✅ Created ChromaDB collection")
            
            # Test basic operations
            collection.add(
                ids=["test1"],
                documents=["This is a test document"],
                embeddings=[[0.1, 0.2, 0.3]]
            )
            print("✅ Added test document to collection")
            
            results = collection.get(ids=["test1"])
            print(f"✅ Retrieved document: {len(results['documents'])} items")
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False


def test_streamlit_import():
    """Test if Streamlit can be imported."""
    print("\nTesting Streamlit import...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        return True
    except Exception as e:
        print(f"❌ Streamlit import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🔍 Codebase Reader - Basic Component Tests\n")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_codebase_reader,
        test_sentence_transformers,
        test_chromadb,
        test_streamlit_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("Note: Some failures may be expected if optional dependencies are not installed.")
    
    print("\nTo run the application:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main() 