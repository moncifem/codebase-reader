# 🔍 Codebase Reader & Analyzer

An AI-powered codebase analysis tool that helps you understand, search, and analyze your code using advanced embedding models and Large Language Models (LLMs). Built following best practices from embeddings and vector database research.

## ✨ Features

- **🔍 Intelligent Code Search**: Semantic search across your entire codebase using cosine similarity
- **🤖 AI-Powered Analysis**: Ask questions about your code and get intelligent responses
- **📊 Codebase Insights**: Comprehensive statistics and language distribution
- **🔄 Multiple Providers**: Support for different embedding and LLM providers
- **⚡ Efficient Chunking**: Smart code chunking with language-aware processing
- **💾 Persistent Storage**: ChromaDB for efficient vector storage with optimized distance metrics
- **🌐 Beautiful UI**: Clean Streamlit interface with multiple pages
- **🔒 Security Analysis**: Built-in security vulnerability detection
- **📝 Documentation Generation**: AI-powered code documentation
- **💡 Code Suggestions**: Get improvement suggestions for your code

## 🏗️ Architecture

The application consists of several modular components:

- **Configuration Manager**: Centralized configuration handling
- **Codebase Reader**: Code scanning, parsing, and chunking
- **Embeddings Module**: Multiple embedding providers (SentenceTransformers, OpenAI) with optimized text preprocessing
- **Vector Store**: ChromaDB integration with cosine distance for efficient similarity search
- **LLM Client**: Flexible LLM integration for code analysis
- **Main Analyzer**: Orchestrates all components
- **Streamlit UI**: User-friendly web interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd codebase-reader
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional for OpenAI features):
   ```bash
   cp env.example .env
   # Edit .env and add your API keys
   ```

4. **Run the application**:
   ```bash
   python run.py
   # OR
   streamlit run app.py
   ```

5. **Test the installation**:
   ```bash
   python test_basic.py
   ```

6. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage Guide

### 1. Index a Codebase

1. Navigate to the "📁 Index Codebase" page
2. Choose how to specify your codebase directory:
   - Type the full path directly
   - Browse and select from current directory
3. Configure indexing options:
   - Update existing files (recommended)
   - Clear index before indexing (if needed)
4. Click "🚀 Start Indexing" and wait for completion

### 2. Search and Query

#### Semantic Search
1. Go to "🔍 Search & Query" page
2. Enter your search query (e.g., "authentication logic", "database connection")
3. Adjust filters:
   - Number of results
   - Programming language
   - File path patterns
4. View results with similarity scores and code context

#### AI Questions
1. Use the "🤖 Ask AI About Your Code" section
2. Enter natural language questions about your codebase
3. Adjust the number of context chunks for better responses
4. Get comprehensive AI-powered answers

#### Code Analysis
For each search result, you can:
- **📖 Explain**: Get AI explanation of what the code does
- **💡 Suggest Improvements**: Get code improvement suggestions
- **🔒 Security Check**: Analyze for potential security issues

### 3. Dashboard Overview

The dashboard provides:
- Total chunks and files indexed
- Programming language distribution
- Codebase size and statistics
- Embedding provider information
- Quick access to indexed files

### 4. Settings and Configuration

- Switch between embedding providers (SentenceTransformers, OpenAI)
- Change LLM providers
- View current configuration
- Manage the vector database index

## ⚙️ Configuration

The application uses `config.yaml` for configuration:

```yaml
# Vector Database Settings
vector_db:
  type: "chromadb"
  persist_directory: "./chroma_db"
  collection_name: "codebase_chunks"
  distance_metric: "cosine"  # Recommended by OpenAI for text embeddings

# Embedding Settings
embeddings:
  default_provider: "sentence_transformers"
  sentence_transformers:
    model_name: "all-MiniLM-L6-v2"
  openai:
    model: "text-embedding-ada-002"  # Latest and most cost-effective
    api_key_env: "OPENAI_API_KEY"
    dimensions: 1536  # text-embedding-ada-002 outputs 1536-dimensional vectors

# LLM Settings
llm:
  default_provider: "openai"
  openai:
    model: "gpt-3.5-turbo"
    api_key_env: "OPENAI_API_KEY"
    max_tokens: 1000
    temperature: 0.1

# Chunking Settings
chunking:
  chunk_size: 1000
  chunk_overlap: 200
  max_file_size_mb: 10
  # Optimized for semantic coherence while staying within embedding limits
```

## 🔧 Supported Languages

The application supports analysis of the following programming languages:

- Python (.py)
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)
- Java (.java)
- C/C++ (.c, .cpp, .h, .hpp)
- C# (.cs)
- Go (.go)
- Rust (.rs)
- PHP (.php)
- Ruby (.rb)
- Swift (.swift)
- Kotlin (.kt)
- Scala (.scala)
- SQL (.sql)
- HTML (.html)
- CSS (.css)
- Markdown (.md)
- Text files (.txt)
- JSON (.json)
- YAML (.yaml, .yml)

## 🤖 AI Providers

### Embedding Providers

1. **SentenceTransformers** (Default, Offline)
   - Model: `all-MiniLM-L6-v2`
   - No API key required
   - Runs locally
   - Good for privacy-sensitive scenarios

2. **OpenAI** (Online, Recommended)
   - Model: `text-embedding-ada-002`
   - Requires OpenAI API key
   - 1536-dimensional vectors
   - Higher quality embeddings
   - Optimized with cosine distance metric

### LLM Providers

1. **OpenAI** (Default)
   - Model: `gpt-3.5-turbo`
   - Requires OpenAI API key
   - High-quality code analysis

## 🔬 Technical Implementation

### Embeddings Best Practices

Based on research from embeddings and vector database experts, our implementation follows these best practices:

- **Text Preprocessing**: Automatic newline removal and whitespace normalization as recommended by OpenAI
- **Distance Metric**: Cosine similarity for measuring semantic similarity between embeddings
- **Batch Processing**: Efficient batch processing with configurable batch sizes to respect API limits
- **Dimension Optimization**: Proper handling of embedding dimensions (1536 for text-embedding-ada-002)
- **Error Handling**: Graceful fallbacks for empty or problematic text chunks

### Vector Database Optimization

- **ChromaDB**: Open-source vector database optimized for similarity search
- **HNSW Algorithm**: Hierarchical Navigable Small World algorithm for fast nearest neighbor search
- **Persistent Storage**: Efficient storage with DuckDB and Parquet format
- **Metadata Filtering**: Advanced filtering by language, file path, and other attributes

## 📁 Project Structure

```
codebase-reader/
├── src/                        # Main application modules
│   ├── __init__.py
│   ├── config_manager.py       # Configuration management
│   ├── codebase_reader.py      # Code scanning and chunking
│   ├── embeddings.py           # Embedding providers with optimizations
│   ├── vector_store.py         # ChromaDB integration with cosine distance
│   ├── llm_client.py          # LLM providers
│   └── codebase_analyzer.py   # Main orchestrator
├── app.py                     # Streamlit application
├── config.yaml               # Configuration file with optimized settings
├── requirements.txt          # Python dependencies (fixed versions)
├── env.example              # Environment variables template
├── run.py                   # Easy startup script
├── test_basic.py           # Component testing
└── README.md               # This file
```

## 🔍 Example Use Cases

### 1. Understanding a New Codebase
- Index the codebase
- Ask: "What is the main entry point of this application?"
- Search for: "main function" or "application startup"

### 2. Finding Security Issues
- Search for: "user input" or "database query"
- Use security analysis on relevant code chunks
- Ask: "Are there any potential security vulnerabilities?"

### 3. Code Review and Improvements
- Search for specific functions or modules
- Get improvement suggestions for code chunks
- Ask: "How can I optimize this database code?"

### 4. Documentation Generation
- Find key functions and classes
- Generate documentation for important code sections
- Ask: "How does the authentication system work?"

## 🛠️ Development

### Adding New Embedding Providers

1. Create a new provider class inheriting from `EmbeddingProvider`
2. Implement required methods: `embed_text`, `embed_batch`, `dimension`
3. Add proper text preprocessing with `_preprocess_text` method
4. Add provider to `EmbeddingManager._create_provider()`
5. Update configuration if needed

### Adding New LLM Providers

1. Create a new provider class inheriting from `LLMProvider`
2. Implement required methods: `generate_response`, `analyze_code`
3. Add provider to `LLMClient._create_provider()`
4. Update configuration and UI

## 🔧 Troubleshooting

### Common Issues

1. **Tree-sitter installation errors**: The requirements.txt has been updated with compatible versions
2. **ChromaDB persistence issues**: Ensure the `chroma_db` directory has write permissions
3. **OpenAI API errors**: Check your API key in the `.env` file
4. **Memory issues with large codebases**: Adjust the `chunk_size` and `batch_size` in configuration

### Performance Optimization

- Use OpenAI embeddings for better semantic quality
- Adjust `chunk_size` based on your use case (smaller for precise search, larger for context)
- Use appropriate `batch_size` for your API rate limits
- Consider using SentenceTransformers for offline/privacy scenarios

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Commit with descriptive messages
6. Push to your fork
7. Create a pull request

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- **ChromaDB** for efficient vector storage
- **SentenceTransformers** for local embeddings
- **OpenAI** for advanced AI capabilities and embedding best practices
- **Streamlit** for the beautiful web interface
- **Research Community** for embeddings and vector database best practices

## 📞 Support

If you encounter any issues or have questions:

1. Run the test suite: `python test_basic.py`
2. Check the configuration in `config.yaml`
3. Ensure all dependencies are installed correctly: `pip install -r requirements.txt`
4. Verify API keys are set up properly (if using OpenAI)
5. Check the logs for error messages

For bugs and feature requests, please open an issue on the repository.

# 🔍 Codebase Reader & Analyzer

An AI-powered codebase analysis tool that helps you understand, search, and analyze your code using advanced embedding models and Large Language Models (LLMs). Built following best practices from embeddings and vector database research.

## ✨ Features

- **🔍 Intelligent Code Search**: Semantic search across your entire codebase using cosine similarity
- **🤖 AI-Powered Analysis**: Ask questions about your code and get intelligent responses
- **📊 Codebase Insights**: Comprehensive statistics and language distribution
- **🔄 Multiple Providers**: Support for different embedding and LLM providers
- **⚡ Efficient Chunking**: Smart code chunking with language-aware processing
- **💾 Persistent Storage**: ChromaDB for efficient vector storage with optimized distance metrics
- **🌐 Beautiful UI**: Clean Streamlit interface with multiple pages
- **🔒 Security Analysis**: Built-in security vulnerability detection
- **📝 Documentation Generation**: AI-powered code documentation
- **💡 Code Suggestions**: Get improvement suggestions for your code

## 🏗️ Architecture

The application consists of several modular components:

- **Configuration Manager**: Centralized configuration handling
- **Codebase Reader**: Code scanning, parsing, and chunking
- **Embeddings Module**: Multiple embedding providers (SentenceTransformers, OpenAI) with optimized text preprocessing
- **Vector Store**: ChromaDB integration with cosine distance for efficient similarity search
- **LLM Client**: Flexible LLM integration for code analysis
- **Main Analyzer**: Orchestrates all components
- **Streamlit UI**: User-friendly web interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd codebase-reader
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional for OpenAI features):
   ```bash
   cp env.example .env
   # Edit .env and add your API keys
   ```

4. **Run the application**:
   ```bash
   python run.py
   # OR
   streamlit run app.py
   ```

5. **Test the installation**:
   ```bash
   python test_basic.py
   ```

6. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage Guide

### 1. Index a Codebase

1. Navigate to the "📁 Index Codebase" page
2. Choose how to specify your codebase directory:
   - Type the full path directly
   - Browse and select from current directory
3. Configure indexing options:
   - Update existing files (recommended)
   - Clear index before indexing (if needed)
4. Click "🚀 Start Indexing" and wait for completion

### 2. Search and Query

#### Semantic Search
1. Go to "🔍 Search & Query" page
2. Enter your search query (e.g., "authentication logic", "database connection")
3. Adjust filters:
   - Number of results
   - Programming language
   - File path patterns
4. View results with similarity scores and code context

#### AI Questions
1. Use the "🤖 Ask AI About Your Code" section
2. Enter natural language questions about your codebase
3. Adjust the number of context chunks for better responses
4. Get comprehensive AI-powered answers

#### Code Analysis
For each search result, you can:
- **📖 Explain**: Get AI explanation of what the code does
- **💡 Suggest Improvements**: Get code improvement suggestions
- **🔒 Security Check**: Analyze for potential security issues

### 3. Dashboard Overview

The dashboard provides:
- Total chunks and files indexed
- Programming language distribution
- Codebase size and statistics
- Embedding provider information
- Quick access to indexed files

### 4. Settings and Configuration

- Switch between embedding providers (SentenceTransformers, OpenAI)
- Change LLM providers
- View current configuration
- Manage the vector database index

## ⚙️ Configuration

The application uses `config.yaml` for configuration:

```yaml
# Vector Database Settings
vector_db:
  type: "chromadb"
  persist_directory: "./chroma_db"
  collection_name: "codebase_chunks"
  distance_metric: "cosine"  # Recommended by OpenAI for text embeddings

# Embedding Settings
embeddings:
  default_provider: "sentence_transformers"
  sentence_transformers:
    model_name: "all-MiniLM-L6-v2"
  openai:
    model: "text-embedding-ada-002"  # Latest and most cost-effective
    api_key_env: "OPENAI_API_KEY"
    dimensions: 1536  # text-embedding-ada-002 outputs 1536-dimensional vectors

# LLM Settings
llm:
  default_provider: "openai"
  openai:
    model: "gpt-3.5-turbo"
    api_key_env: "OPENAI_API_KEY"
    max_tokens: 1000
    temperature: 0.1

# Chunking Settings
chunking:
  chunk_size: 1000
  chunk_overlap: 200
  max_file_size_mb: 10
  # Optimized for semantic coherence while staying within embedding limits
```

## 🔧 Supported Languages

The application supports analysis of the following programming languages:

- Python (.py)
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)
- Java (.java)
- C/C++ (.c, .cpp, .h, .hpp)
- C# (.cs)
- Go (.go)
- Rust (.rs)
- PHP (.php)
- Ruby (.rb)
- Swift (.swift)
- Kotlin (.kt)
- Scala (.scala)
- SQL (.sql)
- HTML (.html)
- CSS (.css)
- Markdown (.md)
- Text files (.txt)
- JSON (.json)
- YAML (.yaml, .yml)

## 🤖 AI Providers

### Embedding Providers

1. **SentenceTransformers** (Default, Offline)
   - Model: `all-MiniLM-L6-v2`
   - No API key required
   - Runs locally
   - Good for privacy-sensitive scenarios

2. **OpenAI** (Online, Recommended)
   - Model: `text-embedding-ada-002`
   - Requires OpenAI API key
   - 1536-dimensional vectors
   - Higher quality embeddings
   - Optimized with cosine distance metric

### LLM Providers

1. **OpenAI** (Default)
   - Model: `gpt-3.5-turbo`
   - Requires OpenAI API key
   - High-quality code analysis

## 🔬 Technical Implementation

### Embeddings Best Practices

Based on research from embeddings and vector database experts, our implementation follows these best practices:

- **Text Preprocessing**: Automatic newline removal and whitespace normalization as recommended by OpenAI
- **Distance Metric**: Cosine similarity for measuring semantic similarity between embeddings
- **Batch Processing**: Efficient batch processing with configurable batch sizes to respect API limits
- **Dimension Optimization**: Proper handling of embedding dimensions (1536 for text-embedding-ada-002)
- **Error Handling**: Graceful fallbacks for empty or problematic text chunks

### Vector Database Optimization

- **ChromaDB**: Open-source vector database optimized for similarity search
- **HNSW Algorithm**: Hierarchical Navigable Small World algorithm for fast nearest neighbor search
- **Persistent Storage**: Efficient storage with DuckDB and Parquet format
- **Metadata Filtering**: Advanced filtering by language, file path, and other attributes

## 📁 Project Structure

```
codebase-reader/
├── src/                        # Main application modules
│   ├── __init__.py
│   ├── config_manager.py       # Configuration management
│   ├── codebase_reader.py      # Code scanning and chunking
│   ├── embeddings.py           # Embedding providers with optimizations
│   ├── vector_store.py         # ChromaDB integration with cosine distance
│   ├── llm_client.py          # LLM providers
│   └── codebase_analyzer.py   # Main orchestrator
├── app.py                     # Streamlit application
├── config.yaml               # Configuration file with optimized settings
├── requirements.txt          # Python dependencies (fixed versions)
├── env.example              # Environment variables template
├── run.py                   # Easy startup script
├── test_basic.py           # Component testing
└── README.md               # This file
```

## 🔍 Example Use Cases

### 1. Understanding a New Codebase
- Index the codebase
- Ask: "What is the main entry point of this application?"
- Search for: "main function" or "application startup"

### 2. Finding Security Issues
- Search for: "user input" or "database query"
- Use security analysis on relevant code chunks
- Ask: "Are there any potential security vulnerabilities?"

### 3. Code Review and Improvements
- Search for specific functions or modules
- Get improvement suggestions for code chunks
- Ask: "How can I optimize this database code?"

### 4. Documentation Generation
- Find key functions and classes
- Generate documentation for important code sections
- Ask: "How does the authentication system work?"

## 🛠️ Development

### Adding New Embedding Providers

1. Create a new provider class inheriting from `EmbeddingProvider`
2. Implement required methods: `embed_text`, `embed_batch`, `dimension`
3. Add proper text preprocessing with `_preprocess_text` method
4. Add provider to `EmbeddingManager._create_provider()`
5. Update configuration if needed

### Adding New LLM Providers

1. Create a new provider class inheriting from `LLMProvider`
2. Implement required methods: `generate_response`, `analyze_code`
3. Add provider to `LLMClient._create_provider()`
4. Update configuration and UI

## 🔧 Troubleshooting

### Common Issues

1. **Tree-sitter installation errors**: The requirements.txt has been updated with compatible versions
2. **ChromaDB persistence issues**: Ensure the `chroma_db` directory has write permissions
3. **OpenAI API errors**: Check your API key in the `.env` file
4. **Memory issues with large codebases**: Adjust the `chunk_size` and `batch_size` in configuration

### Performance Optimization

- Use OpenAI embeddings for better semantic quality
- Adjust `chunk_size` based on your use case (smaller for precise search, larger for context)
- Use appropriate `batch_size` for your API rate limits
- Consider using SentenceTransformers for offline/privacy scenarios

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Commit with descriptive messages
6. Push to your fork
7. Create a pull request

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- **ChromaDB** for efficient vector storage
- **SentenceTransformers** for local embeddings
- **OpenAI** for advanced AI capabilities and embedding best practices
- **Streamlit** for the beautiful web interface
- **Research Community** for embeddings and vector database best practices

## 📞 Support

If you encounter any issues or have questions:

1. Run the test suite: `python test_basic.py`
2. Check the configuration in `config.yaml`
3. Ensure all dependencies are installed correctly: `pip install -r requirements.txt`
4. Verify API keys are set up properly (if using OpenAI)
5. Check the logs for error messages

For bugs and feature requests, please open an issue on the repository.

---

Happy coding! 🚀 