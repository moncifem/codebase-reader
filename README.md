# 🔍 Codebase Reader & Analyzer

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)](https://openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-orange)](https://anthropic.com/)

**🚀 AI-Powered Codebase Analysis with Flexible Provider Support**

*Index, search, and analyze any codebase with semantic understanding using local or cloud AI models*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Configuration](#-configuration) • [Contributing](#-contributing)

![Codebase Analyzer Demo](https://via.placeholder.com/800x400/1a1a1a/00d4aa?text=🔍+Codebase+Analyzer+Demo)

</div>

---

## ✨ Features

### 🧠 **Flexible AI Provider System**
- **🌐 Multi-Provider Support**: OpenAI, Anthropic Claude, or local SentenceTransformers
- **🔄 Hot-Swappable**: Switch between providers without restart
- **🏠 Works Offline**: Default mini-LLM model (`all-MiniLM-L6-v2`) requires no API keys
- **💪 Graceful Fallbacks**: Automatic degradation when providers unavailable

### 📊 **Intelligent Code Analysis**
- **🔍 Semantic Search**: Find code by meaning, not just keywords
- **❓ AI Q&A**: Ask natural language questions about your codebase
- **📝 Code Explanation**: AI-powered code documentation and explanations
- **🔒 Security Analysis**: Automated vulnerability detection
- **💡 Improvement Suggestions**: Code quality recommendations

### 🚀 **Developer Experience**
- **🎨 Beautiful UI**: Modern Streamlit interface with dark theme
- **📁 Smart Indexing**: Comprehensive ignore patterns (70+ file types)
- **⚡ Real-time Status**: Live provider monitoring and health checks
- **🔧 Easy Setup**: One-command installation and configuration
- **📈 Rich Analytics**: Codebase statistics and language distribution

### 🛡️ **Robust & Reliable**
- **🔍 Works with ANY configuration**: No API keys? No problem!
- **📋 Comprehensive File Support**: 25+ programming languages
- **🚫 Smart Exclusions**: Automatically ignores build artifacts, dependencies
- **💾 Persistent Storage**: ChromaDB vector database with incremental updates
- **🔄 Update Detection**: Automatically tracks file changes

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/codebase-reader.git
cd codebase-reader

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run flexible_app.py
```

### 🔑 Optional: Configure API Keys

Create a `.env` file for enhanced AI capabilities:

```bash
# Optional: For OpenAI GPT models and embeddings
OPENAI_API_KEY=your_openai_api_key_here

# Optional: For Anthropic Claude models  
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**💡 Note**: The app works perfectly without any API keys using local models!

---

## 🎯 Usage

### 1. **Launch the Application**
```bash
streamlit run flexible_app.py
```
Open your browser to `http://localhost:8501`

### 2. **Index Your Codebase**
- Navigate to "📁 Index Codebase"
- Enter your project path or use the path verification
- Review the 70+ ignore patterns (node_modules, .git, build artifacts, etc.)
- Click "🚀 Start Indexing"

### 3. **Search & Analyze**
- **🔍 Semantic Search**: Find code by functionality
- **🤖 AI Chat**: Ask questions like "How does authentication work?"
- **📊 Dashboard**: View codebase statistics and language distribution

### 4. **Configure Providers**
- Visit "⚙️ Settings" to switch between AI providers
- Monitor provider status in "🔧 Provider Status"
- Real-time switching without restart!

---

## ⚙️ Configuration

### Supported Languages
```
Python • JavaScript • TypeScript • Java • C++ • C# • Go • Rust
PHP • Ruby • Swift • Kotlin • Scala • SQL • HTML • CSS
Markdown • JSON • YAML • and many more...
```

### AI Providers

| Provider | Embeddings | LLM | API Key Required | Best For |
|----------|------------|-----|------------------|----------|
| **SentenceTransformers** | ✅ | ❌ | ❌ No | Privacy, offline use |
| **OpenAI** | ✅ | ✅ | ✅ Yes | High quality, fast |
| **Anthropic** | ❌ | ✅ | ✅ Yes | Detailed analysis |

### Ignored Patterns
The system automatically excludes:
- **Dependencies**: `node_modules`, `vendor`, `packages`
- **Build outputs**: `dist`, `build`, `target`, `bin`
- **Version control**: `.git`, `.svn`, `.hg`
- **IDE files**: `.vscode`, `.idea`, editor configs
- **Logs & temp**: `*.log`, `tmp`, cache directories
- **Media files**: Videos, audio, large binaries
- [View full list in config.yaml](config.yaml)

---

## 🏗️ Architecture

```
📦 codebase-reader/
├── 🎨 flexible_app.py              # Modern Streamlit UI
├── ⚙️ config.yaml                 # Comprehensive configuration
├── 🧠 src/
│   ├── 📊 codebase_analyzer.py     # Main orchestrator
│   ├── 🔄 provider_manager.py      # Dynamic provider system
│   ├── 🎯 flexible_embeddings.py   # Smart embedding management
│   ├── 🤖 flexible_llm.py         # LLM client with fallbacks
│   ├── 💾 vector_store.py          # ChromaDB integration
│   ├── 📚 codebase_reader.py       # Code parsing & chunking
│   └── 🔌 providers/               # Modular AI providers
│       ├── sentence_transformers_provider.py
│       ├── openai_provider.py
│       └── anthropic_provider.py
└── 📋 requirements.txt             # Dependencies
```

---

## 🤝 Contributing

We love contributions! Here's how to get started:

### 🐛 Bug Reports
Found a bug? [Open an issue](https://github.com/yourusername/codebase-reader/issues) with:
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)

### ✨ Feature Requests
Have an idea? [Open an issue](https://github.com/yourusername/codebase-reader/issues) with:
- Use case description
- Proposed solution
- Example usage

### 🔧 Development Setup
```bash
# Fork the repo and clone your fork
git clone https://github.com/yourusername/codebase-reader.git

# Create a feature branch
git checkout -b feature/amazing-feature

# Make your changes and test
python -m pytest tests/

# Commit with descriptive message
git commit -m "✨ Add amazing feature"

# Push and create pull request
git push origin feature/amazing-feature
```

### 📋 Adding New Providers
Want to add support for a new AI provider? Check out the [Provider Development Guide](docs/provider-development.md).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/codebase-reader&type=Date)](https://star-history.com/#yourusername/codebase-reader&Date)

---

## 🙏 Acknowledgments

- **[ChromaDB](https://www.trychroma.com/)** - Vector database for embeddings
- **[Streamlit](https://streamlit.io/)** - Beautiful web app framework
- **[SentenceTransformers](https://www.sbert.net/)** - Local embedding models
- **[OpenAI](https://openai.com/)** - GPT models and embeddings
- **[Anthropic](https://anthropic.com/)** - Claude AI models
- **[Tree-sitter](https://tree-sitter.github.io/)** - Code parsing

---

<div align="center">

**Made with ❤️ by the open source community**

[⭐ Star this repo](https://github.com/moncifem/codebase-reader) • [🐛 Report Bug](https://github.com/moncifem/codebase-reader/issues) • [✨ Request Feature](https://github.com/moncifem/codebase-reader/issues)

</div> 