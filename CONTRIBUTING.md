# 🤝 Contributing to Codebase Reader & Analyzer

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## 🚀 Development Process

We use GitHub to sync code, track issues and feature requests, and accept pull requests.

### 🔧 Setting Up Development Environment

1. **Fork the repo** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/codebase-reader.git
   cd codebase-reader
   ```

3. **Set up Python environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

5. **Set up pre-commit hooks** (recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 📝 Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** following our coding standards

3. **Add or update tests** as appropriate

4. **Update documentation** if you're adding/changing functionality

5. **Ensure all tests pass**:
   ```bash
   python -m pytest tests/
   ```

6. **Commit your changes** with descriptive messages:
   ```bash
   git commit -m "✨ Add amazing feature that does X"
   ```

7. **Push to your fork** and submit a pull request:
   ```bash
   git push origin feature/amazing-feature
   ```

### Pull Request Guidelines

- **Use descriptive titles** and include relevant emojis (✨ for features, 🐛 for bugs, 📚 for docs)
- **Fill out the pull request template** completely
- **Link to any relevant issues** using "Fixes #123" or "Closes #123"
- **Keep changes focused** - one feature/fix per PR
- **Add tests** for new functionality
- **Update documentation** as needed

## 🐛 Bug Reports

Great bug reports are extremely helpful! Please include:

### Bug Report Template
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. Windows 10, macOS 12.1, Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- Browser: [e.g. Chrome 95.0]
- Version: [e.g. v1.2.0]

**Additional context**
Any other context about the problem.
```

## ✨ Feature Requests

We love feature ideas! Please include:

### Feature Request Template
```markdown
**Is your feature request related to a problem?**
A clear description of the problem you're trying to solve.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Mockups, examples, or additional details.
```

## 🏗️ Code Contribution Guidelines

### Code Style

- **Follow PEP 8** for Python code style
- **Use type hints** where possible
- **Write descriptive variable names**
- **Keep functions small and focused**
- **Add docstrings** to all public functions and classes

### Architecture Principles

- **Modular design**: Keep components loosely coupled
- **Provider pattern**: Follow existing patterns for new AI providers
- **Graceful degradation**: Handle missing dependencies elegantly
- **Configuration driven**: Use `config.yaml` for settings
- **Error handling**: Provide clear, actionable error messages

### Adding New AI Providers

To add support for a new AI provider:

1. **Create provider class** in `src/providers/`:
   ```python
   from .base_provider import BaseProvider
   
   class NewProvider(BaseProvider):
       name = "new_provider"
       display_name = "New Provider"
       # ... implement required methods
   ```

2. **Register in provider manager**: Add to `src/provider_manager.py`

3. **Update configuration**: Add settings to `config.yaml`

4. **Add tests**: Create tests in `tests/providers/`

5. **Update documentation**: Add to README and docs

### Testing

- **Write tests** for new functionality
- **Use pytest** for testing framework
- **Mock external dependencies** (APIs, file systems)
- **Test error conditions** and edge cases
- **Maintain >80% code coverage**

Run tests with:
```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src --cov-report=html
```

## 📚 Documentation

- **Update README.md** for user-facing changes
- **Add docstrings** following Google/NumPy style
- **Include code examples** in documentation
- **Keep docs up to date** with code changes

## 🔍 Code Review Process

All submissions go through code review:

1. **Automated checks** must pass (tests, linting)
2. **Maintainer review** for code quality and architecture
3. **Community feedback** on significant changes
4. **Documentation review** for completeness

## 🎯 Good First Issues

Look for issues labeled `good first issue` or `help wanted`. These are:
- **Well-defined** problems
- **Self-contained** tasks
- **Good learning opportunities**
- **Guided by maintainers**

## 📞 Community

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bugs and feature requests
- **Code Review**: On pull requests

## 🏆 Recognition

Contributors are recognized in:
- **README.md**: Major contributors
- **CHANGELOG.md**: Feature and fix credits
- **GitHub**: Contributor graphs and stats

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Thank You!

Your contributions make this project better for everyone. Every pull request, issue report, and suggestion helps build a more robust tool for developers worldwide.

Happy coding! 🚀 