# Contributing to AI Resume Screener

Thank you for your interest in contributing to the AI Resume Screener project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Error messages and logs

### Suggesting Features

1. **Check the roadmap** in README.md first
2. **Open a feature request** with:
   - Clear description of the feature
   - Use cases and benefits
   - Potential implementation approach
   - Any relevant examples or mockups

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request**

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool (venv, conda, etc.)

### Local Development

```bash
# Clone your fork
git clone https://github.com/your-username/assignment.git
cd assignment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Download required models
python -m spacy download en_core_web_sm
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_resume_parser.py

# Run with verbose output
pytest -v
```

### Code Quality Checks

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

## 📝 Coding Standards

### Code Style

- **Formatter**: Black (line length: 127)
- **Import sorting**: isort with Black profile
- **Linting**: flake8 with custom configuration
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style docstrings

### Example Code Style

```python
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ExampleClass:
    """Example class demonstrating coding standards.
    
    This class shows the expected code style including type hints,
    docstrings, and error handling patterns.
    
    Args:
        param1: Description of parameter
        param2: Optional parameter with default
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None):
        self.param1 = param1
        self.param2 = param2 or 0
    
    def process_data(self, data: List[Dict[str, Any]]) -> List[str]:
        """Process input data and return results.
        
        Args:
            data: List of dictionaries containing input data
            
        Returns:
            List of processed strings
            
        Raises:
            ValueError: If data is empty or invalid
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        results = []
        for item in data:
            try:
                processed = self._process_item(item)
                results.append(processed)
            except Exception as e:
                logger.error(f"Error processing item {item}: {str(e)}")
                continue
        
        return results
    
    def _process_item(self, item: Dict[str, Any]) -> str:
        """Private method to process individual item."""
        return str(item.get('value', ''))
```

### Testing Standards

- **Framework**: pytest
- **Coverage**: Aim for 90%+ on new code
- **Test types**: Unit tests, integration tests, property-based tests
- **Naming**: `test_function_name_scenario`
- **Structure**: Arrange-Act-Assert pattern

### Example Test

```python
import pytest
from src.example_module import ExampleClass


class TestExampleClass:
    """Test cases for ExampleClass."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.example = ExampleClass("test_param")
    
    def test_process_data_success(self):
        """Test successful data processing."""
        # Arrange
        input_data = [{"value": "test1"}, {"value": "test2"}]
        expected = ["test1", "test2"]
        
        # Act
        result = self.example.process_data(input_data)
        
        # Assert
        assert result == expected
    
    def test_process_data_empty_input(self):
        """Test processing with empty input raises ValueError."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            self.example.process_data([])
    
    @pytest.mark.parametrize("input_data,expected", [
        ([{"value": "a"}], ["a"]),
        ([{"value": "b"}, {"value": "c"}], ["b", "c"]),
    ])
    def test_process_data_parametrized(self, input_data, expected):
        """Test data processing with various inputs."""
        result = self.example.process_data(input_data)
        assert result == expected
```

## 🏗️ Architecture Guidelines

### Project Structure

```
src/
├── models/          # Data models and schemas
├── parsers/         # Resume parsing components
├── embeddings/      # Embedding generation and caching
├── ranking/         # Ranking and scoring algorithms
├── utils/           # Utility functions
└── api/             # API endpoints (future)

tests/
├── unit/            # Unit tests
├── integration/     # Integration tests
└── fixtures/        # Test data and fixtures
```

### Design Principles

1. **Single Responsibility**: Each class/function has one clear purpose
2. **Dependency Injection**: Use dependency injection for testability
3. **Error Handling**: Graceful error handling with logging
4. **Type Safety**: Use type hints throughout
5. **Documentation**: Clear docstrings and comments
6. **Performance**: Consider performance implications
7. **Extensibility**: Design for future enhancements

### Adding New Features

When adding new features:

1. **Design first**: Consider the architecture and interfaces
2. **Write tests**: Test-driven development preferred
3. **Document**: Update docstrings and README
4. **Consider backwards compatibility**
5. **Add configuration options** when appropriate
6. **Include error handling and logging**

## 📋 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] No merge conflicts with main branch
- [ ] Commit messages are clear and descriptive

### PR Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation review**
5. **Approval** and merge

## 🐛 Bug Reports

### Good Bug Report Includes

- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Environment details**:
  - OS and version
  - Python version
  - Package versions
  - Browser (for UI issues)
- **Error messages** and stack traces
- **Screenshots** if applicable

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. Windows 10, macOS 12.0, Ubuntu 20.04]
- Python Version: [e.g. 3.9.7]
- Package Version: [e.g. 1.0.0]

**Additional Context**
Any other context about the problem.
```

## 🚀 Feature Requests

### Good Feature Request Includes

- **Clear description** of the feature
- **Problem it solves** or use case
- **Proposed solution** (if you have one)
- **Alternatives considered**
- **Additional context** or examples

## 📚 Documentation

### Types of Documentation

1. **Code documentation**: Docstrings and comments
2. **API documentation**: Function/class interfaces
3. **User documentation**: README, tutorials
4. **Developer documentation**: Architecture, contributing

### Documentation Standards

- **Clear and concise** language
- **Examples** where helpful
- **Up-to-date** with code changes
- **Proper formatting** (Markdown, reStructuredText)

## 🏷️ Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🤔 Questions?

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private inquiries

## 🙏 Recognition

Contributors will be recognized in:

- **README.md**: Contributors section
- **CHANGELOG.md**: Release notes
- **GitHub**: Contributor graphs and statistics

Thank you for contributing to AI Resume Screener! 🎉