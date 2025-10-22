# Contributing to AutoPrepML

Thank you for your interest in contributing to AutoPrepML! We welcome contributions from the community.

## 🚀 Quick Start

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/autoprepml.git
cd autoprepml

# 3. Create a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 4. Install in development mode
pip install -e ".[dev]"

# 5. Create a feature branch
git checkout -b feature/your-feature-name

# 6. Make your changes and run tests
pytest tests/ -v

# 7. Commit and push
git add .
git commit -m "Description of your changes"
git push origin feature/your-feature-name

# 8. Open a Pull Request on GitHub
```

## 🎯 Ways to Contribute

### 1. Report Bugs
- Open an [issue](https://github.com/mdshoaibuddinchanda/autoprepml/issues) with detailed steps to reproduce
- Include Python version, OS, and error messages
- Provide a minimal code example if possible

### 2. Suggest Features
- Check [existing issues](https://github.com/mdshoaibuddinchanda/autoprepml/issues) first
- Describe the feature and its use case
- Explain why it would benefit the community

### 3. Submit Pull Requests
- Fix bugs or implement new features
- Write tests for your changes (maintain >90% coverage)
- Update documentation as needed
- Follow code style guidelines (see below)

### 4. Improve Documentation
- Fix typos or clarify explanations
- Add examples or tutorials
- Improve API documentation

### 5. Write Tests
- Increase test coverage
- Add edge case tests
- Improve test clarity

## 📝 Code Style Guidelines

### Python Style
- Follow [PEP 8](https://pep8.org/) guidelines
- Use meaningful variable and function names
- Keep functions focused and modular (max 50 lines)
- Add docstrings to all public functions and classes

### Docstring Format
```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Example:
        >>> function_name("test", 42)
        True
    """
    pass
```

### Testing Guidelines
- Write tests for all new features
- Use descriptive test names: `test_should_remove_duplicates_when_flag_is_true`
- Organize tests by module: `tests/test_<module_name>.py`
- Aim for >90% code coverage
- Run tests before committing: `pytest tests/ -v`

### Commit Messages
Use clear, descriptive commit messages:
```
Add text cleaning functionality for URLs and HTML

- Added clean_text() method with URL/HTML removal
- Added unit tests for text cleaning
- Updated documentation with examples
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_text.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=autoprepml --cov-report=html
```

### Check Coverage Report
```bash
# Open in browser (Windows)
start htmlcov/index.html

# macOS
open htmlcov/index.html

# Linux
xdg-open htmlcov/index.html
```

## 📦 Adding New Features

### 1. New Data Type Module
If adding a new data type (e.g., `ImagePrepML`):

1. Create `autoprepml/image.py` with the class
2. Add comprehensive tests in `tests/test_image.py`
3. Update `autoprepml/__init__.py` to export the class
4. Create `examples/demo_image.py` demo script
5. Update README with the new data type
6. Add documentation to `docs/`

### 2. New Cleaning Function
If adding a new cleaning function:

1. Add function to appropriate module (e.g., `cleaning.py`)
2. Write unit tests in corresponding test file
3. Update docstrings with examples
4. Add to relevant class methods
5. Update documentation

### 3. New CLI Feature
If adding a CLI option:

1. Update `autoprepml/cli.py` with new argument
2. Add logic to handle the new option
3. Write tests for CLI functionality
4. Update README CLI reference table
5. Add examples to documentation

## 🔍 Code Review Process

1. **Automated Checks**: GitHub Actions will run tests and linting
2. **Manual Review**: Maintainers will review code quality and design
3. **Feedback**: Address any requested changes
4. **Merge**: Once approved, your PR will be merged

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows PEP 8 style guidelines
- [ ] All tests pass: `pytest tests/ -v`
- [ ] New tests added for new features
- [ ] Code coverage >90%: `pytest tests/ --cov=autoprepml`
- [ ] Docstrings added to new functions/classes
- [ ] Documentation updated (README, docs/)
- [ ] Examples added if applicable
- [ ] Commit messages are descriptive
- [ ] No merge conflicts with main branch

## 🐛 Reporting Security Issues

If you discover a security vulnerability, please email directly instead of opening a public issue.

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

All contributors will be recognized in:
- GitHub contributors page
- CHANGELOG.md
- README.md acknowledgments section

## 📧 Questions?

- **GitHub Discussions**: [Ask questions](https://github.com/mdshoaibuddinchanda/autoprepml/discussions)
- **Issues**: [Technical questions](https://github.com/mdshoaibuddinchanda/autoprepml/issues)

Thank you for contributing to AutoPrepML! 🎉
