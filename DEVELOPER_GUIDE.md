# Developer Guide

Development standards and workflows for the Omnes energy system modeling platform.

## Setup

```bash
git clone https://github.com/ntatrishvili/Omnes.git
cd Omnes
python -m venv venv
# Activate virtual environment (platform-specific)
pip install -r requirements.txt
python main.py  # Verify installation
```

## Testing

### Run all tests with coverage
```bash
python -m pytest --cov=app --cov-report=html --cov-report=term-missing -v
```

### Specific test commands
```bash
python -m pytest test/test_conversion/ -v          # Conversion module only
python -m pytest test/path/to/test_file.py -v      # Single test file
python -m pytest -v --tb=short                     # Concise error output
```

Coverage reports are generated in `htmlcov/index.html`.

## Code Quality

### Pre-commit checklist
```bash
black .                                    # Code formatting
isort .                                    # Import sorting  
python -m pytest -v                       # All tests pass
```

### Optional tools
```bash
mypy app/                                  # Type checking
flake8 app/                               # Linting
pylint app/                               # Additional linting
```

## Development Workflow

1. **Branch**: `git checkout -b feature/feature-name`
2. **Develop**: Write code + tests, following existing patterns
3. **Quality check**: Run pre-commit checklist
4. **Commit**: Descriptive messages following conventional commits
5. **Push**: Create pull request with clear description

## Architecture

```
app/
├── conversion/         # Data transformation utilities
│   ├── pulp_converter.py
│   ├── validation_utils.py
│   └── pulp_utils.py
├── infra/             # Core infrastructure
├── model/             # Data models
└── operation/         # Optimization logic
test/                  # Test suite mirroring app/ structure
```

## Standards

- **Docstrings**: NumPy style (see `DOCSTRING_STYLE_GUIDE.md`)
- **Type hints**: Required for all public functions
- **Testing**: Comprehensive tests for new functionality
- **Coverage**: Target >90% for new code
- **Naming**: Follow PEP 8 conventions

## Debugging

```bash
python -m pytest test/specific_test.py::TestClass::test_method -v -s  # Single test with output
python -m pytest --pdb                                               # Drop to debugger on failure
```

## Release Process

1. All tests pass with good coverage
2. Documentation updated
3. Pull request with detailed description
4. Code review and approval
5. Merge to main

