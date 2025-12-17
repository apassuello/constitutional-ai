# GitHub Actions CI/CD

This directory contains the CI/CD workflows for the Constitutional AI project.

## Workflows

### 1. CI (`ci.yml`)
**Triggers:** Push and Pull Requests to `main`, `master`, `develop` branches

**What it does:**
- Runs tests on Python 3.8, 3.9, 3.10, 3.11, and 3.12
- Generates code coverage reports (XML, HTML, and terminal)
- Uploads coverage to Codecov (Python 3.11 only)
- Stores HTML coverage report as artifact (available for 30 days)
- Adds coverage summary to PR comments

**Coverage Requirements:**
- Target: 80% coverage
- Configured in `codecov.yml`

### 2. Code Quality (`code-quality.yml`)
**Triggers:** Push and Pull Requests to `main`, `master`, `develop` branches

**What it checks:**
- **Black**: Code formatting (100 char line length)
- **isort**: Import statement sorting
- **Flake8**: PEP 8 style compliance
- **Ruff**: Fast Python linter with multiple rule sets
- **MyPy**: Static type checking (currently non-blocking)

### 3. Dependency Review (`dependency-review.yml`)
**Triggers:** Pull Requests only

**What it does:**
- Scans for vulnerable or malicious dependencies
- Reviews license compliance
- Provides security alerts for dependency changes

### 4. Dependabot (`../dependabot.yml`)
**Schedule:** Weekly

**What it does:**
- Automatically updates GitHub Actions versions
- Automatically updates Python dependencies
- Creates PRs for dependency updates

## Local Development

### Run tests with coverage
```bash
pytest tests/ -v --cov=constitutional_ai --cov-report=html --cov-report=term
```

### Check code quality
```bash
# Format check
black --check constitutional_ai/ tests/ examples/ demos/

# Auto-format
black constitutional_ai/ tests/ examples/ demos/

# Sort imports
isort constitutional_ai/ tests/ examples/ demos/

# Lint with Flake8
flake8 constitutional_ai/ tests/ examples/ demos/ --max-line-length=100 --extend-ignore=E203,W503

# Lint with Ruff
ruff check constitutional_ai/ tests/ examples/ demos/

# Type check
mypy constitutional_ai/ --ignore-missing-imports
```

### Run all checks at once
```bash
# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Or run manually
black constitutional_ai/ tests/ examples/ demos/ && \
isort constitutional_ai/ tests/ examples/ demos/ && \
flake8 constitutional_ai/ tests/ examples/ demos/ --max-line-length=100 --extend-ignore=E203,W503 && \
ruff check constitutional_ai/ tests/ examples/ demos/ && \
pytest tests/ -v --cov=constitutional_ai
```

## Configuration Files

- **pyproject.toml**: Main configuration for Black, isort, MyPy, pytest, Ruff, and coverage
- **codecov.yml**: Codecov configuration (coverage targets, ignored files)
- **.github/dependabot.yml**: Dependabot configuration

## Codecov Integration

To enable Codecov:
1. Go to https://codecov.io/
2. Sign in with GitHub
3. Enable the repository
4. Add `CODECOV_TOKEN` to repository secrets (if private repo)

## Badge URLs

Update the `yourusername` placeholder in README.md with your actual GitHub username/org:

```markdown
[![CI](https://github.com/yourusername/constitutional-ai/workflows/CI/badge.svg)](https://github.com/yourusername/constitutional-ai/actions/workflows/ci.yml)
[![Code Quality](https://github.com/yourusername/constitutional-ai/workflows/Code%20Quality/badge.svg)](https://github.com/yourusername/constitutional-ai/actions/workflows/code-quality.yml)
[![codecov](https://codecov.io/gh/yourusername/constitutional-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/constitutional-ai)
```
