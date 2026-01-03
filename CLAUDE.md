# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM plugin (`llm-gemini-code-assist`) that provides access to Google's Gemini models via the Code Assist API with OAuth authentication. It's a fork of `llm-gemini` modified to use OAuth instead of API keys.

## Development Commands

```bash
# Install dependencies and set up dev environment
uv sync
uv run pre-commit install

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_llm_gemini_code_assist.py::test_name

# Run tests with coverage
uv run pytest --cov

# Run pre-commit hooks manually
uv run pre-commit run --all-files

# Type checking
uv run mypy llm_gemini_code_assist.py

# Linting/formatting
uv run ruff check --fix .
uv run ruff format .
```

## Architecture

Single-file plugin (`llm_gemini_code_assist.py`) that registers with the LLM framework via entry points.

### Key Components

- **Authentication**: OAuth 2.0 flow using credentials from `~/.gemini/oauth_creds.json` (gemini-cli compatible) or plugin cache. The `authenticate()` function handles credential loading, validation, and refresh.

- **Model Classes**: `_SharedGemini` base class with shared logic, `GeminiPro` (sync) and `AsyncGeminiPro` (async) for execution. Models are registered with `gemini-ca/` prefix.

- **Code Assist API Wrapper**: Requests are wrapped in Code Assist format (`wrap_code_assist_request`) and responses unwrapped (`unwrap_code_assist_response`). The API endpoint is `cloudcode-pa.googleapis.com`.

- **Plugin Cache**: Credentials and project IDs cached in `<llm_user_dir>/gemini-code-assist/`.

### Testing

Tests use `pytest-recording` (VCR) for HTTP interaction recording. Key fixtures in `conftest.py`:
- `mock_oauth_credentials`: Mocks credentials during playback, copies real credentials during recording
- `shared_mock_llm_user_path`/`mock_llm_user_path`: Isolate test directories

Recording new tests requires real OAuth credentials from `llm gemini-ca auth`.

## Code Style

- Line length: 100 characters
- Ruff for linting (E, F, W, I, N, UP, B, A, C4, PT, S, RUF rules)
- Mypy strict mode
- Python 3.10+

## Releasing

Releases are tag-driven. Version is derived automatically from git tags via `uv-dynamic-versioning`.

```bash
# 1. Ensure tests and pre-commit pass
uv run pytest
uv run pre-commit run --all-files

# 2. Create and push a version tag
git tag v0.27.0
git push origin v0.27.0
```

The `publish.yml` workflow will automatically:
- Build the package
- Generate changelog from commits
- Create a GitHub Release with artifacts
- Publish to PyPI (uses OIDC trusted publishing)
