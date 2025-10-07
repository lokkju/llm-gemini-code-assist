# Cleanup, Testing, Linting, and UV Migration Plan

## Phase 1: Cleanup

### 1. Remove generated files
- Delete `llm_gemini_code_assist.egg-info/` and `__pycache__/` directories

### 2. Update README.md
- Replace all references to `llm-gemini` with `llm-gemini-code-assist`
- Update installation instructions (no PyPI yet, install from git)
- Document OAuth authentication flow (`gemini auth`)
- Update model prefix from `gemini/` to `code-assist/`
- Remove API key instructions (OAuth only)
- Update repository URLs
- Remove cogapp model listing (not needed for this plugin)
- Add Code Assist API specific documentation

### 3. Fix test imports
- Update `tests/test_gemini.py` line 10: `from llm_gemini` → `from llm_gemini_code_assist`
- Update test model names to use `code-assist/` prefix
- Remove tests that rely on API keys (OAuth only)

### 4. Update GitHub workflows
- Fix `cache-dependency-path`: `setup.py` → `pyproject.toml`
- Remove `cogapp --check` from test workflow
- Update PyPI package name in publish workflow

### 5. Stage git changes
- Add `llm_gemini_code_assist.py` and `tests/test_basic_oauth.py`
- Remove `llm_gemini.py`
- Commit pyproject.toml changes

## Phase 2: Testing

### 1. Update existing tests
- Modify `test_gemini.py` for Code Assist API format
- Mock OAuth credentials for tests
- Update cassettes to capture Code Assist API responses
- Add environment variable for test OAuth path

### 2. Add new tests
- OAuth credential loading (valid, expired, missing)
- Project assignment caching
- Request/response wrapping
- Error handling (no credentials, expired token, API errors)
- Model listing with `code-assist/` prefix

### 3. Recording test cassettes
- Set up test OAuth credentials
- Record interactions with Code Assist API
- Verify cassette filtering for sensitive data

## Phase 3: Linting & Code Quality

### 1. Add ruff configuration to `pyproject.toml`
- Configure line length, linting rules
- Add import sorting

### 2. Format code
- Run `ruff format` on all Python files
- Run `ruff check --fix` for auto-fixable issues

### 3. Type hints
- Add missing type hints to functions
- Configure mypy in `pyproject.toml`
- Run mypy and fix type issues

### 4. Add pre-commit config (optional)
- Set up `.pre-commit-config.yaml` with ruff

## Phase 4: UV Migration

### 1. Update pyproject.toml
- Add `[build-system]` if missing
- Ensure compatibility with uv

### 2. Update GitHub workflows
- Replace `pip install` with `uv pip install` or `uv sync`
- Add uv cache configuration

### 3. Generate uv.lock
- Run `uv pip compile pyproject.toml` → `requirements.txt`
- Or use `uv sync` for modern workflow

### 4. Update development docs
- Replace pip commands with uv equivalents
- Document uv installation process

## Estimated Order of Execution

1. Cleanup (20 mins)
2. Fix tests enough to run (15 mins)
3. Add linting/formatting (10 mins)
4. UV migration (15 mins)
5. Complete testing suite (30 mins)

**Total: ~90 minutes**
