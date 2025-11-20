# Cleanup, Testing, Linting, and UV Migration Plan

## Phase 1: Cleanup

### 4. Update GitHub workflows
- Fix `cache-dependency-path`: `setup.py` → `pyproject.toml`
- Remove `cogapp --check` from test workflow
- Update PyPI package name in publish workflow

## Phase 2: Testing

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
