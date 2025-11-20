# Future Code Quality Improvements

This document contains additional code quality tools and improvements to consider implementing later.

## Additional Tools to Consider

### 1. Vulture - Dead Code Detection
**Purpose:** Finds unused code in Python programs

**Installation:**
```bash
uv add --dev vulture
```

**Usage:**
```bash
vulture llm_gemini_code_assist.py tests/
```

**Pre-commit hook:**
```yaml
- repo: https://github.com/jendrikseipp/vulture
  rev: v2.11
  hooks:
    - id: vulture
```

**Configuration in pyproject.toml:**
```toml
[tool.vulture]
min_confidence = 80
paths = ["llm_gemini_code_assist.py", "tests/"]
```

---

### 2. Interrogate - Docstring Coverage
**Purpose:** Checks Python code for missing docstrings

**Installation:**
```bash
uv add --dev interrogate
```

**Usage:**
```bash
interrogate -v llm_gemini_code_assist.py
```

**CI Integration:**
```yaml
- name: Check docstring coverage
  run: uv run interrogate -v -f 80 llm_gemini_code_assist.py
```

**Configuration in pyproject.toml:**
```toml
[tool.interrogate]
ignore-init-method = true
ignore-init-module = true
ignore-magic = true
ignore-semiprivate = true
ignore-private = true
ignore-property-decorators = true
ignore-module = true
ignore-nested-functions = true
ignore-nested-classes = true
fail-under = 80
exclude = ["tests", "docs", "build"]
verbose = 1
```

---

### 3. Coverage Threshold Enforcement
**Purpose:** Ensure minimum test coverage is maintained

**Current setup:** You already have pytest-cov in dev dependencies

**CI Integration:**
```yaml
- name: Run tests with coverage
  run: uv run pytest --cov=llm_gemini_code_assist --cov-report=term-missing --cov-fail-under=80 tests
```

**Configuration in pyproject.toml:**
```toml
[tool.coverage.run]
source = ["llm_gemini_code_assist"]
omit = ["tests/*", "*/site-packages/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]
```

---

### 4. Additional Ruff Rules to Consider

Once you're comfortable with the current rule set, consider adding:

```toml
[tool.ruff.lint]
select = [
    # ... existing rules ...
    "D",    # pydocstyle (docstring conventions)
    "ANN",  # flake8-annotations (type annotations)
    "T20",  # flake8-print (prevent print statements)
    "SIM",  # flake8-simplify (code simplification)
    "TCH",  # flake8-type-checking (typing imports)
    "ARG",  # flake8-unused-arguments
    "PTH",  # flake8-use-pathlib (prefer pathlib over os.path)
    "ERA",  # eradicate (commented-out code)
    "PL",   # pylint rules
]
```

**Note:** These are more opinionated and may require significant refactoring.

---

### 5. Stricter Mypy Configuration

Once you have types throughout your codebase, consider:

```toml
[tool.mypy]
# Stricter type checking
disallow_untyped_defs = true
disallow_incomplete_defs = true
disallow_any_explicit = true
disallow_any_generics = true
strict = true  # Enables all optional error checking flags
```

---

### 6. Pre-commit CI
**Purpose:** Run pre-commit checks on all files in CI

**GitHub Action:**
```yaml
- name: Run pre-commit on all files
  run: |
    uv run pre-commit run --all-files
```

---

### 7. Dependency Scanning

Consider adding:
- **Safety** - Check for known security vulnerabilities in dependencies
- **pip-audit** - Audit Python packages for known vulnerabilities
- **Dependabot** - GitHub's automated dependency updates

**Example for pip-audit:**
```yaml
- name: Check for vulnerabilities
  run: |
    uv pip install pip-audit
    uv run pip-audit
```

---

## Implementation Priority

1. **Coverage threshold** - Quick win, you already have pytest-cov
2. **Interrogate** - If you want to improve documentation
3. **Vulture** - If you suspect dead code
4. **Stricter mypy** - After adding type hints throughout
5. **Additional ruff rules** - Gradually, as you refactor
6. **Dependency scanning** - Security improvement

---

## Notes

- Start with lower thresholds (e.g., 60-70% coverage) and gradually increase
- Some tools may report false positives - adjust configurations as needed
- Consider your team's preferences and coding style when choosing rules
- Too many strict rules at once can be overwhelming - add incrementally
