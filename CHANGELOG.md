## v0.27.0 (2025-12-13)

### Feat

- Configure commitizen and versioning

## v0.26.3 (2025-12-13)

### Feat

- updating version

### Fix

- Correct AsyncGenerator type annotation for Python 3.10 compatibility
- Update mypy strictness and remove cogapp check

## v0.26.2 (2025-11-19)

### Feat

- Update README and version for Gemini Code Assist plugin - Updates the `README.md` to reflect the new authentication command `llm gemini-ca auth`.

## v0.26.1 (2025-11-19)

### Feat

- Add uv setup script for easy dev environment configuration
- Add linting and type checking to CI Integrates ruff and mypy into the CI pipeline for improved code quality. - **CI:** Add a `lint` job to `.github/workflows/test.yml` that runs ruff and mypy. - **Pre-commit:** Update `.pre-commit-config.yaml` to use newer versions of pre-commit hooks and add mypy. - **Dependencies:** Update `pyproject.toml` to include development dependencies for linting and type checking. - **Tests:** Add a `tests/__init__.py` file and update test fixtures to improve test organization and reliability. - **Error Handling:** Improve error handling and logging in authentication-related functions within `llm_gemini_code_assist.py`.
- Update gemini code assist plugin - Update README to reflect changes for the code assist API. - Remove obsolete cleanup plan items.
- Remove API key auth and update OAuth logic This commit refactors the Gemini Code Assist plugin to exclusively use OAuth 2.0 for authentication, removing all API key-related code and documentation. Changes include: - **CLEANUP_PLAN.md**: Removed instructions related to API keys and updated documentation references to OAuth. - **llm_gemini_code_assist.py**:     - Removed functions and logic for handling API keys.     - Renamed `credentials_from_oauth_creds_json` to `credentials_from_oauth_creds_data` for clarity.     - Updated `get_oauth_credentials` to use the new function name and correctly handle token expiration and refresh.     - Corrected the format of scopes when saving credentials to the cache.     - Raised `AuthenticationError` instead of `OAuthError` for consistency. - **tests/**:     - Removed VCR cassette `test_authenticate.yaml` as it's no longer relevant.     - Updated `test_oauth_token_refresh_failed_request.yaml` to reflect the new error response from Google's API.     - Censor `project` field in cassette requests.     - Updated `conftest.py` to include `mock_oauth_credentials` fixture for better isolation and control over credential mocking during tests. This fixture handles both recording and playback scenarios for OAuth.     - Updated existing tests to use the new fixture and reflect the removal of API key authentication.     - Added checks to ensure `LLM_USER_PATH` is set correctly for tests using `mock_oauth_credentials`.     - Improved error handling and validation in `mock_oauth_credentials` when copying real credentials during recording.     - Updated test assertions to expect `AuthenticationError` instead of `OAuthError`.     - Added `test_cli_gemini_ca_models` to test CLI model listing without API keys.
- Refactor Gemini authentication flow Refactors the authentication logic for the Gemini Code Assist to be more robust and testable. Key changes include: - **Centralized `authenticate` function:** This function now orchestrates the entire authentication process, providing a clear flow for credential discovery. - **Modular credential loading:** Separate helper functions (`_load_gemini_cli_credentials`, `_run_oauth_flow`) handle specific credential sources (gemini-cli cache, full OAuth flow). - **Credential validation and refresh:** A new helper `_validate_and_refresh_creds` robustly checks credential validity and handles token refreshes, logging issues encountered. - **Improved error handling:** Custom `AuthenticationError` is used consistently, and exceptions during the OAuth flow are properly caught and wrapped. - **Testability enhancements:**     - `conftest.py` now includes a `shared_tmpdir` fixture for shared temporary directories and `mock_llm_user_path` to properly set the `LLM_USER_PATH` environment variable for tests, isolating test environments.     - The `test_authenticate` function in `test_llm_gemini_code_assist.py` is updated to use these fixtures and test the new `authenticate` function. - **Reduced `click` dependency:** The core authentication logic is less reliant on `click` for prompts and output, making it more reusable. `click` is now primarily used in the CLI command `auth`. - **Clearer logging:** Added more informative log messages throughout the authentication process.
- Update CI to use Python 3.10+ and uv - Update CI workflows to use Python versions 3.10 and above, removing 3.9. - Switch from pip to uv for faster dependency management in CI. - Update actions/checkout and actions/setup-python to their latest versions. - Modify `cache-dependency-path` to `pyproject.toml` in CI workflows. - Add `tests.asserts` module for better assertion handling. - Refactor OAuth handling in `llm_gemini_code_assist.py` for clarity and robustness. - Introduce `AuthenticationError` and `clear_cache` command for better error handling and cache management. - Update several package dependencies to their latest versions. - Add `pytest-dependency` to dev dependencies.
- Set up pre-commit hooks and update Python version requirement This commit introduces pre-commit hooks to enforce code style and quality. It also updates the minimum Python version requirement from 3.11 to 3.10 and removes a deleted test cassette file. Changes: - Add `.pre-commit-config.yaml` to define pre-commit hooks for trailing whitespace, end-of-file fixer, YAML checking, large file detection, Ruff linting and formatting, and Gitleaks secret scanning. - Update `requires-python` in `pyproject.toml` from `>=3.11` to `>=3.10`. - Add `pre-commit` to the `dev` dependency group in `pyproject.toml`. - Remove the deleted test cassette file `tests/cassettes/test_gemini_code_assist/test_prompt_async.yaml`. - Remove the deleted test cassette file `tests/cassettes/test_gemini_code_assist/test_prompt_sync.yaml`. - Remove the deleted test cassette file `tests/cassettes/test_gemini_code_assist/test_tools.yaml`. - Add a new test cassette file `tests/cassettes/test_llm_gemini_code_assist/test_prompt_with_pydantic_schema.yaml`. - Add sys.path manipulation to `tests/conftest.py` to ensure correct module import. - Remove the deleted test file `tests/test_basic_oauth.py`. - Modify `tests/test_llm_gemini_code_assist.py` to make `test_prompt_with_pydantic_schema` a regular function and adjust assertions.

### Fix

- Add form-encoded body censoring for OAuth tests
- Remove unsupported tool.uv.scripts and fix syntax errors
- Increment version in pyproject.toml and uv.lock The `pyproject.toml` file and the `uv.lock` file have been updated to reflect a new version number. - `pyproject.toml`: The `version` in the `[project]` table has been updated from `0.26.0` to `0.26.1`.
- README updates for Gemini 2.0 (#57)
