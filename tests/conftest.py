import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

import pytest


def before_record_request(request):
    """Censor sensitive data from request bodies before recording/matching"""
    if request.body:
        # Try JSON bodies first
        try:
            body = json.loads(request.body)
            # Censor sensitive fields - these will be censored in both
            # the saved cassette AND during playback matching
            if "user_prompt_id" in body:
                body["user_prompt_id"] = "CENSORED-USER-PROMPT-ID"
            if "project" in body:
                body["project"] = "CENSORED-PROJECT-ID"
            request.body = json.dumps(body).encode("utf-8")
        except (json.JSONDecodeError, AttributeError):
            # Handle form-encoded bodies (OAuth token requests)
            from urllib.parse import parse_qs, urlencode

            try:
                if isinstance(request.body, bytes):
                    body_str = request.body.decode("utf-8")
                else:
                    body_str = request.body

                # Parse form data
                params = parse_qs(body_str, keep_blank_values=True)

                # Censor OAuth-related fields
                if "client_id" in params:
                    params["client_id"] = ["CENSORED-CLIENT-ID"]
                if "client_secret" in params:
                    params["client_secret"] = ["CENSORED-CLIENT-SECRET"]
                if "refresh_token" in params:
                    params["refresh_token"] = ["CENSORED-REFRESH-TOKEN"]
                if "access_token" in params:
                    params["access_token"] = ["CENSORED-ACCESS-TOKEN"]

                # Rebuild the body
                request.body = urlencode(params, doseq=True)
            except (ValueError, AttributeError):
                pass
    return request


@pytest.fixture(scope="module")
def vcr_config():
    return {
        # 1. This prevents your secret token from being saved in the YAML file
        "filter_headers": [("authorization", "CENSORED-OAUTH-TOKEN")],
        "before_record_request": before_record_request,
        # 2. This tells VCR to ignore the auth header when matching
        #    This is the key to making playback work!
        "match_on": [
            "method",
            "scheme",
            "host",
            "port",
            "path",
            "query",
            "body",
        ],
    }


@pytest.fixture(scope="module")
def module_tmp_path(tmp_path_factory):
    """
    Creates a temporary directory shared across all tests in a module.

    The directory is created by the tmp_path_factory and will be unique
    for each module. It's automatically cleaned up by pytest after
    all tests in the module have run.
    """
    # Create a base temporary directory for the module
    module_tmp_dir = tmp_path_factory.mktemp("shared_module_dir")

    # Yield the path to the tests
    return module_tmp_dir

    # No cleanup code is needed here; pytest's tmp_path_factory
    # handles the removal of the directory and its contents.


@pytest.fixture(scope="module")
def shared_mock_llm_user_path(module_tmp_path):
    """
    Creates a shared 'llm.datasette.io' user directory for all tests in the module.
    Use this when tests need to share state (e.g., when using test dependencies).
    Sets LLM_USER_PATH for the module scope.
    """
    import os

    user_dir = module_tmp_path / "llm.datasette.io"
    user_dir.mkdir()

    # Set environment variable for the entire module
    old_value = os.environ.get("LLM_USER_PATH")
    os.environ["LLM_USER_PATH"] = str(user_dir)

    yield user_dir

    # Cleanup: restore original environment variable
    if old_value is None:
        os.environ.pop("LLM_USER_PATH", None)
    else:
        os.environ["LLM_USER_PATH"] = old_value


@pytest.fixture
def mock_llm_user_path(tmp_path, monkeypatch):
    """
    Creates an isolated 'llm.datasette.io' user directory for a single test.
    Use this when tests need their own clean directory for isolation.
    Function-scoped - each test gets its own directory.
    """
    user_dir = tmp_path / "llm.datasette.io"
    user_dir.mkdir()

    # monkeypatch automatically reverts this after the test
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))

    return user_dir


@pytest.fixture
def mock_oauth_credentials(monkeypatch, request, record_mode):
    """
    Provides mock OAuth credentials for tests that need them.
    During VCR playback, this mocks the credential loading to prevent any HTTP requests.
    During VCR recording, copies real credentials from the actual environment.

    Requires that LLM_USER_PATH has been set by one of
    - shared_mock_llm_user_path
    - mock_llm_user_path.
    """
    import os
    import shutil
    from datetime import datetime, timedelta
    from pathlib import Path

    import llm
    from google.oauth2.credentials import Credentials

    import llm_gemini_code_assist

    # Safety check: ensure LLM_USER_PATH is set to a test directory
    llm_user_path = os.environ.get("LLM_USER_PATH")
    if not llm_user_path or "pytest" not in str(llm_user_path):
        raise RuntimeError(
            "mock_oauth_credentials requires LLM_USER_PATH to be set to a test directory. "
            "Use either @pytest.mark.usefixtures('shared_mock_llm_user_path') or "
            "@pytest.mark.usefixtures('mock_llm_user_path') on your test."
        )

    # Check if the test uses VCR
    uses_vcr = request.node.get_closest_marker("vcr") is not None

    # Determine if we're in playback mode
    is_playback = False
    if uses_vcr:
        # For VCR tests, check record mode first, then cassette existence
        test_name = request.node.name
        cassette_path = (
            Path(__file__).parent
            / "cassettes"
            / "test_llm_gemini_code_assist"
            / f"{test_name}.yaml"
        )
        cassette_exists = cassette_path.exists()

        # If record_mode is 'none', we must be in playback mode
        if record_mode == "none":
            if not cassette_exists:
                pytest.fail(
                    f"VCR cassette not found at {cassette_path} but --record-mode=none "
                    "was specified. Either create the cassette by recording the test first, "
                    "or use a different record mode."
                )
            is_playback = True
        else:
            # For other record modes (or default), check if cassette exists
            is_playback = cassette_exists
    else:
        # For non-VCR tests (like test_authenticate), check pytest record mode
        # If --record-mode=none, we're in playback mode
        is_playback = record_mode == "none"

    if is_playback:
        # During playback: mock credential functions to prevent any HTTP requests
        def mock_get_oauth_credentials():
            # Create credentials that are valid and won't trigger refresh
            # Note: Credentials.token must be set for valid=True
            # Use a future expiry time (1 hour from now)
            creds = Credentials(
                token="mock_access_token_12345",
                refresh_token="mock_refresh_token_67890",
                id_token="mock_id_token",
                token_uri="https://oauth2.googleapis.com/token",
                client_id=llm_gemini_code_assist.CLIENT_ID,
                client_secret=llm_gemini_code_assist.CLIENT_SECRET,
                scopes=llm_gemini_code_assist.SCOPES,
                expiry=datetime.utcnow() + timedelta(hours=1),
            )
            return creds

        def mock_validate_and_refresh_creds(creds):
            # In playback mode, just return the credentials as valid without attempting refresh
            if creds is None:
                return None
            return creds

        def mock_run_oauth_flow():
            # In playback mode, return mock credentials without triggering browser flow
            return mock_get_oauth_credentials()

        # Mock get_code_assist_project to return fake values
        # The actual project ID doesn't matter since it gets censored in VCR
        def mock_get_code_assist_project(credentials):
            return ("mock-project-id", "PREMIUM")

        monkeypatch.setattr(
            llm_gemini_code_assist, "get_oauth_credentials", mock_get_oauth_credentials
        )
        monkeypatch.setattr(
            llm_gemini_code_assist, "_validate_and_refresh_creds", mock_validate_and_refresh_creds
        )
        monkeypatch.setattr(llm_gemini_code_assist, "_run_oauth_flow", mock_run_oauth_flow)
        monkeypatch.setattr(
            llm_gemini_code_assist, "get_code_assist_project", mock_get_code_assist_project
        )
    else:
        # During recording: copy real credentials from actual environment
        # User should have run `llm gemini-ca auth` before recording
        real_llm_dir = Path.home() / ".config" / "io.datasette.llm"
        real_plugin_cache = real_llm_dir / llm_gemini_code_assist.GEMINI_CODE_ASSIST_PLUGIN_SLUG
        real_oauth_file = real_plugin_cache / llm_gemini_code_assist.OAUTH_CREDENTIALS_FILE

        if not real_oauth_file.exists():
            raise FileNotFoundError(
                f"No real OAuth credentials found at {real_oauth_file}. "
                "Please run `llm gemini-ca auth` before recording VCR cassettes."
            )

        # Validate the credentials file before copying
        try:
            creds_data = json.loads(real_oauth_file.read_text())
            # Validate required fields exist
            required_fields = ["access_token", "refresh_token"]
            missing_fields = [field for field in required_fields if field not in creds_data]
            if missing_fields:
                raise ValueError(
                    "OAuth credentials file is missing required fields: "
                    f"{', '.join(missing_fields)}. "
                    "Please run `llm gemini-ca auth` to generate valid credentials."
                )

            # Validate that the values are non-empty strings
            empty_fields = [field for field in required_fields if not creds_data.get(field)]
            if empty_fields:
                raise ValueError(
                    f"OAuth credentials file has empty values for: {', '.join(empty_fields)}. "
                    "Please run `llm gemini-ca auth` to generate valid credentials."
                )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"OAuth credentials file at {real_oauth_file} is not valid JSON: {e}. "
                "Please run `llm gemini-ca auth` to regenerate credentials."
            ) from e

        # Copy real credentials to test environment (uses LLM_USER_PATH)
        test_llm_user_dir = llm.user_dir()
        test_plugin_cache = (
            test_llm_user_dir / llm_gemini_code_assist.GEMINI_CODE_ASSIST_PLUGIN_SLUG
        )
        test_plugin_cache.mkdir(exist_ok=True)
        test_oauth_file = test_plugin_cache / llm_gemini_code_assist.OAUTH_CREDENTIALS_FILE
        shutil.copy(real_oauth_file, test_oauth_file)

        # Also copy project_id cache if it exists
        real_project_cache = real_plugin_cache / llm_gemini_code_assist.PROJECT_ID_CACHE_FILE
        if real_project_cache.exists():
            test_project_cache = test_plugin_cache / llm_gemini_code_assist.PROJECT_ID_CACHE_FILE
            shutil.copy(real_project_cache, test_project_cache)

    return
