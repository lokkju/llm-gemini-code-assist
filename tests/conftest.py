import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from google.oauth2.credentials import Credentials

def before_record_request(request):
    if request.body:
        try:
            body = json.loads(request.body)
            if 'user_prompt_id' in body:
                body['user_prompt_id'] = 'CENSORED-USER-PROMPT-ID'
            request.body = json.dumps(body).encode('utf-8')
        except (json.JSONDecodeError, AttributeError):
            pass
    return request

@pytest.fixture(scope="module")
def vcr_config():
    return {
        # 1. This prevents your secret token from being saved in the YAML file
        "filter_headers": [
            ("authorization", "CENSORED-OAUTH-TOKEN")
        ],
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

@pytest.fixture(autouse=True)
def mock_oauth_for_vcr_tests(request):
    """
    Automatically mock OAuth credentials and project ID cache for VCR tests.

    This fixture:
    1. Detects if the test is marked with @pytest.mark.vcr
    2. Checks if we're in playback mode (cassette exists or --record-mode=none)
    3. Mocks get_oauth_credentials() to return mock credentials
    4. Mocks _load_project_id_cache() to return cached project ID

    This allows tests to run without requiring actual credential files when
    VCR cassettes are available.
    """
    # Check if test is marked with vcr
    vcr_marker = request.node.get_closest_marker('vcr')
    if not vcr_marker:
        yield
        return

    # Check if we should mock (cassette exists or --record-mode=none)
    record_mode = request.config.getoption('--record-mode', default='once')
    should_mock = record_mode in ['none', 'once']

    # For 'once' mode, also check if cassette exists
    if record_mode == 'once':
        cassette_dir = os.path.join(
            os.path.dirname(__file__),
            'cassettes',
            request.node.parent.name,
        )
        cassette_path = os.path.join(cassette_dir, f"{request.node.name}.yaml")
        should_mock = os.path.exists(cassette_path)

    if should_mock:
        # Create mock credentials
        mock_creds = MagicMock(spec=Credentials)
        mock_creds.token = "mock-oauth-token"
        mock_creds.id_token = "mock-id-token"
        mock_creds.refresh_token = "mock-refresh-token"
        mock_creds.token_uri = "https://oauth2.googleapis.com/token"
        mock_creds.valid = True
        mock_creds.expiry = datetime.now() + timedelta(hours=1)

        # Mock get_oauth_credentials
        with patch('llm_gemini_code_assist.get_oauth_credentials', return_value=mock_creds):
            # Mock _load_project_id_cache to return a cached project ID
            # This prevents the API call to get project assignment
            mock_cache = {
                "mock-user@example.com": ("prismatic-acronym-xzpzh", "PREMIUM")
            }
            with patch('llm_gemini_code_assist._load_project_id_cache', return_value=mock_cache):
                # Mock get_user_email to return consistent email
                with patch('llm_gemini_code_assist.get_user_email', return_value="mock-user@example.com"):
                    yield
    else:
        # Not in playback mode, use real credentials
        yield
